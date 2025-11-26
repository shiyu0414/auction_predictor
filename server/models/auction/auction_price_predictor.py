import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import pickle
import os

warnings.filterwarnings('ignore')


class AuctionPricePredictor:
    def __init__(self, config=None):
        self.model = None
        self.feature_columns = None
        self.df = None
        self.scaler = StandardScaler()
        self.config = config if config is not None else {}
        
        # 默认配置参数，用于降低预测值
        self.default_config = {
            'model_params': {
                'objective': 'reg:squarederror',
                'n_estimators': 90,  # 减少树的数量 (原为100)
                'max_depth': 4,  # 降低树的深度 (原为6)
                'learning_rate': 0.08,  # 降低学习率 (原为0.1)
                'random_state': 42
            },
            'decrease_factor': 0.85,  # 降低下调因子 (原为0.9)
            'price_validation': {
                'min_multiplier': 0.5,
                'max_multiplier': 1.3,  # 降低价格上限倍数
                'upper_bound_std_multiplier': 2.0  # 降低标准差倍数
            }
        }

    def load_data(self, file_path):
        """加载拍卖数据"""
        print(f"DEBUG: 正在从 {file_path} 加载数据...")
        self.df = pd.read_csv(file_path)
        print(f"DEBUG: 加载了 {len(self.df)} 条记录")
        print(f"DEBUG: 列名: {list(self.df.columns)}")
        
        # 数据预处理
        self.df['createtime'] = pd.to_datetime(self.df['createtime'], dayfirst=True)
        # 将空字符串转换为NaN
        self.df['transaction_price'] = pd.to_numeric(self.df['transaction_price'], errors='coerce')
        # 标记是否流拍（transaction_price为空或0）
        self.df['is_failed'] = (self.df['transaction_price'].isna()) | (self.df['transaction_price'] == 0)
        
        print(f"DEBUG: 流拍记录数: {self.df['is_failed'].sum()}")
        print(f"DEBUG: 成功记录数: {(~self.df['is_failed']).sum()}")

    def _create_features_for_training(self):
        """
        为训练创建特征
        """
        if self.df is None:
            raise ValueError("请先加载数据")

        # 按照productid和quantity分组，按时间排序
        grouped = self.df.sort_values(['productid', 'quantity', 'createtime']).groupby(['productid', 'quantity'])

        features_list = []
        for name, group in grouped:
            group = group.reset_index(drop=True)

            for i in range(1, len(group)):
                # 当前记录是流拍记录才需要预测下次起拍价
                if group.loc[i - 1, 'is_failed']:
                    # 获取到目前为止的历史成交价格
                    past_transactions = group.loc[:i - 1].loc[
                        ~group.loc[:i - 1, 'is_failed'], 'transaction_price']

                    if len(past_transactions) > 0:
                        # 计算该产品的整体统计信息
                        product_transactions = self.df[
                            (self.df['productid'] == group.loc[i - 1, 'productid']) &
                            (self.df['quantity'] == group.loc[i - 1, 'quantity']) &
                            (~self.df['is_failed'])
                            ]['transaction_price']

                        record = {
                            'productid': group.loc[i - 1, 'productid'],
                            'quantity': group.loc[i - 1, 'quantity'],
                            'last_starting_price': group.loc[i - 1, 'price'],
                            'history_avg_price': past_transactions.mean(),
                            'history_median_price': past_transactions.median(),
                            'history_min_price': past_transactions.min(),
                            'history_max_price': past_transactions.max(),
                            'history_std_price': past_transactions.std() if len(past_transactions) > 1 else 0,
                            'product_avg_price': product_transactions.mean() if len(
                                product_transactions) > 0 else np.nan,
                            'product_median_price': product_transactions.median() if len(
                                product_transactions) > 0 else np.nan,
                            'times_failed': group.loc[:i - 1, 'is_failed'].sum(),
                            'total_auctions': i,
                            'success_rate': (~group.loc[:i - 1, 'is_failed']).sum() / i if i > 0 else 0,
                            'target_price': group.loc[i, 'price']  # 下次起拍价作为目标
                        }
                        features_list.append(record)

        return pd.DataFrame(features_list)

    def optimize_hyperparameters(self, X, y, cv=3):
        """
        使用网格搜索优化XGBoost模型参数
        
        参数:
        X: 特征数据
        y: 目标变量
        cv: 交叉验证折数
        
        返回:
        最佳参数组合
        """
        print("开始参数调优...")

        # 定义参数网格
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.7, 0.8, 1.0],
            'colsample_bytree': [0.7, 0.8, 1.0]
        }

        # 初始化基础模型
        base_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42
        )

        # 标准化数据
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 执行网格搜索
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_scaled, y)

        # 获取最佳参数
        best_params = grid_search.best_params_
        print(f"参数调优完成")
        print(f"最佳参数: {best_params}")
        print(f"最佳CV分数: {-grid_search.best_score_:.2f}")

        return best_params

    def train(self, use_optimization=False, cv_folds=3):
        """训练模型"""
        if self.df is None:
            raise ValueError("请先加载数据")

        # 准备训练数据
        all_features = self._create_features_for_training()

        if all_features.empty:
            raise ValueError("没有足够的历史数据来创建特征")

        # 删除包含NaN的行
        all_features = all_features.dropna()

        if all_features.empty:
            raise ValueError("清理数据后没有可用的训练样本")

        # 分离特征和目标变量
        self.feature_columns = [col for col in all_features.columns if col not in ['target_price', 'productid']]
        X = all_features[self.feature_columns]
        y = all_features['target_price']

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 特征标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 交叉验证评估
        cv_metrics = self._cross_validate(X, y)
        print("交叉验证结果:")
        print(f"平均 MSE: {cv_metrics['avg_mse']:.2f} ± {cv_metrics['std_mse']:.2f}")
        print(f"平均 MAE: {cv_metrics['avg_mae']:.2f} ± {cv_metrics['std_mae']:.2f}")
        print(f"平均 R²: {cv_metrics['avg_r2']:.4f} ± {cv_metrics['std_r2']:.4f}")

        # 确定模型参数
        if use_optimization:
            # 使用参数调优
            best_params = self.optimize_hyperparameters(X, y, cv=cv_folds)
            # 确保objective参数存在
            best_params['objective'] = 'reg:squarederror'
            best_params['random_state'] = 42

            # 使用最佳参数创建模型
            self.model = xgb.XGBRegressor(**best_params)
        else:
            # 使用配置中的参数或默认参数
            model_params = self.config.get('model_params', self.default_config['model_params'])
            self.model = xgb.XGBRegressor(**model_params)

        self.model.fit(X_train_scaled, y_train)

        # 在测试集上预测并评估
        y_pred = self.model.predict(X_test_scaled)
        test_mse = mean_squared_error(y_test, y_pred)
        test_mae = mean_absolute_error(y_test, y_pred)
        test_r2 = r2_score(y_test, y_pred)

        print("\n最终模型测试结果:")
        print(f"测试集 MSE: {test_mse:.2f}")
        print(f"测试集 MAE: {test_mae:.2f}")
        print(f"测试集 R²: {test_r2:.4f}")

        # 特征重要性
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return {
            'mse': test_mse,
            'mae': test_mae,
            'r2': test_r2,
            'feature_importance': feature_importance,
            'cv_metrics': cv_metrics
        }

    def predict_next_starting_price(self, product_id, quantity, last_starting_price):
        """
        预测特定产品的下次起拍价（优化版本）

        参数:
        product_id: 产品ID
        quantity: 数量
        last_starting_price: 上次起拍价

        返回:
        预测的下次起拍价和相关信息（包含置信度等）
        """
        if self.model is None:
            raise ValueError("模型未训练完成，无法进行预测")

        # 输入验证和边界检查
        if last_starting_price <= 0:
            raise ValueError("上次起拍价必须大于0")
        if quantity <= 0:
            raise ValueError("拍卖数量必须大于0")

        # 从原始数据中查找该产品的历史信息
        # 处理产品ID和数量的数据类型问题
        print(f"DEBUG: 正在查找 product_id={product_id}, quantity={quantity}")
        
        # 确保product_id和quantity转换为适当的类型进行匹配
        try:
            # 尝试将product_id转换为整数进行匹配
            product_id_for_match = int(product_id)
            product_match = (self.df['productid'] == product_id_for_match)
        except (ValueError, TypeError):
            # 如果不能转换为整数，则使用字符串匹配
            product_match = (self.df['productid'] == str(product_id))
            
        try:
            # 尝试将quantity转换为整数进行匹配
            quantity_for_match = int(quantity)
            quantity_match = (self.df['quantity'] == quantity_for_match)
        except (ValueError, TypeError):
            # 如果不能转换为整数，则使用字符串匹配
            quantity_match = (self.df['quantity'] == str(quantity))
        
        print(f"DEBUG: 匹配产品ID的记录数: {product_match.sum()}")
        print(f"DEBUG: 匹配数量的记录数: {quantity_match.sum()}")
        
        product_history = self.df[product_match & quantity_match]
        has_history = not product_history.empty
        history_count = len(product_history)
        
        print(f"DEBUG: 最终匹配的记录数: {history_count}")
        if has_history:
            print("DEBUG: 匹配的记录:")
            print(product_history[['id', 'productid', 'quantity', 'price', 'transaction_price', 'is_failed']].head().to_string())

        # 获取历史成交价格
        transaction_prices = product_history[~product_history['is_failed']]['transaction_price'].dropna().tolist()
        failed_count = product_history['is_failed'].sum()
        total_auctions = len(product_history)

        successful_records = product_history[~product_history['is_failed']]
        # print(f"DEBUG: 成功交易记录数: {self.df}")

        # 创建特征字典
        features = {
            'quantity': quantity,
            'last_starting_price': last_starting_price,
        }

        if has_history:
            # 计算历史统计特征
            if transaction_prices:
                prices_array = np.array(transaction_prices)
                features['history_avg_price'] = np.mean(prices_array)
                features['history_median_price'] = np.median(prices_array)
                features['history_min_price'] = np.min(prices_array)
                features['history_max_price'] = np.max(prices_array)
                features['history_std_price'] = np.std(prices_array) if len(prices_array) > 1 else 0
            else:
                features['history_avg_price'] = last_starting_price
                features['history_median_price'] = last_starting_price
                features['history_min_price'] = last_starting_price
                features['history_max_price'] = last_starting_price
                features['history_std_price'] = 0

            # 计算该产品的整体成交价格统计
            product_transactions = self.df[
                (self.df['productid'] == product_id) &
                (self.df['quantity'] == quantity) &
                (~self.df['is_failed'])
                ]['transaction_price'].dropna()

            features['product_avg_price'] = product_transactions.mean() if len(product_transactions) > 0 else np.nan
            features['product_median_price'] = product_transactions.median() if len(
                product_transactions) > 0 else np.nan

            # 添加成功相关特征
            features['times_failed'] = failed_count
            features['total_auctions'] = total_auctions
            features['success_rate'] = (total_auctions - failed_count) / total_auctions if total_auctions > 0 else 0

            # 新增特征：价格偏差率
            avg_price = features['history_avg_price']
            if avg_price > 0:
                features['price_deviation_ratio'] = min(3.0, max(0.3, last_starting_price / avg_price))
            else:
                features['price_deviation_ratio'] = 1.0

            # 新增特征：失败趋势（最近3次拍卖）
            recent_auctions = min(3, total_auctions)
            if recent_auctions > 0:
                recent_failures = product_history.tail(recent_auctions)['is_failed'].sum()
                features['recent_failure_rate'] = recent_failures / recent_auctions
            else:
                features['recent_failure_rate'] = 0
                
            # 新增特征：长期趋势（与整体平均比较）
            if features['product_avg_price'] is not None and features['product_avg_price'] > 0:
                features['long_term_trend'] = features['history_avg_price'] / features['product_avg_price']
            else:
                features['long_term_trend'] = 1.0
        else:
            # 如果没有历史数据，使用默认值
            features['history_avg_price'] = last_starting_price
            features['history_median_price'] = last_starting_price
            features['history_min_price'] = last_starting_price
            features['history_max_price'] = last_starting_price
            features['history_std_price'] = 0
            features['product_avg_price'] = last_starting_price
            features['product_median_price'] = last_starting_price
            features['times_failed'] = 0
            features['total_auctions'] = 0
            features['success_rate'] = 0
            features['price_deviation_ratio'] = 1.0
            features['recent_failure_rate'] = 0
            features['long_term_trend'] = 1.0

        # 按照训练时的特征顺序构造特征向量
        try:
            features_np = np.array([[features.get(col, 0) for col in self.feature_columns]])
        except KeyError as e:
            raise KeyError(f"特征不匹配: {e}. 训练时的特征列: {self.feature_columns}")

        # 检测异常值
        features_np = self._detect_and_handle_outliers(features_np)

        # 应用特征标准化
        features_scaled = self.scaler.transform(features_np)

        # 预测
        predicted_price = self.model.predict(features_scaled)[0]

        # 结果合理性验证和调整
        predicted_price = self._validate_prediction(predicted_price, features)

        # 计算置信度
        confidence = self._calculate_confidence(features, has_history, history_count, transaction_prices)

        # 构建返回信息
        market_condition = self._determine_market_condition(features)
        prediction_basis = self._generate_prediction_basis(features, has_history, market_condition)

        info = {
            # 预测方法：如果有历史数据和成交记录则使用模型预测，否则使用简单降价策略
            'method': 'model_prediction' if has_history and transaction_prices else 'simple_decrease',
            # 历史成交记录数量
            'historical_transactions': len(transaction_prices),
            # 历史平均成交价格
            'history_avg_price': features['history_avg_price'],
            # 历史最低成交价格
            'history_min_price': features['history_min_price'],
            # 历史最高成交价格
            'history_max_price': features['history_max_price'],
            # 拍卖成功率（成交次数/总拍卖次数）
            'success_rate': features['success_rate'],
            # 预测置信度（0-1之间的值）
            'confidence': confidence,
            # 市场状况评估（strong/weak/declining/stable/very_weak）
            'market_condition': market_condition,
            # 预测依据的文本描述
            'prediction_basis': prediction_basis,
            # 是否有历史数据
            'has_history_data': has_history,
            # 历史记录总数
            'history_record_count': history_count
        }

        return predicted_price, info

    def _detect_and_handle_outliers(self, features):
        """
        检测并处理特征中的异常值
        """
        # 对数值进行边界处理
        result = features.copy()
        result = np.clip(result, -1e6, 1e6)  # 设置合理的上下限
        return result

    def _validate_prediction(self, predicted_price, features):
        """
        验证预测结果的合理性并进行必要的调整
        """
        # 确保价格为正数
        predicted_price = max(0.01, predicted_price)

        # 获取配置中的价格验证参数或使用默认参数
        price_validation_config = self.config.get('price_validation', self.default_config['price_validation'])
        decrease_factor = self.config.get('decrease_factor', self.default_config['decrease_factor'])

        # 基于历史价格范围进行调整
        avg_price = features.get('history_avg_price', 0)
        std_price = features.get('history_std_price', 0)
        last_price = features.get('last_starting_price', 0)

        # 如果有历史数据，使用历史价格范围约束预测结果
        if avg_price > 0:
            # 动态计算边界，考虑市场趋势
            lower_bound = max(avg_price * 0.3, avg_price - 3.0 * std_price)
            upper_bound = avg_price + price_validation_config['upper_bound_std_multiplier'] * std_price

            # 应用软约束
            if predicted_price < lower_bound:
                predicted_price = lower_bound + (predicted_price - lower_bound) * 0.2
            elif predicted_price > upper_bound:
                predicted_price = upper_bound - (predicted_price - upper_bound) * 0.1

        # 确保预测价格与上次起拍价有合理的关系
        if last_price > 0:
            min_reasonable_price = last_price * price_validation_config['min_multiplier']
            max_reasonable_price = last_price * price_validation_config['max_multiplier']
            predicted_price = max(min_reasonable_price, min(max_reasonable_price, predicted_price))

            # 根据市场条件调整价格
            # 在下降市场中，更倾向于降价
            if predicted_price > last_price:
                excess = (predicted_price - last_price) / last_price
                # 应用更积极的下调策略
                predicted_price = last_price + (predicted_price - last_price) * max(decrease_factor, 1.0 - excess * 0.7)

        return predicted_price

    def _calculate_confidence(self, features, has_history, history_count, transaction_prices):
        """
        计算预测置信度
        """
        # 基础置信度基于是否有历史数据
        if has_history and transaction_prices:
            base_confidence = 0.7
            
            # 基于历史数据量调整（数据越多，置信度越高）
            history_factor = min(history_count / 100, 1.0) * 0.15

            # 基于价格波动性调整（波动越小，置信度越高）
            volatility = features['history_std_price'] / (features['history_avg_price'] or 1)
            volatility_factor = max(0, 0.1 - volatility * 0.15)

            # 基于成功率调整
            success_rate = features['success_rate']
            success_factor = (success_rate - 0.5) * 0.1 if success_rate > 0.5 else -(0.5 - success_rate) * 0.05

            confidence = base_confidence + history_factor + volatility_factor + success_factor
        else:
            # 无历史数据或无成交记录时降低置信度
            confidence = 0.4

        # 确保置信度在合理范围内
        return min(0.95, max(0.1, confidence))

    def _determine_market_condition(self, features):
        """
        确定当前市场状况
        """
        # 基于成功率和失败趋势判断市场状况
        success_rate = features['success_rate']
        recent_failure_rate = features['recent_failure_rate']

        if recent_failure_rate > 0.66:
            return 'declining'
        elif success_rate > 0.8:
            return 'strong'
        elif success_rate > 0.6:
            return 'stable'
        elif success_rate > 0.4:
            return 'weak'
        else:
            return 'very_weak'

    def _generate_prediction_basis(self, features, has_history, market_condition):
        """
        生成预测依据的描述
        """
        basis_parts = []

        if has_history:
            basis_parts.append("基于历史交易数据分析")

            # 根据市场状况添加说明
            if market_condition == 'strong':
                basis_parts.append("当前市场状况良好")
            elif market_condition == 'weak':
                basis_parts.append("当前市场状况较弱")
            elif market_condition == 'declining':
                basis_parts.append("近期拍卖成功率下降")
            else:
                basis_parts.append("市场相对稳定")

            # 考虑价格偏差
            if features['price_deviation_ratio'] > 1.5:
                basis_parts.append("上次起拍价高于历史平均水平")
            elif features['price_deviation_ratio'] < 0.7:
                basis_parts.append("上次起拍价低于历史平均水平")
        else:
            basis_parts.append("基于当前拍卖参数和类似产品模式")
            basis_parts.append("缺乏历史数据，建议适当调整")

        return ", ".join(basis_parts)

    def save_model(self, model_path):
        """
        保存模型、标准化器和特征列信息到指定路径
        
        参数:
        model_path: 模型保存路径（不包含扩展名）
        
        返回:
        bool: 保存是否成功
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)

            # 保存模型组件
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns
            }

            with open(f"{model_path}.pkl", 'wb') as f:
                pickle.dump(model_data, f)

            print(f"模型已成功保存到 {model_path}.pkl")
            return True
        except Exception as e:
            print(f"保存模型时出错: {e}")
            return False

    def load_model(self, model_path):
        """
        从指定路径加载模型、标准化器和特征列信息
        
        参数:
        model_path: 模型加载路径（不包含扩展名）
        
        返回:
        bool: 加载是否成功
        """
        try:
            with open(f"{model_path}.pkl", 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']

            print(f"模型已成功从 {model_path}.pkl 加载")
            print(f"加载的特征列数量: {len(self.feature_columns)}")
            return True
        except Exception as e:
            print(f"加载模型时出错: {e}")
            return False

    def _cross_validate(self, X, y, n_splits=5):
        """
        执行KFold交叉验证，评估模型性能
        
        参数:
        X: 特征数据
        y: 目标变量
        n_splits: 交叉验证的折数
        
        返回:
        包含平均和标准差性能指标的字典
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        mse_scores = []
        mae_scores = []
        r2_scores = []

        for train_idx, val_idx in kf.split(X):
            X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
            y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

            # 标准化
            scaler_cv = StandardScaler()
            X_train_scaled = scaler_cv.fit_transform(X_train_cv)
            X_val_scaled = scaler_cv.transform(X_val_cv)

            # 训练模型
            model_cv = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            model_cv.fit(X_train_scaled, y_train_cv)

            # 预测和评估
            y_pred_cv = model_cv.predict(X_val_scaled)
            mse_scores.append(mean_squared_error(y_val_cv, y_pred_cv))
            mae_scores.append(mean_absolute_error(y_val_cv, y_pred_cv))
            r2_scores.append(r2_score(y_val_cv, y_pred_cv))

        # 计算平均和标准差
        return {
            'avg_mse': np.mean(mse_scores),
            'std_mse': np.std(mse_scores),
            'avg_mae': np.mean(mae_scores),
            'std_mae': np.std(mae_scores),
            'avg_r2': np.mean(r2_scores),
            'std_r2': np.std(r2_scores),
            'all_mse': mse_scores,
            'all_mae': mae_scores,
            'all_r2': r2_scores
        }


# # 使用示例
# if __name__ == "__main__":
#     # 使用示例 - 修改这些参数来降低预测值
#
#     # 调整配置以降低预测值
#     config = {
#         'features': [
#             'last_starting_price',
#             'history_avg_price',
#             'history_median_price',
#             'times_failed',
#             'success_rate',
#             'quantity'
#             # 移除了可能导致预测值偏高的特征: history_max_price, history_std_price
#         ],
#         'model_params': {
#             'objective': 'reg:squarederror',
#             'n_estimators': 90,  # 减少树的数量 (原为100)
#             'max_depth': 4,  # 降低树的深度 (原为6)
#             'learning_rate': 0.08,  # 降低学习率 (原为0.1)
#             'random_state': 42
#         },
#         'decrease_factor': 0.85  # 降低下调因子 (原为0.9)
#     }
#
#     # 初始化预测器
# predictor = AuctionPricePredictor(config)
#
# # 加载数据
# print("正在加载数据...")
# try:
#     predictor.load_data(r'C:\Users\P15\Desktop\auction.csv')
#     print("数据加载完成")
# except Exception as e:
#     print(f"加载数据时出错: {e}")
#     exit(1)
#
# # 训练模型（可以选择是否使用参数调优）
# print("正在训练模型...")
# try:
#     use_tuning = False  # 设置为True启用参数调优
#     metrics = predictor.train(use_optimization=use_tuning)
#     print(f"模型训练完成:")
#     print(f"  均方误差 (MSE): {metrics['mse']:.2f}")
#     print(f"  平均绝对误差 (MAE): {metrics['mae']:.2f}")
#     print("\n特征重要性:")
#     print(metrics['feature_importance'].head(10))
#
#     # 保存模型
#     predictor.save_model("auction_price_model")
# except Exception as e:
#     print(f"训练模型时出错: {e}")
#     exit(1)
#
# # 针对具体数据进行预测
# # 示例：产品ID: 1868477355743612930，数量: 5000，当前起拍价: 280.0000
# print("\n针对流拍数据进行预测:")
# product_id = 1868477355743612930
# quantity = 5000
# last_price = 280.0000
#
# try:
#     predicted_price, info = predictor.predict_next_starting_price(product_id, quantity, last_price)
#     print(f"产品ID: {product_id}")
#     print(f"数量: {quantity}")
#     print(f"上次起拍价: {last_price}")
#     print(f"预测下次起拍价: {predicted_price:.4f}")
#     print(f"价格调整: {predicted_price - last_price:+.4f} ({((predicted_price / last_price - 1) * 100):+.2f}%)")
#     print(f"预测方法: {info['method']}")
#     if info['method'] == 'model_prediction':
#         print(f"历史成交次数: {info['historical_transactions']}")
#         print(f"历史平均成交价: {info['history_avg_price']:.4f}")
#         print(f"成功率: {info['success_rate']:.2%}")
# except Exception as e:
#     print(f"预测时出错: {e}")

# 示例：加载模型（实际使用时）
# new_predictor = AuctionPricePredictor()
# new_predictor.load_model("auction_price_model")
# new_predicted_price, new_info = new_predictor.predict_next_starting_price(...)

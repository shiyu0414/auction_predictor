import os
import sys
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.models.auction.auction_price_predictor import AuctionPricePredictor

class TestAuctionPricePredictor(unittest.TestCase):
    """
    拍卖价格预测模型测试类
    """
    
    def setUp(self):
        """
        测试前的设置
        """
        self.predictor = AuctionPricePredictor()
        
        # 创建测试数据
        self.test_data = pd.DataFrame({
            'productid': [12345, 12345, 12345, 67890, 67890],
            'quantity': [1000, 1000, 1000, 2000, 2000],
            'price': [100.0, 90.0, 85.0, 200.0, 180.0],
            'createtime': pd.to_datetime(['2023-01-01', '2023-01-15', '2023-02-01', '2023-01-01', '2023-01-15']),
            'transaction_price': [95.0, 0, 85.0, 0, 175.0]
        })
        
        # 模拟数据加载
        with patch.object(self.predictor, 'load_data', return_value=True) as mock_load:
            self.predictor.load_data('dummy_path.csv')
            self.predictor.df = self.test_data.copy()
            mock_load.assert_called_once_with('dummy_path.csv')
    
    def test_data_preprocessing(self):
        """
        测试数据预处理功能
        """
        # 检查数据预处理
        self.predictor.df['createtime'] = pd.to_datetime(self.predictor.df['createtime'])
        self.predictor.df['is_failed'] = (self.predictor.df['transaction_price'].isna()) | \
                                       (self.predictor.df['transaction_price'] == 0)
        
        # 验证预处理结果
        self.assertTrue('is_failed' in self.predictor.df.columns)
        self.assertEqual(self.predictor.df['is_failed'].sum(), 2)  # 两条流拍记录
        
        # 验证时间特征
        self.predictor.df['month'] = self.predictor.df['createtime'].dt.month
        self.predictor.df['season'] = (self.predictor.df['month'] - 1) // 3 + 1
        
        self.assertTrue('month' in self.predictor.df.columns)
        self.assertTrue('season' in self.predictor.df.columns)
    
    def test_feature_creation(self):
        """
        测试特征创建功能
        """
        # 预处理数据
        self.predictor.df['is_failed'] = (self.predictor.df['transaction_price'].isna()) | \
                                       (self.predictor.df['transaction_price'] == 0)
        self.predictor.df['month'] = self.predictor.df['createtime'].dt.month
        self.predictor.df['season'] = (self.predictor.df['month'] - 1) // 3 + 1
        
        # 手动创建特征数据用于测试
        self.predictor.train_data = pd.DataFrame({
            'productid': [12345, 67890],
            'quantity': [1000, 2000],
            'last_starting_price': [100.0, 200.0],
            'avg_transaction_price': [95.0, 0.0],
            'median_transaction_price': [95.0, 0.0],
            'min_transaction_price': [95.0, 0.0],
            'max_transaction_price': [95.0, 0.0],
            'std_transaction_price': [0.0, 0.0],
            'last_transaction_price': [95.0, 0.0],
            'price_change_rate': [0.0, 0.0],
            'times_failed': [0, 0],
            'total_auctions': [1, 1],
            'success_rate': [1.0, 0.0],
            'consecutive_fails': [0, 0],
            'avg_failure_interval': [0.0, 0.0],
            'month': [1, 1],
            'season': [1, 1],
            'days_since_last_auction': [0, 0],
            'target_price': [90.0, 180.0]
        })
        
        # 验证特征数据
        self.assertEqual(len(self.predictor.train_data), 2)
        self.assertTrue('last_starting_price' in self.predictor.train_data.columns)
        self.assertTrue('target_price' in self.predictor.train_data.columns)
    
    @patch('server.models.aaaa.xgb.XGBRegressor')
    def test_model_training(self, mock_xgb):
        """
        测试模型训练功能
        """
        # 设置模拟的训练数据
        self.predictor.train_data = pd.DataFrame({
            'productid': [12345, 67890],
            'last_starting_price': [100.0, 200.0],
            'avg_transaction_price': [95.0, 190.0],
            'success_rate': [0.8, 0.7],
            'target_price': [90.0, 180.0]
        })
        
        # 模拟模型
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([89.5, 179.0])
        mock_xgb.return_value = mock_model
        
        # 训练模型
        metrics = self.predictor.train_model()
        
        # 验证结果
        mock_xgb.assert_called_once()
        mock_model.fit.assert_called_once()
        self.assertTrue('mse' in metrics)
        self.assertTrue('mae' in metrics)
        self.assertTrue('r2' in metrics)
    
    def test_predict_with_no_model(self):
        """
        测试在模型未训练时的预测行为
        """
        # 当模型为None时，应该返回默认降价
        predicted_price, info = self.predictor.predict_next_starting_price(
            12345, 1000, 100.0
        )
        
        self.assertEqual(predicted_price, 90.0)  # 100.0 * 0.9
        self.assertEqual(info['status'], 'error')
        self.assertEqual(info['confidence'], 0.0)
    
    def test_predict_with_no_data(self):
        """
        测试在没有数据时的预测行为
        """
        # 没有历史数据的产品
        with patch.object(self.predictor, 'df', None):
            predicted_price, info = self.predictor.predict_next_starting_price(
                99999, 5000, 100.0
            )
            
            self.assertEqual(predicted_price, 90.0)
            self.assertEqual(info['status'], 'error')
    
    @patch('server.models.aaaa.xgb.XGBRegressor')
    def test_predict_with_model(self, mock_xgb):
        """
        测试有模型时的预测功能
        """
        # 设置模拟的训练数据和模型
        self.predictor.train_data = pd.DataFrame({
            'productid': [12345, 67890],
            'last_starting_price': [100.0, 200.0],
            'avg_transaction_price': [95.0, 190.0],
            'success_rate': [0.8, 0.7],
            'target_price': [90.0, 180.0]
        })
        
        # 训练模拟模型
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([89.5, 179.0])
        mock_xgb.return_value = mock_model
        
        self.predictor.feature_columns = ['last_starting_price', 'avg_transaction_price', 'success_rate']
        self.predictor.model = mock_model
        
        # 模拟特征标准化器
        self.predictor.scaler = MagicMock()
        self.predictor.scaler.transform.return_value = np.array([[100.0, 95.0, 0.8]])
        mock_model.predict.return_value = np.array([90.5])
        
        # 预测
        predicted_price, info = self.predictor.predict_next_starting_price(
            12345, 1000, 100.0
        )
        
        self.assertEqual(predicted_price, 90.5)
        self.assertEqual(info['status'], 'success')
        self.predictor.scaler.transform.assert_called_once()
        mock_model.predict.assert_called_once()
    
    def test_predict_with_no_transaction_history(self):
        """
        测试没有成交记录的预测行为
        """
        # 产品只有流拍记录
        product_history = pd.DataFrame({
            'productid': [54321, 54321],
            'quantity': [1000, 1000],
            'price': [100.0, 90.0],
            'transaction_price': [0, 0],
            'is_failed': [True, True],
            'createtime': pd.to_datetime(['2023-01-01', '2023-01-15']),
            'month': [1, 1],
            'season': [1, 1]
        })
        
        # 模拟产品查询返回只有流拍的历史
        with patch.object(self.predictor.df, 'query', return_value=product_history):
            # 模拟没有模型
            self.predictor.model = None
            
            # 第一次流拍，应该降价5%
            predicted_price, info = self.predictor.predict_next_starting_price(
                54321, 1000, 100.0
            )
            
            self.assertEqual(predicted_price, 95.0)  # 100.0 * 0.95
            self.assertEqual(info['status'], 'warning')
            self.assertEqual(info['confidence'], 0.4)
    
    def test_price_adjustment_strategy(self):
        """
        测试不同流拍次数的价格调整策略
        """
        # 测试不同流拍次数的降价幅度
        test_cases = [
            (1, 100.0, 95.0),  # 第一次流拍，降价5%
            (2, 100.0, 90.0),  # 第二次流拍，降价10%
            (3, 100.0, 85.0)   # 多次流拍，降价15%
        ]
        
        for failed_count, last_price, expected_price in test_cases:
            with self.subTest(failed_count=failed_count):
                # 创建模拟的产品历史
                product_history = pd.DataFrame({
                    'productid': [123] * failed_count,
                    'quantity': [100] * failed_count,
                    'price': [last_price] * failed_count,
                    'transaction_price': [0] * failed_count,
                    'is_failed': [True] * failed_count,
                    'createtime': pd.date_range('2023-01-01', periods=failed_count),
                    'month': [1] * failed_count,
                    'season': [1] * failed_count
                })
                
                # 模拟查询结果
                with patch.object(pd.DataFrame, '__getitem__', return_value=MagicMock(
                    __and__=MagicMock(return_value=product_history)
                )):
                    # 禁用模型
                    self.predictor.model = None
                    
                    price, _ = self.predictor.predict_next_starting_price(123, 100, last_price)
                    self.assertEqual(price, expected_price)
    
    def test_confidence_estimation(self):
        """
        测试预测置信度估计
        """
        # 模拟不同数据量的情况
        test_cases = [
            (3, 0.6),   # 少量数据，低置信度
            (7, 0.8),   # 中等数据量，中等置信度
            (12, 0.95)  # 大量数据，高置信度
        ]
        
        for data_count, expected_confidence in test_cases:
            with self.subTest(data_count=data_count):
                # 创建模拟数据
                product_history = pd.DataFrame({
                    'productid': [123] * data_count,
                    'quantity': [100] * data_count,
                    'price': [100.0] * data_count,
                    'transaction_price': [95.0] * data_count,
                    'is_failed': [False] * data_count,
                    'createtime': pd.date_range('2023-01-01', periods=data_count),
                    'month': [1] * data_count,
                    'season': [1] * data_count
                })
                
                # 模拟查询和预测
                with patch.object(pd.DataFrame, '__getitem__', return_value=MagicMock(
                    __and__=MagicMock(return_value=product_history)
                )):
                    # 设置模型
                    self.predictor.model = MagicMock()
                    self.predictor.model.predict.return_value = np.array([95.0])
                    self.predictor.feature_columns = ['last_starting_price']
                    self.predictor.scaler = MagicMock()
                    self.predictor.scaler.transform.return_value = np.array([[100.0]])
                    
                    _, info = self.predictor.predict_next_starting_price(123, 100, 100.0)
                    self.assertEqual(info['confidence'], expected_confidence)

    def test_invalid_price_handling(self):
        """
        测试无效价格处理
        """
        # 设置模型返回无效价格
        self.predictor.model = MagicMock()
        self.predictor.model.predict.return_value = np.array([-10.0])  # 负数价格
        self.predictor.feature_columns = ['last_starting_price']
        self.predictor.scaler = MagicMock()
        self.predictor.scaler.transform.return_value = np.array([[100.0]])
        
        # 创建模拟数据
        product_history = pd.DataFrame({
            'productid': [123],
            'quantity': [100],
            'price': [100.0],
            'transaction_price': [95.0],
            'is_failed': [False],
            'createtime': ['2023-01-01'],
            'month': [1],
            'season': [1]
        })
        
        with patch.object(pd.DataFrame, '__getitem__', return_value=MagicMock(
            __and__=MagicMock(return_value=product_history)
        )):
            price, info = self.predictor.predict_next_starting_price(123, 100, 100.0)
            
            # 应该调整为默认降价
            self.assertEqual(price, 90.0)
            self.assertEqual(info['status'], 'adjusted')
            self.assertEqual(info['confidence'], 0.5)

if __name__ == '__main__':
    # 运行所有测试
    unittest.main()
    
    # 或者运行特定测试
    # test_suite = unittest.TestSuite()
    # test_suite.addTest(TestAuctionPricePredictor('test_predict_with_model'))
    # unittest.TextTestRunner().run(test_suite)
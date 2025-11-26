from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import logging
import os

# 导入日志工具
from util.log import get_logger
# 导入预测模型
from server.models.auction.auction_price_predictor import AuctionPricePredictor

# 创建日志记录器
logger = get_logger("server.routes.auction")

# 创建路由实例
router = APIRouter(prefix="/api/auction", tags=["auction"])

# 初始化预测器
predictor = None

# 模型文件路径
MODEL_FILE_PATH = os.path.join("models", "auction_price_model")
TRAINING_DATA_PATH =os.path.join("data", "auction.csv")

def init_predictor():
    """
    初始化预测器实例

    Returns:
        AuctionPricePredictor: 预测器实例
    """
    global predictor
    if predictor is None:
        try:
            config = {}
            # 创建新的预测器实例
            predictor = AuctionPricePredictor(config)

            # 尝试加载已保存的模型
            if os.path.exists(MODEL_FILE_PATH):
                success = predictor.load_model(MODEL_FILE_PATH)
                if success:
                    logger.info("从 %s 加载现有模型成功", MODEL_FILE_PATH)
                else:
                    logger.warning("从 %s 加载模型失败", MODEL_FILE_PATH)
                    # 如果加载失败，尝试训练新模型
                    train_new_model(predictor)
            else:
                # 如果没有保存的模型，使用默认数据训练新模型
                logger.info("模型文件 %s 不存在，训练新模型", MODEL_FILE_PATH)
                train_new_model(predictor)

        except Exception as e:
            logger.error("初始化预测器失败: %s", str(e))
            # 即使出错也要创建基本实例
            predictor = AuctionPricePredictor(config)
            # 尝试训练新模型
            try:
                train_new_model(predictor)
            except Exception as train_e:
                logger.error("训练新模型也失败: %s", str(train_e))
    return predictor

def train_new_model(predictor_instance):
    """
    训练新模型
    
    Args:
        predictor_instance: 预测器实例
        
    Raises:
        FileNotFoundError: 当训练数据文件不存在时
    """
    if os.path.exists(TRAINING_DATA_PATH):
        logger.info("使用 %s 的数据训练新模型", TRAINING_DATA_PATH)
        predictor_instance.load_data(TRAINING_DATA_PATH)
        metrics = predictor_instance.train(use_optimization=False)
        success = predictor_instance.save_model(MODEL_FILE_PATH)
        if success:
            logger.info("新模型训练并保存成功，MSE: %.2f", metrics['mse'])
        else:
            logger.warning("新模型训练成功但保存失败，MSE: %.2f", metrics['mse'])
    else:
        logger.warning("找不到训练数据文件: %s", TRAINING_DATA_PATH)
        raise FileNotFoundError(f"训练数据文件不存在: {TRAINING_DATA_PATH}")

# 定义请求模型
class AuctionPredictRequest(BaseModel):
    """
    拍卖价格预测请求模型
    """
    product_id: str = Field(..., description="产品ID")
    quantity: int = Field(..., gt=0, description="数量，必须大于0")
    last_price: float = Field(..., ge=0, description="上次拍卖价格，必须大于等于0")

# 定义响应模型
class AuctionPredictResponse(BaseModel):
    """
    拍卖价格预测响应模型
    """
    success: bool = Field(..., description="请求是否成功")
    predicted_price: float = Field(..., description="预测的拍卖价格")
    # price_reduction: Optional[float] = Field(None, description="价格减少金额")
    price_reduction_percentage: Optional[str] = Field(None, description="价格减少百分比")
    confidence: Optional[float] = Field(None, description="预测置信度")
    prediction_basis: Optional[str] = Field(None, description="预测依据详情")
    market_condition: Optional[str] = Field(None, description="市场状况评估")
    historical_transactions: Optional[int] = Field(None, description="历史成交次数")
    has_history_data: Optional[bool] = Field(None, description="是否有历史数据")
    message: str = Field("", description="响应消息")

@router.post("/predict", response_model=AuctionPredictResponse)
async def predict_auction_price(request: AuctionPredictRequest) -> AuctionPredictResponse:
    """
    预测拍卖价格接口

    根据产品ID、数量和上次拍卖价格，预测下一次拍卖的起拍价格。

    Args:
        request: 包含预测所需参数的请求对象
            product_id: 产品ID
            quantity: 拍卖数量
            last_price: 上次拍卖价格

    Returns:
        AuctionPredictResponse: 包含预测结果和依据的响应对象

    Raises:
        HTTPException: 当预测过程中发生错误时
    """
    logger.info(
        "收到预测请求: product_id=%s, quantity=%d, last_price=%.2f",
        request.product_id,
        request.quantity,
        request.last_price
    )

    try:
        # 确保预测器已初始化
        predictor_instance = init_predictor()

        # 检查数据是否已加载
        if predictor_instance.df is None:
            logger.warning("数据未加载，尝试重新加载训练数据")
            try:
                if os.path.exists(TRAINING_DATA_PATH):
                    predictor_instance.load_data(TRAINING_DATA_PATH)
                    logger.info("训练数据加载成功")
                else:
                    logger.error("训练数据文件不存在: %s", TRAINING_DATA_PATH)
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"训练数据文件不存在: {TRAINING_DATA_PATH}"
                    )
            except Exception as load_error:
                logger.error("加载训练数据失败: %s", str(load_error))
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"加载训练数据失败: {str(load_error)}"
                )

        # 检查模型是否已训练
        if predictor_instance.model is None:
            logger.warning("模型未训练完成，尝试重新加载或训练")
            try:
                # 尝试重新加载模型
                if os.path.exists(MODEL_FILE_PATH):
                    success = predictor_instance.load_model(MODEL_FILE_PATH)
                    if not success:
                        # 如果加载失败，尝试重新训练
                        train_new_model(predictor_instance)
                else:
                    # 模型文件不存在，训练新模型
                    train_new_model(predictor_instance)
            except Exception as reload_e:
                logger.error("重新加载或训练模型失败: %s", str(reload_e))
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"模型不可用且无法重新加载或训练: {str(reload_e)}"
                )

        # 调用预测方法
        predicted_price, prediction_info = predictor_instance.predict_next_starting_price(
            product_id=request.product_id,
            quantity=request.quantity,
            last_starting_price=request.last_price
        )

        # 计算价格减少金额和减少百分比（保留一位小数）
        logger.debug(f"原始预测价格: {predicted_price}")
        predicted_price_rounded = round(float(predicted_price), 1)
        logger.debug(f"四舍五入后价格: {predicted_price_rounded}")
        price_reduction = request.last_price - predicted_price_rounded
        price_reduction_percentage_value = (price_reduction / request.last_price) * 100 if request.last_price > 0 else 0.0
        # 先四舍五入到两位小数，再格式化为一位小数的字符串
        price_reduction_percentage = f"{round(price_reduction_percentage_value, 2):.1f}%" if price_reduction_percentage_value != 0 else "0%"

        # 记录预测详细信息
        logger.debug(f"预测信息: product_id={request.product_id}, quantity={request.quantity}")
        logger.debug(f"  上次起拍价: {request.last_price}")
        logger.debug(f"  预测方法: {prediction_info['method']}")
        logger.debug(f"  是否有历史数据: {prediction_info['has_history_data']}")
        logger.debug(f"  历史成交记录数: {prediction_info['historical_transactions']}")
        logger.debug(f"  市场状况: {prediction_info['market_condition']}")
        logger.debug(f"  置信度: {prediction_info['confidence']:.2f}")
        logger.debug(f"  成功率: {prediction_info['success_rate']:.2f}")
        if prediction_info['has_history_data']:
            logger.debug(f"  历史平均价格: {prediction_info['history_avg_price']:.2f}")
            logger.debug(f"  历史价格范围: {prediction_info['history_min_price']:.2f} - {prediction_info['history_max_price']:.2f}")

        # 构建响应
        response = AuctionPredictResponse(
            success=True,
            predicted_price=predicted_price_rounded,
            price_reduction_percentage=price_reduction_percentage,
            confidence=prediction_info.get('confidence'),
            prediction_basis=prediction_info.get('prediction_basis'),
            market_condition=prediction_info.get('market_condition'),
            historical_transactions=prediction_info.get('historical_transactions'),
            has_history_data=prediction_info.get('has_history_data'),
            message="价格预测成功"
        )

        logger.info(
            "预测完成 product_id=%s: predicted_price=%.1f (%s)",
            request.product_id,
            response.predicted_price,
            response.price_reduction_percentage
        )

        return response

    except ValueError as e:
        logger.error("预测期间验证错误: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("预测期间发生错误: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"预测失败: {str(e)}"
        )

@router.get("/test")
async def test_auction_route():
    """
    测试拍卖路由是否正常工作

    Returns:
        dict: 包含成功消息的字典
    """
    logger.info("测试拍卖路由被调用")
    return {
        "message": "拍卖路由工作正常",
        "predictor_initialized": predictor is not None,
        "model_trained": predictor is not None and predictor.model is not None
    }

# 初始化预测器
init_predictor()

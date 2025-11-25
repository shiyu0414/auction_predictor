from uvicorn import run
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from typing import Any, Dict

from util.data_model import BaseRes, VersionRes, DataReq, DataRes
from util.conf import get_conf
from util.log import get_logger
from util.db import test_connection
# 正确的导入方式
from server.routes import users_router

# 初始化配置和日志
logger = get_logger("server.main")
conf = get_conf()
NAME = conf["server"]["name"]
VERSION = conf["server"]["version"]

# 创建FastAPI应用实例
app = FastAPI(
    title=NAME,
    version=VERSION,
    description="Auction Predictor API"
)

# 注册路由
app.include_router(users_router)
logger.info("Users router registered")

# 打印所有注册的路由进行调试
for route in app.routes:
    logger.info(f"Registered route: {route.path} - {route.methods}")

# 数据库连接测试
try:
    if test_connection():
        logger.info("Database connection successful")
    else:
        logger.warning("Database connection failed")
except Exception as e:
    logger.error(f"Database configuration error: {e}")

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """
    处理请求验证异常
    
    Args:
        request: 请求对象
        exc: 异常对象
        
    Returns:
        JSONResponse: 错误响应
    """
    logger.error(f"Request validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content=BaseRes(success=False, message=str(exc)).model_dump()
    )

@app.get("/")
async def api_home() -> BaseRes:
    """
    API根路径，返回欢迎信息和文档链接
    
    Returns:
        BaseRes: 包含欢迎信息和文档链接的响应
    """
    url = "http://" + conf["server"]["run"]["host"] + ":" + str(conf["server"]["run"]["port"]) + "/docs"
    return BaseRes(success=True, message=f"Welcome to {NAME}, docs in {url}")

@app.get("/version")
async def api_version() -> VersionRes:
    """
    获取API版本信息
    
    Returns:
        VersionRes: 包含版本号的响应
    """
    return VersionRes(version=VERSION, success=True, message="")

@app.get("/test/{data}")
async def api_get_data(data: str) -> DataRes:
    """
    测试GET端点，接收路径参数并返回
    
    Args:
        data (str): 从路径中获取的数据
        
    Returns:
        DataRes: 包含接收到数据的响应
    """
    logger.info(f"Received GET test data: {data}")
    try:
        res = DataRes.model_validate({
            "success": True,
            "message": "",
            "data": data
        })
    except Exception as e:
        logger.error(f"Error processing GET test data: {e}")
        res = DataRes(success=False, message=str(e))
    logger.info(f"Returning response: {res}")
    return res

@app.post("/test")
async def api_post_test(msg_req: DataReq) -> DataRes:
    """
    测试POST端点，接收请求体并返回
    
    Args:
        msg_req (DataReq): 包含请求数据的对象
        
    Returns:
        DataRes: 包含接收到数据的响应
    """
    logger.info(f"Received POST test data: {msg_req}")
    try:
        res = DataRes.model_validate({
            "success": True,
            "message": "",
            "data": msg_req.data
        })
    except Exception as e:
        logger.error(f"Error processing POST test data: {e}")
        res = DataRes(success=False, message=str(e))
    logger.info(f"Returning response: {res}")
    return res

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    健康检查端点
    
    Returns:
        Dict[str, Any]: 健康状态信息
    """
    db_status = "unknown"
    try:
        db_status = "connected" if test_connection() else "disconnected"
    except Exception:
        db_status = "error"
    
    return {
        "status": "healthy",
        "service": NAME,
        "version": VERSION,
        "database": db_status
    }

if __name__ == "__main__":
    # 启动服务器
    run("server.main:app", **conf["server"]["run"])

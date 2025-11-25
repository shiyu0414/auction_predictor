from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
import logging
from typing import Optional
from util.conf import get_conf

try:
    from urllib.parse import quote_plus
except ImportError:
    from urllib import quote_plus

Base = declarative_base()
logger = logging.getLogger(__name__)


class DatabaseConfig:
    def __init__(self):
        self.conf = get_conf()
        self.mysql_config = self.conf.get("database", {}).get("mysql", {})

    def get_connection_string(self) -> Optional[str]:
        """
        构建MySQL连接字符串
        格式: mysql+pymysql://user:password@host:port/database
        """
        if not self.mysql_config:
            logger.warning("MySQL configuration not found")
            return None

        user = self.mysql_config.get("user")
        password = self.mysql_config.get("password")
        host = self.mysql_config.get("host", "localhost")
        port = self.mysql_config.get("port", 3306)
        database = self.mysql_config.get("database")

        if not all([user, password, database]):
            logger.warning("Missing required MySQL configuration parameters")
            return None

        # 对用户名和密码进行URL编码，防止特殊字符导致解析错误
        encoded_user = quote_plus(user)
        encoded_password = quote_plus(password)

        return f"mysql+pymysql://{encoded_user}:{encoded_password}@{host}:{port}/{database}"

    def create_engine(self):
        """创建数据库引擎"""
        connection_string = self.get_connection_string()
        if not connection_string:
            raise ValueError("Unable to create database connection string")

        engine_options = self.mysql_config.get("engine_options", {})
        return create_engine(connection_string, **engine_options)

    def get_session_factory(self):
        """获取会话工厂"""
        engine = self.create_engine()
        session_options = self.mysql_config.get("session_options", {})
        return sessionmaker(bind=engine, **session_options)


# 全局数据库配置实例
db_config = DatabaseConfig()


def get_db_session() -> Session:
    """获取数据库会话"""
    session_factory = db_config.get_session_factory()
    return session_factory()


def test_connection():
    """测试数据库连接"""
    try:
        engine = db_config.create_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            return result.fetchone()[0] == 1
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False

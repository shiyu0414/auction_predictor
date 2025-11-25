from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Dict, Any

from util.db import get_db_session
from server.models.user import User

router = APIRouter(prefix="/api/users", tags=["users"])


# 添加一个简单的测试端点来验证路由是否注册
@router.get("/test")
def test_route():
    """
    测试端点
    
    用于验证用户路由是否正确注册和工作。
    
    Returns:
        dict: 包含成功消息的字典
    """
    return {"message": "Users route is working"}


@router.get("/", response_model=List[Dict[str, Any]])
def get_all_users(db: Session = Depends(get_db_session)):
    """
    获取所有用户数据
    
    从数据库中查询所有用户记录，并以列表形式返回。
    
    Args:
        db (Session): 数据库会话对象，通过依赖注入提供
        
    Returns:
        List[Dict[str, Any]]: 用户信息列表，每个用户以字典形式表示
        
    Raises:
        HTTPException: 当数据库查询失败时抛出500错误
    """
    try:
        users = db.query(User).all()
        return [user.to_dict() for user in users]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to fetch users: {str(e)}"
        )


@router.get("/{user_id}", response_model=Dict[str, Any])
def get_user_by_id(user_id: int, db: Session = Depends(get_db_session)):
    """
    根据ID获取单个用户
    
    通过用户ID在数据库中查找特定用户，并返回用户详细信息。
    
    Args:
        user_id (int): 要查询的用户ID
        db (Session): 数据库会话对象，通过依赖注入提供
        
    Returns:
        Dict[str, Any]: 用户信息字典
        
    Raises:
        HTTPException: 
            - 当用户不存在时抛出404错误
            - 当数据库查询失败时抛出500错误
    """
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail="User not found"
            )
        return user.to_dict()
    except HTTPException:
        # 重新抛出已知的HTTP异常
        raise
    except Exception as e:
        # 处理未预期的异常
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to fetch user: {str(e)}"
        )

# 新增：创建新用户的POST端点
@router.post("/", response_model=Dict[str, Any])
def create_user(user_data: Dict[str, Any], db: Session = Depends(get_db_session)):
    """
    创建新用户
    
    在数据库中创建一个新的用户记录。
    
    Args:
        user_data (Dict[str, Any]): 包含用户信息的字典
        db (Session): 数据库会话对象，通过依赖注入提供
        
    Returns:
        Dict[str, Any]: 新创建的用户信息
        
    Raises:
        HTTPException: 当创建用户失败时抛出500错误
    """
    try:
        # 从传入数据中提取字段
        new_user = User(
            username=user_data.get("username"),
            email=user_data.get("email"),
            hashed_password=user_data.get("hashed_password"),
            is_active=user_data.get("is_active", True),
            is_superuser=user_data.get("is_superuser", False)
        )
        
        # 添加到数据库
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        return new_user.to_dict()
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create user: {str(e)}"
        )

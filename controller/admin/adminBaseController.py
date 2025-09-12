import logging
from pydantic import BaseModel
from typing import Optional
from fastapi import APIRouter, Request, HTTPException
from controller.helper.controllerHelper import require_admin_access
from service.database.models.user import User
from controller.helper.singletonHelper import get_config_composer, get_vector_manager, get_rag_service, get_document_processor, get_db_manager
logger = logging.getLogger("admin-controller")
router = APIRouter(prefix="/base", tags=["Admin"])

class SignupRequest(BaseModel):
    """회원가입 요청 모델"""
    username: str
    email: str
    password: str
    full_name: Optional[str] = None

class SignupResponse(BaseModel):
    """회원가입 응답 모델"""
    success: bool
    message: str
    username: Optional[str] = None

@router.get("/superuser")
async def check_superuser(request: Request):
    try:
        app_db = get_db_manager(request)
        super_users = app_db.find_by_condition(
            User,
            {
                "user_type": "superuser"
            },
            limit=1
        )

        if super_users:
            return {"superuser": True}

        else:
            return {"superuser": False}

    except Exception as e:
        logger.error(f"Token refresh error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

async def is_superuser(request: Request, user_id: int) -> bool:
    try:
        app_db = get_db_manager(request)
        super_users = app_db.find_by_condition(
            User,
            {
                "id": user_id,
                "user_type": "superuser"
            },
            limit=1
        )

        if super_users:
            return {"superuser": True}

        else:
            return {"superuser": False}

    except Exception as e:
        logger.error(f"Token refresh error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/validate/superuser")
async def validate_superuser(request: Request):
    session = require_admin_access(request)
    user_id = session['user_id']
    try:
        app_db = get_db_manager(request)
        super_users = app_db.find_by_condition(
            User,
            {
                "id": user_id,
                "user_type": "superuser"
            },
            limit=1
        )

        if super_users:
            return {"superuser": True}

        else:
            return {"superuser": False}

    except Exception as e:
        logger.error(f"Token refresh error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.post("/create-superuser", response_model=SignupResponse)
async def create_superuser(request: Request, signup_data: SignupRequest):
    try:
        app_db = get_db_manager(request)

        if not signup_data.username or not signup_data.email or not signup_data.password:
            raise HTTPException(
                status_code=400,
                detail="Username, email, and password are required"
            )

        existing_superuser = app_db.find_by_condition(
            User,
            {"user_type": "superuser"},
            limit=1
        )

        if existing_superuser:
            raise HTTPException(
                status_code=409,
                detail="Superuser already exists"
            )

        existing_user_by_username = app_db.find_by_condition(
            User,
            {"username": signup_data.username},
            limit=1
        )

        if existing_user_by_username:
            raise HTTPException(
                status_code=409,
                detail="Username already exists"
            )

        existing_user_by_email = app_db.find_by_condition(
            User,
            {"email": signup_data.email},
            limit=1
        )

        if existing_user_by_email:
            raise HTTPException(
                status_code=422,
                detail="Email already exists"
            )

        new_user = User(
            username=signup_data.username,
            email=signup_data.email,
            password_hash=signup_data.password,
            full_name=signup_data.full_name,
            is_active=True,
            is_admin=True,
            last_login=None,
            preferences={},
            user_type="superuser",
            group_name="superuser",
            groups={},
        )

        insert_result = app_db.insert(new_user)
        if insert_result and insert_result.get("result") == "success":
            username = signup_data.username
        else:
            username = None

        if username:
            logger.info(f"New user created: {signup_data.username} (User Name: {username})")
            return SignupResponse(
                success=True,
                message="User created successfully",
                username=username
            )
        else:
            logger.error(f"Failed to create user: {signup_data.username}")
            raise HTTPException(
                status_code=500,
                detail="Failed to create user"
            )

    except Exception as e:
        logger.error(f"Signup error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

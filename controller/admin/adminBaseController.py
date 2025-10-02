import logging
from pydantic import BaseModel
from typing import Optional
from fastapi import APIRouter, Request, HTTPException
from controller.helper.controllerHelper import require_admin_access
from service.database.models.user import User
from service.database.models.backend import BackendLogs
from controller.helper.singletonHelper import get_config_composer, get_vector_manager, get_rag_service, get_document_processor, get_db_manager
from service.database.logger_helper import create_logger
from controller.utils.section_config import available_sections, available_admin_sections

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
        user_info = app_db.find_by_condition(
            User,
            {
                "id": user_id,
            },
            limit=1
        )
        user_info = user_info[0] if user_info and len(user_info) > 0 else None

        if user_info.user_type == "superuser":
            return {"superuser": True, "user_id": user_id, "available_admin_sections": available_admin_sections, "user_type": "superuser"}

        elif user_info.user_type == "admin" and user_info.available_admin_sections is not None and user_info.available_admin_sections != []:
            return {"superuser": True, "user_id": user_id, "available_admin_sections": user_info.available_admin_sections, "user_type": "admin"}

        else:
            return {"superuser": False, "user_id": None, "available_admin_sections": [], "user_type": "standard"}

    except Exception as e:
        logger.error(f"Token refresh error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.post("/create-superuser", response_model=SignupResponse)
async def create_superuser(request: Request, signup_data: SignupRequest):
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, None, request)

    try:
        if not signup_data.username or not signup_data.email or not signup_data.password:
            backend_log.warn("Superuser creation attempted with missing required fields")
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
            backend_log.warn("Attempt to create superuser when one already exists")
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
            backend_log.warn(f"Superuser creation failed - username already exists: {signup_data.username}")
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
            backend_log.warn(f"Superuser creation failed - email already exists: {signup_data.email}")
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
            backend_log.success(f"Successfully created superuser: {signup_data.username}",
                              metadata={"username": signup_data.username, "email": signup_data.email})
            logger.info(f"New user created: {signup_data.username} (User Name: {username})")
            return SignupResponse(
                success=True,
                message="User created successfully",
                username=username
            )
        else:
            backend_log.error(f"Failed to create superuser: {signup_data.username}")
            logger.error(f"Failed to create user: {signup_data.username}")
            raise HTTPException(
                status_code=500,
                detail="Failed to create user"
            )

    except Exception as e:
        backend_log.error("Superuser creation error", exception=e,
                         metadata={"username": signup_data.username, "email": signup_data.email})
        logger.error(f"Signup error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/backend/logs")
async def get_backend_logs(request: Request, page: int = 1, page_size: int = 250):
    val_superuser = await validate_superuser(request)
    if not val_superuser.get("superuser", False):
        raise HTTPException(
            status_code=403,
            detail="Superuser access required"
        )

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, val_superuser.get("user_id"), request)

    try:
        if page < 1:
            page = 1
        if page_size < 1 or page_size > 1000:
            page_size = 250

        offset = (page - 1) * page_size

        logs = app_db.find_by_condition(
            BackendLogs,
            {},
            orderby="created_at",
            limit=page_size,
            offset=offset
        )

        return {
            "logs": logs,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "offset": offset,
                "total_returned": len(logs)
            }
        }
    except Exception as e:
        backend_log.error("Error fetching backend logs", exception=e,
                         metadata={"page": page, "page_size": page_size})
        logger.error(f"Error fetching backend logs: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

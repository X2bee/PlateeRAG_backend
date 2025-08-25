import logging
from pydantic import BaseModel
from typing import Optional
from fastapi import APIRouter, Request, HTTPException
from controller.controller_helper import require_admin_access
from service.database.models.user import User
from controller.singletonHelper import get_config_composer, get_vector_manager, get_rag_service, get_document_processor, get_db_manager
from controller.admin.adminBaseController import validate_superuser

logger = logging.getLogger("admin-controller")
router = APIRouter(prefix="/user", tags=["Admin"])

@router.get("/all-users")
async def get_all_users(request: Request):
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )

    try:
        app_db = get_db_manager(request)
        users = app_db.find_all(User)
        return {"users": users}
    except Exception as e:
        logger.error(f"Error fetching all users: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

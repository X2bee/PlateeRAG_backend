import logging
from pydantic import BaseModel
from typing import Optional
from fastapi import APIRouter, Request, HTTPException
from controller.helper.controllerHelper import require_admin_access
from service.database.models.user import User
from controller.helper.singletonHelper import get_config_composer, get_vector_manager, get_rag_service, get_document_processor, get_db_manager
logger = logging.getLogger("admin-controller")
router = APIRouter(prefix="/base", tags=["Admin"])

async def get_user_id_from_admin(request: Request) -> Optional[int]:
    try:
        session = require_admin_access(request)
        user_id = session['user_id']

        return user_id

    except Exception as e:
        logger.error(f"Token refresh error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

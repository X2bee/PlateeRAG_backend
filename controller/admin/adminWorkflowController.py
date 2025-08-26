import logging
from pydantic import BaseModel
from typing import Optional
from fastapi import APIRouter, Request, HTTPException
from controller.helper.controllerHelper import require_admin_access
from controller.helper.singletonHelper import get_config_composer, get_vector_manager, get_rag_service, get_document_processor, get_db_manager
from controller.admin.adminBaseController import validate_superuser

from service.database.models.user import User
from service.database.models.executor import ExecutionIO

logger = logging.getLogger("admin-controller")
router = APIRouter(prefix="/workflow", tags=["Admin"])

@router.get("/all-io-logs")
async def get_all_workflows(request: Request):
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )

    try:
        app_db = get_db_manager(request)
        io_logs = app_db.find_all(ExecutionIO)
        return {"io_logs": io_logs}
    except Exception as e:
        logger.error(f"Error fetching all IO logs: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

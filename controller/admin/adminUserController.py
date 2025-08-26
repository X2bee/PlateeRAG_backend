import logging
from pydantic import BaseModel
from typing import Optional
from fastapi import APIRouter, Request, HTTPException
from controller.helper.controllerHelper import require_admin_access
from controller.helper.singletonHelper import get_config_composer, get_vector_manager, get_rag_service, get_document_processor, get_db_manager
from controller.admin.adminBaseController import validate_superuser

from service.database.models.user import User
from service.database.models.executor import ExecutionIO, ExecutionMeta
from service.database.models.workflow import WorkflowMeta
from service.database.models.performance import WorkflowExecution, NodePerformance
from service.database.models.vectordb import VectorDB, VectorDBChunkMeta, VectorDBChunkEdge

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

@router.delete("/user-account")
async def delete_user(request: Request, user_data: dict):
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )

    try:
        app_db = get_db_manager(request)
        user_id = user_data.get("id")
        username = user_data.get("username")
        user_email = user_data.get("email")

        db_user_info = app_db.find_by_id(User, user_id)
        if not db_user_info:
            raise HTTPException(
                status_code=404,
                detail="User not found"
            )

        if db_user_info.username != username or db_user_info.email != user_email:
            raise HTTPException(
                status_code=400,
                detail="User information mismatch"
            )

        # 사용자와 관련된 모든 데이터를 삭제
        logger.info(f"Starting deletion of user {user_id} and all related data")

        # 1. ExecutionIO 삭제
        execution_io_deleted = app_db.delete_by_condition(ExecutionIO, {"user_id": user_id})
        logger.info(f"ExecutionIO deletion result for user {user_id}: {execution_io_deleted}")

        # 2. ExecutionMeta 삭제
        execution_meta_deleted = app_db.delete_by_condition(ExecutionMeta, {"user_id": user_id})
        logger.info(f"ExecutionMeta deletion result for user {user_id}: {execution_meta_deleted}")

        # 3. WorkflowMeta 삭제
        workflow_meta_deleted = app_db.delete_by_condition(WorkflowMeta, {"user_id": user_id})
        logger.info(f"WorkflowMeta deletion result for user {user_id}: {workflow_meta_deleted}")

        # 4. WorkflowExecution 삭제
        workflow_execution_deleted = app_db.delete_by_condition(WorkflowExecution, {"user_id": user_id})
        logger.info(f"WorkflowExecution deletion result for user {user_id}: {workflow_execution_deleted}")

        # 5. NodePerformance 삭제
        node_performance_deleted = app_db.delete_by_condition(NodePerformance, {"user_id": user_id})
        logger.info(f"NodePerformance deletion result for user {user_id}: {node_performance_deleted}")

        # 6. VectorDB 삭제
        vector_db_deleted = app_db.delete_by_condition(VectorDB, {"user_id": user_id})
        logger.info(f"VectorDB deletion result for user {user_id}: {vector_db_deleted}")

        # 7. VectorDBChunkMeta 삭제
        chunk_meta_deleted = app_db.delete_by_condition(VectorDBChunkMeta, {"user_id": user_id})
        logger.info(f"VectorDBChunkMeta deletion result for user {user_id}: {chunk_meta_deleted}")

        # 8. VectorDBChunkEdge 삭제
        chunk_edge_deleted = app_db.delete_by_condition(VectorDBChunkEdge, {"user_id": user_id})
        logger.info(f"VectorDBChunkEdge deletion result for user {user_id}: {chunk_edge_deleted}")

        # 9. 마지막으로 User 삭제
        result = app_db.delete(User, user_id)
        if not result:
            raise HTTPException(
                status_code=404,
                detail="User not found"
            )

        logger.info(f"Successfully deleted user {user_id} and all related data")
        return {
            "detail": "User and all related data deleted successfully",
            "deleted_records": {
                "execution_io": execution_io_deleted,
                "execution_meta": execution_meta_deleted,
                "workflow_meta": workflow_meta_deleted,
                "workflow_execution": workflow_execution_deleted,
                "node_performance": node_performance_deleted,
                "vector_db": vector_db_deleted,
                "vector_db_chunk_meta": chunk_meta_deleted,
                "vector_db_chunk_edge": chunk_edge_deleted,
                "user": result
            }
        }
    except Exception as e:
        logger.error(f"Error deleting user and related data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

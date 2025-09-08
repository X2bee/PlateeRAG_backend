import logging
from pydantic import BaseModel
from typing import Optional
from fastapi import APIRouter, Request, HTTPException
from controller.helper.controllerHelper import require_admin_access
from controller.helper.singletonHelper import get_config_composer, get_vector_manager, get_rag_service, get_document_processor, get_db_manager
from controller.admin.adminBaseController import validate_superuser

# authController에서 필요한 함수들과 모델들 import
from controller.authController import (LoginRequest, LoginResponse, login, find_user_by_email)

from service.database.models.user import User
from service.database.models.executor import ExecutionIO, ExecutionMeta
from service.database.models.workflow import WorkflowMeta
from service.database.models.performance import WorkflowExecution, NodePerformance
from service.database.models.vectordb import VectorDB, VectorDBChunkMeta, VectorDBChunkEdge

logger = logging.getLogger("admin-user-controller")
router = APIRouter(prefix="/user", tags=["Admin"])

@router.post("/superuser-login", response_model=LoginResponse)
async def superuser_login(request: Request, login_data: LoginRequest):
    """
    슈퍼유저 로그인 API

    기본 로그인 함수를 사용하되 슈퍼유저 검증만 추가
    """
    try:
        # 먼저 사용자가 슈퍼유저인지 확인
        app_db = get_db_manager(request)
        user = find_user_by_email(app_db, login_data.email)

        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid email or password"
            )

        # 슈퍼유저 여부 확인
        if user.user_type != "superuser":
            raise HTTPException(
                status_code=403,
                detail="Superuser privileges required"
            )

        # 슈퍼유저가 맞으면 기본 로그인 함수 호출
        result = await login(request, login_data)

        # 로그인 성공 시 메시지 수정
        if result.success:
            result.message = "Superuser login successful"
            logger.info(f"Superuser logged in: {login_data.email}")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Superuser login error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/all-users")
async def get_all_users(request: Request, page: int = 1, page_size: int = 100):
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )

    try:
        # 페이지 번호 검증
        if page < 1:
            page = 1
        if page_size < 1 or page_size > 1000:
            page_size = 100

        offset = (page - 1) * page_size

        app_db = get_db_manager(request)
        users = app_db.find_all(User, limit=page_size, offset=offset)

        return {
            "users": users,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "offset": offset,
                "total_returned": len(users)
            }
        }
    except Exception as e:
        logger.error(f"Error fetching all users: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/standby-users")
async def get_standby_users(request: Request):
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )

    try:
        app_db = get_db_manager(request)
        users = app_db.find_by_condition(User, {"is_active": False})
        return {"users": users}
    except Exception as e:
        logger.error(f"Error fetching standby users: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.post("/approve-user")
async def approve_user(request: Request, user_data: dict):
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

        # 사용자 존재 여부 확인
        db_user_info = app_db.find_by_condition(User, {"id": user_id, "is_active": False})
        if not db_user_info:
            raise HTTPException(
                status_code=404,
                detail="User not found"
            )

        db_user_info = db_user_info[0]

        # 사용자 정보 일치 확인
        if db_user_info.username != username or db_user_info.email != user_email:
            raise HTTPException(
                status_code=400,
                detail="User information mismatch"
            )

        # 이미 활성화된 사용자인지 확인
        if db_user_info.is_active:
            raise HTTPException(
                status_code=400,
                detail="User is already active"
            )

        db_user_info.is_active = True
        app_db.update(db_user_info)

        logger.info(f"Successfully approved user {user_id} ({username})")
        return {
            "detail": "User approved successfully",
            "user": {
                "id": user_id,
                "username": username,
                "email": user_email,
                "is_active": True
            }
        }
    except Exception as e:
        logger.error(f"Error approving user: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.put("/edit-user")
async def edit_user(request: Request, user_data: dict):
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )

    try:
        app_db = get_db_manager(request)
        user_id = user_data.get("id")

        # 사용자 존재 여부 확인
        db_user_info = app_db.find_by_condition(User, {"id": user_id})
        if not db_user_info:
            raise HTTPException(
                status_code=404,
                detail="User not found"
            )

        db_user_info = db_user_info[0]

        db_user_info.email = user_data.get("email", db_user_info.email)
        db_user_info.username = user_data.get("username", db_user_info.username)
        db_user_info.full_name = user_data.get("full_name", db_user_info.full_name)
        db_user_info.is_admin = user_data.get("is_admin", db_user_info.is_admin)
        db_user_info.user_type = user_data.get("user_type", db_user_info.user_type)
        db_user_info.preferences = user_data.get("preferences", db_user_info.preferences)
        db_user_info.is_active = user_data.get("is_active", db_user_info.is_active)
        db_user_info.password_hash = user_data.get("password_hash", db_user_info.password_hash)
        app_db.update(db_user_info)

        logger.info(f"Successfully edited user {user_id}")
        return {
            "detail": "User approved successfully",
            "user": {
                "id": user_id,
            }
        }
    except Exception as e:
        logger.error(f"Error approving user: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.put("/edit-user/groups")
async def edit_user_groups(request: Request, user_data: dict):
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )

    try:
        app_db = get_db_manager(request)
        user_id = user_data.get("id")
        db_user_info = app_db.find_by_condition(User, {"id": user_id})
        if not db_user_info:
            raise HTTPException(
                status_code=404,
                detail="User not found"
            )

        db_user_info = db_user_info[0]
        existing_groups = db_user_info.groups if db_user_info.groups else []
        add_group = user_data.get("group_name", db_user_info.group_name)

        if isinstance(add_group, str):
            add_group = [add_group]
        elif not isinstance(add_group, list):
            add_group = []

        new_groups = list(set(existing_groups) | set(add_group))

        app_db.update_list_columns(User, {"groups": new_groups}, {"id": user_id})

        logger.info(f"Successfully edited user {user_id}")
        return {
            "detail": "User groups updated successfully",
            "user": {
                "id": user_id,
                "groups": new_groups
            }
        }
    except Exception as e:
        logger.error(f"Error approving user: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.delete("/edit-user/groups")
async def delete_user_groups(request: Request, user_data: dict):
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )

    try:
        app_db = get_db_manager(request)
        user_id = user_data.get("id")

        # 사용자 존재 여부 확인
        db_user_info = app_db.find_by_condition(User, {"id": user_id})
        if not db_user_info:
            raise HTTPException(
                status_code=404,
                detail="User not found"
            )

        db_user_info = db_user_info[0]
        existing_groups = db_user_info.groups if db_user_info.groups else []
        remove_group = user_data.get("group_name", "")

        if isinstance(remove_group, str):
            remove_group = [remove_group]
        elif not isinstance(remove_group, list):
            remove_group = []

        new_groups = [group for group in existing_groups if group not in remove_group]
        app_db.update_list_columns(User, {"groups": new_groups}, {"id": user_id})

        logger.info(f"Successfully removed groups from user {user_id}")
        return {
            "detail": "User groups updated successfully",
            "user": {
                "id": user_id,
                "groups": new_groups,
                "removed_groups": remove_group
            }
        }
    except Exception as e:
        logger.error(f"Error removing user groups: {str(e)}")
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

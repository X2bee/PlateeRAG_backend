import json
import logging
from pydantic import BaseModel
from typing import Optional
from fastapi import APIRouter, Request, HTTPException
from controller.helper.singletonHelper import get_config_composer, get_vector_manager, get_rag_service, get_document_processor, get_db_manager
from controller.admin.adminBaseController import validate_superuser
from service.database.logger_helper import create_logger

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
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, None, request)

    try:
        # 먼저 사용자가 슈퍼유저인지 확인
        user = find_user_by_email(app_db, login_data.email)

        if not user:
            backend_log.warn(f"Superuser login attempt with invalid email: {login_data.email}")
            raise HTTPException(
                status_code=401,
                detail="Invalid email or password"
            )

        # 슈퍼유저 여부 확인
        if user.user_type != "superuser":
            backend_log.warn(f"Non-superuser attempted superuser login: {login_data.email}",
                           metadata={"user_type": user.user_type, "user_id": user.id})
            raise HTTPException(
                status_code=403,
                detail="Superuser privileges required"
            )

        # 슈퍼유저가 맞으면 기본 로그인 함수 호출
        result = await login(request, login_data)

        # 로그인 성공 시 메시지 수정
        if result.success:
            result.message = "Superuser login successful"
            backend_log.success(f"Superuser login successful: {login_data.email}",
                              metadata={"user_id": user.id, "email": login_data.email})
            logger.info(f"Superuser logged in: {login_data.email}")

        return result

    except HTTPException:
        raise
    except Exception as e:
        backend_log.error("Superuser login error", exception=e,
                         metadata={"email": login_data.email})
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

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, val_superuser.get("user_id"), request)

    try:
        # 페이지 번호 검증
        if page < 1:
            page = 1
        if page_size < 1 or page_size > 1000:
            page_size = 100

        offset = (page - 1) * page_size

        users = app_db.find_all(User, limit=page_size, offset=offset)

        backend_log.success("Successfully fetched all users",
                          metadata={"page": page, "page_size": page_size, "returned_count": len(users)})
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
        backend_log.error("Error fetching all users", exception=e,
                         metadata={"page": page, "page_size": page_size})
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

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, val_superuser.get("user_id"), request)

    try:
        users = app_db.find_by_condition(User, {"is_active": False})
        backend_log.success("Successfully fetched standby users",
                          metadata={"standby_user_count": len(users)})
        return {"users": users}
    except Exception as e:
        backend_log.error("Error fetching standby users", exception=e)
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

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, val_superuser.get("user_id"), request)

    try:
        user_id = user_data.get("id")
        username = user_data.get("username")
        user_email = user_data.get("email")

        # 사용자 존재 여부 확인
        db_user_info = app_db.find_by_condition(User, {"id": user_id, "is_active": False})
        if not db_user_info:
            backend_log.warn(f"User not found for approval: {user_id}")
            raise HTTPException(
                status_code=404,
                detail="User not found"
            )

        db_user_info = db_user_info[0]

        # 사용자 정보 일치 확인
        if db_user_info.username != username or db_user_info.email != user_email:
            backend_log.warn(f"User information mismatch during approval: {user_id}",
                           metadata={"provided_username": username, "provided_email": user_email,
                                   "db_username": db_user_info.username, "db_email": db_user_info.email})
            raise HTTPException(
                status_code=400,
                detail="User information mismatch"
            )

        # 이미 활성화된 사용자인지 확인
        if db_user_info.is_active:
            backend_log.warn(f"User already active during approval attempt: {user_id}")
            raise HTTPException(
                status_code=400,
                detail="User is already active"
            )

        db_user_info.is_active = True
        app_db.update(db_user_info)

        backend_log.success(f"Successfully approved user: {username} ({user_id})",
                          metadata={"user_id": user_id, "username": username, "email": user_email})
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
        backend_log.error("Error approving user", exception=e,
                         metadata={"user_data": user_data})
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

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, val_superuser.get("user_id"), request)

    try:
        user_id = user_data.get("id")

        # 사용자 존재 여부 확인
        db_user_info = app_db.find_by_condition(User, {"id": user_id})
        if not db_user_info:
            backend_log.warn(f"User not found for editing: {user_id}")
            raise HTTPException(
                status_code=404,
                detail="User not found"
            )

        db_user_info = db_user_info[0]

        # 기존 값 저장 (변경 전 상태 확인용)
        original_is_admin = db_user_info.is_admin
        original_user_type = db_user_info.user_type

        db_user_info.email = user_data.get("email", db_user_info.email)
        db_user_info.username = user_data.get("username", db_user_info.username)
        db_user_info.full_name = user_data.get("full_name", db_user_info.full_name)
        db_user_info.is_admin = user_data.get("is_admin", db_user_info.is_admin)
        db_user_info.user_type = user_data.get("user_type", db_user_info.user_type)
        db_user_info.preferences = user_data.get("preferences", db_user_info.preferences)
        db_user_info.is_active = user_data.get("is_active", db_user_info.is_active)
        db_user_info.password_hash = user_data.get("password_hash", db_user_info.password_hash)

        app_db.update(db_user_info)

        # admin 권한이 제거되는 경우 (is_admin이 False가 되거나 user_type이 standard가 되는 경우)
        is_demoted = False
        if original_is_admin and not db_user_info.is_admin:
            is_demoted = True
        if original_user_type != "standard" and db_user_info.user_type == "standard":
            is_demoted = True

        # admin 권한이 제거되면 __admin__ 그룹들도 모두 제거
        if is_demoted:
            existing_groups = db_user_info.groups if db_user_info.groups else []
            # __admin__으로 끝나지 않는 그룹만 유지
            new_groups = [group for group in existing_groups if not group.endswith("__admin__")]
            app_db.update_list_columns(User, {"groups": new_groups}, {"id": user_id})
            logger.info(f"User {user_id} demoted from admin - removed all __admin__ groups. Groups: {existing_groups} -> {new_groups}")

        backend_log.success(f"Successfully edited user: {user_id}",
                          metadata={"user_id": user_id, "updated_fields": list(user_data.keys())})
        logger.info(f"Successfully edited user {user_id}")
        return {
            "detail": "User approved successfully",
            "user": {
                "id": user_id,
            }
        }
    except Exception as e:
        backend_log.error("Error editing user", exception=e,
                         metadata={"user_data": user_data})
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

        has_admin_group = any(group.endswith("__admin__") for group in add_group)
        if has_admin_group and not db_user_info.is_admin and db_user_info.user_type == "standard":
            db_user_info.is_admin = True
            db_user_info.user_type = "admin"
            app_db.update(db_user_info)
            logger.info(f"User {user_id} promoted to admin due to __admin__ group assignment")

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

@router.post("/update-user/available-admin-sections")
async def edit_user_available_admin_sections(request: Request, user_data: dict):
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
        available_sections = user_data.get("available_admin_sections", db_user_info.available_admin_sections)
        if available_sections is not None:
            if isinstance(available_sections, str):
                try:
                    available_sections = json.loads(available_sections)
                except json.JSONDecodeError:
                    available_sections = [s.strip() for s in available_sections.split(',') if s.strip()]

            if not isinstance(available_sections, list):
                available_sections = [str(available_sections)]

        updates = {}
        if available_sections is not None:
            updates['available_admin_sections'] = available_sections

        app_db.update_list_columns(User, updates, {"id": user_id})

        logger.info(f"Successfully edited user {user_id}")
        return {
            "detail": "User groups updated successfully",
            "user": {
                "id": user_id,
                "available_admin_sections": available_sections
            }
        }
    except Exception as e:
        logger.error(f"Error approving user: {str(e)}")
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

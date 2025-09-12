import logging
import json
from pydantic import BaseModel
from fastapi import APIRouter, Request, HTTPException
from controller.helper.singletonHelper import get_db_manager
from controller.manager.managerBaseController import get_user_id_from_admin
from service.database.models.group import GroupMeta
from service.database.models.user import User
from service.database.logger_helper import create_logger

logger = logging.getLogger("admin-group-controller")
router = APIRouter(prefix="/group", tags=["Admin"])

@router.get("/all-groups")
async def get_all_groups(request: Request):
    manager_user_id = await get_user_id_from_admin(request)
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, manager_user_id, request)

    manager_account = app_db.find_by_condition(User, {"id": manager_user_id})
    manager_groups = manager_account[0].groups

    try:
        groups = []
        for group_name in manager_groups:
            group_meta = app_db.find_by_condition(GroupMeta, {"group_name": group_name})
            if group_meta:
                groups.extend(group_meta)

        backend_log.success("Successfully fetched all groups")
        return {"groups": groups}
    except Exception as e:
        backend_log.error("Error fetching manager groups", exception=e)
        logger.error("Error fetching manager groups: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        ) from e

@router.get("/group-users")
async def get_group_users(request: Request, group_name: str):
    manager_user_id = await get_user_id_from_admin(request)
    backend_log = create_logger(app_db=None, user_id=manager_user_id, request=request)

    try:
        app_db = get_db_manager(request)
        # 로거에 app_db 설정
        backend_log.app_db = app_db
        db_type = app_db.config_db_manager.db_type

        manager_account = app_db.find_by_condition(User, {"id": manager_user_id})
        manager_groups = manager_account[0].groups

        if group_name not in manager_groups:
            backend_log.warn(f"Access denied to group: {group_name}")
            raise HTTPException(
                status_code=403,
                detail="You do not have access to this group"
            )

        if db_type == "postgresql":
            query = "SELECT * FROM users WHERE %s = ANY(groups)"
            params = (group_name,)
            results = app_db.config_db_manager.execute_query(query, params)
            users = [User.from_dict(dict(row)) for row in results] if results else []
        else:
            query = "SELECT * FROM users WHERE groups LIKE ?"
            params = (f'%"{group_name}"%',)
            results = app_db.config_db_manager.execute_query(query, params)
            users = [User.from_dict(dict(row)) for row in results] if results else []

        backend_log.success(f"Successfully fetched users for group: {group_name}",
                          metadata={"group_name": group_name, "user_count": len(users)})
        return {"users": users}

    except Exception as e:
        backend_log.error("Error fetching group users", exception=e,
                         metadata={"group_name": group_name})
        logger.error("Error fetching group users: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        ) from e

@router.delete("/group")
async def delete_group(request: Request, group_name: str):
    manager_user_id = await get_user_id_from_admin(request)
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, manager_user_id, request)

    manager_account = app_db.find_by_condition(User, {"id": manager_user_id})
    manager_groups = manager_account[0].groups

    if group_name not in manager_groups:
        backend_log.warn(f"Access denied to delete group: {group_name}")
        raise HTTPException(
            status_code=403,
            detail="You do not have access to this group"
        )

    try:
        group = app_db.find_by_condition(GroupMeta, {'group_name': group_name})
        if not group:
            backend_log.warn(f"Group not found: {group_name}")
            raise HTTPException(
                status_code=404,
                detail="Group not found"
            )

        app_db.delete_by_condition(GroupMeta, {'group_name': group_name})
        db_type = app_db.config_db_manager.db_type

        if db_type == "postgresql":
            # PostgreSQL: 배열에서 특정 값 포함 여부 확인
            query = "SELECT * FROM users WHERE %s = ANY(groups)"
            params = (group_name,)
            results = app_db.config_db_manager.execute_query(query, params)
            users = [User.from_dict(dict(row)) for row in results] if results else []
        else:
            # SQLite: JSON 배열에서 검색 (LIKE 사용)
            query = "SELECT * FROM users WHERE groups LIKE ?"
            params = (f'%"{group_name}"%',)
            results = app_db.config_db_manager.execute_query(query, params)
            users = [User.from_dict(dict(row)) for row in results] if results else []

        # 각 사용자의 groups에서 해당 그룹명 제거
        for user in users:
            existing_groups = user.groups if user.groups else []
            new_groups = [group for group in existing_groups if group != group_name]

            # update_list_columns 메서드로 업데이트 (첫 번째 인자는 모델 클래스가 아닌 인스턴스)
            app_db.update_list_columns(User, {"groups": new_groups}, {"id": user.id})

        backend_log.success(f"Successfully deleted group: {group_name}",
                          metadata={"group_name": group_name, "affected_users": len(users)})
        return {"detail": f"Group '{group_name}' deleted successfully and {len(users)} users moved to 'none' group"}

    except Exception as e:
        backend_log.error("Error deleting group", exception=e,
                         metadata={"group_name": group_name})
        logger.error("Error deleting group: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        ) from e

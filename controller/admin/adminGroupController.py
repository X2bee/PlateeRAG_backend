import logging
import json
from pydantic import BaseModel
from fastapi import APIRouter, Request, HTTPException
from controller.helper.singletonHelper import get_db_manager
from controller.admin.adminBaseController import validate_superuser

from service.database.models.group import GroupMeta
from service.database.models.user import User

logger = logging.getLogger("admin-group-controller")
router = APIRouter(prefix="/group", tags=["Admin"])

class GroupData(BaseModel):
    group_name: str
    available: bool
    available_sections: list

@router.get("/all-groups")
async def get_all_groups(request: Request):
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )

    try:
        app_db = get_db_manager(request)
        groups = app_db.find_all(GroupMeta)
        return {"groups": groups}
    except Exception as e:
        logger.error("Error fetching all groups: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        ) from e

@router.get("/all-groups/list")
async def get_all_groups_list(request: Request):
    try:
        app_db = get_db_manager(request)
        groups = app_db.find_by_condition(GroupMeta, {'available': True})

        # group_name만 추출해서 리스트로 반환
        group_names = [group.group_name for group in groups]

        return {"groups": group_names}
    except Exception as e:
        logger.error("Error fetching all groups list: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        ) from e

@router.post("/create")
async def create_group(request: Request, group_data: GroupData):
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )

    try:
        app_db = get_db_manager(request)
        exist_group = app_db.find_by_condition(GroupMeta, {'group_name': group_data.group_name})

        if exist_group:
            raise HTTPException(
                status_code=400,
                detail="Group with this name already exists"
            )

        group = GroupMeta(
            group_name=group_data.group_name,
            available=group_data.available,
            available_sections=[],
        )

        app_db.insert(group)
        return {"detail": "Group created successfully"}
    except Exception as e:
        logger.error("Error creating group: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        ) from e

@router.get("/group-users")
async def get_group_users(request: Request, group_name: str):
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )

    try:
        app_db = get_db_manager(request)
        db_type = app_db.config_db_manager.db_type

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

        return {"users": users}

    except Exception as e:
        logger.error("Error fetching group users: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        ) from e

@router.get("/group-available-section")
async def get_group_available_sections(request: Request, group_name: str):
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )

    try:
        app_db = get_db_manager(request)
        group = app_db.find_by_condition(GroupMeta, {'group_name': group_name})

        if group:
            group = group[0]
            group_available_sections = group.available_sections
        else:
            raise HTTPException(
                status_code=404,
                detail="Group not found"
            )

        return {"available_sections": group_available_sections}

    except Exception as e:
        logger.error("Error fetching all users: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        ) from e

@router.delete("/group")
async def delete_group(request: Request, group_name: str):
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )

    try:
        app_db = get_db_manager(request)
        group = app_db.find_by_condition(GroupMeta, {'group_name': group_name})
        if not group:
            raise HTTPException(
                status_code=404,
                detail="Group not found"
            )

        app_db.delete_by_condition(GroupMeta, {'group_name': group_name})

        # 해당 그룹을 groups 배열에 가지고 있는 모든 사용자 찾기
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
        return {"detail": f"Group '{group_name}' deleted successfully and {len(users)} users moved to 'none' group"}

    except Exception as e:
        logger.error("Error deleting group: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        ) from e

@router.post("/update")
async def update_group_permissions(request: Request, group_data: dict):
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )

    try:
        app_db = get_db_manager(request)
        group = app_db.find_by_condition(GroupMeta, {'group_name': group_data.get("group_name")})
        if not group:
            raise HTTPException(
                status_code=404,
                detail="Group not found"
            )

        if group:
            group = group[0]

        available_sections = group_data.get("available_sections", group.available_sections)
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
            updates['available_sections'] = available_sections
        if "available" in group_data:
            updates['available'] = group_data.get("available", group.available)

        app_db.update_list_columns(GroupMeta, updates, {'group_name': group.group_name})

        return {"detail": "Group permissions updated successfully"}
    except Exception as e:
        logger.error("Error updating group permissions: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        ) from e

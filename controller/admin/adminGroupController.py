import logging
import json
from pydantic import BaseModel
from fastapi import APIRouter, Request, HTTPException
from controller.helper.singletonHelper import get_db_manager
from controller.admin.adminBaseController import validate_superuser

from service.database.models.group import GroupMeta
from service.database.models.user import User

logger = logging.getLogger("admin-controller")
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
        users = app_db.find_by_condition(User, {'group_name': group_name})
        return {"users": users}

    except Exception as e:
        logger.error("Error fetching all users: %s", str(e))
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

        # Handle available_sections properly for PostgreSQL array type
        available_sections = group_data.get("available_sections")
        if available_sections is not None:
            # If it's a string (JSON), parse it to list
            if isinstance(available_sections, str):
                try:
                    available_sections = json.loads(available_sections)
                except json.JSONDecodeError:
                    # If parsing fails, split by comma or keep as is
                    available_sections = [s.strip() for s in available_sections.split(',') if s.strip()]

            # Ensure it's a list
            if not isinstance(available_sections, list):
                available_sections = [str(available_sections)]

            group.available_sections = available_sections

        # Update other fields
        if "available" in group_data:
            group.available = group_data.get("available", group.available)

        #TODO sqlite에서도 생각
        app_db = get_db_manager(request)
        if available_sections is not None:
            array_literal = "{" + ",".join([f'"{item}"' for item in available_sections]) + "}"

            update_query = """
            UPDATE group_meta
            SET available = %s, available_sections = %s, updated_at = CURRENT_TIMESTAMP
            WHERE group_name = %s
            """

            params = (group.available, array_literal, group.group_name)
            app_db.config_db_manager.execute_update_delete(update_query, params)
        else:
            update_query = """
            UPDATE group_meta
            SET available = %s, updated_at = CURRENT_TIMESTAMP
            WHERE group_name = %s
            """
            params = (group.available, group.group_name)
            app_db.config_db_manager.execute_update_delete(update_query, params)

        return {"detail": "Group permissions updated successfully"}
    except Exception as e:
        logger.error("Error updating group permissions: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        ) from e

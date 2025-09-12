import logging
from pydantic import BaseModel
from typing import Optional
from fastapi import APIRouter, Request, HTTPException
from controller.helper.singletonHelper import get_db_manager
from controller.manager.managerBaseController import get_user_id_from_admin
from service.database.models.user import User

logger = logging.getLogger("manager-user-controller")
router = APIRouter(prefix="/user", tags=["Manager"])

@router.delete("/edit-user/groups")
async def delete_user_groups(request: Request, user_data: dict):
    manager_user_id = await get_user_id_from_admin(request)

    try:
        app_db = get_db_manager(request)
        user_id = user_data.get("user_id")

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

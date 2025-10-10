"""
Admin Prompt 컨트롤러

관리자용 프롬프트 관리 API 엔드포인트를 제공합니다.
전체 프롬프트 조회, 필터링, 수정, 삭제 등을 담당합니다.
"""

from fastapi import APIRouter, HTTPException, Request, Query
from pydantic import BaseModel
from typing import Optional
import logging
import uuid

from service.database.models.prompts import Prompts
from controller.helper.singletonHelper import get_db_manager
from controller.admin.adminBaseController import validate_superuser
from service.database.logger_helper import create_logger
from controller.admin.adminHelper import get_manager_groups, get_manager_accessible_users, manager_section_access, get_manager_accessible_workflows_ids

logger = logging.getLogger("admin-prompt-controller")
router = APIRouter(prefix="/prompt", tags=["Admin"])

class CreatePromptRequest(BaseModel):
    prompt_title: str
    prompt_content: str
    public_available: bool = False
    language: Optional[str] = "ko"
    is_template: Optional[bool] = False

class DeletePromptRequest(BaseModel):
    prompt_uid: str

class UpdatePromptRequest(BaseModel):
    prompt_uid: str
    prompt_title: Optional[str] = None
    prompt_content: Optional[str] = None
    public_available: Optional[bool] = None
    language: Optional[str] = None
    is_template: Optional[bool] = None

@router.get("/list")
async def get_all_prompts(
    request: Request,
    limit: int = Query(1000, ge=1, le=5000, description="Number of prompts to return"),
    offset: int = Query(0, ge=0, description="Number of prompts to skip"),
    language: Optional[str] = Query(None, description="Filter by language (en, ko)"),
    user_id: Optional[int] = Query(None, description="Filter by user_id"),
    is_template: Optional[bool] = Query(None, description="Filter by template status"),
):
    """관리자용: 모든 프롬프트 목록을 반환합니다."""
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, val_superuser.get("user_id"), request)
    section_access = manager_section_access(app_db, val_superuser.get("user_id"), ["prompt-store"])
    if not section_access:
        backend_log.warn(f"User {val_superuser.get('user_id')} attempted to access prompt store without permission")
        raise HTTPException(
            status_code=403,
            detail="Prompt store access required"
        )

    backend_log.info(
        "Retrieving all prompts (admin)",
        metadata={
            "limit": limit,
            "offset": offset,
            "language": language,
            "user_id": user_id,
            "is_template": is_template,
        },
    )

    try:
        if user_id:
            if val_superuser.get("user_type") != "superuser":
                accessible_user_ids = [user.id for user in get_manager_accessible_users(app_db, val_superuser.get("user_id"))]
                if user_id not in accessible_user_ids:
                    backend_log.warn(f"User {val_superuser.get('user_id')} attempted to access prompts of user {user_id} without permission")
                    raise HTTPException(
                        status_code=403,
                        detail="You do not have access to prompts of this user"
                    )
            conditions = {"user_id": user_id}
        else:
            if val_superuser.get("user_type") != "superuser":
                accessible_user_ids = [user.id for user in get_manager_accessible_users(app_db, val_superuser.get("user_id"))]
                conditions = {"user_id__in__": accessible_user_ids}
            else:
                conditions = {}
        if language:
            conditions["language"] = language
        if is_template is not None:
            conditions["is_template"] = is_template

        # 모든 프롬프트 조회
        all_prompts = app_db.find_by_condition(
            Prompts,
            conditions=conditions if conditions else {},
            limit=limit,
            offset=offset,
            orderby="updated_at",
            orderby_asc=False,
            join_user=True,
        )

        prompt_list = []
        for prompt in all_prompts:
            prompt_data = {
                "id": prompt.id,
                "prompt_uid": prompt.prompt_uid,
                "prompt_title": prompt.prompt_title,
                "prompt_content": prompt.prompt_content,
                "public_available": prompt.public_available,
                "is_template": prompt.is_template,
                "language": prompt.language,
                "user_id": prompt.user_id,
                "username": getattr(prompt, 'username', None),
                "full_name": getattr(prompt, 'full_name', None),
                "created_at": prompt.created_at,
                "updated_at": prompt.updated_at,
                "metadata": prompt.metadata,
            }
            prompt_list.append(prompt_data)

        # id 기준 중복 제거
        unique_prompts = {prompt['id']: prompt for prompt in prompt_list}.values()
        prompt_list = list(unique_prompts)

        response_data = {
            "prompts": prompt_list,
            "total_count": len(prompt_list),
            "limit": limit,
            "offset": offset,
            "filters_applied": {
                "language": language,
                "user_id": user_id,
                "is_template": is_template,
            },
        }

        backend_log.success(
            "Successfully retrieved all prompts (admin)",
            metadata={
                "prompt_count": len(prompt_list),
                "limit": limit,
                "offset": offset,
            },
        )

        return response_data

    except Exception as e:
        backend_log.error(
            "Failed to retrieve all prompts (admin)",
            exception=e,
            metadata={"limit": limit, "offset": offset, "language": language, "user_id": user_id},
        )
        logger.error("Error getting all prompts (admin): %s", e)
        raise HTTPException(status_code=500, detail="Failed to retrieve prompt list")


@router.post("/create")
async def create_prompt(
    request: Request,
    prompt_data: CreatePromptRequest
):
    """관리자용: 새로운 프롬프트를 생성합니다."""
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, val_superuser.get("user_id"), request)
    section_access = manager_section_access(app_db, val_superuser.get("user_id"), ["prompt-store"])
    if not section_access:
        backend_log.warn(f"User {val_superuser.get('user_id')} attempted to access prompt store without permission")
        raise HTTPException(
            status_code=403,
            detail="Prompt store access required"
        )
    backend_log.info(
        "Creating new prompt (admin)",
        metadata={
            "prompt_title": prompt_data.prompt_title,
            "public_available": prompt_data.public_available,
            "language": prompt_data.language,
            "is_template": prompt_data.is_template,
        },
    )

    try:
        # prompt_uid 생성 (타이틀 기반)
        base_uid = prompt_data.prompt_title.replace(' ', '_').lower()
        base_uid = ''.join(c for c in base_uid if c.isalnum() or c == '_')
        unique_suffix = str(uuid.uuid4())[:8]
        prompt_uid = f"{base_uid}_{unique_suffix}"

        # 새로운 Prompt 객체 생성
        new_prompt = Prompts(
            user_id=val_superuser.get("user_id"),
            prompt_uid=prompt_uid,
            prompt_title=prompt_data.prompt_title,
            prompt_content=prompt_data.prompt_content,
            public_available=prompt_data.public_available,
            is_template=prompt_data.is_template,
            language=prompt_data.language,
            metadata={}
        )

        # 데이터베이스에 저장
        result = app_db.insert(new_prompt)

        if result and result.get("result") == "success":
            created_prompt = {
                "id": result.get("id"),
                "prompt_uid": prompt_uid,
                "prompt_title": prompt_data.prompt_title,
                "prompt_content": prompt_data.prompt_content,
                "public_available": prompt_data.public_available,
                "is_template": prompt_data.is_template,
                "language": prompt_data.language,
                "user_id": val_superuser.get("user_id"),
                "metadata": {}
            }

            backend_log.success(
                "Successfully created new prompt (admin)",
                metadata={
                    "prompt_id": result.get("id"),
                    "prompt_uid": prompt_uid,
                    "prompt_title": prompt_data.prompt_title,
                },
            )

            return {
                "success": True,
                "message": "Prompt created successfully",
                "prompt": created_prompt
            }
        else:
            backend_log.error(
                "Failed to create prompt (admin) - database insert failed",
                metadata={"result": result},
            )
            raise HTTPException(status_code=500, detail="Failed to create prompt")

    except Exception as e:
        backend_log.error(
            "Failed to create prompt (admin)",
            exception=e,
            metadata={
                "prompt_title": prompt_data.prompt_title,
            },
        )
        logger.error("Error creating prompt (admin): %s", e)
        raise HTTPException(status_code=500, detail="Failed to create prompt")


@router.post("/update")
async def update_prompt(
    request: Request,
    prompt_data: UpdatePromptRequest
):
    """관리자용: 프롬프트를 업데이트합니다."""
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, val_superuser.get("user_id"), request)
    section_access = manager_section_access(app_db, val_superuser.get("user_id"), ["prompt-store"])
    if not section_access:
        backend_log.warn(f"User {val_superuser.get('user_id')} attempted to access prompt store without permission")
        raise HTTPException(
            status_code=403,
            detail="Prompt store access required"
        )

    backend_log.info(
        "Updating prompt (admin)",
        metadata={
            "prompt_uid": prompt_data.prompt_uid,
        },
    )

    try:
        # 프롬프트 조회 (user_id 조건 없이)
        exist_prompt_data = app_db.find_by_condition(
            Prompts,
            {"prompt_uid": prompt_data.prompt_uid},
            limit=1
        )

        if not exist_prompt_data:
            backend_log.warn(
                "Prompt not found (admin)",
                metadata={"prompt_uid": prompt_data.prompt_uid},
            )
            raise HTTPException(status_code=404, detail="Prompt not found")

        if val_superuser.get("user_type") != "superuser":
            accessible_user_ids = [user.id for user in get_manager_accessible_users(app_db, val_superuser.get("user_id"))]
            if exist_prompt_data[0].user_id not in accessible_user_ids:
                backend_log.warn(f"User {val_superuser.get('user_id')} attempted to update prompt of user {exist_prompt_data[0].user_id} without permission")
                raise HTTPException(
                    status_code=403,
                    detail="You do not have access to update this prompt"
                )

        exist_prompt_data = exist_prompt_data[0]

        # 업데이트
        if prompt_data.prompt_title is not None:
            exist_prompt_data.prompt_title = prompt_data.prompt_title
        if prompt_data.prompt_content is not None:
            exist_prompt_data.prompt_content = prompt_data.prompt_content
        if prompt_data.public_available is not None:
            exist_prompt_data.public_available = prompt_data.public_available
        if prompt_data.language is not None:
            exist_prompt_data.language = prompt_data.language
        if prompt_data.is_template is not None:
            exist_prompt_data.is_template = prompt_data.is_template

        app_db.update(exist_prompt_data)

        backend_log.success(
            "Successfully updated prompt (admin)",
            metadata={
                "prompt_uid": prompt_data.prompt_uid,
                "prompt_id": exist_prompt_data.id,
            },
        )

        return {
            "success": True,
            "message": "Prompt updated successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        backend_log.error(
            "Failed to update prompt (admin)",
            exception=e,
            metadata={
                "prompt_uid": prompt_data.prompt_uid,
            },
        )
        logger.error("Error updating prompt (admin): %s", e)
        raise HTTPException(status_code=500, detail="Failed to update prompt")


@router.delete("/delete")
async def delete_prompt(
    request: Request,
    prompt_data: DeletePromptRequest
):
    """관리자용: 프롬프트를 삭제합니다."""
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, val_superuser.get("user_id"), request)
    section_access = manager_section_access(app_db, val_superuser.get("user_id"), ["prompt-store"])
    if not section_access:
        backend_log.warn(f"User {val_superuser.get('user_id')} attempted to access prompt store without permission")
        raise HTTPException(
            status_code=403,
            detail="Prompt store access required"
        )

    backend_log.info(
        "Deleting prompt (admin)",
        metadata={
            "prompt_uid": prompt_data.prompt_uid,
        },
    )

    try:
        # 프롬프트 조회 (user_id 조건 없이)
        prompt = app_db.find_by_condition(
            Prompts,
            {"prompt_uid": prompt_data.prompt_uid},
            limit=1
        )

        if not prompt:
            backend_log.warn(
                "Prompt not found (admin)",
                metadata={"prompt_uid": prompt_data.prompt_uid},
            )
            raise HTTPException(status_code=404, detail="Prompt not found")

        if val_superuser.get("user_type") != "superuser":
            accessible_user_ids = [user.id for user in get_manager_accessible_users(app_db, val_superuser.get("user_id"))]
            if prompt[0].user_id not in accessible_user_ids:
                backend_log.warn(f"User {val_superuser.get('user_id')} attempted to delete prompt of user {prompt[0].user_id} without permission")
                raise HTTPException(
                    status_code=403,
                    detail="You do not have access to delete this prompt"
                )

        delete_result = app_db.delete_by_condition(
            Prompts,
            {"prompt_uid": prompt_data.prompt_uid}
        )

        if delete_result:
            backend_log.success(
                "Successfully deleted prompt (admin)",
                metadata={
                    "prompt_uid": prompt_data.prompt_uid,
                    "prompt_id": prompt[0].id,
                },
            )
            return {
                "success": True,
                "message": "Prompt deleted successfully"
            }
        else:
            backend_log.error(
                "Failed to delete prompt (admin) - database delete failed",
                metadata={"prompt_uid": prompt_data.prompt_uid},
            )
            raise HTTPException(status_code=500, detail="Failed to delete prompt")

    except HTTPException:
        raise
    except Exception as e:
        backend_log.error(
            "Failed to delete prompt (admin)",
            exception=e,
            metadata={
                "prompt_uid": prompt_data.prompt_uid,
            },
        )
        logger.error("Error deleting prompt (admin): %s", e)
        raise HTTPException(status_code=500, detail="Failed to delete prompt")

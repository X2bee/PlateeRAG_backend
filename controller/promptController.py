"""
Prompt 컨트롤러

프롬프트 관리 API 엔드포인트를 제공합니다.
프롬프트 조회, 필터링 등을 담당합니다.
"""

from fastapi import APIRouter, HTTPException, Request, Query
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import logging
import uuid

from service.database.models.prompts import Prompts
from controller.helper.singletonHelper import get_db_manager
from controller.helper.controllerHelper import extract_user_id_from_request
from service.database.logger_helper import create_logger

logger = logging.getLogger("prompt-controller")
router = APIRouter(prefix="/api/prompt", tags=["prompt"])

class CreatePromptRequest(BaseModel):
    prompt_title: str
    prompt_content: str
    public_available: bool = False
    language: Optional[str] = "ko"

@router.get("/list")
async def get_prompt_list(
    request: Request,
    limit: int = Query(300, ge=1, le=500, description="Number of prompts to return"),
    offset: int = Query(0, ge=0, description="Number of prompts to skip"),
    language: Optional[str] = Query(None, description="Filter by language (en, ko)"),
):
    """프롬프트 목록을 반환합니다."""
    try:
        user_id = extract_user_id_from_request(request)
    except:
        user_id = None

    app_db = get_db_manager(request)
    # 응답 데이터 구성
    prompt_list = []

    # user_id가 있는 경우에만 로깅
    if user_id:
        backend_log = create_logger(app_db, user_id, request)
        backend_log.info(
            "Retrieving prompt list",
            metadata={
                "limit": limit,
                "offset": offset,
                "language": language,
            },
        )

        my_prompts = app_db.find_by_condition(
            Prompts,
            conditions={"user_id": user_id},
            limit=1000,
            orderby="id",
            orderby_asc=False,
            join_user=True,
        )
        for prompt in my_prompts:
            prompt_data = {
                "id": prompt.id,
                "prompt_uid": prompt.prompt_uid,
                "prompt_title": prompt.prompt_title,
                "prompt_content": prompt.prompt_content,
                "public_available": prompt.public_available,
                "is_template": prompt.is_template,
                "language": prompt.language,
                "user_id": prompt.user_id,
                "username": prompt.username,
                "full_name": prompt.full_name,
                "created_at": prompt.created_at,
                "updated_at": prompt.updated_at,
                "metadata": prompt.metadata,
            }
            prompt_list.append(prompt_data)

    try:
        template_prompts = app_db.find_by_condition(
            Prompts,
            conditions={"is_template": True},
            limit=limit,
            offset=offset,
            orderby="id",
            orderby_asc=False,
        )

        for prompt in template_prompts:
            prompt_data = {
                "id": prompt.id,
                "prompt_uid": prompt.prompt_uid,
                "prompt_title": prompt.prompt_title,
                "prompt_content": prompt.prompt_content,
                "public_available": prompt.public_available,
                "is_template": prompt.is_template,
                "language": prompt.language,
                "user_id": prompt.user_id,
                "created_at": prompt.created_at,
                "updated_at": prompt.updated_at,
                "metadata": prompt.metadata,
            }
            prompt_list.append(prompt_data)

        shared_prompts = app_db.find_by_condition(
            Prompts,
            conditions={"public_available": True, "is_template": False, "user_id__not__": user_id} if user_id else {"public_available": True, "is_template": False},
            limit=limit,
            offset=offset,
            orderby="id",
            orderby_asc=False,
            join_user=True,
        )

        for prompt in shared_prompts:
            prompt_data = {
                "id": prompt.id,
                "prompt_uid": prompt.prompt_uid,
                "prompt_title": prompt.prompt_title,
                "prompt_content": prompt.prompt_content,
                "public_available": prompt.public_available,
                "is_template": prompt.is_template,
                "language": prompt.language,
                "user_id": prompt.user_id,
                "username": prompt.username,
                "full_name": prompt.full_name,
                "created_at": prompt.created_at,
                "updated_at": prompt.updated_at,
                "metadata": prompt.metadata,
            }
            prompt_list.append(prompt_data)

        response_data = {
            "prompts": prompt_list,
            "total_count": len(prompt_list),
            "limit": limit,
            "offset": offset,
            "filters_applied": {
                "language": language,
            },
        }

        if user_id:
            backend_log.success(
                "Successfully retrieved prompt list",
                metadata={
                    "prompt_count": len(prompt_list),
                    "limit": limit,
                    "offset": offset,
                },
            )

        return response_data

    except Exception as e:
        if user_id:
            backend_log.error(
                "Failed to retrieve prompt list",
                exception=e,
                metadata={"limit": limit, "offset": offset, "language": language},
            )
        logger.error("Error getting prompt list: %s", e)
        raise HTTPException(status_code=500, detail="Failed to retrieve prompt list")


@router.post("/create")
async def create_prompt(
    request: Request,
    prompt_data: CreatePromptRequest
):
    """새로운 프롬프트를 생성합니다."""
    try:
        user_id = extract_user_id_from_request(request)
    except:
        raise HTTPException(status_code=401, detail="Authentication required")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    backend_log.info(
        "Creating new prompt",
        metadata={
            "prompt_title": prompt_data.prompt_title,
            "public_available": prompt_data.public_available,
            "language": prompt_data.language,
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
            user_id=user_id,
            prompt_uid=prompt_uid,
            prompt_title=prompt_data.prompt_title,
            prompt_content=prompt_data.prompt_content,
            public_available=prompt_data.public_available,
            is_template=False,
            language=prompt_data.language,
            metadata={}
        )

        # 데이터베이스에 저장
        result = app_db.insert(new_prompt)

        if result and result.get("result") == "success":
            # 생성된 프롬프트 정보 반환
            created_prompt = {
                "id": result.get("id"),
                "prompt_uid": prompt_uid,
                "prompt_title": prompt_data.prompt_title,
                "prompt_content": prompt_data.prompt_content,
                "public_available": prompt_data.public_available,
                "is_template": False,
                "language": prompt_data.language,
                "user_id": user_id,
                "metadata": {}
            }

            backend_log.success(
                "Successfully created new prompt",
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
                "Failed to create prompt - database insert failed",
                metadata={"result": result},
            )
            raise HTTPException(status_code=500, detail="Failed to create prompt")

    except Exception as e:
        backend_log.error(
            "Failed to create prompt",
            exception=e,
            metadata={
                "prompt_title": prompt_data.prompt_title,
                "user_id": user_id,
            },
        )
        logger.error("Error creating prompt: %s", e)
        raise HTTPException(status_code=500, detail="Failed to create prompt")

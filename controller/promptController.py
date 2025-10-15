"""
Prompt 컨트롤러

프롬프트 관리 API 엔드포인트를 제공합니다.
프롬프트 조회, 필터링 등을 담당합니다.
"""

from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import logging
import uuid
import io
import json
import pandas as pd
from datetime import datetime, timezone, timedelta

from service.database.models.prompts import Prompts, PromptStoreRating
from controller.helper.singletonHelper import get_db_manager
from controller.helper.controllerHelper import extract_user_id_from_request, require_admin_access
from service.database.logger_helper import create_logger

logger = logging.getLogger("prompt-controller")
router = APIRouter(prefix="/api/prompt", tags=["prompt"])

class CreatePromptRequest(BaseModel):
    prompt_title: str
    prompt_content: str
    public_available: bool = False
    language: Optional[str] = "ko"

class DeletePromptRequest(BaseModel):
    prompt_uid: str

class UpdatePromptRequest(BaseModel):
    prompt_uid: str
    prompt_title: Optional[str] = None
    prompt_content: Optional[str] = None
    public_available: Optional[bool] = None
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

        my_prompt_conditions = {"user_id": user_id}
        if language:
            my_prompt_conditions["language"] = language

        my_prompts = app_db.find_by_condition(
            Prompts,
            conditions=my_prompt_conditions,
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
                "rating_count": prompt.rating_count,
                "rating_sum": prompt.rating_sum,
            }
            prompt_list.append(prompt_data)

    try:
        template_conditions = {"is_template": True}
        if language:
            template_conditions["language"] = language

        template_prompts = app_db.find_by_condition(
            Prompts,
            conditions=template_conditions,
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
                "username": None,
                "full_name": None,
                "created_at": prompt.created_at,
                "updated_at": prompt.updated_at,
                "metadata": prompt.metadata,
                "rating_count": prompt.rating_count,
                "rating_sum": prompt.rating_sum,
            }
            prompt_list.append(prompt_data)

        shared_conditions = {"public_available": True, "is_template": False, "user_id__not__": user_id} if user_id else {"public_available": True, "is_template": False}
        if language:
            shared_conditions["language"] = language

        shared_prompts = app_db.find_by_condition(
            Prompts,
            conditions=shared_conditions,
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
                "rating_count": prompt.rating_count,
                "rating_sum": prompt.rating_sum,
            }
            prompt_list.append(prompt_data)

        # id 기준 중복 제거 (딕셔너리 사용)
        unique_prompts = {prompt['id']: prompt for prompt in prompt_list}.values()
        prompt_list = list(unique_prompts)

        # updated_at 기준 최신순 정렬
        prompt_list.sort(key=lambda x: x.get('updated_at', ''), reverse=True)

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

@router.delete("/delete")
async def delete_prompt(
    request: Request,
    prompt_data: DeletePromptRequest
):
    """프롬프트를 삭제합니다."""
    try:
        user_id = extract_user_id_from_request(request)
    except:
        raise HTTPException(status_code=401, detail="Authentication required")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    backend_log.info(
        "Deleting prompt",
        metadata={
            "prompt_title": prompt_data.prompt_title,
            "public_available": prompt_data.public_available,
            "language": prompt_data.language,
        },
    )

    try:
        existing_prompt_data = app_db.find_by_condition(Prompts, {"prompt_uid": prompt_data.prompt_uid, "user_id": user_id}, limit=1)
        if not existing_prompt_data:
            backend_log.warn(
                "Prompt not found or access denied",
                metadata={"prompt_uid": prompt_data.prompt_uid, "user_id": user_id},
            )
            raise HTTPException(status_code=404, detail="Prompt not found or access denied")
        delete_result = app_db.delete_by_condition(Prompts, {"prompt_uid": existing_prompt_data.prompt_uid, "user_id": user_id})
        if delete_result and delete_result.get("result") == "success":
            backend_log.success(
                "Successfully deleted prompt",
                metadata={
                    "prompt_uid": existing_prompt_data.prompt_uid,
                    "user_id": user_id,
                },
            )

            app_db.delete_by_condition(PromptStoreRating, {"prompt_uid": existing_prompt_data.prompt_uid, "prompt_store_id": existing_prompt_data.id})
            return {
                "success": True,
                "message": "Prompt deleted successfully"
            }

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

@router.post("/update")
async def update_prompt(
    request: Request,
    prompt_data: UpdatePromptRequest
):
    """프롬프트를 업데이트합니다."""
    try:
        user_id = extract_user_id_from_request(request)
    except:
        raise HTTPException(status_code=401, detail="Authentication required")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    backend_log.info(
        "Updating prompt",
        metadata={
            "prompt_title": prompt_data.prompt_title,
            "public_available": prompt_data.public_available,
            "language": prompt_data.language,
        },
    )

    try:
        exist_prompt_data = app_db.find_by_condition(Prompts, {"prompt_uid": prompt_data.prompt_uid, "user_id": user_id}, limit=1)
        if not exist_prompt_data:
            backend_log.warn(
                "Prompt not found or access denied",
                metadata={"prompt_uid": prompt_data.prompt_uid, "user_id": user_id},
            )
            raise HTTPException(status_code=404, detail="Prompt not found or access denied")

        exist_prompt_data = exist_prompt_data[0]
        exist_prompt_data.prompt_title = prompt_data.prompt_title if prompt_data.prompt_title is not None else exist_prompt_data.prompt_title
        exist_prompt_data.prompt_content = prompt_data.prompt_content if prompt_data.prompt_content is not None else exist_prompt_data.prompt_content
        exist_prompt_data.public_available = prompt_data.public_available if prompt_data.public_available is not None else exist_prompt_data.public_available
        exist_prompt_data.language = prompt_data.language if prompt_data.language is not None else exist_prompt_data.language
        app_db.update(exist_prompt_data)
        backend_log.success(
            "Successfully updated prompt",
            metadata={
                "prompt_uid": prompt_data.prompt_uid,
                "user_id": user_id,
            },
        )
        return {
            "success": True,
            "message": "Prompt updated successfully"
        }
    except Exception as e:
        backend_log.error(
            "Failed to update prompt",
            exception=e,
            metadata={
                "prompt_title": prompt_data.prompt_title,
                "user_id": user_id,
            },
        )
        logger.error("Error updating prompt: %s", e)
        raise HTTPException(status_code=500, detail="Failed to update prompt")


@router.post("/rating/{prompt_uid}")
async def rate_prompt(
    request: Request,
    prompt_uid: str,
    user_id: int,
    is_template: bool,
    rating: int
):
    """
    특정 prompt에 대한 평가를 추가합니다.
    """
    login_user_id = extract_user_id_from_request(request)
    if not login_user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    if rating < 1 or rating > 5:
        raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, login_user_id, request)

    search_conditions = {
        "user_id": user_id,
        "prompt_uid": prompt_uid,
    }

    if is_template:
        search_conditions.pop("user_id", None)
        search_conditions["is_template"] = True

    try:
        backend_log.info(
            "Starting prompt rating",
            metadata={"prompt_uid": prompt_uid}
        )

        existing_data = app_db.find_by_condition(
            Prompts,
            search_conditions,
            limit=1
        )

        if not existing_data:
            backend_log.warn(
                "Prompt not found for rating",
                metadata={"prompt_uid": prompt_uid}
            )
            raise HTTPException(status_code=404, detail=f"Prompt '{prompt_uid}' not found")

        existing_data = existing_data[0]

        existing_rating = app_db.find_by_condition(
            PromptStoreRating,
            {
                "user_id": login_user_id,
                "prompt_store_id": existing_data.id,
            },
            limit=1
        )

        if existing_rating:
            # 기존 평가 업데이트
            existing_rating = existing_rating[0]
            existing_rating_score = existing_rating.rating
            existing_rating.rating = rating
            existing_data.rating_sum = existing_data.rating_sum - existing_rating_score + rating
            app_db.update(existing_rating)
            app_db.update(existing_data)
            backend_log.info(
                "Updated existing prompt rating",
                metadata={
                    "prompt_uid": prompt_uid,
                    "rating": rating,
                    "prompt_store_id": existing_data.id,
                }
            )
        else:
            # 새로운 평가 추가
            new_rating = PromptStoreRating(
                user_id=login_user_id,
                prompt_store_id=existing_data.id,
                prompt_uid=existing_data.prompt_uid,
                prompt_title=existing_data.prompt_title,
                rating=rating
            )
            app_db.insert(new_rating)
            existing_data.rating_count += 1
            existing_data.rating_sum += rating
            app_db.update(existing_data)
            backend_log.info(
                "Inserted new prompt rating",
                metadata={
                    "prompt_uid": prompt_uid,
                    "rating": rating,
                    "prompt_store_id": existing_data.id,
                }
            )

        backend_log.success(
            "Prompt rating completed successfully",
            metadata={
                "prompt_uid": prompt_uid,
                "rating": rating
            }
        )

        logger.info(f"Prompt rated successfully: {prompt_uid}")
        return {
            "success": True,
            "message": f"Prompt '{prompt_uid}' rated successfully",
            "rating": rating,
            "average_rating": existing_data.rating_sum / existing_data.rating_count if existing_data.rating_count > 0 else 0
        }

    except HTTPException:
        raise
    except Exception as e:
        backend_log.error(
            "Prompt rating failed",
            exception=e,
            metadata={"prompt_uid": prompt_uid}
        )
        logger.error(f"Error rating prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to rate prompt: {str(e)}")

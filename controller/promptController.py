"""
Prompt 컨트롤러

프롬프트 관리 API 엔드포인트를 제공합니다.
프롬프트 조회, 필터링 등을 담당합니다.
"""

from fastapi import APIRouter, HTTPException, Request, Query
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import logging

from service.database.models.prompts import Prompts
from controller.helper.singletonHelper import get_db_manager
from controller.helper.controllerHelper import extract_user_id_from_request
from service.database.logger_helper import create_logger

logger = logging.getLogger("prompt-controller")
router = APIRouter(prefix="/prompt", tags=["prompt"])

@router.get("/list")
async def get_prompt_list(
    request: Request,
    limit: int = Query(300, ge=1, le=500, description="Number of prompts to return"),
    offset: int = Query(0, ge=0, description="Number of prompts to skip"),
    language: Optional[str] = Query(None, description="Filter by language (en, ko)"),
    is_template: Optional[bool] = Query(None, description="Filter by template status"),
    public_available: Optional[bool] = Query(None, description="Filter by public availability"),
    search: Optional[str] = Query(None, description="Search in prompt title and content")
):
    """프롬프트 목록을 반환합니다."""
    try:
        user_id = extract_user_id_from_request(request)
    except:
        user_id = None

    app_db = get_db_manager(request)

    # user_id가 있는 경우에만 로깅
    if user_id:
        backend_log = create_logger(app_db, user_id, request)
        backend_log.info("Retrieving prompt list",
                        metadata={"limit": limit, "offset": offset, "language": language,
                                "is_template": is_template, "public_available": public_available,
                                "search": search})

    try:
        # 필터 조건 구성
        conditions = {}

        if language:
            conditions["language"] = language
        if is_template is not None:
            conditions["is_template"] = is_template
        if public_available is not None:
            conditions["public_available"] = public_available
        if search:
            # 제목이나 내용에서 검색 (LIKE 검색)
            conditions["prompt_title__like__"] = search

        # 프롬프트 조회
        prompts = app_db.find_by_condition(
            Prompts,
            conditions=conditions,
            limit=limit,
            offset=offset,
            orderby="id",
            orderby_asc=False
        )

        # 응답 데이터 구성
        prompt_list = []
        for prompt in prompts:
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
                "metadata": prompt.metadata
            }
            prompt_list.append(prompt_data)

        response_data = {
            "prompts": prompt_list,
            "total_count": len(prompt_list),
            "limit": limit,
            "offset": offset,
            "filters_applied": {
                "language": language,
                "is_template": is_template,
                "public_available": public_available,
                "search": search
            }
        }

        if user_id:
            backend_log.success("Successfully retrieved prompt list",
                              metadata={"prompt_count": len(prompt_list),
                                      "limit": limit, "offset": offset,
                                      "filters": conditions})

        return response_data

    except Exception as e:
        if user_id:
            backend_log.error("Failed to retrieve prompt list", exception=e,
                            metadata={"limit": limit, "offset": offset, "language": language})
        logger.error("Error getting prompt list: %s", e)
        raise HTTPException(status_code=500, detail="Failed to retrieve prompt list")

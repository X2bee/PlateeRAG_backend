"""
Admin Prompt 컨트롤러

관리자용 프롬프트 관리 API 엔드포인트를 제공합니다.
전체 프롬프트 조회, 필터링, 수정, 삭제 등을 담당합니다.
"""

from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import logging
import uuid
import io
import json
import pandas as pd
from datetime import datetime, timezone, timedelta

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


def clean_excel_string(text):
    """
    Excel 호환성을 위한 문자열 정제
    - XML 1.0 스펙 완전 준수
    - openpyxl 엔진 호환성 보장
    - 모든 비정상 문자 제거
    """
    if text is None or text == '':
        return ''

    import re
    import unicodedata

    try:
        # 1단계: 기본 문자열 변환 및 타입 체크
        if isinstance(text, (dict, list)):
            text = json.dumps(text, ensure_ascii=False, indent=None)
        else:
            text = str(text)

        # 2단계: 바이트 레벨 정규화 (NFC 정규화)
        try:
            text = unicodedata.normalize('NFC', text)
        except Exception:
            pass

        # 3단계: UTF-16 서러게이트 및 비문자 제거
        text = re.sub(r'[\uD800-\uDFFF\uFDD0-\uFDEF\uFFFE\uFFFF]', '', text)

        # 4단계: NULL 및 제어 문자 완전 제거
        text = text.replace('\x00', '').replace('\uFEFF', '')
        text = re.sub(r'[\x01-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)

        # 5단계: XML 1.0 완전 준수 문자 필터링
        # 허용 범위만 명시적으로 유지
        valid_chars = []
        for char in text:
            try:
                cp = ord(char)
                # XML 1.0 허용 범위: 0x09, 0x0A, 0x0D, 0x20-0xD7FF, 0xE000-0xFFFD
                if (cp == 0x09 or cp == 0x0A or cp == 0x0D or
                    (0x20 <= cp <= 0xD7FF) or
                    (0xE000 <= cp <= 0xFFFD)):
                    valid_chars.append(char)
                # 0x10000-0x10FFFF 범위는 len(char) == 2로 체크
                elif len(char.encode('utf-16-le')) == 4:  # 서러게이트 페어
                    if 0x10000 <= cp <= 0x10FFFF:
                        valid_chars.append(char)
            except (ValueError, TypeError, UnicodeDecodeError):
                continue

        text = ''.join(valid_chars)

        # 6단계: 공백 정규화
        text = re.sub(r'[ \t]+', ' ', text)  # 연속 공백
        text = re.sub(r'\n{3,}', '\n\n', text)  # 3개 이상 줄바꿈 -> 2개로
        text = re.sub(r'[\r\n]+', '\n', text)  # CRLF 혼용 정리

        # 7단계: Excel 수식 방지 (보안)
        if text.startswith(('=', '+', '-', '@', '\t', '\r')):
            text = "'" + text

        # 8단계: 길이 제한 (Excel 셀 최대 32,767자)
        max_len = 32760
        if len(text) > max_len:
            text = text[:max_len] + '...'

        # 9단계: 최종 정리
        text = text.strip()

        # 10단계: 빈 문자열 체크
        if not text or text.isspace():
            return ''

        return text

    except Exception as e:
        logger.error(f"Critical error in clean_excel_string: {str(e)}, input type: {type(text)}")
        return '[ERROR: 데이터 정제 실패]'


@router.get("/download/all-prompts")
async def download_all_prompts(
    request: Request,
    file_format: str = Query("excel", regex="^(excel|csv)$", description="Download format: excel or csv", alias="format"),
    user_id: Optional[int] = None,
    language: Optional[str] = None,
    public_available: Optional[bool] = None,
    is_template: Optional[bool] = None,
):
    """
    관리자용: 모든 프롬프트를 Excel 또는 CSV 파일로 다운로드합니다.
    관리자 권한이 필요합니다.

    Parameters:
    - format: 다운로드 형식 (excel 또는 csv, 기본값: excel)
    - user_id: 특정 사용자의 프롬프트만 필터링 (optional)
    - language: 특정 언어로 필터링 (optional)
    - public_available: 공개 여부로 필터링 (optional)
    - is_template: 템플릿 여부로 필터링 (optional)
    """
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
        backend_log.warn(f"User {val_superuser.get('user_id')} attempted to download prompts without permission")
        raise HTTPException(
            status_code=403,
            detail="Prompt store access required"
        )

    backend_log.info(
        f"Starting prompt {file_format.upper()} download (admin)",
        metadata={
            "format": file_format,
            "user_id_filter": user_id,
            "language": language,
            "public_available": public_available,
            "is_template": is_template,
        },
    )

    try:
        # 조건 설정
        if user_id:
            if val_superuser.get("user_type") != "superuser":
                accessible_user_ids = [user.id for user in get_manager_accessible_users(app_db, val_superuser.get("user_id"))]
                if user_id not in accessible_user_ids:
                    backend_log.warn(f"User {val_superuser.get('user_id')} attempted to download prompts of user {user_id} without permission")
                    raise HTTPException(
                        status_code=403,
                        detail="You do not have access to prompts of this user"
                    )
                conditions = {"user_id": user_id}
            else:
                conditions = {"user_id": user_id}
        else:
            if val_superuser.get("user_type") != "superuser":
                accessible_user_ids = [user.id for user in get_manager_accessible_users(app_db, val_superuser.get("user_id"))]
                conditions = {"user_id__in__": accessible_user_ids}
            else:
                conditions = {}

        if language is not None:
            conditions['language'] = language
        if public_available is not None:
            conditions['public_available'] = public_available
        if is_template is not None:
            conditions['is_template'] = is_template

        # 전체 프롬프트 조회
        prompts_result = app_db.find_by_condition(
            Prompts,
            conditions,
            limit=1000000,  # 대량 조회
            orderby="updated_at",
            orderby_asc=False,
            return_list=True,
            join_user=True
        )

        if not prompts_result:
            backend_log.info(
                "No prompts found for download (admin)",
                metadata={"conditions": conditions}
            )
            raise HTTPException(
                status_code=404,
                detail="No prompts found for the given criteria"
            )

        # 데이터 프레임 생성을 위한 리스트
        excel_data = []

        for prompt in prompts_result:
            try:
                # prompt_content를 문자열로 변환
                prompt_content = str(prompt['prompt_content']) if prompt.get('prompt_content') else ''

                # 메타데이터를 JSON 문자열로 변환
                metadata_str = ''
                if prompt.get('metadata'):
                    try:
                        metadata_str = json.dumps(prompt['metadata'], ensure_ascii=False, indent=None, separators=(',', ':'))
                    except Exception as e:
                        logger.warning(f"Failed to serialize metadata: {e}")
                        metadata_str = str(prompt['metadata'])

                row = {
                    "ID": int(prompt['id']) if prompt.get('id') else 0,
                    "Prompt UID": clean_excel_string(prompt.get('prompt_uid')),
                    "Title": clean_excel_string(prompt.get('prompt_title')),
                    "Content": clean_excel_string(prompt_content),
                    "Language": clean_excel_string(prompt.get('language')),
                    "Public": "Yes" if prompt.get('public_available') else "No",
                    "Template": "Yes" if prompt.get('is_template') else "No",
                    "User ID": int(prompt['user_id']) if prompt.get('user_id') else 0,
                    "Username": clean_excel_string(prompt.get('username')),
                    "Full Name": clean_excel_string(prompt.get('full_name')),
                    "Metadata": clean_excel_string(metadata_str),
                    "Created At": prompt['created_at'].isoformat() if isinstance(prompt.get('created_at'), datetime) else str(prompt.get('created_at', '')),
                    "Updated At": prompt['updated_at'].isoformat() if isinstance(prompt.get('updated_at'), datetime) else str(prompt.get('updated_at', '')),
                }
                excel_data.append(row)

            except Exception as e:
                logger.error(f"Error processing prompt entry: {e}")
                # 에러 발생 시 기본값으로 row 추가
                excel_data.append({
                    "ID": 0,
                    "Prompt UID": "",
                    "Title": "",
                    "Content": "[ERROR: 데이터 처리 실패]",
                    "Language": "",
                    "Public": "",
                    "Template": "",
                    "User ID": 0,
                    "Username": "",
                    "Full Name": "",
                    "Metadata": "",
                    "Created At": "",
                    "Updated At": "",
                })

        # DataFrame 생성 (명시적 dtype 지정)
        df = pd.DataFrame(excel_data, dtype=str)

        # ID와 User ID만 정수형으로 변환
        try:
            df['ID'] = pd.to_numeric(df['ID'], errors='coerce').fillna(0).astype(int)
            df['User ID'] = pd.to_numeric(df['User ID'], errors='coerce').fillna(0).astype(int)
        except Exception:
            pass

        # 파일명 생성 (날짜 포함) - Asia/Seoul timezone (UTC+9)
        kst = timezone(timedelta(hours=9))
        timestamp = datetime.now(kst).strftime("%Y%m%d_%H%M%S")

        output = io.BytesIO()

        # 형식에 따라 파일 생성
        if file_format == "csv":
            # CSV 파일 생성
            try:
                # CSV로 변환 (UTF-8 with BOM for Excel compatibility)
                csv_string = df.to_csv(index=False, encoding='utf-8-sig', lineterminator='\n')
                output.write(csv_string.encode('utf-8-sig'))
                output.seek(0)

                filename = f"prompts_{timestamp}.csv"
                media_type = "text/csv"

                backend_log.success(
                    "Successfully generated CSV file for prompts (admin)",
                    metadata={
                        "conditions": conditions,
                        "prompt_count": len(excel_data),
                        "filename": filename
                    }
                )

            except Exception as e:
                logger.error(f"Error creating CSV file: {e}", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail=f"CSV 파일 생성 실패: {str(e)}"
                )

        else:  # file_format == "excel"
            # Excel 파일 생성
            try:
                with pd.ExcelWriter(
                    output,
                    engine='openpyxl',
                    engine_kwargs={'write_only': False}
                ) as writer:
                    df.to_excel(
                        writer,
                        index=False,
                        sheet_name='Prompts',
                        na_rep='',
                    )

                    # 워크시트 접근하여 추가 설정
                    worksheet = writer.sheets['Prompts']
                    workbook = writer.book

                    # 워크북 레벨 설정
                    try:
                        if hasattr(workbook, 'calculation'):
                            workbook.calculation.calcMode = 'manual'
                            workbook.calculation.fullCalcOnLoad = False
                    except AttributeError:
                        pass

                    # 열 너비 자동 조정 및 셀 타입 설정
                    for column in worksheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter

                        for cell in column:
                            try:
                                if cell.value is not None:
                                    cell_value_str = str(cell.value)
                                    max_length = max(max_length, len(cell_value_str))

                                    # 데이터 행만 문자열로 강제 (헤더 제외, ID와 User ID는 숫자 유지)
                                    if cell.row > 1 and column_letter not in ['A', 'H']:  # A=ID, H=User ID
                                        # 수식으로 해석될 수 있는 문자 처리
                                        if cell_value_str.startswith(('=', '+', '-', '@')):
                                            cell.value = "'" + cell_value_str
                                        cell.data_type = 's'  # 문자열 타입
                            except Exception as e:
                                logger.debug(f"Error processing cell {cell.coordinate}: {e}")
                                continue

                        # 열 너비 설정
                        # Content 열은 더 넓게
                        if column_letter == 'D':  # Content 열
                            adjusted_width = 80
                        else:
                            adjusted_width = min(max(max_length + 2, 10), 100)

                        worksheet.column_dimensions[column_letter].width = adjusted_width

                output.seek(0)

                filename = f"prompts_{timestamp}.xlsx"
                media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

                backend_log.success(
                    "Successfully generated Excel file for prompts (admin)",
                    metadata={
                        "conditions": conditions,
                        "prompt_count": len(excel_data),
                        "filename": filename
                    }
                )

            except Exception as e:
                logger.error(f"Error creating Excel file: {e}", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail=f"Excel 파일 생성 실패: {str(e)}"
                )

        # StreamingResponse로 파일 반환
        return StreamingResponse(
            output,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        backend_log.error(
            f"Error generating {file_format.upper()} file for prompts (admin)",
            exception=e,
            metadata={
                "format": file_format,
                "conditions": {
                    "user_id": user_id,
                    "language": language,
                    "public_available": public_available,
                    "is_template": is_template
                }
            }
        )
        logger.error(f"Error generating {file_format.upper()} file (admin): {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        ) from e

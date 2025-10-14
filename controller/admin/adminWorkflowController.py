import logging
import json
import string
import secrets
import os
from datetime import datetime
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from controller.helper.singletonHelper import get_db_manager
import io
import pandas as pd
from controller.admin.adminBaseController import validate_superuser
from controller.helper.utils.data_parsers import parse_input_data
from service.database.logger_helper import create_logger
from controller.admin.adminHelper import get_manager_groups, get_manager_accessible_users, manager_section_access, get_manager_accessible_workflows_ids
from service.database.models.executor import ExecutionIO
from service.database.models.workflow import WorkflowMeta
from service.database.models.deploy import DeployMeta
from service.database.models.user import User
from controller.helper.utils.data_parsers import safe_round_float
from service.database.models.performance import NodePerformance

logger = logging.getLogger("admin-workflow-controller")
router = APIRouter(prefix="/workflow", tags=["Admin"])

def extract_result_from_json(json_string):
    """
    Extract the 'result' field from a JSON string.
    If parsing fails or 'result' is not found, return the original string.
    """
    if not json_string:
        return json_string

    try:
        data = json.loads(json_string)
        return data.get("result", json_string)
    except (json.JSONDecodeError, TypeError):
        return json_string

def clean_excel_string(text):
    """
    Excel 호환성을 위한 최강 문자열 정제
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

def parse_actual_output(output):
    """
    Parse and clean output by removing specific tags and patterns.
    Enhanced version matching frontend parseActualOutput logic.

    Removes:
    - <think> tags (AI thinking process)
    - <TOOLUSELOG>, <TOOLOUTPUTLOG> tags (tool execution logs)
    - <at> tags (attention/mention tags)
    - [Cite.{...}] patterns (citation references)
    - <FEEDBACK_LOOP> tags (feedback iteration data)
    - <FEEDBACK_STATUS>, <FEEDBACK_RESULT>, <FEEDBACK_REPORT> tags
    - <TODO_DETAILS> tags (task tracking information)
    """
    if not output:
        return ''

    import re

    # Convert to string if needed (handle dict, list, None, etc.)
    if isinstance(output, (dict, list)):
        try:
            output = json.dumps(output, ensure_ascii=False)
        except Exception:
            output = str(output)
    else:
        processed_output = str(output) if output else ''

    processed_output = str(output)

    # 1. Remove <think> tags (case-insensitive)
    processed_output = re.sub(r'<think>[\s\S]*?</think>', '', processed_output, flags=re.IGNORECASE)

    # 2. Remove <TOOLUSELOG> tags
    if '<TOOLUSELOG>' in processed_output and '</TOOLUSELOG>' in processed_output:
        processed_output = re.sub(r'<TOOLUSELOG>[\s\S]*?</TOOLUSELOG>', '', processed_output)

    # 3. Remove <TOOLOUTPUTLOG> tags
    if '<TOOLOUTPUTLOG>' in processed_output and '</TOOLOUTPUTLOG>' in processed_output:
        processed_output = re.sub(r'<TOOLOUTPUTLOG>[\s\S]*?</TOOLOUTPUTLOG>', '', processed_output)

    # 4. Remove <at> tags (case-insensitive)
    if '<at>' in processed_output.lower() and '</at>' in processed_output.lower():
        processed_output = re.sub(r'<at>[\s\S]*?</at>', '', processed_output, flags=re.IGNORECASE)

    # 5. Remove [Cite.{...}] patterns
    if '[Cite.' in processed_output and '}]' in processed_output:
        processed_output = re.sub(r'\[Cite\.\s*\{[\s\S]*?\}\]', '', processed_output)

    # 6. Remove <FEEDBACK_LOOP> tags
    if '<FEEDBACK_LOOP>' in processed_output and '</FEEDBACK_LOOP>' in processed_output:
        processed_output = re.sub(r'<FEEDBACK_LOOP>[\s\S]*?</FEEDBACK_LOOP>', '', processed_output)

    # 7. Remove <FEEDBACK_STATUS> tags
    if '<FEEDBACK_STATUS>' in processed_output and '</FEEDBACK_STATUS>' in processed_output:
        processed_output = re.sub(r'<FEEDBACK_STATUS>[\s\S]*?</FEEDBACK_STATUS>', '', processed_output)

    # 8. Remove <FEEDBACK_RESULT> tags
    if '<FEEDBACK_RESULT>' in processed_output and '</FEEDBACK_RESULT>' in processed_output:
        processed_output = re.sub(r'<FEEDBACK_RESULT>[\s\S]*?</FEEDBACK_RESULT>', '', processed_output)

    # 9. Remove <FEEDBACK_REPORT> tags
    if '<FEEDBACK_REPORT>' in processed_output and '</FEEDBACK_REPORT>' in processed_output:
        processed_output = re.sub(r'<FEEDBACK_REPORT>[\s\S]*?</FEEDBACK_REPORT>', '', processed_output)

    # 10. Remove <TODO_DETAILS> tags
    if '<TODO_DETAILS>' in processed_output and '</TODO_DETAILS>' in processed_output:
        processed_output = re.sub(r'<TODO_DETAILS>[\s\S]*?</TODO_DETAILS>', '', processed_output)

    # 11. Additional aggressive cleaning - remove any remaining internal tags
    # (case-insensitive for common variations)
    processed_output = re.sub(r'<(?:TOOL|FEEDBACK|TODO)[^>]*>[\s\S]*?</(?:TOOL|FEEDBACK|TODO)[^>]*>', '', processed_output, flags=re.IGNORECASE)

    # 12. Clean up excessive whitespace created by tag removal
    processed_output = re.sub(r'\n{3,}', '\n\n', processed_output)  # Max 2 consecutive newlines
    processed_output = re.sub(r' {2,}', ' ', processed_output)  # Multiple spaces to single

    return processed_output.strip()

def process_io_logs_efficient(io_logs):
    """
    Efficiently process io_logs using map and dictionary unpacking.
    """
    def process_single_log(log):
        log_dict = {k: v for k, v in log.__dict__.items() if not k.startswith('_')}
        log_dict.update({
            "input_data": extract_result_from_json(log.input_data),
            "output_data": extract_result_from_json(log.output_data)
        })
        return log_dict

    return list(map(process_single_log, io_logs))

@router.get("/admin-io-logs")
async def get_io_logs_by_id(request: Request, user_id: int = None, workflow_name: str = None, workflow_id: str = None):
    """
    관리자용 ExecutionIO 로그를 반환합니다. user_id가 없으면 모든 로그를 가져올 수 있습니다.
    """
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, val_superuser.get("user_id"), request)
    section_access = manager_section_access(app_db, val_superuser.get("user_id"), ["workflow-monitoring", "chat-monitoring", "workflow-management"])
    if not section_access:
        backend_log.warn(f"User {val_superuser.get('user_id')} attempted to access IO logs without permission")
        raise HTTPException(
            status_code=403,
            detail="Group permissions access required"
        )
    try:
        if user_id:
            if val_superuser.get("user_type") != "superuser":
                manager_accessible_users = get_manager_accessible_users(app_db, val_superuser.get("user_id"))
                if user_id not in [user.id for user in manager_accessible_users]:
                    raise HTTPException(
                        status_code=403,
                        detail="You do not have permission to access logs for this user"
                    )
            conditions = {'user_id': user_id}
        else:
            if val_superuser.get("user_type") != "superuser":
                manager_accessible_users = get_manager_accessible_users(app_db, val_superuser.get("user_id"))
                accessible_user_ids = [user.id for user in manager_accessible_users]
                logger.info(f"Manager accessible users: {accessible_user_ids}")
                conditions = {"user_id__in__": accessible_user_ids}
            else:
                conditions = {}

        if workflow_name:
            conditions['workflow_name'] = workflow_name
        if workflow_id:
            conditions['workflow_id'] = workflow_id

        if conditions:
            # find_by_condition을 사용하여 ExecutionIO와 users 테이블 조인
            result = app_db.find_by_condition(
                ExecutionIO,
                conditions,
                limit=10000000,
                offset=0,
                orderby="updated_at",
                orderby_asc=True,
                return_list=True,
                join_user=True
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="At least one filter parameter (user_id, workflow_name, workflow_id) must be provided"
            )

        if not result:
            backend_log.info("No IO logs found for given conditions",
                           metadata={"conditions": conditions})
            logger.info("No IO logs found")
            return JSONResponse(content={
                "io_logs": [],
                "message": "No IO logs found"
            })

        io_logs = []
        for idx, row in enumerate(result):
            # input_data 파싱 (row는 이제 딕셔너리 형태)
            raw_input_data = json.loads(row['input_data']).get('result', None) if row['input_data'] else None
            parsed_input_data = parse_input_data(raw_input_data) if raw_input_data else None

            log_entry = {
                "log_id": idx + 1,
                "io_id": row['id'],
                "user_id": row['user_id'],
                "username": row['username'],
                "full_name": row['full_name'],
                "interaction_id": row['interaction_id'],
                "workflow_name": row['workflow_name'],
                "workflow_id": row['workflow_id'],
                "input_data": parsed_input_data,
                "output_data": json.loads(row['output_data']).get('result', None) if row['output_data'] else None,
                "updated_at": row['updated_at'].isoformat() if isinstance(row['updated_at'], datetime) else row['updated_at'],
                "created_at": row['created_at'].isoformat() if isinstance(row['created_at'], datetime) else row['created_at']
            }
            io_logs.append(log_entry)

        response_data = {
            "workflow_name": workflow_name,
            "workflow_id": workflow_id,
            "in_out_logs": io_logs,
            "message": "In/Out logs retrieved successfully"
        }

        backend_log.success("Successfully retrieved IO logs",
                          metadata={"conditions": conditions, "log_count": len(io_logs)})
        return JSONResponse(content=response_data)

    except Exception as e:
        backend_log.error("Error fetching IO logs", exception=e,
                         metadata={"conditions": {"user_id": user_id, "workflow_name": workflow_name, "workflow_id": workflow_id}})
        logger.error(f"Error fetching IO logs: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        ) from e

@router.get("/all-io-logs")
async def get_all_workflows_by_id(request: Request, page: int = 1, page_size: int = 250, user_id: int = None, workflow_id: str = None, workflow_name: str = None, start_date: str = None, end_date: str = None):
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, val_superuser.get("user_id"), request)
    section_access = manager_section_access(app_db, val_superuser.get("user_id"), ["workflow-monitoring", "chat-monitoring", "workflow-management"])

    if not section_access:
        backend_log.warn(f"User {val_superuser.get('user_id')} attempted to access IO logs without permission")
        raise HTTPException(
            status_code=403,
            detail="Group permissions access required"
        )

    try:
        if page < 1:
            page = 1
        if page_size < 1 or page_size > 1000:
            page_size = 250

        offset = (page - 1) * page_size

        if user_id:
            if val_superuser.get("user_type") != "superuser":
                manager_accessible_users = get_manager_accessible_users(app_db, val_superuser.get("user_id"))
                if user_id not in [user.id for user in manager_accessible_users]:
                    raise HTTPException(
                        status_code=403,
                        detail="You do not have permission to access logs for this user"
                    )
            conditions = {'user_id': user_id}
        else:
            if val_superuser.get("user_type") != "superuser":
                manager_accessible_users = get_manager_accessible_users(app_db, val_superuser.get("user_id"))
                accessible_user_ids = [user.id for user in manager_accessible_users]
                logger.info(f"Manager accessible users: {accessible_user_ids}")
                conditions = {"user_id__in__": accessible_user_ids}
            else:
                conditions = {}
        if workflow_id:
            conditions['workflow_id'] = workflow_id
        if workflow_name:
            conditions['workflow_name'] = workflow_name

        # 날짜 필터 추가
        if start_date:
            try:
                start_datetime = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                conditions['created_at__gte__'] = start_datetime
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid start_date format. Use ISO 8601 format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")

        if end_date:
            try:
                end_datetime = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                conditions['created_at__lte__'] = end_datetime
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid end_date format. Use ISO 8601 format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")

        io_logs_result = app_db.find_by_condition(
            ExecutionIO,
            conditions,
            limit=page_size,
            offset=offset,
            orderby="updated_at",
            orderby_asc=False,
            return_list=True,
            join_user=True
        )

        # 딕셔너리 형태의 결과를 처리
        processed_logs = []
        for log in io_logs_result:
            log_dict = dict(log)
            log_dict.update({
                "input_data": extract_result_from_json(log_dict["input_data"]),
                "output_data": extract_result_from_json(log_dict["output_data"])
            })
            processed_logs.append(log_dict)

        return {
            "io_logs": processed_logs,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "offset": offset,
                "total_returned": len(processed_logs)
            }
        }
    except Exception as e:
        logger.error("Error fetching all IO logs: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        ) from e


@router.get("/download/excel/all-io-logs")
async def download_all_io_logs_excel(
    request: Request,
    user_id: int = None,
    workflow_id: str = None,
    workflow_name: str = None,
    start_date: str = None,
    end_date: str = None,
    data_processing: bool = True
):
    """
    IO 로그를 Excel 파일로 다운로드합니다.
    start_date와 end_date는 ISO 8601 형식 (YYYY-MM-DD 또는 YYYY-MM-DDTHH:MM:SS)
    data_processing: True이면 input/output 데이터에서 태그 및 불필요한 요소 제거
    """
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, val_superuser.get("user_id"), request)
    section_access = manager_section_access(app_db, val_superuser.get("user_id"), ["workflow-monitoring", "chat-monitoring", "workflow-management"])

    if not section_access:
        backend_log.warn(f"User {val_superuser.get('user_id')} attempted to access IO logs without permission")
        raise HTTPException(
            status_code=403,
            detail="Group permissions access required"
        )

    try:
        # 조건 설정
        if user_id:
            if val_superuser.get("user_type") != "superuser":
                manager_accessible_users = get_manager_accessible_users(app_db, val_superuser.get("user_id"))
                if user_id not in [user.id for user in manager_accessible_users]:
                    raise HTTPException(
                        status_code=403,
                        detail="You do not have permission to access logs for this user"
                    )
            conditions = {'user_id': user_id}
        else:
            if val_superuser.get("user_type") != "superuser":
                manager_accessible_users = get_manager_accessible_users(app_db, val_superuser.get("user_id"))
                accessible_user_ids = [user.id for user in manager_accessible_users]
                logger.info(f"Manager accessible users: {accessible_user_ids}")
                conditions = {"user_id__in__": accessible_user_ids}
            else:
                conditions = {}

        if workflow_id:
            conditions['workflow_id'] = workflow_id
        if workflow_name:
            conditions['workflow_name'] = workflow_name

        # 날짜 필터 추가 (시간은 무시하고 날짜만 사용)
        if start_date:
            try:
                # 날짜만 파싱하고 시간은 00:00:00으로 설정
                start_datetime = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                start_datetime = start_datetime.replace(hour=0, minute=0, second=0, microsecond=0)
                conditions['created_at__gte__'] = start_datetime
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid start_date format. Use ISO 8601 format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")

        if end_date:
            try:
                # 날짜만 파싱하고 시간은 23:59:59로 설정
                end_datetime = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                end_datetime = end_datetime.replace(hour=23, minute=59, second=59, microsecond=999999)
                conditions['created_at__lte__'] = end_datetime
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid end_date format. Use ISO 8601 format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")

        # 전체 데이터 조회 (페이징 없이)
        io_logs_result = app_db.find_by_condition(
            ExecutionIO,
            conditions,
            limit=1000000,  # 대량 조회
            orderby="updated_at",
            orderby_asc=False,
            return_list=True,
            join_user=True
        )

        if not io_logs_result:
            backend_log.info("No IO logs found for Excel download",
                           metadata={"conditions": conditions})
            raise HTTPException(status_code=404, detail="No IO logs found for the given criteria")

        # 데이터 프레임 생성을 위한 리스트
        excel_data = []

        for log in io_logs_result:
            try:
                # 1단계: JSON에서 result 추출
                input_data = extract_result_from_json(log['input_data'])
                output_data = extract_result_from_json(log['output_data'])

                # 2단계: data_processing이 True이면 태그 제거
                if data_processing:
                    input_data = parse_actual_output(input_data)
                    output_data = parse_actual_output(output_data)

                # 3단계: dict/list를 JSON 문자열로 먼저 변환
                if isinstance(input_data, (dict, list)):
                    try:
                        input_data = json.dumps(input_data, ensure_ascii=False, indent=None, separators=(',', ':'))
                    except Exception as e:
                        logger.warning(f"Failed to serialize input_data: {e}")
                        input_data = str(input_data)
                else:
                    input_data = str(input_data) if input_data is not None else ''

                if isinstance(output_data, (dict, list)):
                    try:
                        output_data = json.dumps(output_data, ensure_ascii=False, indent=None, separators=(',', ':'))
                    except Exception as e:
                        logger.warning(f"Failed to serialize output_data: {e}")
                        output_data = str(output_data)
                else:
                    output_data = str(output_data) if output_data is not None else ''

                # 4단계: 모든 필드에 Excel 정제 적용
                row = {
                    "User ID": int(log['user_id']) if log.get('user_id') else 0,
                    "Username": clean_excel_string(log.get('username')),
                    "Full Name": clean_excel_string(log.get('full_name')),
                    "Interaction ID": clean_excel_string(log.get('interaction_id')),
                    "Workflow Name": clean_excel_string(log.get('workflow_name')),
                    "Workflow ID": clean_excel_string(log.get('workflow_id')),
                    "Input Data": clean_excel_string(input_data),
                    "Output Data": clean_excel_string(output_data),
                    "Created At": log['created_at'].isoformat() if isinstance(log['created_at'], datetime) else str(log.get('created_at', '')),
                }
                excel_data.append(row)

            except Exception as e:
                logger.error(f"Error processing log entry: {e}")
                # 에러 발생 시 기본값으로 row 추가
                excel_data.append({
                    "User ID": 0,
                    "Username": "",
                    "Full Name": "",
                    "Interaction ID": "",
                    "Workflow Name": "",
                    "Workflow ID": "",
                    "Input Data": "[ERROR: 데이터 처리 실패]",
                    "Output Data": "[ERROR: 데이터 처리 실패]",
                    "Created At": "",
                })

        # 5단계: DataFrame 생성 (명시적 dtype 지정)
        df = pd.DataFrame(excel_data, dtype=str)

        # User ID만 정수형으로 변환
        try:
            df['User ID'] = pd.to_numeric(df['User ID'], errors='coerce').fillna(0).astype(int)
        except Exception:
            pass

        # 6단계: Excel 파일 생성
        output = io.BytesIO()
        try:
            # openpyxl의 write_only 모드는 사용하지 않음 (셀 스타일 접근 필요)
            with pd.ExcelWriter(
                output,
                engine='openpyxl',
                engine_kwargs={'write_only': False}  # 일반 모드 명시
            ) as writer:
                df.to_excel(
                    writer,
                    index=False,
                    sheet_name='IO Logs',
                    na_rep='',  # NA 값을 빈 문자열로
                )

                # 워크시트 접근하여 추가 설정
                worksheet = writer.sheets['IO Logs']
                workbook = writer.book

                # 워크북 레벨 설정
                try:
                    # 수식 계산 비활성화 (보안 및 성능)
                    if hasattr(workbook, 'calculation'):
                        workbook.calculation.calcMode = 'manual'
                        workbook.calculation.fullCalcOnLoad = False
                except AttributeError:
                    pass  # 일부 openpyxl 버전에서는 미지원

                # 열 너비 자동 조정 및 셀 타입 설정
                for idx, column in enumerate(worksheet.columns, 1):
                    max_length = 0
                    column_letter = column[0].column_letter

                    for cell in column:
                        try:
                            if cell.value is not None:
                                cell_value_str = str(cell.value)
                                max_length = max(max_length, len(cell_value_str))

                                # 데이터 행만 문자열로 강제 (헤더 제외)
                                if cell.row > 1 and column_letter not in ['A']:  # User ID는 숫자 유지
                                    # 수식으로 해석될 수 있는 문자 처리
                                    if cell_value_str.startswith(('=', '+', '-', '@')):
                                        cell.value = "'" + cell_value_str  # apostrophe prefix
                                    cell.data_type = 's'  # 문자열 타입
                        except Exception as e:
                            logger.debug(f"Error processing cell {cell.coordinate}: {e}")
                            continue

                    # 열 너비 설정 (최소 10, 최대 100)
                    adjusted_width = min(max(max_length + 2, 10), 100)
                    worksheet.column_dimensions[column_letter].width = adjusted_width

        except Exception as e:
            logger.error(f"Error creating Excel file: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Excel 파일 생성 실패: {str(e)}")

        output.seek(0)

        # 파일명 생성 (날짜 포함) - Asia/Seoul timezone (UTC+9)
        from datetime import timezone, timedelta
        kst = timezone(timedelta(hours=9))
        timestamp = datetime.now(kst).strftime("%Y%m%d_%H%M%S")
        filename = f"workflow_chat_log_{timestamp}.xlsx"

        backend_log.success("Successfully generated Excel file for IO logs",
                          metadata={"conditions": conditions, "log_count": len(excel_data), "filename": filename})

        # StreamingResponse로 파일 반환
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        backend_log.error("Error generating Excel file for IO logs", exception=e,
                         metadata={"conditions": {"user_id": user_id, "workflow_id": workflow_id, "workflow_name": workflow_name}})
        logger.error(f"Error generating Excel file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        ) from e


@router.get("/all-io-logs/tester")
async def get_workflow_io_logs_for_tester(request: Request, user_id: int = None, workflow_name: str = None):
    """
    특정 워크플로우의 ExecutionIO 로그를 interaction_batch_id별로 그룹화하여 반환합니다.
    """
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, val_superuser.get("user_id"), request)
    section_access = manager_section_access(app_db, val_superuser.get("user_id"), ["workflow-monitoring"])
    if not section_access:
        backend_log.warn(f"User {val_superuser.get('user_id')} attempted to access IO logs without permission")
        raise HTTPException(
            status_code=403,
            detail="Group permissions access required"
        )

    if val_superuser.get("user_type") != "superuser":
        manager_accessible_users = get_manager_accessible_users(app_db, val_superuser.get("user_id"))
        accessible_user_ids = [user.id for user in manager_accessible_users]
        if user_id not in accessible_user_ids:
            logger.warning(f"User {val_superuser.get('user_id')} attempted to access IO logs for user {user_id} without permission. Accessible users: {accessible_user_ids}")
            raise HTTPException(
                status_code=403,
                detail="You do not have permission to access logs for this user"
            )

    try:
        backend_log.info("Retrieving workflow tester IO logs",
                        metadata={"workflow_name": workflow_name})

        result = app_db.find_by_condition(
            ExecutionIO,
            {
                "user_id": user_id,
                "workflow_name": workflow_name,
                "test_mode": True
            },
            limit=1000000,
            orderby="updated_at",
            orderby_asc=True,
            return_list=True
        )

        if not result:
            backend_log.info("No tester IO logs found",
                           metadata={"workflow_name": workflow_name})
            logger.info(f"No performance data found for workflow: {workflow_name}")
            return JSONResponse(content={
                "workflow_name": workflow_name,
                "response_data_list": [],
                "message": "No in_out_logs data found for this workflow"
            })

        # interaction_batch_id별로 그룹화
        tester_groups = {}

        for idx, row in enumerate(result):
            interaction_id = row['interaction_id']

            # interaction_id에서 마지막 숫자를 제외한 배치 ID 추출
            parts = interaction_id.split('____')
            if len(parts) >= 4:
                interaction_batch_id = '____'.join(parts[:-1])
            else:
                interaction_batch_id = interaction_id

            if interaction_batch_id not in tester_groups:
                tester_groups[interaction_batch_id] = []

            # input_data 파싱
            raw_input_data = json.loads(row['input_data']).get('result', None) if row['input_data'] else None
            parsed_input_data = parse_input_data(raw_input_data) if raw_input_data else None

            # interaction_id에서 마지막 번호를 추출하여 log_id로 사용
            parts = interaction_id.split('____')
            if len(parts) >= 4 and parts[-1].isdigit():
                log_id = int(parts[-1])
            else:
                log_id = len(tester_groups[interaction_batch_id]) + 1

            log_entry = {
                "log_id": log_id,
                "interaction_id": row['interaction_id'],
                "workflow_name": row['workflow_name'],
                "workflow_id": row['workflow_id'],
                "input_data": parsed_input_data,
                "output_data": json.loads(row['output_data']).get('result', None) if row['output_data'] else None,
                "expected_output": row['expected_output'],
                "llm_eval_score": row['llm_eval_score'],
                "updated_at": row['updated_at'].isoformat() if isinstance(row['updated_at'], datetime) else row['updated_at']
            }
            tester_groups[interaction_batch_id].append(log_entry)

        # 각 테스터 그룹을 response_data 형태로 변환
        response_data_list = []
        for interaction_batch_id, performance_stats in tester_groups.items():
            response_data = {
                "workflow_name": workflow_name,
                "interaction_batch_id": interaction_batch_id,
                "in_out_logs": performance_stats,
                "message": "In/Out logs retrieved successfully"
            }
            response_data_list.append(response_data)

        final_response = {
            "workflow_name": workflow_name,
            "response_data_list": response_data_list,
            "message": f"In/Out logs retrieved successfully for {len(response_data_list)} tester groups"
        }

        backend_log.success("Successfully retrieved workflow tester IO logs",
                          metadata={"workflow_name": workflow_name,
                                  "tester_groups": len(response_data_list),
                                  "total_logs": len(result)})

        logger.info(f"Performance stats retrieved for workflow: {workflow_name}, {len(response_data_list)} tester groups")
        return JSONResponse(content=final_response)

    except Exception as e:
        backend_log.error("Failed to retrieve workflow tester IO logs", exception=e,
                         metadata={"workflow_name": workflow_name})
        logger.error(f"Error retrieving workflow performance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve performance data: {str(e)}")

@router.get("/all-list")
async def get_all_workflows(request: Request, page: int = 1, page_size: int = 250, user_id = None):
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, val_superuser.get("user_id"), request)
    section_access = manager_section_access(app_db, val_superuser.get("user_id"), ["workflow-monitoring", "workflow-management"])
    if not section_access:
        backend_log.warn(f"User {val_superuser.get('user_id')} attempted to access IO logs without permission")
        raise HTTPException(
            status_code=403,
            detail="Group permissions access required"
        )
    db_type = app_db.config_db_manager.db_type
    try:
        # 페이지 번호 검증
        if page < 1:
            page = 1
        if page_size < 1 or page_size > 1000:
            page_size = 250

        offset = (page - 1) * page_size

        if user_id:
            if val_superuser.get("user_type") != "superuser":
                manager_accessible_users = get_manager_accessible_users(app_db, val_superuser.get("user_id"))
                if user_id not in [user.id for user in manager_accessible_users]:
                    raise HTTPException(
                        status_code=403,
                        detail="You do not have permission to access logs for this user"
                    )
            query = """
                SELECT
                    wm.id, wm.created_at, wm.updated_at,
                    wm.user_id, wm.workflow_id, wm.workflow_name,
                    wm.node_count, wm.edge_count, wm.has_startnode, wm.has_endnode,
                    wm.is_completed, wm.metadata, wm.is_shared, wm.share_group, wm.share_permissions,
                    u.full_name, u.username,
                    dm.is_deployed, dm.deploy_key, dm.is_accepted, dm.inquire_deploy
                FROM workflow_meta wm
                LEFT JOIN users u ON wm.user_id = u.id
                LEFT JOIN deploy_meta dm ON wm.workflow_id = dm.workflow_id AND wm.workflow_name = dm.workflow_name AND wm.user_id = dm.user_id
                WHERE wm.user_id = %s
                ORDER BY wm.created_at DESC
                LIMIT %s OFFSET %s
            """
            if db_type != "postgresql":
                query = query.replace("%s", "?")
            all_workflows = app_db.config_db_manager.execute_query(query, (user_id, page_size, offset))
        else:
            query = """
                SELECT
                    wm.id, wm.created_at, wm.updated_at,
                    wm.user_id, wm.workflow_id, wm.workflow_name,
                    wm.node_count, wm.edge_count, wm.has_startnode, wm.has_endnode,
                    wm.is_completed, wm.metadata, wm.is_shared, wm.share_group, wm.share_permissions,
                    u.full_name, u.username,
                    dm.is_deployed, dm.deploy_key, dm.is_accepted, dm.inquire_deploy
                FROM workflow_meta wm
                LEFT JOIN users u ON wm.user_id = u.id
                LEFT JOIN deploy_meta dm ON wm.workflow_id = dm.workflow_id AND wm.workflow_name = dm.workflow_name AND wm.user_id = dm.user_id
                ORDER BY wm.created_at DESC
                LIMIT %s OFFSET %s
            """
            if val_superuser.get("user_type") != "superuser":
                accessible_workflow_ids = get_manager_accessible_workflows_ids(app_db, val_superuser.get("user_id"))
                manager_accessible_users = get_manager_accessible_users(app_db, val_superuser.get("user_id"))
                accessible_user_ids = [user.id for user in manager_accessible_users]

                if db_type == "postgresql":
                    workflow_placeholders = ', '.join(['%s'] * len(accessible_workflow_ids))
                    user_placeholders = ', '.join(['%s'] * len(accessible_user_ids))
                else:
                    workflow_placeholders = ', '.join(['?'] * len(accessible_workflow_ids))
                    user_placeholders = ', '.join(['?'] * len(accessible_user_ids))
                query = query.replace("ORDER BY", f"WHERE wm.workflow_id IN ({workflow_placeholders}) AND wm.user_id IN ({user_placeholders}) ORDER BY")
                params = accessible_workflow_ids + accessible_user_ids + [page_size, offset]
                all_workflows = app_db.config_db_manager.execute_query(query, params)
            else:
                if db_type != "postgresql":
                    query = query.replace("%s", "?")
                all_workflows = app_db.config_db_manager.execute_query(query, (page_size, offset))

        # id 중복 제거 - id를 기준으로 중복된 항목은 첫 번째 것만 유지
        seen_ids = set()
        unique_workflows = []
        for workflow in all_workflows:
            workflow_id = workflow['id']
            if workflow_id not in seen_ids:
                seen_ids.add(workflow_id)
                unique_workflows.append(workflow)

        return {
            "workflows": unique_workflows,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "offset": offset,
                "total_returned": len(unique_workflows)
            }
        }
    except Exception as e:
        logger.error("Error fetching all IO logs: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        ) from e

@router.post("/update/{workflow_name}")
async def update_workflow(request: Request, workflow_name: str, update_dict: dict):
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, val_superuser.get("user_id"), request)
    section_access = manager_section_access(app_db, val_superuser.get("user_id"), ["workflow-management"])
    if not section_access:
        backend_log.warn(f"User {val_superuser.get('user_id')} attempted to access IO logs without permission")
        raise HTTPException(
            status_code=403,
            detail="Group permissions access required"
        )

    user_id = update_dict.get("user_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required in the update data")
    if val_superuser.get("user_type") != "superuser":
        manager_accessible_users = get_manager_accessible_users(app_db, val_superuser.get("user_id"))
        if user_id not in [user.id for user in manager_accessible_users]:
            raise HTTPException(
                status_code=403,
                detail="You do not have permission to update workflows for this user"
            )

    try:
        existing_data = app_db.find_by_condition(
            WorkflowMeta,
            {
                "user_id": user_id,
                "workflow_name": workflow_name
            },
            limit=1
        )

        if not existing_data:
            raise HTTPException(status_code=404, detail="Workflow not found")
        existing_data = existing_data[0]

        deploy_data = app_db.find_by_condition(
            DeployMeta,
            {
                "user_id": user_id,
                "workflow_name": workflow_name,
            },
            limit=1
        )
        if not deploy_data:
            raise HTTPException(status_code=404, detail="배포 메타데이터를 찾을 수 없습니다")
        deploy_meta = deploy_data[0]

        existing_data.is_shared = update_dict.get("is_shared", existing_data.is_shared)
        existing_data.share_group = update_dict.get("share_group", existing_data.share_group)
        existing_data.share_permissions = update_dict.get("share_permissions", existing_data.share_permissions)

        deploy_enabled = update_dict.get("enable_deploy", deploy_meta.is_deployed)
        deploy_meta.is_deployed = deploy_enabled

        deploy_meta.is_accepted = update_dict.get("is_accepted", deploy_meta.is_accepted)
        deploy_meta.inquire_deploy = update_dict.get("inquire_deploy", deploy_meta.inquire_deploy)

        if deploy_enabled:
            alphabet = string.ascii_letters + string.digits
            deploy_key = ''.join(secrets.choice(alphabet) for _ in range(32))
            deploy_meta.deploy_key = deploy_key
            deploy_meta.inquire_deploy = False

            logger.info(f"Generated new deploy key for workflow: {workflow_name}")
        else:
            deploy_meta.deploy_key = ""
            logger.info(f"Cleared deploy key for workflow: {workflow_name}")

        app_db.update(existing_data)
        app_db.update(deploy_meta)

        return {
            "message": "Workflow updated successfully",
            "workflow_name": existing_data.workflow_name,
            "deploy_key": deploy_meta.deploy_key if deploy_meta.is_deployed else None,
        }

    except Exception as e:
        logger.error(f"Failed to update workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update workflow: {str(e)}")

@router.delete("/delete/{workflow_name}")
async def delete_workflow(request: Request, user_id, workflow_name: str):
    """
    특정 workflow를 삭제합니다.
    """
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, val_superuser.get("user_id"), request)
    section_access = manager_section_access(app_db, val_superuser.get("user_id"), ["workflow-management"])
    if not section_access:
        backend_log.warn(f"User {val_superuser.get('user_id')} attempted to access IO logs without permission")
        raise HTTPException(
            status_code=403,
            detail="Group permissions access required"
        )
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required in the update data")
    if val_superuser.get("user_type") != "superuser":
        manager_accessible_users = get_manager_accessible_users(app_db, val_superuser.get("user_id"))
        if user_id not in [user.id for user in manager_accessible_users]:
            raise HTTPException(
                status_code=403,
                detail="You do not have permission to update workflows for this user"
            )

    try:
        existing_data = app_db.find_by_condition(
            WorkflowMeta,
            {
                "user_id": user_id,
                "workflow_name": workflow_name,
            },
            ignore_columns=['workflow_data'],
            limit=1
        )

        app_db.delete(WorkflowMeta, existing_data[0].id if existing_data else None)
        app_db.delete_by_condition(DeployMeta, {
            "user_id": user_id,
            "workflow_id": existing_data[0].workflow_id,
            "workflow_name": workflow_name,
        })

        logger.info(f"Workflow deleted successfully: {workflow_name}")
        return JSONResponse(content={
            "success": True,
            "message": f"Workflow '{workflow_name}' deleted successfully"
        })

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_name}' not found")
    except Exception as e:
        logger.error(f"Error deleting workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete workflow: {str(e)}")

@router.get("/performance")
async def get_workflow_performance(request: Request, user_id: int, workflow_name: str, workflow_id: str):
    """
    특정 워크플로우의 성능 통계를 반환합니다.
    node_id와 node_name별로 평균 성능 지표를 계산합니다.
    """
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, val_superuser.get("user_id"), request)
    section_access = manager_section_access(app_db, val_superuser.get("user_id"), ["workflow-monitoring"])
    if not section_access:
        backend_log.warn(f"User {val_superuser.get('user_id')} attempted to access IO logs without permission")
        raise HTTPException(
            status_code=403,
            detail="Group permissions access required"
        )
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required in the update data")
    if val_superuser.get("user_type") != "superuser":
        manager_accessible_users = get_manager_accessible_users(app_db, val_superuser.get("user_id"))
        if user_id not in [user.id for user in manager_accessible_users]:
            raise HTTPException(
                status_code=403,
                detail="You do not have permission to update workflows for this user"
            )
    try:
        backend_log.info("Retrieving workflow performance statistics",
                        metadata={"workflow_name": workflow_name, "workflow_id": workflow_id})

        # SQL 쿼리 작성
        query = """
        SELECT
            node_id,
            node_name,
            AVG(processing_time_ms) as avg_processing_time_ms,
            AVG(cpu_usage_percent) as avg_cpu_usage_percent,
            AVG(ram_usage_mb) as avg_ram_usage_mb,
            AVG(CASE WHEN gpu_usage_percent IS NOT NULL THEN gpu_usage_percent END) as avg_gpu_usage_percent,
            AVG(CASE WHEN gpu_memory_mb IS NOT NULL THEN gpu_memory_mb END) as avg_gpu_memory_mb,
            COUNT(*) as execution_count,
            COUNT(CASE WHEN gpu_usage_percent IS NOT NULL THEN 1 END) as gpu_execution_count
        FROM node_performance
        WHERE workflow_name = %s AND workflow_id = %s AND user_id = %s
        GROUP BY node_id, node_name
        ORDER BY node_id
        """

        # SQLite인 경우 파라미터 플레이스홀더 변경
        if app_db.config_db_manager.db_type == "sqlite":
            query = query.replace("%s", "?")

        # 쿼리 실행
        result = app_db.config_db_manager.execute_query(query, (workflow_name, workflow_id, user_id))

        if not result:
            backend_log.info("No performance data found",
                           metadata={"workflow_name": workflow_name, "workflow_id": workflow_id})
            logger.info(f"No performance data found for workflow: {workflow_name} ({workflow_id})")
            return JSONResponse(content={
                "workflow_name": workflow_name,
                "workflow_id": workflow_id,
                "performance_stats": [],
                "message": "No performance data found for this workflow"
            })

        # 결과 포맷팅
        performance_stats = []
        for row in result:
            stats = {
                "node_id": row['node_id'],
                "node_name": row['node_name'],
                "avg_processing_time_ms": safe_round_float(row['avg_processing_time_ms']) if row['avg_processing_time_ms'] else 0.0,
                "avg_cpu_usage_percent": safe_round_float(row['avg_cpu_usage_percent']) if row['avg_cpu_usage_percent'] else 0.0,
                "avg_ram_usage_mb": safe_round_float(row['avg_ram_usage_mb']) if row['avg_ram_usage_mb'] else 0.0,
                "avg_gpu_usage_percent": safe_round_float(row['avg_gpu_usage_percent']) if row['avg_gpu_usage_percent'] else None,
                "avg_gpu_memory_mb": safe_round_float(row['avg_gpu_memory_mb']) if row['avg_gpu_memory_mb'] else None,
                "execution_count": int(row['execution_count']) if row['execution_count'] else 0,
                "gpu_execution_count": int(row['gpu_execution_count']) if row['gpu_execution_count'] else 0
            }
            performance_stats.append(stats)

        # 전체 워크플로우 통계 계산
        total_executions = sum(stat['execution_count'] for stat in performance_stats)
        avg_total_processing_time = sum(float(stat['avg_processing_time_ms']) * stat['execution_count'] for stat in performance_stats) / total_executions if total_executions > 0 else 0.0
        avg_total_cpu_usage = sum(float(stat['avg_cpu_usage_percent']) * stat['execution_count'] for stat in performance_stats) / total_executions if total_executions > 0 else 0.0
        avg_total_ram_usage = sum(float(stat['avg_ram_usage_mb']) * stat['execution_count'] for stat in performance_stats) / total_executions if total_executions > 0 else 0.0

        # GPU 통계
        gpu_stats = None
        total_gpu_executions = sum(stat['gpu_execution_count'] for stat in performance_stats)
        if total_gpu_executions > 0:
            gpu_usage_sum = sum(float(stat['avg_gpu_usage_percent']) * stat['gpu_execution_count'] for stat in performance_stats if stat['avg_gpu_usage_percent'] is not None)
            gpu_memory_sum = sum(float(stat['avg_gpu_memory_mb']) * stat['gpu_execution_count'] for stat in performance_stats if stat['avg_gpu_memory_mb'] is not None)

            gpu_stats = {
                "avg_gpu_usage_percent": round(float(gpu_usage_sum / total_gpu_executions), 2) if total_gpu_executions > 0 else None,
                "avg_gpu_memory_mb": round(float(gpu_memory_sum / total_gpu_executions), 2) if total_gpu_executions > 0 else None,
                "gpu_execution_count": total_gpu_executions
            }

        response_data = {
            "workflow_name": workflow_name,
            "workflow_id": workflow_id,
            "summary": {
                "total_executions": total_executions,
                "avg_total_processing_time_ms": round(float(avg_total_processing_time), 2),
                "avg_total_cpu_usage_percent": round(float(avg_total_cpu_usage), 2),
                "avg_total_ram_usage_mb": round(float(avg_total_ram_usage), 2),
                "gpu_stats": gpu_stats
            },
            "performance_stats": performance_stats
        }

        backend_log.success("Successfully retrieved workflow performance statistics",
                          metadata={"workflow_name": workflow_name,
                                  "workflow_id": workflow_id,
                                  "total_executions": total_executions,
                                  "nodes_analyzed": len(performance_stats),
                                  "gpu_executions": total_gpu_executions,
                                  "avg_processing_time": round(float(avg_total_processing_time), 2)})

        logger.info(f"Performance stats retrieved for workflow: {workflow_name} ({workflow_id})")
        return JSONResponse(content=response_data)

    except Exception as e:
        backend_log.error("Failed to retrieve workflow performance statistics", exception=e,
                         metadata={"workflow_name": workflow_name, "workflow_id": workflow_id})
        logger.error(f"Error retrieving workflow performance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve performance data: {str(e)}")

@router.delete("/performance")
async def delete_workflow_performance(request: Request, user_id: int, workflow_name: str, workflow_id: str):
    """
    특정 워크플로우의 성능 데이터를 삭제합니다.
    """
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, val_superuser.get("user_id"), request)
    section_access = manager_section_access(app_db, val_superuser.get("user_id"), ["workflow-monitoring"])
    if not section_access:
        backend_log.warn(f"User {val_superuser.get('user_id')} attempted to access IO logs without permission")
        raise HTTPException(
            status_code=403,
            detail="Group permissions access required"
        )
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required in the update data")
    if val_superuser.get("user_type") != "superuser":
        manager_accessible_users = get_manager_accessible_users(app_db, val_superuser.get("user_id"))
        if user_id not in [user.id for user in manager_accessible_users]:
            raise HTTPException(
                status_code=403,
                detail="You do not have permission to update workflows for this user"
            )
    try:
        backend_log.info("Starting workflow performance data deletion",
                        metadata={"workflow_name": workflow_name, "workflow_id": workflow_id})

        existing_data = app_db.find_by_condition(
            NodePerformance,
            {
                "user_id": user_id,
                "workflow_name": workflow_name,
                "workflow_id": workflow_id
            },
            limit=1000000
        )

        delete_count = len(existing_data) if existing_data else 0

        if delete_count == 0:
            backend_log.info("No performance data found to delete",
                           metadata={"workflow_name": workflow_name, "workflow_id": workflow_id})
            logger.info(f"No performance data found to delete for workflow: {workflow_name} ({workflow_id})")
            return JSONResponse(content={
                "workflow_name": workflow_name,
                "workflow_id": workflow_id,
                "deleted_count": 0,
                "message": "No performance data found to delete"
            })

        app_db.delete_by_condition(
            NodePerformance,
            {
                "user_id": user_id,
                "workflow_name": workflow_name,
                "workflow_id": workflow_id
            }
        )

        response_data = {
            "user_id": user_id,
            "workflow_name": workflow_name,
            "workflow_id": workflow_id,
            "deleted_count": delete_count,
            "message": f"Successfully deleted {delete_count} performance records"
        }

        backend_log.success("Successfully deleted workflow performance data",
                          metadata={"workflow_name": workflow_name,
                                  "workflow_id": workflow_id,
                                  "deleted_count": delete_count})

        logger.info(f"Deleted {delete_count} performance records for workflow: {workflow_name} ({workflow_id})")
        return JSONResponse(content=response_data)

    except Exception as e:
        backend_log.error("Failed to delete workflow performance data", exception=e,
                         metadata={"workflow_name": workflow_name,
                                 "workflow_id": workflow_id,
                                 "expected_delete_count": delete_count if 'delete_count' in locals() else 0})
        logger.error(f"Error deleting performance data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete performance data: {str(e)}")

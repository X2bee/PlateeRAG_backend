"""
Tool Storage CRUD 작업 관련 엔드포인트들
사용자가 생성한 툴을 저장, 수정, 삭제하는 기능
"""
import json
import logging
import httpx
from datetime import datetime
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from controller.helper.controllerHelper import extract_user_id_from_request
from controller.helper.singletonHelper import get_db_manager
from controller.tools.models.requests import SaveToolRequest, ToolData
from service.database.models.tools import Tools
from service.database.models.user import User
from service.database.logger_helper import create_logger

logger = logging.getLogger("tool-storage")
router = APIRouter()

def html_element_to_dict(element):
    """
    BeautifulSoup element를 간단한 dict로 변환합니다.
    태그 이름을 key로, 자식 요소와 content를 value로 하는 계층 구조를 생성합니다.
    스타일, 클래스, id 등은 제거하고 content만 포함합니다.

    Args:
        element: BeautifulSoup element

    Returns:
        dict 형태로 변환된 element
    """
    from bs4 import NavigableString, Comment

    # 주석은 무시
    if isinstance(element, Comment):
        return None

    # 텍스트 노드
    if isinstance(element, NavigableString):
        text = str(element).strip()
        return text if text else None

    # 요소 노드 - 태그 이름을 key로 사용
    tag_name = element.name
    result = {}

    # 자식 노드들을 수집
    children = []
    text_content = []

    for child in element.children:
        if isinstance(child, NavigableString):
            text = str(child).strip()
            if text:
                text_content.append(text)
        elif not isinstance(child, Comment):
            child_dict = html_element_to_dict(child)
            if child_dict is not None:
                if isinstance(child_dict, dict):
                    children.append(child_dict)
                elif isinstance(child_dict, str):
                    text_content.append(child_dict)

    # content가 있으면 추가
    if text_content:
        result["content"] = " ".join(text_content)

    # 자식 태그들을 태그 이름으로 그룹화
    for child_dict in children:
        if isinstance(child_dict, dict):
            for key, value in child_dict.items():
                if key in result:
                    # 같은 태그가 여러 개면 리스트로
                    if not isinstance(result[key], list):
                        result[key] = [result[key]]
                    result[key].append(value)
                else:
                    result[key] = value

    # 결과가 비어있으면 None 반환
    if not result:
        return None

    return {tag_name: result} if result else None


def parse_html_response(html_text: str) -> dict:
    """
    HTML 응답을 간단한 계층 구조의 dict로 변환합니다.
    스크립트와 스타일은 제거하고, 태그 계층 구조와 content만 포함합니다.

    Args:
        html_text: 원본 HTML 문자열

    Returns:
        계층 구조의 dict
    """
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_text, 'html.parser')

        # script, style 태그 제거
        for script in soup(["script", "style"]):
            script.decompose()

        # HTML을 dict 구조로 변환
        html_dict = None
        if soup.html:
            html_dict = html_element_to_dict(soup.html)
        elif soup.body:
            html_dict = html_element_to_dict(soup.body)
        else:
            # html, body 태그가 없으면 첫 번째 요소 사용
            first_element = next((child for child in soup.children if hasattr(child, 'name')), None)
            if first_element:
                html_dict = html_element_to_dict(first_element)

        logger.info("Successfully parsed HTML response to dict structure")

        return html_dict if html_dict else {}

    except ImportError:
        logger.warning("BeautifulSoup not available, returning error")
        return {"error": "BeautifulSoup not installed"}
    except Exception as html_parse_error:
        logger.warning(f"Failed to parse HTML: {str(html_parse_error)}")
        return {"error": str(html_parse_error)}


@router.get("/simple-list")
async def simple_list_tools(request: Request):
    """
    사용자의 모든 툴 목록을 반환합니다.
    """
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        tools_data = app_db.find_by_condition(
            Tools,
            {"user_id": user_id},
            limit=1000,
            orderby="updated_at",
            join_user=True,
            return_list=True,
        )

        backend_log.success("Tool list retrieved successfully",
                          metadata={"tool_count": len(tools_data)})

        logger.info(f"Found {len(tools_data)} tools for user {user_id}")
        return tools_data

    except Exception as e:
        backend_log.error("Failed to list tools", exception=e)
        logger.error(f"Error listing tools: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list tools: {str(e)}")

@router.get("/list")
async def list_tools(request: Request):
    """
    사용자의 모든 툴 목록을 반환합니다.
    """
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        tools_data = app_db.find_by_condition(
            Tools,
            {"user_id": user_id},
            limit=1000,
            orderby="updated_at",
            join_user=True,
            return_list=True
        )

        backend_log.success("Tool list retrieved successfully",
                          metadata={"tool_count": len(tools_data)})

        logger.info(f"Found {len(tools_data)} tools for user {user_id}")
        return {"tools": tools_data}

    except Exception as e:
        backend_log.error("Failed to list tools", exception=e)
        logger.error(f"Error listing tools: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list tools: {str(e)}")

@router.get("/list/detail")
async def list_tools_detail(request: Request):
    """
    사용자의 모든 툴 상세 정보를 반환합니다.
    """
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        user = app_db.find_by_id(User, user_id)
        groups = user.groups if user else []
        user_name = user.username if user else "Unknown User"

        backend_log.info("Starting detailed tool list retrieval",
                        metadata={"user_name": user_name, "groups": groups})

        # 자신의 툴
        own_tools = app_db.find_by_condition(
            Tools,
            {"user_id": user_id},
            limit=10000,
            orderby="updated_at",
            return_list=True,
            join_user=True
        )
        own_tool_count = len(own_tools) if own_tools else 0

        # 그룹 공유 툴 추가
        shared_tool_count = 0
        all_tools = list(own_tools) if own_tools else []

        if groups and groups != None and groups != [] and len(groups) > 0:
            for group_name in groups:
                shared_tools = app_db.find_by_condition(
                    Tools,
                    {"share_group": group_name, "is_shared": True},
                    limit=10000,
                    orderby="updated_at",
                    return_list=True,
                    join_user=True
                )
                if shared_tools:
                    all_tools.extend(shared_tools)
                    shared_tool_count += len(shared_tools)

        # 중복 제거
        seen_ids = set()
        unique_data = []
        for item in all_tools:
            if item.get("id") not in seen_ids:
                seen_ids.add(item.get("id"))
                item_dict = dict(item)
                if 'created_at' in item_dict and item_dict['created_at']:
                    item_dict['created_at'] = item_dict['created_at'].isoformat() if hasattr(item_dict['created_at'], 'isoformat') else str(item_dict['created_at'])
                if 'updated_at' in item_dict and item_dict['updated_at']:
                    item_dict['updated_at'] = item_dict['updated_at'].isoformat() if hasattr(item_dict['updated_at'], 'isoformat') else str(item_dict['updated_at'])
                unique_data.append(item_dict)

        backend_log.success("Detailed tool list retrieved successfully",
                          metadata={"own_tools": own_tool_count,
                                  "shared_tools": shared_tool_count,
                                  "total_unique_tools": len(unique_data)})

        logger.info(f"Found {len(unique_data)} tools with detailed information")

        return JSONResponse(content={"tools": unique_data})

    except Exception as e:
        backend_log.error("Failed to retrieve detailed tool list", exception=e)
        logger.error(f"Error listing tool details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list tool details: {str(e)}")

@router.post("/save")
async def save_tool(request: Request, tool_request: SaveToolRequest):
    """
    툴 정보를 데이터베이스에 저장합니다.
    """
    login_user_id = extract_user_id_from_request(request)
    if not login_user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    if tool_request.user_id and str(tool_request.user_id) != str(login_user_id):
        user_id = str(tool_request.user_id)
    else:
        user_id = login_user_id

    logger.info(f"Saving tool for user: {user_id}, function name: {tool_request.function_name}")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        tool_data = tool_request.content.model_dump()

        backend_log.info("Starting tool save operation",
                        metadata={"function_name": tool_request.function_name,
                                "function_id": tool_request.content.function_id})

        # 기존 툴 확인
        existing_data = app_db.find_by_condition(
            Tools,
            {
                "user_id": user_id,
                "function_id": tool_request.content.function_id,
            },
            limit=1,
        )

        # api_header와 api_body를 JSON 문자열로 변환
        api_header = tool_data.get('api_header', {})
        api_body = tool_data.get('api_body', {})
        metadata = tool_data.get('metadata', {})
        static_body = tool_data.get('static_body', {})
        body_type = tool_data.get('body_type', 'application/json')

        # GET이 아닌 경우 body_type에 따라 Content-Type 헤더 자동 설정
        if tool_data.get('api_method', 'GET').upper() != 'GET':
            if not api_header:
                api_header = {}

            # body_type에 따라 Content-Type 설정 (multipart/form-data 제외)
            if body_type != 'multipart/form-data' and body_type != 'url-params':
                api_header['Content-Type'] = body_type

        tool_meta = Tools(
            user_id=user_id,
            function_name=tool_request.function_name,
            function_id=tool_request.content.function_id,
            description=tool_data.get('description', ''),
            api_header=api_header,
            api_body=api_body,
            api_url=tool_data.get('api_url', ''),
            api_method=tool_data.get('api_method', 'GET'),
            api_timeout=tool_data.get('api_timeout', 30),
            response_filter=tool_data.get('response_filter', False),
            html_parser=tool_data.get('html_parser', False),
            response_filter_path=tool_data.get('response_filter_path', ''),
            response_filter_field=tool_data.get('response_filter_field', ''),
            status=tool_data.get('status', ''),
            is_shared=existing_data[0].is_shared if existing_data and len(existing_data) > 0 else False,
            share_group=existing_data[0].share_group if existing_data and len(existing_data) > 0 else None,
            share_permissions=existing_data[0].share_permissions if existing_data and len(existing_data) > 0 else 'read',
            metadata=metadata,
            body_type=body_type,
            static_body=static_body,
            is_query_string=tool_data.get('is_query_string', False)
        )

        if existing_data and len(existing_data) > 0:
            tool_meta.id = existing_data[0].id
            insert_result = app_db.update(tool_meta)
            operation = "update"
        else:
            insert_result = app_db.insert(tool_meta)
            operation = "create"

        if insert_result and insert_result.get("result") == "success":
            backend_log.success("Tool saved successfully",
                              metadata={"function_name": tool_request.function_name,
                                      "function_id": tool_request.content.function_id,
                                      "operation": operation})

            logger.info(f"Tool saved successfully: {tool_request.function_name}")
        else:
            backend_log.error("Failed to save tool metadata",
                            metadata={"function_name": tool_request.function_name,
                                    "error": insert_result.get('error', 'Unknown error')})
            logger.error(f"Failed to save tool metadata: {insert_result.get('error', 'Unknown error')}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save tool metadata: {insert_result.get('error', 'Unknown error')}"
            )

        return JSONResponse(content={
            "success": True,
            "message": f"Tool '{tool_request.function_name}' saved successfully",
            "function_id": tool_request.content.function_id
        })

    except HTTPException:
        raise
    except Exception as e:
        backend_log.error("Tool save operation failed", exception=e,
                         metadata={"function_name": tool_request.function_name})
        logger.error(f"Error saving tool: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save tool: {str(e)}")

@router.post("/update/{function_id}")
async def update_tool(request: Request, tool_id: int, function_id: str, update_dict: dict):
    """
    툴 정보를 업데이트합니다. (모든 필드 업데이트 가능)
    """
    user_id = extract_user_id_from_request(request)
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        existing_data = app_db.find_by_condition(
            Tools,
            {
                "id": tool_id,
                "user_id": user_id,
                "function_id": function_id
            },
            limit=1
        )

        if not existing_data:
            raise HTTPException(status_code=404, detail="Tool not found")

        tool = existing_data[0]

        # 업데이트 가능한 모든 필드들
        if "function_name" in update_dict:
            tool.function_name = update_dict["function_name"]
        if "description" in update_dict:
            tool.description = update_dict["description"]
        if "api_header" in update_dict:
            tool.api_header = update_dict["api_header"]
        if "api_body" in update_dict:
            tool.api_body = update_dict["api_body"]
        if "static_body" in update_dict:
            tool.static_body = update_dict["static_body"]
        if "body_type" in update_dict:
            tool.body_type = update_dict["body_type"]
        if "api_url" in update_dict:
            tool.api_url = update_dict["api_url"]
        if "api_method" in update_dict:
            tool.api_method = update_dict["api_method"]
        if "api_timeout" in update_dict:
            tool.api_timeout = update_dict["api_timeout"]
        if "response_filter" in update_dict:
            tool.response_filter = update_dict["response_filter"]
        if "html_parser" in update_dict:
            tool.html_parser = update_dict["html_parser"]
        if "response_filter_path" in update_dict:
            tool.response_filter_path = update_dict["response_filter_path"]
        if "response_filter_field" in update_dict:
            tool.response_filter_field = update_dict["response_filter_field"]
        if "status" in update_dict:
            tool.status = update_dict["status"]
        if "is_shared" in update_dict:
            tool.is_shared = update_dict["is_shared"]
        if "share_group" in update_dict:
            tool.share_group = update_dict["share_group"]
        if "share_permissions" in update_dict:
            tool.share_permissions = update_dict["share_permissions"]
        if "metadata" in update_dict:
            tool.metadata = update_dict["metadata"]
        if "is_query_string" in update_dict:
            tool.is_query_string = update_dict["is_query_string"]

        # body_type이 변경되고 GET이 아닌 경우 Content-Type 헤더 자동 업데이트
        if "body_type" in update_dict or "api_method" in update_dict:
            api_method = tool.api_method or 'GET'
            body_type = tool.body_type or 'application/json'

            if api_method.upper() != 'GET':
                if not tool.api_header:
                    tool.api_header = {}

                # body_type에 따라 Content-Type 설정 (multipart/form-data, url-params 제외)
                if body_type != 'multipart/form-data' and body_type != 'url-params':
                    tool.api_header['Content-Type'] = body_type
                elif body_type == 'url-params' and 'Content-Type' in tool.api_header:
                    # url-params인 경우 Content-Type 제거
                    del tool.api_header['Content-Type']

        app_db.update(tool)

        backend_log.success("Tool updated successfully",
                          metadata={"function_id": function_id})

        return {
            "message": "Tool updated successfully",
            "function_id": function_id,
            "function_name": tool.function_name
        }

    except HTTPException:
        raise
    except Exception as e:
        backend_log.error("Tool update failed", exception=e,
                         metadata={"function_id": function_id})
        logger.error(f"Failed to update tool: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update tool: {str(e)}")

@router.delete("/delete/{function_id}")
async def delete_tool(request: Request, function_id: str):
    """
    특정 툴을 삭제합니다.
    """
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        backend_log.info("Starting tool deletion",
                        metadata={"function_id": function_id})

        existing_data = app_db.find_by_condition(
            Tools,
            {
                "user_id": user_id,
                "function_id": function_id,
            },
            limit=1
        )

        if not existing_data:
            backend_log.warn("Tool not found for deletion",
                           metadata={"function_id": function_id})
            raise HTTPException(status_code=404, detail=f"Tool '{function_id}' not found")

        # 데이터베이스에서 삭제
        app_db.delete(Tools, existing_data[0].id)

        backend_log.success("Tool deleted successfully",
                          metadata={"function_id": function_id,
                                  "function_name": existing_data[0].function_name})

        logger.info(f"Tool deleted successfully: {function_id}")
        return JSONResponse(content={
            "success": True,
            "message": f"Tool '{existing_data[0].function_name}' deleted successfully"
        })

    except HTTPException:
        raise
    except Exception as e:
        backend_log.error("Tool deletion failed", exception=e,
                         metadata={"function_id": function_id})
        logger.error(f"Error deleting tool: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete tool: {str(e)}")

@router.post("/api-test")
async def test_api(request: Request, test_request: dict):
    """
    API 엔드포인트를 테스트합니다. (CORS 우회)
    브라우저의 CORS 제한을 피하기 위해 백엔드에서 요청을 프록시합니다.
    """
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        # 요청 파라미터 추출
        api_url = test_request.get('api_url', '')
        api_method = test_request.get('api_method', 'GET').upper()
        api_headers = test_request.get('api_headers', {})
        api_body = test_request.get('api_body', {})  # schema (사용 안 함)
        static_body = test_request.get('static_body', {})  # 실제 key-value 데이터
        body_type = test_request.get('body_type', 'application/json')
        api_timeout = test_request.get('api_timeout', 30)
        is_query_string = test_request.get('is_query_string', False)
        html_parser = test_request.get('html_parser', True)  # HTML 파싱 여부 (기본값: True)

        if not api_url or not api_url.strip():
            raise HTTPException(status_code=400, detail="API URL is required")

        # is_query_string이 True이면 URL의 query string placeholder를 실제 값으로 치환
        import re
        if is_query_string and static_body:
            logger.info(f"Processing query string placeholders in URL: {api_url}")

            # URL에서 {key} 패턴을 찾아서 static_body의 값으로 치환
            for key, value in static_body.items():
                placeholder = f"{{{key}}}"
                if placeholder in api_url:
                    api_url = api_url.replace(placeholder, str(value))
                    logger.info(f"Replaced {placeholder} with {value}")

            logger.info(f"Final URL with replaced query string: {api_url}")
            # static_body는 이미 URL에 포함되었으므로 비움
            static_body = {}

        backend_log.info("Testing API endpoint",
                        metadata={
                            "api_url": api_url,
                            "api_method": api_method,
                            "has_headers": bool(api_headers),
                            "has_static_body": bool(static_body),
                            "body_type": body_type,
                            "is_query_string": is_query_string,
                            "html_parser": html_parser
                        })

        # httpx를 사용하여 요청 전송
        async with httpx.AsyncClient(timeout=api_timeout, follow_redirects=True) as client:
            try:
                # 요청 옵션 구성
                request_kwargs = {
                    "headers": api_headers if api_headers else {},
                }

                # body_type에 따라 Content-Type과 데이터 처리
                if api_method != 'GET':
                    if body_type == 'application/json':
                        # JSON 형식
                        request_kwargs["headers"]["Content-Type"] = "application/json"
                        if static_body:
                            request_kwargs["json"] = static_body

                    elif body_type == 'application/xml':
                        # XML 형식
                        request_kwargs["headers"]["Content-Type"] = "application/xml"
                        if static_body:
                            request_kwargs["content"] = str(static_body) if not isinstance(static_body, str) else static_body

                    elif body_type == 'application/x-www-form-urlencoded':
                        # application/x-www-form-urlencoded
                        request_kwargs["headers"]["Content-Type"] = "application/x-www-form-urlencoded"
                        if static_body:
                            request_kwargs["data"] = static_body

                    elif body_type == 'multipart/form-data':
                        # multipart/form-data (httpx가 자동으로 Content-Type 설정)
                        if static_body:
                            request_kwargs["files"] = {k: (None, str(v)) for k, v in static_body.items()}

                    elif body_type == 'text/plain':
                        # Plain text
                        request_kwargs["headers"]["Content-Type"] = "text/plain"
                        if static_body:
                            request_kwargs["content"] = str(static_body) if not isinstance(static_body, str) else static_body

                    elif body_type == 'text/html':
                        # HTML
                        request_kwargs["headers"]["Content-Type"] = "text/html"
                        if static_body:
                            request_kwargs["content"] = str(static_body) if not isinstance(static_body, str) else static_body

                    elif body_type == 'text/csv':
                        # CSV
                        request_kwargs["headers"]["Content-Type"] = "text/csv"
                        if static_body:
                            request_kwargs["content"] = str(static_body) if not isinstance(static_body, str) else static_body

                    elif body_type == 'url-params':
                        # URL 파라미터로 추가 (GET 스타일, body 없음)
                        if static_body:
                            request_kwargs["params"] = static_body

                    else:
                        # 기본값: application/json
                        request_kwargs["headers"]["Content-Type"] = "application/json"
                        if static_body:
                            request_kwargs["json"] = static_body

                elif api_method == 'GET' and static_body:
                    # GET 요청인 경우 항상 URL 파라미터로 처리
                    request_kwargs["params"] = static_body

                # 요청 전송
                logger.info(f"Sending {api_method} request to {api_url} with body_type={body_type}")
                response = await client.request(
                    method=api_method,
                    url=api_url,
                    **request_kwargs
                )

                # 응답 처리
                content_type = response.headers.get('content-type', '').lower()

                # 응답 데이터 파싱
                response_data = None
                try:
                    if 'application/json' in content_type:
                        # JSON 응답
                        response_data = response.json()
                    elif ('text/html' in content_type or 'application/xhtml' in content_type) and html_parser:
                        # HTML 응답 - html_parser가 True일 때만 파싱
                        response_data = parse_html_response(response.text)
                    elif 'text/html' in content_type or 'application/xhtml' in content_type:
                        # HTML 응답 - html_parser가 False면 텍스트 그대로
                        response_data = response.text
                    elif 'text/xml' in content_type or 'application/xml' in content_type:
                        # XML 응답 - 텍스트로 반환
                        response_data = response.text
                    elif 'text/plain' in content_type:
                        # Plain text 응답
                        response_data = response.text
                    elif 'text/csv' in content_type:
                        # CSV 응답
                        response_data = response.text
                    else:
                        # 기타 타입은 텍스트로 읽고 JSON 파싱 시도
                        text = response.text
                        try:
                            # JSON 파싱 시도
                            response_data = json.loads(text)
                        except json.JSONDecodeError:
                            # JSON 파싱 실패시 텍스트 그대로
                            response_data = text
                except Exception as parse_error:
                    logger.warning(f"Failed to parse response: {str(parse_error)}")
                    response_data = response.text

                # 응답 헤더 추출
                response_headers = dict(response.headers)

                result = {
                    "success": response.is_success,
                    "data": {
                        "status": response.status_code,
                        "statusText": response.reason_phrase,
                        "contentType": content_type or 'unknown',
                        "headers": response_headers,
                        "response": response_data
                    }
                }

                backend_log.success("API test completed",
                                  metadata={
                                      "api_url": api_url,
                                      "status_code": response.status_code,
                                      "success": response.is_success
                                  })

                logger.info(f"API test successful: {api_url} - Status: {response.status_code}")
                return JSONResponse(content=result)

            except httpx.TimeoutException:
                error_msg = f"요청 시간 초과 ({api_timeout}초)"
                backend_log.error("API test timeout",
                                metadata={"api_url": api_url, "timeout": api_timeout})
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": False,
                        "data": None,
                        "error": error_msg
                    }
                )

            except httpx.ConnectError as e:
                error_msg = f"연결 실패: {str(e)}"
                backend_log.error("API test connection error",
                                exception=e,
                                metadata={"api_url": api_url})
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": False,
                        "data": None,
                        "error": error_msg
                    }
                )

            except httpx.HTTPStatusError as e:
                error_msg = f"HTTP 오류: {e.response.status_code} {e.response.reason_phrase}"
                backend_log.error("API test HTTP error",
                                exception=e,
                                metadata={"api_url": api_url})
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": False,
                        "data": {
                            "status": e.response.status_code,
                            "statusText": e.response.reason_phrase,
                            "response": e.response.text
                        },
                        "error": error_msg
                    }
                )

            except Exception as e:
                error_msg = f"요청 실패: {str(e)}"
                backend_log.error("API test failed",
                                exception=e,
                                metadata={"api_url": api_url})
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": False,
                        "data": None,
                        "error": error_msg
                    }
                )

    except HTTPException:
        raise
    except Exception as e:
        backend_log.error("API test operation failed", exception=e)
        logger.error(f"Error testing API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to test API: {str(e)}")

@router.post("/tool-test")
async def test_tool(request: Request, tool_id: int, function_id: str):
    """
    툴의 API 엔드포인트를 테스트하고 결과에 따라 status를 업데이트합니다.
    성공 시 status를 'active'로, 실패 시 'inactive'로 변경합니다.
    """
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        # 툴 정보 조회
        tool_data = app_db.find_by_condition(
            Tools,
            {
                "id": tool_id,
                "user_id": user_id,
                "function_id": function_id
            },
            limit=1
        )

        if not tool_data or len(tool_data) == 0:
            backend_log.warn("Tool not found for testing",
                           metadata={"function_id": function_id})
            raise HTTPException(status_code=404, detail=f"Tool '{function_id}' not found")

        tool = tool_data[0]

        # 툴의 API 정보 추출
        api_url = tool.api_url
        api_method = tool.api_method or 'GET'
        api_timeout = tool.api_timeout or 30

        # api_header와 api_body 파싱
        api_headers = tool.api_header
        if isinstance(api_headers, str):
            api_headers = json.loads(api_headers) if api_headers else {}

        api_body = tool.api_body
        if isinstance(api_body, str):
            api_body = json.loads(api_body) if api_body else {}

        if not api_url or not api_url.strip():
            backend_log.warn("Tool has no API URL",
                           metadata={"function_id": function_id})
            raise HTTPException(status_code=400, detail="Tool has no API URL configured")

        backend_log.info("Testing tool API endpoint",
                        metadata={
                            "function_id": function_id,
                            "api_url": api_url,
                            "api_method": api_method,
                            "has_headers": bool(api_headers),
                            "has_body": bool(api_body)
                        })

        # httpx를 사용하여 요청 전송
        test_success = False
        error_message = None

        async with httpx.AsyncClient(timeout=api_timeout, follow_redirects=True) as client:
            try:
                # 요청 옵션 구성
                request_kwargs = {
                    "headers": api_headers if api_headers else {},
                }

                # GET이 아니고 body가 있는 경우에만 추가
                if api_method.upper() != 'GET' and api_body:
                    # Content-Type이 명시되지 않은 경우 기본값 설정
                    if 'content-type' not in {k.lower() for k in request_kwargs["headers"].keys()}:
                        request_kwargs["headers"]["Content-Type"] = "application/json"
                    request_kwargs["json"] = api_body

                # 요청 전송
                logger.info(f"Sending {api_method} request to {api_url}")
                response = await client.request(
                    method=api_method.upper(),
                    url=api_url,
                    **request_kwargs
                )

                # 응답 처리
                content_type = response.headers.get('content-type', '').lower()

                # 응답 데이터 파싱
                response_data = None
                try:
                    if 'application/json' in content_type:
                        # JSON 응답
                        response_data = response.json()
                    elif 'text/html' in content_type or 'application/xhtml' in content_type:
                        # HTML 응답 - 파싱하여 구조화된 데이터로 반환
                        response_data = parse_html_response(response.text)
                    elif 'text/xml' in content_type or 'application/xml' in content_type:
                        # XML 응답 - 텍스트로 반환
                        response_data = response.text
                    elif 'text/plain' in content_type:
                        # Plain text 응답
                        response_data = response.text
                    elif 'text/csv' in content_type:
                        # CSV 응답
                        response_data = response.text
                    else:
                        # 기타 타입은 텍스트로 읽고 JSON 파싱 시도
                        text = response.text
                        try:
                            # JSON 파싱 시도
                            response_data = json.loads(text)
                        except json.JSONDecodeError:
                            # JSON 파싱 실패시 텍스트 그대로
                            response_data = text
                except Exception as parse_error:
                    logger.warning(f"Failed to parse response: {str(parse_error)}")
                    response_data = response.text

                # 응답 헤더 추출
                response_headers = dict(response.headers)

                # 성공 여부 판단
                test_success = response.is_success

                result = {
                    "success": test_success,
                    "data": {
                        "status": response.status_code,
                        "statusText": response.reason_phrase,
                        "contentType": content_type or 'unknown',
                        "headers": response_headers,
                        "response": response_data
                    }
                }

            except httpx.TimeoutException:
                test_success = False
                error_message = f"요청 시간 초과 ({api_timeout}초)"
                result = {
                    "success": False,
                    "data": None,
                    "error": error_message
                }

            except httpx.ConnectError as e:
                test_success = False
                error_message = f"연결 실패: {str(e)}"
                result = {
                    "success": False,
                    "data": None,
                    "error": error_message
                }

            except httpx.HTTPStatusError as e:
                test_success = False
                error_message = f"HTTP 오류: {e.response.status_code} {e.response.reason_phrase}"
                result = {
                    "success": False,
                    "data": {
                        "status": e.response.status_code,
                        "statusText": e.response.reason_phrase,
                        "response": e.response.text
                    },
                    "error": error_message
                }

            except Exception as e:
                test_success = False
                error_message = f"요청 실패: {str(e)}"
                result = {
                    "success": False,
                    "data": None,
                    "error": error_message
                }

        # 테스트 결과에 따라 status 업데이트
        new_status = "active" if test_success else "inactive"
        tool.status = new_status

        update_response = app_db.update(tool)
        if not update_response or update_response.get("result") != "success":
            backend_log.error("Failed to update tool status",
                            metadata={
                                "function_id": function_id,
                                "new_status": new_status
                            })
            raise HTTPException(status_code=500, detail="Failed to update tool status")

        backend_log.success("Tool test completed and status updated",
                          metadata={
                              "function_id": function_id,
                              "api_url": api_url,
                              "test_success": test_success,
                              "new_status": new_status
                          })

        logger.info(f"Tool test completed: {function_id} - Status: {new_status}")

        # 결과에 status 정보 추가
        result["tool_status"] = new_status
        result["function_id"] = function_id
        result["function_name"] = tool.function_name

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        backend_log.error("Tool test operation failed", exception=e,
                         metadata={"function_id": function_id})
        logger.error(f"Error testing tool: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to test tool: {str(e)}")

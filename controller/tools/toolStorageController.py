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
            select_columns=["function_name", "function_id", "updated_at"],
            orderby="updated_at"
        )

        tools_list = []
        for tool in tools_data:
            tools_list.append({
                "function_name": tool.function_name,
                "function_id": tool.function_id
            })

        backend_log.success("Tool list retrieved successfully",
                          metadata={"tool_count": len(tools_list)})

        logger.info(f"Found {len(tools_list)} tools for user {user_id}")
        return JSONResponse(content={"tools": tools_list})

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
            response_filter_path=tool_data.get('response_filter_path', ''),
            response_filter_field=tool_data.get('response_filter_field', ''),
            status=tool_data.get('status', ''),
            is_shared=existing_data[0].is_shared if existing_data and len(existing_data) > 0 else False,
            share_group=existing_data[0].share_group if existing_data and len(existing_data) > 0 else None,
            share_permissions=existing_data[0].share_permissions if existing_data and len(existing_data) > 0 else 'read',
            metadata=metadata,
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

@router.get("/load/{function_id}")
async def load_tool(request: Request, function_id: str):
    """
    특정 툴을 로드합니다.
    """
    login_user_id = extract_user_id_from_request(request)
    if not login_user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, login_user_id, request)

    try:
        backend_log.info("Starting tool load operation",
                        metadata={"function_id": function_id})

        # 자신의 툴 또는 공유된 툴 검색
        tool_meta = app_db.find_by_condition(
            Tools,
            {"function_id": function_id},
            limit=1
        )

        if not tool_meta or len(tool_meta) == 0:
            backend_log.warn("Tool not found",
                           metadata={"function_id": function_id})
            raise HTTPException(status_code=404, detail=f"Tool '{function_id}' not found")

        tool = tool_meta[0]

        # 권한 확인 (자신의 툴이거나 공유된 툴인지)
        user = app_db.find_by_id(User, login_user_id)
        user_groups = user.groups if user else []

        if tool.user_id != int(login_user_id):
            if not tool.is_shared or (tool.share_group and tool.share_group not in user_groups):
                backend_log.warn("Tool access denied",
                               metadata={"function_id": function_id})
                raise HTTPException(status_code=403, detail="Access denied to this tool")

        # api_header와 api_body가 문자열이면 파싱
        api_header = tool.api_header
        if isinstance(api_header, str):
            api_header = json.loads(api_header) if api_header else {}

        api_body = tool.api_body
        if isinstance(api_body, str):
            api_body = json.loads(api_body) if api_body else {}

        metadata = tool.metadata
        if isinstance(metadata, str):
            metadata = json.loads(metadata) if metadata else {}

        tool_data = {
            "function_name": tool.function_name,
            "function_id": tool.function_id,
            "description": tool.description,
            "api_header": api_header,
            "api_body": api_body,
            "api_url": tool.api_url,
            "api_method": tool.api_method,
            "api_timeout": tool.api_timeout,
            "response_filter": tool.response_filter,
            "response_filter_path": tool.response_filter_path,
            "response_filter_field": tool.response_filter_field,
            "status": tool.status,
            "metadata": metadata,
        }

        backend_log.success("Tool loaded successfully",
                          metadata={"function_id": function_id,
                                  "function_name": tool.function_name})

        logger.info(f"Tool loaded successfully: {function_id}")
        return JSONResponse(content=tool_data)

    except HTTPException:
        raise
    except Exception as e:
        backend_log.error("Tool load operation failed", exception=e,
                         metadata={"function_id": function_id})
        logger.error(f"Error loading tool: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load tool: {str(e)}")

@router.post("/update/{function_id}")
async def update_tool(request: Request, function_id: str, update_dict: dict):
    """
    툴 정보를 업데이트합니다. (이름, 공유 설정 등)
    """
    user_id = extract_user_id_from_request(request)
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        existing_data = app_db.find_by_condition(
            Tools,
            {
                "user_id": user_id,
                "function_id": function_id
            },
            limit=1
        )

        if not existing_data:
            raise HTTPException(status_code=404, detail="Tool not found")

        tool = existing_data[0]

        # 업데이트 가능한 필드들
        if "function_name" in update_dict:
            tool.function_name = update_dict["function_name"]
        if "is_shared" in update_dict:
            tool.is_shared = update_dict["is_shared"]
        if "share_group" in update_dict:
            tool.share_group = update_dict["share_group"]
        if "share_permissions" in update_dict:
            tool.share_permissions = update_dict["share_permissions"]

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
        api_body = test_request.get('api_body', {})
        api_timeout = test_request.get('api_timeout', 30)

        if not api_url or not api_url.strip():
            raise HTTPException(status_code=400, detail="API URL is required")

        backend_log.info("Testing API endpoint",
                        metadata={
                            "api_url": api_url,
                            "api_method": api_method,
                            "has_headers": bool(api_headers),
                            "has_body": bool(api_body)
                        })

        # httpx를 사용하여 요청 전송
        async with httpx.AsyncClient(timeout=api_timeout, follow_redirects=True) as client:
            try:
                # 요청 옵션 구성
                request_kwargs = {
                    "headers": api_headers if api_headers else {},
                }

                # GET이 아니고 body가 있는 경우에만 추가
                if api_method != 'GET' and api_body:
                    # Content-Type이 명시되지 않은 경우 기본값 설정
                    if 'content-type' not in {k.lower() for k in request_kwargs["headers"].keys()}:
                        request_kwargs["headers"]["Content-Type"] = "application/json"
                    request_kwargs["json"] = api_body

                # 요청 전송
                logger.info(f"Sending {api_method} request to {api_url}")
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
                        response_data = response.json()
                    else:
                        # JSON이 아닌 경우 텍스트로 읽기
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

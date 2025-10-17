"""
Tool Store CRUD 작업 관련 엔드포인트들
사용자가 생성한 툴을 Tool Store에 업로드, 수정, 삭제하는 기능
"""
import json
import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from controller.helper.controllerHelper import extract_user_id_from_request
from controller.helper.singletonHelper import get_db_manager
from controller.tools.models.requests import UploadToolStoreRequest
from service.database.models.tools import Tools, ToolStoreMeta
from service.database.models.user import User
from service.database.logger_helper import create_logger

logger = logging.getLogger("tool-store")
router = APIRouter()

@router.get("/list")
async def list_tool_store(request: Request):
    """
    Tool Store의 모든 툴 목록을 반환합니다.
    """
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        tools_data = app_db.find_by_condition(
            ToolStoreMeta,
            {},
            limit=1000,
            orderby="updated_at",
            join_user=True,
            return_list=True
        )

        # Parse function_data if it's a JSON string
        parsed_tools = []
        for tool in tools_data:
            tool_dict = dict(tool)

            # Parse function_data if it's a string
            if 'function_data' in tool_dict and isinstance(tool_dict['function_data'], str):
                try:
                    tool_dict['function_data'] = json.loads(tool_dict['function_data'])
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse function_data for tool: {tool_dict.get('function_upload_id')}")

            # Parse metadata if it's a string
            if 'metadata' in tool_dict and isinstance(tool_dict['metadata'], str):
                try:
                    tool_dict['metadata'] = json.loads(tool_dict['metadata'])
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse metadata for tool: {tool_dict.get('function_upload_id')}")

            parsed_tools.append(tool_dict)

        backend_log.success("Tool store list retrieved successfully",
                          metadata={"tool_count": len(parsed_tools)})

        logger.info(f"Found {len(parsed_tools)} tools in store")
        return {"tools": parsed_tools}

    except Exception as e:
        backend_log.error("Failed to list tool store", exception=e)
        logger.error(f"Error listing tool store: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list tool store: {str(e)}")

@router.get("/list/detail")
async def list_tool_store_detail(request: Request):
    """
    Tool Store의 모든 툴 상세 정보를 반환합니다.
    """
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        # Tool Store의 모든 툴 조회 (join_user 사용)
        tool_store_data = app_db.find_by_condition(
            ToolStoreMeta,
            {},
            limit=10000,
            orderby="updated_at",
            return_list=True,
            join_user=True
        )

        # 데이터 포맷팅
        unique_data = []
        for item in tool_store_data:
            item_dict = dict(item)
            if 'created_at' in item_dict and item_dict['created_at']:
                item_dict['created_at'] = item_dict['created_at'].isoformat() if hasattr(item_dict['created_at'], 'isoformat') else str(item_dict['created_at'])
            if 'updated_at' in item_dict and item_dict['updated_at']:
                item_dict['updated_at'] = item_dict['updated_at'].isoformat() if hasattr(item_dict['updated_at'], 'isoformat') else str(item_dict['updated_at'])
            unique_data.append(item_dict)

        backend_log.success("Detailed tool store list retrieved successfully",
                          metadata={"total_tools": len(unique_data)})

        logger.info(f"Found {len(unique_data)} tools in store with detailed information")

        return JSONResponse(content={"tools": unique_data})

    except Exception as e:
        backend_log.error("Failed to retrieve detailed tool store list", exception=e)
        logger.error(f"Error listing tool store details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list tool store details: {str(e)}")

@router.post("/upload")
async def upload_tool_store(request: Request, upload_request: UploadToolStoreRequest):
    """
    툴을 Tool Store에 업로드합니다.
    """
    user_id = extract_user_id_from_request(request)
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        # 기존 툴 데이터 가져오기
        existing_tool = app_db.find_by_condition(
            Tools,
            {
                "id": upload_request.function_upload_id,
                "user_id": user_id,
            },
            limit=1
        )

        if not existing_tool:
            raise HTTPException(status_code=404, detail="Tool not found")

        existing_tool = existing_tool[0]

        # function_upload_id 확인
        if not upload_request.function_upload_id or upload_request.function_upload_id.strip() == "":
            raise HTTPException(status_code=400, detail="업로드할 툴의 고유 ID를 입력하세요.")

        # 이미 업로드된 툴인지 확인
        existing_upload_data = app_db.find_by_condition(
            ToolStoreMeta,
            {
                "user_id": user_id,
                "function_upload_id": upload_request.function_upload_id,
            },
            limit=1
        )
        if existing_upload_data:
            raise HTTPException(status_code=400, detail="이미 동일한 이름으로 업로드한 툴이 존재합니다. 다른 이름을 사용하세요.")

        # api_header와 api_body 파싱
        api_header = existing_tool.api_header
        if isinstance(api_header, str):
            api_header = json.loads(api_header) if api_header else {}

        api_body = existing_tool.api_body
        if isinstance(api_body, str):
            api_body = json.loads(api_body) if api_body else {}

        tool_metadata = existing_tool.metadata
        if isinstance(tool_metadata, str):
            tool_metadata = json.loads(tool_metadata) if tool_metadata else {}

        # 전체 툴 데이터를 function_data로 구성
        function_data = {
            "function_name": existing_tool.function_name,
            "function_id": existing_tool.function_id,
            "description": existing_tool.description,
            "api_header": api_header,
            "api_body": api_body,
            "api_url": existing_tool.api_url,
            "api_method": existing_tool.api_method,
            "api_timeout": existing_tool.api_timeout,
            "response_filter": existing_tool.response_filter,
            "response_filter_path": existing_tool.response_filter_path,
            "response_filter_field": existing_tool.response_filter_field,
            "status": existing_tool.status,
        }
        # Tool Store에 저장
        tool_store_data = ToolStoreMeta(
            user_id=user_id,
            function_upload_id=upload_request.function_upload_id,
            function_data=function_data,
        )

        insert_result = app_db.insert(tool_store_data)

        if insert_result and insert_result.get("result") == "success":
            backend_log.success("Tool uploaded to store successfully",
                              metadata={
                                  "function_upload_id": upload_request.function_upload_id,
                                  "description": upload_request.description
                              })

            logger.info(f"Tool uploaded to store successfully: {upload_request.function_upload_id}")
        else:
            backend_log.error("Failed to upload tool to store",
                            metadata={
                                "function_upload_id": upload_request.function_upload_id,
                                "error": insert_result.get('error', 'Unknown error')
                            })
            raise HTTPException(
                status_code=500,
                detail=f"Failed to upload tool: {insert_result.get('error', 'Unknown error')}"
            )

        return {
            "message": "Tool uploaded to store successfully",
            "function_upload_id": upload_request.function_upload_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        backend_log.error("Tool upload to store failed", exception=e,
                         metadata={"function_upload_id": upload_request.function_upload_id})
        logger.error(f"Failed to upload tool to store: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload tool to store: {str(e)}")

@router.post("/update/{function_upload_id}")
async def update_tool_store(request: Request, function_upload_id: str, update_dict: dict):
    """
    Tool Store의 툴 정보를 업데이트합니다.
    """
    user_id = extract_user_id_from_request(request)
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        existing_data = app_db.find_by_condition(
            ToolStoreMeta,
            {
                "user_id": user_id,
                "function_upload_id": function_upload_id
            },
            limit=1
        )

        if not existing_data:
            raise HTTPException(status_code=404, detail="Tool not found in store")

        tool_store = existing_data[0]

        # metadata 업데이트
        current_metadata = tool_store.metadata
        if isinstance(current_metadata, str):
            current_metadata = json.loads(current_metadata) if current_metadata else {}

        if "description" in update_dict:
            current_metadata["description"] = update_dict["description"]
        if "tags" in update_dict:
            current_metadata["tags"] = update_dict["tags"]

        tool_store.metadata = current_metadata

        # function_data 업데이트 (필요한 경우)
        if "function_data" in update_dict:
            tool_store.function_data = update_dict["function_data"]

        app_db.update(tool_store)

        backend_log.success("Tool store entry updated successfully",
                          metadata={"function_upload_id": function_upload_id})

        return {
            "message": "Tool store entry updated successfully",
            "function_upload_id": function_upload_id
        }

    except HTTPException:
        raise
    except Exception as e:
        backend_log.error("Tool store update failed", exception=e,
                         metadata={"function_upload_id": function_upload_id})
        logger.error(f"Failed to update tool store: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update tool store: {str(e)}")

@router.delete("/delete/{function_upload_id}")
async def delete_tool_from_store(request: Request, function_upload_id: str):
    """
    Tool Store에서 특정 툴을 삭제합니다.
    """
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        backend_log.info("Starting tool deletion from store",
                        metadata={"function_upload_id": function_upload_id})

        existing_data = app_db.find_by_condition(
            ToolStoreMeta,
            {
                "user_id": user_id,
                "function_upload_id": function_upload_id,
            },
            limit=1
        )

        if not existing_data:
            backend_log.warn("Tool not found in store for deletion",
                           metadata={"function_upload_id": function_upload_id})
            raise HTTPException(status_code=404, detail=f"Tool '{function_upload_id}' not found in store")

        # 데이터베이스에서 삭제
        app_db.delete(ToolStoreMeta, existing_data[0].id)

        backend_log.success("Tool deleted from store successfully",
                          metadata={"function_upload_id": function_upload_id})

        logger.info(f"Tool deleted from store successfully: {function_upload_id}")
        return JSONResponse(content={
            "success": True,
            "message": f"Tool '{function_upload_id}' deleted from store successfully"
        })

    except HTTPException:
        raise
    except Exception as e:
        backend_log.error("Tool deletion from store failed", exception=e,
                         metadata={"function_upload_id": function_upload_id})
        logger.error(f"Error deleting tool from store: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete tool from store: {str(e)}")

@router.post("/download/{store_tool_id}")
async def download_tool_from_store(request: Request, store_tool_id: str, function_upload_id: str):
    """
    Tool Store에서 툴을 다운로드하여 사용자의 툴 목록에 추가합니다.
    """
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        backend_log.info("Starting tool download from store",
                        metadata={"id": store_tool_id})

        # Tool Store에서 툴 가져오기
        tool_store_meta = app_db.find_by_condition(
            ToolStoreMeta,
            {
                "id": store_tool_id,
                "function_upload_id": function_upload_id,
            },
            limit=1
        )

        if not tool_store_meta or len(tool_store_meta) == 0:
            raise HTTPException(status_code=404, detail=f"Tool '{store_tool_id}' not found in store")

        tool_store = tool_store_meta[0]

        # function_data 파싱
        function_data = tool_store.function_data
        if isinstance(function_data, str):
            function_data = json.loads(function_data) if function_data else {}

        # 새로운 function_id 생성 (중복 방지)
        import uuid
        new_function_id = f"{function_data.get('function_id')}__{uuid.uuid4().hex[:8]}"

        # 사용자의 툴 목록에 추가
        new_tool = Tools(
            user_id=user_id,
            function_name=function_data.get("function_name", "Downloaded Tool"),
            function_id=new_function_id,
            description=function_data.get("description", ""),
            api_header=function_data.get("api_header", {}),
            api_body=function_data.get("api_body", {}),
            api_url=function_data.get("api_url", ""),
            api_method=function_data.get("api_method", "GET"),
            api_timeout=function_data.get("api_timeout", 30),
            response_filter=function_data.get("response_filter", False),
            response_filter_path=function_data.get("response_filter_path", ""),
            response_filter_field=function_data.get("response_filter_field", ""),
            status=function_data.get("status", ""),
            is_shared=False,
            share_group=None,
            share_permissions='read',
            metadata={"downloaded_from": function_upload_id},
        )

        insert_result = app_db.insert(new_tool)

        if insert_result and insert_result.get("result") == "success":
            backend_log.success("Tool downloaded from store successfully",
                              metadata={
                                  "function_upload_id": function_upload_id,
                                  "new_function_id": new_function_id,
                              })

            logger.info(f"Tool downloaded from store successfully: {function_upload_id} -> {new_function_id}")
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to download tool: {insert_result.get('error', 'Unknown error')}"
            )

        return JSONResponse(content={
            "success": True,
            "message": f"Tool '{function_data.get('function_name', 'Downloaded Tool')}' downloaded successfully",
            "function_id": new_function_id,
            "function_name": function_data.get('function_name', 'Downloaded Tool')
        })

    except HTTPException:
        raise
    except Exception as e:
        backend_log.error("Tool download from store failed", exception=e,
                         metadata={"function_upload_id": function_upload_id})
        logger.error(f"Error downloading tool from store: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to download tool from store: {str(e)}")

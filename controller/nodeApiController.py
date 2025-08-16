"""
Node API Controller
노드의 API 함수들을 동적으로 라우터에 등록하고 호출하는 컨트롤러
"""

from fastapi import APIRouter, HTTPException, Request, Body
from fastapi.datastructures import Headers
from pydantic import BaseModel
from typing import Dict, Any, Optional, Union
import json
import inspect
from editor.node_composer import get_node_api_registry, get_node_class_registry
from controller.singletonHelper import get_config_composer, get_vector_manager, get_rag_service, get_document_processor, get_db_manager

router = APIRouter(prefix="/api/editor", tags=["Node API"])

# Pydantic 모델 정의
class ApiCallRequest(BaseModel):
    """API 호출 요청 모델"""
    parameters: Dict[str, Any] = {}

class ApiCallResponse(BaseModel):
    """API 호출 응답 모델"""
    success: bool
    node_id: str
    api_name: str
    result: Any

# 동적으로 등록된 API 함수들을 저장할 딕셔너리
_registered_routes: Dict[str, Dict[str, Any]] = {}

async def call_node_api(node_id: str, api_name: str, request_data: Dict[str, Any], original_request: Request = None) -> ApiCallResponse:
    """노드의 API 함수를 호출하는 핵심 함수"""

    # 1. 노드 클래스 가져오기
    node_class_registry = get_node_class_registry()
    node_class = node_class_registry.get(node_id)

    if not node_class:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")

    # 2. API 함수 가져오기
    api_registry = get_node_api_registry()
    node_apis = api_registry.get(node_id, {})
    api_function = node_apis.get(api_name)

    if not api_function:
        raise HTTPException(
            status_code=404,
            detail=f"API function '{api_name}' not found in node '{node_id}'"
        )

    try:
        # 3. 노드 인스턴스 생성
        node_instance = node_class()

        # 4. API 함수 호출
        api_signature = inspect.signature(api_function)
        api_params = list(api_signature.parameters.keys())[1:]  # self 제외

        # 요청 데이터에서 필요한 파라미터만 추출
        filtered_kwargs = {}
        for param_name in api_params:
            if param_name == "request" and original_request:
                # Request 객체가 필요한 경우 원본 Request 전달
                filtered_kwargs[param_name] = original_request
            elif param_name in request_data:
                filtered_kwargs[param_name] = request_data[param_name]

        # API 함수 실행
        result = api_function(node_instance, **filtered_kwargs)

        return ApiCallResponse(
            success=True,
            node_id=node_id,
            api_name=api_name,
            result=result
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calling API function '{api_name}' in node '{node_id}': {str(e)}"
        )

def register_node_api_routes():
    """등록된 모든 노드의 API 함수들을 FastAPI 라우터에 등록"""

    api_registry = get_node_api_registry()

    for node_id, api_functions in api_registry.items():
        for api_name, api_function in api_functions.items():
            # nodeID 변환: / -> _, 대문자 -> 소문자
            safe_node_id = node_id.replace("/", "_").lower()
            route_path = f"/{safe_node_id}/{api_name}"

            # 등록된 라우트 기록
            _registered_routes[route_path] = {
                "node_id": node_id,
                "api_name": api_name,
                "function": api_function
            }

            print(f"  -> Registered API route: {route_path}")# 노드 API 목록 조회 엔드포인트들 (먼저 등록)
@router.get("/nodes/apis")
async def list_all_node_apis():
    """등록된 모든 노드 API 함수들의 목록을 반환"""
    api_registry = get_node_api_registry()

    result = {}
    for node_id, api_functions in api_registry.items():
        result[node_id] = []
        for api_name, api_function in api_functions.items():
            # API 함수의 시그니처 정보 추출
            api_signature = inspect.signature(api_function)
            api_params = list(api_signature.parameters.keys())[1:]  # self 제외

            safe_node_id = node_id.replace("/", "_").lower()
            result[node_id].append({
                "api_name": api_name,
                "parameters": api_params,
                "route": f"/api/{safe_node_id}/{api_name}",
                "doc": api_function.__doc__ or "No documentation available"
            })

    return {
        "success": True,
        "node_apis": result,
        "total_nodes": len(result),
        "total_apis": sum(len(apis) for apis in result.values())
    }

@router.get("/nodes/{node_id}/apis")
async def list_node_apis(node_id: str):
    """특정 노드의 API 함수들 목록을 반환"""
    api_registry = get_node_api_registry()
    node_apis = api_registry.get(node_id, {})

    if not node_apis:
        raise HTTPException(
            status_code=404,
            detail=f"No API functions found for node '{node_id}'"
        )

    result = []
    for api_name, api_function in node_apis.items():
        # API 함수의 시그니처 정보 추출
        api_signature = inspect.signature(api_function)
        api_params = list(api_signature.parameters.keys())[1:]  # self 제외

        safe_node_id = node_id.replace("/", "_").lower()
        result.append({
            "api_name": api_name,
            "parameters": api_params,
            "route": f"/api/{safe_node_id}/{api_name}",
            "doc": api_function.__doc__ or "No documentation available"
        })

    return {
        "success": True,
        "node_id": node_id,
        "apis": result,
        "total_apis": len(result)
    }

# 통합 API 호출 엔드포인트 (나중에 등록)
@router.post("/{node_id}/{api_name}")
async def call_node_api_endpoint(
    node_id: str,
    api_name: str,
    request: Request,
    request_data: ApiCallRequest = Body(...)
) -> ApiCallResponse:
    """노드의 API 함수를 호출하는 통합 엔드포인트"""

    # 원래 node_id로 복원
    original_node_id = restore_original_node_id(node_id)
    return await call_node_api(original_node_id, api_name, request_data.parameters, request)

@router.get("/{node_id}/{api_name}")
async def call_node_api_get_endpoint(
    node_id: str,
    api_name: str,
    request: Request
) -> ApiCallResponse:
    """노드의 API 함수를 GET 방식으로 호출하는 엔드포인트"""

    # 원래 node_id로 복원
    original_node_id = restore_original_node_id(node_id)

    # GET 요청의 경우 쿼리 파라미터에서 데이터 추출
    request_data = dict(request.query_params)

    return await call_node_api(original_node_id, api_name, request_data, request)

def restore_original_node_id(safe_node_id: str) -> str:
    """안전한 node_id를 원래 node_id로 복원"""
    # 등록된 라우트에서 실제 node_id 찾기
    for route_path, route_info in _registered_routes.items():
        if route_path.startswith(f"/{safe_node_id}/"):
            return route_info["node_id"]

    # 변환 규칙 역적용 (하위 호환성을 위한 기본 로직)
    # 이 로직은 실제로는 사용되지 않을 것임 (위에서 찾아야 함)
    return safe_node_id.replace("_", "/")

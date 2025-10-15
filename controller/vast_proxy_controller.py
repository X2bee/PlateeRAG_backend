"""VastAI proxy controller

All VastAI operations are proxied to a dedicated remote VastAI backend.
"""

from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from enum import Enum
import os
from controller.helper.controllerHelper import extract_user_id_from_request
from controller.helper.singletonHelper import get_config_composer, get_db_manager, get_vast_proxy_client
from service.database.logger_helper import create_logger

router = APIRouter(prefix="/api/vast", tags=["vastAI-proxy"])


def _forward_auth_headers(request: Request) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    auth_header = request.headers.get("Authorization")
    if auth_header:
        headers["Authorization"] = auth_header

    user_header = request.headers.get("X-User-ID")
    if user_header:
        headers["X-User-ID"] = user_header

    return headers


class SortBy(str, Enum):
    price = "price"
    gpu_ram = "gpu_ram"
    num_gpus = "num_gpus"


class OfferSearchRequest(BaseModel):
    gpu_name: Optional[str] = Field(None)
    max_price: Optional[float] = Field(None, ge=0)
    min_gpu_ram: Optional[int] = Field(None, ge=1)
    num_gpus: Optional[int] = Field(None, ge=1)
    inet_down: Optional[float] = Field(None, ge=0)
    inet_up: Optional[float] = Field(None, ge=0)
    rentable: Optional[bool] = Field(None)
    sort_by: SortBy = Field(SortBy.price)
    limit: Optional[int] = Field(20, ge=1, le=100)


class VLLMConfigRequest(BaseModel):
    vllm_serve_model_name: str = "Qwen/Qwen3-1.7B"
    vllm_max_model_len: int = 4096
    vllm_host_ip: str = "0.0.0.0"
    vllm_port: int = 12434
    vllm_controller_port: int = 12435
    vllm_gpu_memory_utilization: float = 0.9
    vllm_pipeline_parallel_size: int = 1
    vllm_tensor_parallel_size: int = 1
    vllm_dtype: str = "auto"
    vllm_tool_call_parser: Optional[str] = None
    vllm_trust_remote_code: bool = True
    vllm_enforce_eager: bool = False
    vllm_max_num_seqs: Optional[int] = None
    vllm_block_size: int = 16
    vllm_swap_space: int = 4
    vllm_disable_log_stats: bool = False


class VLLMServeConfigRequest(BaseModel):
    model_id: str = "Qwen/Qwen3-1.7B"
    tokenizer: Optional[str] = None
    max_model_len: int = 4096
    host: str = "0.0.0.0"
    port: int = 12434
    gpu_memory_utilization: float = 0.9
    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    kv_cache_dtype: Optional[str] = None
    tool_call_parser: Optional[str] = None


class CreateInstanceRequest(BaseModel):
    offer_id: Optional[str] = None
    offer_info: Optional[Dict[str, Any]] = None
    hf_hub_token: Optional[str] = None
    template_name: Optional[str] = None
    auto_destroy: Optional[bool] = None
    vllm_config: Optional[VLLMConfigRequest] = None


class SetVLLMConfigRequest(BaseModel):
    api_base_url: str
    model_name: str


class VLLMHealthCheckRequest(BaseModel):
    ip: str
    port: int


class ProxyApiKeyRequest(BaseModel):
    api_key: str


class OfferInfo(BaseModel):
    id: str
    gpu_name: str
    num_gpus: int
    gpu_ram: float
    dph_total: float
    rentable: bool
    cpu_cores: int
    cpu_name: Optional[str]
    ram: float
    cuda_max_good: float
    public_ipaddr: Optional[str]
    inet_down: float
    inet_up: float


class OfferSearchResponse(BaseModel):
    offers: List[OfferInfo]
    total: int
    filtered_count: int
    search_query: Optional[str]
    sort_info: Dict[str, str]


class InstanceStatusResponse(BaseModel):
    instance_id: str
    status: str
    public_ip: Optional[str] = None
    urls: Dict[str, str] = Field(default_factory=dict)
    port_mappings: Dict[str, Any] = Field(default_factory=dict)
    gpu_info: Optional[Dict[str, Any]] = None
    cost_per_hour: Optional[float] = None
    uptime: Optional[str] = None
    vllm_status: Optional[Dict[str, Any]] = None


class InstanceListResponse(BaseModel):
    instances: List[Dict[str, Any]]
    total: int


def _log_proxy_success(backend_log, message: str, metadata: Optional[Dict[str, Any]] = None):
    if backend_log:
        backend_log.success(message, metadata=metadata or {})


@router.get("/health")
async def health_check(request: Request):
    try:
        client = get_vast_proxy_client(request)
        return await client.health_check()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="서비스 사용 불가") from exc


@router.post("/search-offers", response_model=OfferSearchResponse)
async def search_offers(request: Request, search_request: OfferSearchRequest) -> OfferSearchResponse:
    user_id = extract_user_id_from_request(request)
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    client = get_vast_proxy_client(request)
    result = await client.search_offers(search_request.model_dump(), headers=_forward_auth_headers(request))
    _log_proxy_success(backend_log, "Offers searched via proxy", {"total_offers": result.get("total", 0)})
    return OfferSearchResponse(**result)


@router.post("/instances")
async def create_instance(request: Request, create_request: CreateInstanceRequest):
    user_id = extract_user_id_from_request(request)
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    client = get_vast_proxy_client(request)
    result = await client.create_instance(create_request.model_dump(), headers=_forward_auth_headers(request))
    _log_proxy_success(backend_log, "Proxy instance creation requested", {"instance_id": result.get("instance_id")})
    return result


@router.get("/instances", response_model=InstanceListResponse)
async def list_instances(
    request: Request,
    status_filter: Optional[str] = Query(None),
    include_destroyed: bool = Query(False),
    sort_by: str = Query("created_at")
) -> InstanceListResponse:
    user_id = extract_user_id_from_request(request)
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    client = get_vast_proxy_client(request)
    params = {
        "status_filter": status_filter,
        "include_destroyed": include_destroyed,
        "sort_by": sort_by,
    }
    result = await client.list_instances(params, headers=_forward_auth_headers(request))
    _log_proxy_success(backend_log, "Proxy instance list retrieved", {"total": result.get("total", 0)})
    return InstanceListResponse(**result)


@router.get("/instances/{instance_id}/status-stream")
async def stream_instance_status(request: Request, instance_id: str):
    client = get_vast_proxy_client(request)
    return await client.stream_instance_status(instance_id, headers=_forward_auth_headers(request))


@router.get("/instances/{instance_id}/status", response_model=Dict[str, Any])
async def get_instance_status(request: Request, instance_id: str):
    user_id = extract_user_id_from_request(request)
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    client = get_vast_proxy_client(request)
    result = await client.get_instance_status(instance_id, headers=_forward_auth_headers(request))
    _log_proxy_success(backend_log, "Proxy instance status fetched", {"instance_id": instance_id})
    return result


@router.delete("/instances/{instance_id}")
async def destroy_instance(request: Request, instance_id: str):
    user_id = extract_user_id_from_request(request)
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    client = get_vast_proxy_client(request)
    result = await client.destroy_instance(instance_id, headers=_forward_auth_headers(request))
    _log_proxy_success(backend_log, "Proxy instance destroy requested", {"instance_id": instance_id})
    return result


@router.post("/instances/{instance_id}/update-ports")
async def update_instance_ports(request: Request, instance_id: str):
    user_id = extract_user_id_from_request(request)
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    client = get_vast_proxy_client(request)
    result = await client.update_ports(instance_id, headers=_forward_auth_headers(request))
    _log_proxy_success(backend_log, "Proxy port update requested", {"instance_id": instance_id})
    return result


@router.post("/instances/{instance_id}/vllm-serve")
async def vllm_serve(request: Request, instance_id: str, vllm_config: VLLMServeConfigRequest):
    user_id = extract_user_id_from_request(request)
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    client = get_vast_proxy_client(request)
    result = await client.vllm_serve(instance_id, vllm_config.model_dump(), headers=_forward_auth_headers(request))
    _log_proxy_success(backend_log, "Proxy VLLM serve requested", {"instance_id": instance_id})
    return result


@router.post("/instances/{instance_id}/vllm-down")
async def vllm_down(request: Request, instance_id: str):
    user_id = extract_user_id_from_request(request)
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    client = get_vast_proxy_client(request)
    result = await client.vllm_down(instance_id, headers=_forward_auth_headers(request))
    _log_proxy_success(backend_log, "Proxy VLLM down requested", {"instance_id": instance_id})
    return result


@router.put("/set-vllm",
    summary="VLLM 설정 업데이트",
    description="VLLM API Base URL과 모델명을 업데이트합니다. 실행 중인 애플리케이션의 설정을 동적으로 변경할 수 있습니다.",
    response_model=Dict[str, Any],
    responses={
        400: {"description": "잘못된 설정 값"},
        404: {"description": "설정을 찾을 수 없음"},
        500: {"description": "서버 오류"}
    })
async def set_vllm_config(request: Request, vllm_config: SetVLLMConfigRequest):
    """VLLM API Base URL과 모델명 설정 업데이트"""
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        # appController의 update_persistent_config 함수를 import
        from controller.appController import update_persistent_config, ConfigUpdateRequest

        try:
            config_composer = get_config_composer(request)
            old_api_base_url = config_composer.get_config_by_name("VLLM_API_BASE_URL").value

            api_base_url_request = ConfigUpdateRequest(value=vllm_config.api_base_url)
            api_base_url_result = await update_persistent_config("VLLM_API_BASE_URL", api_base_url_request, request)

            backend_log.info(f"VLLM_API_BASE_URL 업데이트: {old_api_base_url} -> {vllm_config.api_base_url}")
        except KeyError:
            backend_log.warning("VLLM_API_BASE_URL 설정을 찾을 수 없습니다")
            raise HTTPException(status_code=404, detail="VLLM_API_BASE_URL 설정을 찾을 수 없습니다")

        try:
            old_model_name = config_composer.get_config_by_name("VLLM_MODEL_NAME").value

            model_name_request = ConfigUpdateRequest(value=vllm_config.model_name)
            model_name_result = await update_persistent_config("VLLM_MODEL_NAME", model_name_request, request)

            backend_log.info(f"VLLM_MODEL_NAME 업데이트: {old_model_name} -> {vllm_config.model_name}")

        except KeyError:
            backend_log.warning("VLLM_MODEL_NAME 설정을 찾을 수 없습니다")
            raise HTTPException(status_code=404, detail="VLLM_MODEL_NAME 설정을 찾을 수 없습니다")
        try:
            provider_request = ConfigUpdateRequest(value='vllm')
            provider_response = await update_persistent_config("DEFAULT_LLM_PROVIDER", provider_request, request)

            backend_log.info(f"DEFAULT_LLM_PROVIDER 업데이트: {provider_request.value}")

        except KeyError:
            backend_log.warning("DEFAULT_LLM_PROVIDER 설정을 찾을 수 없습니다")
            raise HTTPException(status_code=404, detail="DEFAULT_LLM_PROVIDER 설정을 찾을 수 없습니다")

        backend_log.info(f"VLLM 설정 업데이트", metadata={"api_base_url": api_base_url_result, "model_name": model_name_result})

        return {
            "success": True,
            "message": "VLLM 설정이 성공적으로 업데이트되었습니다",
            "api_base_url": vllm_config.api_base_url,
            "model_name": vllm_config.model_name,
            "method": "update_persistent_config"
        }

    except HTTPException:
        raise
    except Exception as e:
        backend_log.error("VLLM 설정 업데이트 실패", metadata={"error": str(e)})
        raise HTTPException(status_code=500, detail=f"VLLM 설정 업데이트 실패: {str(e)}")


@router.post("/instances/vllm-health")
async def vllm_health_check(request: Request, health_request: VLLMHealthCheckRequest):
    client = get_vast_proxy_client(request)
    return await client.check_vllm_health(health_request.model_dump(), headers=_forward_auth_headers(request))


@router.post("/proxy/api-key")
async def update_proxy_api_key(request: Request, payload: ProxyApiKeyRequest):
    expected_token = os.getenv("VAST_PROXY_API_TOKEN", "").strip()
    auth_header = request.headers.get("Authorization", "").strip()

    if expected_token:
        expected_header = f"Bearer {expected_token}"
        if auth_header != expected_header:
            raise HTTPException(status_code=401, detail="Unauthorized")

    config_composer = get_config_composer(request)
    update_result = config_composer.update_config("VAST_API_KEY", payload.api_key)
    os.environ["VAST_API_KEY"] = payload.api_key
    return {
        "success": True,
        "message": "Vast API key updated",
        "old_value": update_result.get("old_value"),
        "new_value": update_result.get("new_value"),
    }

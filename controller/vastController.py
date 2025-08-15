"""
VastAI 인스턴스 관리 컨트롤러

VastAI 클라우드 GPU 인스턴스의 검색, 생성, 관리 및 모니터링을 위한 RESTful API
"""

from fastapi import APIRouter, HTTPException, Request, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import logging
from typing import Dict, Any, Optional, List, Literal
from enum import Enum
import json
import asyncio
from urllib import request as urllib_request
from service.vast.vast_service import VastService

router = APIRouter(prefix="/api/vast", tags=["vastAI"])
logger = logging.getLogger("vast-controller")

# SSE 연결 관리를 위한 전역 변수
sse_connections: Dict[str, List[asyncio.Queue]] = {}

# ========== Enums ==========
class SortBy(str, Enum):
    price = "price"
    gpu_ram = "gpu_ram"
    num_gpus = "num_gpus"

# ========== Request Models ==========
class OfferSearchRequest(BaseModel):
    """GPU 오퍼 검색 요청"""
    gpu_name: Optional[str] = Field(None, description="GPU 모델명", example="RTX4090")
    max_price: Optional[float] = Field(None, description="최대 시간당 가격 ($)", example=2.0, ge=0)
    min_gpu_ram: Optional[int] = Field(None, description="최소 GPU RAM (GB)", example=16, ge=1)
    num_gpus: Optional[int] = Field(None, description="GPU 개수", example=1, ge=1)
    inet_down: Optional[float] = Field(None, description="최소 다운로드 속도 (Mbps)", example=100.0, ge=0)
    inet_up: Optional[float] = Field(None, description="최소 업로드 속도 (Mbps)", example=100.0, ge=0)
    rentable: Optional[bool] = Field(None, description="렌트 가능 여부", example=True)
    sort_by: SortBy = Field(SortBy.price, description="정렬 기준")
    limit: Optional[int] = Field(20, description="결과 제한 개수", example=10, ge=1, le=100)

class VLLMConfigRequest(BaseModel):
    """VLLM 설정 요청"""
    # 모델 설정
    vllm_serve_model_name: str = Field("Qwen/Qwen3-1.7B", description="사용할 모델명", example="Qwen/Qwen3-1.7B")
    vllm_max_model_len: int = Field(4096, description="최대 모델 길이", example=2048, ge=512, le=32768)

    # 네트워크 설정
    vllm_host_ip: str = Field("0.0.0.0", description="호스트 IP", example="0.0.0.0")
    vllm_port: int = Field(12434, description="VLLM 서비스 포트", example=12434, ge=1024, le=65535)
    vllm_controller_port: int = Field(12435, description="VLLM 컨트롤러 포트", example=12435, ge=1024, le=65535)

    # 성능 설정
    vllm_gpu_memory_utilization: float = Field(0.9, description="GPU 메모리 사용률", example=0.5, ge=0.1, le=1.0)
    vllm_pipeline_parallel_size: int = Field(1, description="파이프라인 병렬 크기", example=1, ge=1)
    vllm_tensor_parallel_size: int = Field(1, description="텐서 병렬 크기", example=1, ge=1)

    # 데이터 타입 및 고급 설정
    vllm_dtype: str = Field('auto', description="데이터 타입")
    vllm_tool_call_parser: Optional[str] = Field(None, description="도구 호출 파서")
    vllm_trust_remote_code: bool = Field(True, description="원격 코드 신뢰 여부")
    vllm_enforce_eager: bool = Field(False, description="즉시 실행 강제 여부")
    vllm_max_num_seqs: Optional[int] = Field(None, description="최대 시퀀스 수", ge=1)
    vllm_block_size: int = Field(16, description="블록 크기", example=16, ge=1)
    vllm_swap_space: int = Field(4, description="스왑 공간 (GiB)", example=4, ge=0)
    vllm_disable_log_stats: bool = Field(False, description="로그 통계 비활성화")

class VLLMServeConfigRequest(BaseModel):
    """VLLM 설정 요청"""
    # 모델 설정
    model_id: str = Field("Qwen/Qwen3-1.7B", description="사용할 모델명", example="Qwen/Qwen3-1.7B")
    tokenizer: Optional[str] = Field(None, description="토크나이저 모델명", example="Qwen/Qwen3-1.7B-tokenizer")
    max_model_len: int = Field(4096, description="최대 모델 길이", example=2048, ge=512, le=32768)

    # 네트워크 설정
    host: str = Field("0.0.0.0", description="호스트 IP", example="0.0.0.0")
    port: int = Field(12434, description="VLLM 서비스 포트", example=12434, ge=1024, le=65535)

    # 성능 설정
    gpu_memory_utilization: float = Field(0.9, description="GPU 메모리 사용률", example=0.5, ge=0.1, le=1.0)
    pipeline_parallel_size: int = Field(1, description="파이프라인 병렬 크기", example=1, ge=1)
    tensor_parallel_size: int = Field(1, description="텐서 병렬 크기", example=1, ge=1)

    # 데이터 타입 및 고급 설정
    dtype: str = Field('auto', description="데이터 타입")
    kv_cache_dtype: Optional[str] = Field(None, description="KV 캐시 데이터 타입", example="bfloat16")
    tool_call_parser: Optional[str] = Field(None, description="도구 호출 파서")

class CreateInstanceRequest(BaseModel):
    """인스턴스 생성 요청"""
    offer_id: str = Field(None, description="특정 오퍼 ID (없으면 자동 선택)", example="12345")
    offer_info: Optional[Dict[str, Any]] = Field(None, description="오퍼 정보")
    hf_hub_token: Optional[str] = Field(None, description="HuggingFace 토큰", example="hf_xxxxx")
    template_name: Optional[str] = Field(None, description="사용할 템플릿 이름 (budget, high_performance, research)", example="budget")
    auto_destroy: Optional[bool] = Field(None, description="자동 삭제 여부", example=False)
    vllm_config: Optional[VLLMConfigRequest] = Field(None, description="VLLM 설정 (선택사항)")

class SetupVLLMRequest(BaseModel):
    """VLLM 설정 및 실행 요청"""
    script_directory: str = Field("/vllm/vllm-script", description="스크립트 디렉토리 경로")
    hf_hub_token: Optional[str] = Field(None, description="HuggingFace 토큰", example="hf_xxxxx")
    main_script: str = Field("main.py", description="메인 스크립트 파일명")
    log_file: str = Field("/tmp/vllm.log", description="로그 파일 경로")
    install_requirements: bool = Field(True, description="requirements.txt 설치 여부")
    vllm_config: VLLMConfigRequest = Field(..., description="VLLM 설정")
    additional_env_vars: Optional[Dict[str, str]] = Field(None, description="추가 환경변수")

class ExecuteCommandRequest(BaseModel):
    """명령어 실행 요청"""
    command: str = Field(..., description="실행할 명령어", example="ps aux | grep python")
    working_directory: Optional[str] = Field(None, description="작업 디렉토리", example="/vllm/vllm-script")
    environment_vars: Optional[Dict[str, str]] = Field(None, description="환경변수")
    background: bool = Field(False, description="백그라운드 실행 여부")
    timeout: Optional[int] = Field(300, description="타임아웃 (초)", ge=1, le=3600)

class SetVLLMConfigRequest(BaseModel):
    """VLLM 설정 업데이트 요청"""
    api_base_url: str = Field(..., description="VLLM API Base URL", example="http://localhost:8000/v1")
    model_name: str = Field(..., description="VLLM 모델명", example="Qwen/Qwen3-1.7B")

class VLLMHealthCheckRequest(BaseModel):
    """VLLM 헬스 체크 요청"""
    ip: str = Field(..., description="VLLM 서비스 IP", example="1.2.3.4")
    port: int = Field(..., description="VLLM 서비스 포트", example=12434)

# ========== Response Models ==========
class OfferInfo(BaseModel):
    """GPU 오퍼 정보"""
    id: str = Field(..., description="오퍼 ID")
    gpu_name: str = Field(..., description="GPU 모델명")
    num_gpus: int = Field(..., description="GPU 개수")
    gpu_ram: float = Field(..., description="GPU RAM (GB)")
    dph_total: float = Field(..., description="시간당 총 가격 ($)")
    rentable: bool = Field(..., description="렌트 가능 여부")
    cpu_cores: int = Field(..., description="CPU 코어 수")
    cpu_name: Optional[str] = Field(..., description="CPU 모델명")
    ram: float = Field(..., description="RAM 용량")
    cuda_max_good: float = Field(..., description="CUDA 최대 버전")
    public_ipaddr: Optional[str] = Field(None, description="공개 IP 주소")
    inet_down: float = Field(..., description="다운로드 속도 (Mbps)")
    inet_up: float = Field(..., description="업로드 속도 (Mbps)")

class OfferSearchResponse(BaseModel):
    """GPU 오퍼 검색 응답"""
    offers: List[OfferInfo] = Field(..., description="검색된 오퍼 목록")
    total: int = Field(..., description="전체 오퍼 수")
    filtered_count: int = Field(..., description="필터링된 오퍼 수")
    search_query: Optional[str] = Field(None, description="사용된 검색 쿼리")
    sort_info: Dict[str, str] = Field(..., description="정렬 정보")

class InstanceStatusResponse(BaseModel):
    """인스턴스 상태 응답"""
    instance_id: str = Field(..., description="인스턴스 ID")
    status: str = Field(..., description="인스턴스 상태")
    public_ip: Optional[str] = Field(None, description="공개 IP")
    urls: Dict[str, str] = Field(default_factory=dict, description="접근 URL들")
    port_mappings: Dict[str, Any] = Field(default_factory=dict, description="포트 매핑")
    gpu_info: Optional[Dict[str, Any]] = Field(None, description="GPU 정보")
    cost_per_hour: Optional[float] = Field(None, description="시간당 비용")
    uptime: Optional[str] = Field(None, description="가동 시간")
    vllm_status: Optional[Dict[str, Any]] = Field(None, description="VLLM 상태")

class InstanceListResponse(BaseModel):
    """인스턴스 목록 응답"""
    instances: List[Dict[str, Any]] = Field(..., description="인스턴스 목록")
    total: int = Field(..., description="총 인스턴스 수")

class CommandExecutionResponse(BaseModel):
    """명령어 실행 응답"""
    success: bool = Field(..., description="실행 성공 여부")
    instance_id: str = Field(..., description="인스턴스 ID")
    command: str = Field(..., description="실행된 명령어")
    stdout: str = Field(..., description="표준 출력")
    stderr: str = Field(..., description="표준 에러")
    background: bool = Field(..., description="백그라운드 실행 여부")
    error: Optional[str] = Field(None, description="에러 메시지")

# ========== Helper Functions ==========
def get_vast_service(request: Request) -> VastService:
    """VastService 인스턴스 생성"""
    try:
        service = request.app.state.vast_service

        # 동기 SSE 브로드캐스트 콜백 설정
        service.set_status_change_callback(_sync_broadcast_status_change)

        return service
    except Exception as e:
        logger.error(f"VastService 초기화 실패: {e}")
        raise HTTPException(status_code=500, detail="VastService 초기화 실패")

def get_config_composer(request: Request):
    """ConfigComposer 인스턴스 가져오기"""
    try:
        return request.app.state.config_composer
    except Exception as e:
        logger.error(f"ConfigComposer 가져오기 실패: {e}")
        raise HTTPException(status_code=500, detail="ConfigComposer 초기화 실패")

# ========== SSE 관련 함수들 ==========
def _sync_broadcast_status_change(instance_id: str, status: str):
    """동기적으로 호출 가능한 상태 변경 브로드캐스트 래퍼"""
    try:
        # 현재 실행 중인 이벤트 루프가 있는지 확인
        try:
            loop = asyncio.get_running_loop()
            # 이벤트 루프가 실행 중이면 태스크로 스케줄링
            loop.create_task(_broadcast_status_change(instance_id, status))
        except RuntimeError:
            # 이벤트 루프가 없으면 새로 생성하여 실행
            asyncio.run(_broadcast_status_change(instance_id, status))
    except Exception as e:
        logger.warning(f"SSE 브로드캐스트 실패: {e}")

async def _broadcast_status_change(instance_id: str, status: str):
    """인스턴스 상태 변경을 모든 SSE 클라이언트에게 브로드캐스트"""
    if instance_id not in sse_connections:
        return

    status_data = {
        "instance_id": instance_id,
        "status": status,
        "timestamp": asyncio.get_event_loop().time()
    }

    # 연결이 끊어진 큐들을 제거하기 위한 리스트
    dead_queues = []

    for queue in sse_connections[instance_id]:
        try:
            await queue.put(status_data)
        except Exception as e:
            logger.warning(f"SSE 큐에 데이터 전송 실패: {e}")
            dead_queues.append(queue)

    # 끊어진 연결 제거
    for dead_queue in dead_queues:
        sse_connections[instance_id].remove(dead_queue)

    # 연결이 없으면 인스턴스 키 제거
    if not sse_connections[instance_id]:
        del sse_connections[instance_id]

def _add_sse_connection(instance_id: str, queue: asyncio.Queue):
    """새로운 SSE 연결 추가"""
    if instance_id not in sse_connections:
        sse_connections[instance_id] = []
    sse_connections[instance_id].append(queue)

def _remove_sse_connection(instance_id: str, queue: asyncio.Queue):
    """SSE 연결 제거"""
    if instance_id in sse_connections and queue in sse_connections[instance_id]:
        sse_connections[instance_id].remove(queue)
        if not sse_connections[instance_id]:
            del sse_connections[instance_id]

# ========== API Endpoints ==========

@router.get("/health",
    summary="서비스 상태 확인",
    description="VastAI 서비스의 상태와 연결을 확인합니다.",
    response_description="서비스 상태 정보",
    responses={
        200: {"description": "서비스 정상"},
        503: {"description": "서비스 사용 불가"}
    })
async def health_check(request: Request):
    try:
        service = get_vast_service(request)
        return {
            "status": "healthy",
            "service": "vast",
            "message": "VastAI 서비스가 정상적으로 작동 중입니다"
        }
    except Exception as e:
        logger.error(f"Health check 실패: {e}")
        raise HTTPException(status_code=503, detail="서비스 사용 불가")

@router.post("/search-offers",
    summary="GPU 오퍼 검색",
    description="사용 가능한 VastAI GPU 오퍼를 검색하고 필터링합니다. 가격, GPU 사양, 가용성 등으로 필터링 가능합니다.",
    response_model=OfferSearchResponse,
    responses={
        400: {"description": "API 키 설정 오류"},
        500: {"description": "검색 실패"}
    })
async def search_offers(request: Request, search_request: OfferSearchRequest) -> OfferSearchResponse:
    try:
        service = get_vast_service(request)

        if not service.vast_manager.setup_api_key():
            raise HTTPException(status_code=400, detail="API 키가 설정되지 않았거나 잘못되었습니다")

        # 검색 쿼리 구성
        query_parts = []
        if search_request.gpu_name:
            query_parts.append(f"gpu_name={search_request.gpu_name}")
        if search_request.max_price:
            query_parts.append(f"dph_total<={search_request.max_price}")
        if search_request.min_gpu_ram:
            query_parts.append(f"gpu_ram>={search_request.min_gpu_ram}")
        if search_request.num_gpus:
            query_parts.append(f"num_gpus={search_request.num_gpus}")
        if search_request.inet_down:
            query_parts.append(f"inet_down>={search_request.inet_down}")
        if search_request.inet_up:
            query_parts.append(f"inet_up>={search_request.inet_up}")
        if search_request.rentable is not None:
            query_parts.append(f"rentable={str(search_request.rentable).lower()}")

        search_query = " ".join(query_parts) if query_parts else None
        offers = service.vast_manager.search_offers(search_query)

        if not offers:
            return OfferSearchResponse(
                offers=[],
                total=0,
                filtered_count=0,
                search_query=search_query,
                sort_info={"sort_by": search_request.sort_by, "order": "ascending"}
            )

        # 정렬 및 제한
        sort_key_map = {"price": "dph_total", "gpu_ram": "gpu_ram", "num_gpus": "num_gpus"}
        sort_key = sort_key_map.get(search_request.sort_by, "dph_total")
        sorted_offers = sorted(offers, key=lambda x: x.get(sort_key, 999))
        limited_offers = sorted_offers[:search_request.limit] if search_request.limit else sorted_offers

        return OfferSearchResponse(
            offers=[OfferInfo(**offer) for offer in limited_offers],
            total=len(offers),
            filtered_count=len(limited_offers),
            search_query=search_query,
            sort_info={"sort_by": search_request.sort_by, "sort_key": sort_key, "order": "ascending"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"오퍼 검색 실패: {e}")
        raise HTTPException(status_code=500, detail="오퍼 검색 실패")

@router.post("/instances",
    summary="인스턴스 생성",
    description="새로운 VastAI 인스턴스를 생성합니다. 템플릿 사용(budget, high_performance, research) 또는 커스텀 설정 가능합니다.",
    response_model=Dict[str, Any],
    responses={
        400: {"description": "잘못된 요청 (템플릿 없음, 인스턴스 생성 실패 등)"},
        500: {"description": "서버 오류"}
    })
async def create_instance(request: Request, create_request: CreateInstanceRequest, background_tasks: BackgroundTasks):
    try:
        service = get_vast_service(request)
        config_composer = request.app.state.config_composer
        vast_config = config_composer.get_config_by_category_name("vast")

        # 템플릿 적용
        if create_request.template_name:
            if create_request.template_name not in service.templates:
                available_templates = list(service.templates.keys())
                raise HTTPException(
                    status_code=400,
                    detail=f"템플릿 '{create_request.template_name}'을 찾을 수 없습니다. 사용 가능한 템플릿: {available_templates}"
                )
            service.apply_template(create_request.template_name)

        if create_request.vllm_config:
            logger.info("VLLM 설정 적용")
            for key, value in create_request.vllm_config.dict().items():
                # env_name을 통해 PersistentConfig 객체 찾기
                env_name = f"VLLM_{key.upper()}" if not key.startswith('vllm_') else key.upper()
                # all_configs에서 env_name으로 찾기
                if hasattr(vast_config, 'configs'):
                    for config_obj in vast_config.configs.values():
                        if hasattr(config_obj, 'env_name') and config_obj.env_name == env_name:
                            config_obj.value = value
                            logger.info("VLLM 설정 적용: %s = %s", key, value)

        # 인스턴스 생성
        instance_id = service.create_vllm_instance(
            offer_id=create_request.offer_id,
            template_name=create_request.template_name,
            create_request=create_request,
            vast_config_request=vast_config.configs,
        )

        if not instance_id:
            raise HTTPException(status_code=400, detail="인스턴스 생성 실패")

        # 인스턴스 생성 후 즉시 상태 확인
        try:
            # 인스턴스가 실제로 생성되었는지 확인
            db_instance = service.get_instance_from_db(instance_id)
            if not db_instance:
                raise HTTPException(status_code=400, detail="인스턴스 생성 후 DB에서 찾을 수 없음")

            # VastAI에서 인스턴스 상태 확인
            vast_instance = service.vast_manager.get_instance_info(instance_id)
            if not vast_instance:
                # 인스턴스가 VastAI에 존재하지 않으면 실패로 처리
                service._update_instance(instance_id, updates={"status": "failed"})
                await _broadcast_status_change(instance_id, "failed")
                raise HTTPException(
                    status_code=400,
                    detail=f"인스턴스 {instance_id} 생성 실패: VastAI에서 인스턴스를 찾을 수 없습니다"
                )

            # 인스턴스 상태가 실패인 경우 처리
            vast_status = vast_instance.get("actual_status", "unknown")
            if vast_status in ["exited", "failed", "error"]:
                service._update_instance(instance_id, updates={"status": "failed"})
                await _broadcast_status_change(instance_id, "failed")
                raise HTTPException(
                    status_code=400,
                    detail=f"인스턴스 {instance_id} 생성 실패: 상태가 {vast_status}입니다"
                )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"인스턴스 {instance_id} 상태 확인 실패: {e}")
            service._update_instance(instance_id, updates={"status": "failed"})
            await _broadcast_status_change(instance_id, "failed")
            raise HTTPException(
                status_code=400,
                detail=f"인스턴스 {instance_id} 생성 후 상태 확인 실패: {str(e)}"
            )

        is_valid_model = False
        if create_request.vllm_config and create_request.vllm_config.vllm_serve_model_name and len(create_request.vllm_config.vllm_serve_model_name) >= 1:
            is_valid_model = True

        background_tasks.add_task(service.wait_and_setup_instance, instance_id, is_valid_model)

        # 상태 브로드캐스트
        await _broadcast_status_change(instance_id, "creating")

        return {
            "success": True,
            "instance_id": instance_id,
            "template_name": create_request.template_name,
            "message": "인스턴스가 생성되었습니다. 설정이 백그라운드에서 진행됩니다.",
            "status": "creating",
            "tracking_endpoints": {
                "detailed_status": f"/api/vast/instances/{instance_id}",
                "detailed_info": f"/api/vast/instances/{instance_id}/info",
                "update_ports": f"/api/vast/instances/{instance_id}/update-ports"
            },
            "next_steps": [
                f"1. /api/vast/instances/{instance_id} 엔드포인트로 상태를 확인하세요",
                f"2. /api/vast/instances/{instance_id}/info 엔드포인트로 상세 정보를 확인하세요",
                f"3. 포트 매핑이 필요하면 /api/vast/instances/{instance_id}/update-ports를 호출하세요"
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"인스턴스 생성 실패: {e}")
        raise HTTPException(status_code=500, detail="인스턴스 생성 실패")

@router.get("/instances",
    summary="인스턴스 목록 조회",
    description="모든 인스턴스의 목록을 조회합니다. 상태별 필터링과 정렬을 지원합니다.",
    response_model=InstanceListResponse)
async def list_instances(
    request: Request,
    status_filter: Optional[str] = Query(None, description="상태별 필터링"),
    include_destroyed: bool = Query(False, description="삭제된 인스턴스 포함"),
    sort_by: str = Query("created_at", description="정렬 기준")
) -> InstanceListResponse:
    try:
        service = get_vast_service(request)

        # VastAI에서 내가 현재 빌린 인스턴스 목록 조회
        my_active_instance_ids = set()
        try:
            result = service.vast_manager.run_command(["vastai", "show", "instances"], parse_json=False)
            if result["success"]:
                vast_instances = service.vast_manager._parse_instances_from_text(result["data"])
                for vast_inst in vast_instances:
                    if isinstance(vast_inst, dict) and vast_inst.get("id"):
                        my_active_instance_ids.add(str(vast_inst["id"]))
        except Exception as e:
            logger.warning(f"VastAI 인스턴스 목록 조회 중 오류: {e}")

        # DB에서 VastAI에 없는 인스턴스들을 deleted로 업데이트
        updated_count = 0
        if service.db_manager:
            from service.database.models.vast import VastInstance
            db_instances = service.db_manager.find_all(VastInstance)

            for db_inst in db_instances:
                instance_id = str(db_inst.instance_id)
                if instance_id not in my_active_instance_ids and db_inst.status not in ["destroyed", "deleted"]:
                    logger.info(f"인스턴스 {instance_id}가 내 빌린 인스턴스 목록에서 발견되지 않음. 상태를 deleted로 업데이트")
                    try:
                        db_inst.status = "deleted"
                        service.db_manager.update(db_inst)
                        updated_count += 1
                    except Exception as e:
                        logger.error(f"인스턴스 {instance_id} 상태 업데이트 실패: {e}")

        # 로그 출력
        if updated_count > 0:
            logger.info(f"총 {updated_count}개 인스턴스의 상태를 deleted로 업데이트했습니다")

        # DB에서 deleted가 아닌 인스턴스들만 가져오기
        instances = service.list_instances()

        # 필터링
        if status_filter:
            instances = [inst for inst in instances
                        if inst.get("actual_status") == status_filter or
                           inst.get("status") == status_filter]

        if not include_destroyed:
            instances = [inst for inst in instances
                        if inst.get("actual_status") not in ["exited", "deleted"]]

        # 정렬
        if sort_by == "created_at":
            instances.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        elif sort_by == "cost":
            instances.sort(key=lambda x: x.get("cost_per_hour", 0))

        return InstanceListResponse(instances=instances, total=len(instances))
    except Exception as e:
        logger.error(f"인스턴스 목록 조회 실패: {e}")
        raise HTTPException(status_code=500, detail="인스턴스 목록 조회 실패")

@router.get("/instances/{instance_id}/status-stream",
    summary="인스턴스 상태 스트리밍",
    description="SSE(Server-Sent Events)를 통해 인스턴스 상태 변경을 실시간으로 스트리밍합니다.",
    responses={
        200: {"description": "SSE 스트림"},
        404: {"description": "인스턴스를 찾을 수 없음"}
    })
async def stream_instance_status(request: Request, instance_id: str):
    """SSE를 통한 인스턴스 상태 실시간 스트리밍"""
    try:
        service = get_vast_service(request)
        db_instance = service.get_instance_from_db(instance_id)

        if not db_instance:
            raise HTTPException(status_code=404, detail=f"인스턴스 '{instance_id}'를 찾을 수 없습니다")

        # SSE용 큐 생성
        queue = asyncio.Queue()
        _add_sse_connection(instance_id, queue)

        async def event_stream():
            try:
                # 현재 상태를 먼저 전송
                current_status = {
                    "instance_id": instance_id,
                    "status": db_instance.status,
                    "timestamp": asyncio.get_event_loop().time()
                }
                yield f"data: {json.dumps(current_status)}\n\n"

                # 상태 변경 이벤트 대기
                while True:
                    try:
                        # 큐에서 상태 변경 이벤트 대기 (타임아웃 30초)
                        status_data = await asyncio.wait_for(queue.get(), timeout=30.0)
                        yield f"data: {json.dumps(status_data)}\n\n"
                    except asyncio.TimeoutError:
                        # 연결 유지를 위한 heartbeat
                        heartbeat = {
                            "type": "heartbeat",
                            "timestamp": asyncio.get_event_loop().time()
                        }
                        yield f"data: {json.dumps(heartbeat)}\n\n"
                    except Exception as e:
                        logger.error(f"SSE 스트림 오류: {e}")
                        break
            finally:
                # 연결 종료 시 큐 제거
                _remove_sse_connection(instance_id, queue)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"SSE 스트림 생성 실패: {e}")
        raise HTTPException(status_code=500, detail="SSE 스트림 생성 실패")

@router.get("/instances/{instance_id}/status",
    summary="인스턴스 상태 조회",
    description="지정된 인스턴스의 현재 상태를 DB에서 조회하여 반환합니다.",
    response_model=Dict[str, Any],
    responses={
        404: {"description": "인스턴스를 찾을 수 없음"},
        500: {"description": "서버 오류"}
    })
async def get_instance_status(request: Request, instance_id: str):
    """인스턴스 상태 조회"""
    try:
        service = get_vast_service(request)
        db_instance = service.get_instance_from_db(instance_id)

        if not db_instance:
            raise HTTPException(status_code=404, detail=f"인스턴스 '{instance_id}'를 찾을 수 없습니다")

        return {
            "instance_id": instance_id,
            "status": db_instance.status
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"인스턴스 상태 조회 실패: {e}")
        raise HTTPException(status_code=500, detail="인스턴스 상태 조회 실패")

@router.delete("/instances/{instance_id}",
    summary="인스턴스 삭제",
    description="지정된 인스턴스를 삭제합니다.",
    response_model=Dict[str, Any],
    responses={
        400: {"description": "인스턴스 삭제 실패"},
        500: {"description": "서버 오류"}
    })
async def destroy_instance(request: Request, instance_id: str):
    try:
        service = get_vast_service(request)
        success = service.destroy_instance(instance_id)

        if not success:
            raise HTTPException(status_code=400, detail="인스턴스 삭제 실패")

        # 상태 업데이트 및 SSE 브로드캐스트
        await _broadcast_status_change(instance_id, "deleted")

        return {
            "success": True,
            "message": f"인스턴스 {instance_id}가 삭제되었습니다",
            "instance_id": instance_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"인스턴스 삭제 실패: {e}")
        raise HTTPException(status_code=500, detail="인스턴스 삭제 실패")


@router.post("/instances/{instance_id}/update-ports",
    summary="포트 매핑 정보 업데이트",
    description="인스턴스의 포트 매핑 정보를 VastAI에서 가져와 DB에 업데이트합니다. 개선된 IP 추출 로직을 사용하여 정확한 공인 IP를 식별합니다.",
    response_model=Dict[str, Any])
async def update_instance_ports(request: Request, instance_id: str):
    try:
        service = get_vast_service(request)
        logger.info(f"인스턴스 {instance_id} 포트 매핑 업데이트 시작")

        # 개선된 포트 매핑 업데이트 실행
        success = service.update_instance_port_mappings(instance_id)

        if success:
            # 업데이트된 정보 조회
            db_instance = service.get_instance_from_db(instance_id)
            port_mappings = db_instance.get_port_mappings_dict() if db_instance else {}
            public_ip = db_instance.public_ip if db_instance else None

            logger.info(f"✅ 포트 매핑 업데이트 성공")
            logger.info(f"  - 공인 IP: {public_ip}")
            logger.info(f"  - 포트 매핑 개수: {len(port_mappings)}")

            return {
                "success": True,
                "instance_id": instance_id,
                "message": "포트 매핑 정보가 성공적으로 업데이트되었습니다",
                "port_mappings": port_mappings,
                "public_ip": public_ip,
                "update_method": "improved_ip_extraction",
                "port_count": len(port_mappings)
            }
        else:
            return {
                "success": False,
                "instance_id": instance_id,
                "message": "포트 매핑 정보 업데이트 실패 - VastAI에서 데이터를 가져올 수 없거나 유효한 IP를 찾을 수 없습니다",
                "troubleshooting": [
                    "1. 인스턴스가 running 상태인지 확인하세요",
                    "2. VastAI API 키가 올바른지 확인하세요",
                    "3. 인스턴스 ID가 정확한지 확인하세요",
                    "4. 잠시 후 다시 시도해보세요"
                ]
            }

    except Exception as e:
        logger.error(f"포트 매핑 업데이트 실패: {e}")
        raise HTTPException(status_code=500, detail="포트 매핑 업데이트 실패")

@router.post("/instances/{instance_id}/vllm-serve",
    summary="VLLM 서비스 시작",
    description="인스턴스의 VLLM 서비스를 시작합니다. VLLM 설정을 기반으로 VLLM 서버를 실행합니다.",
    response_model=Dict[str, Any])
async def vllm_serve(request: Request, instance_id: str, vllm_config: VLLMServeConfigRequest):
    try:
        service = get_vast_service(request)
        db_instance = service.get_instance_from_db(instance_id)
        port_mappings = db_instance.get_port_mappings_dict()
        controller_url = f"http://{port_mappings.get('12435', {}).get('external_ip')}:{port_mappings.get('12435', {}).get('external_port')}/api/vllm/serve"

        # VLLM 서비스 시작 요청
        data = json.dumps(vllm_config.dict()).encode('utf-8')
        req = urllib_request.Request(
            controller_url,
            data=data,
            headers={
                'Content-Type': 'application/json',
                'Content-Length': str(len(data))
            },
            method='POST'
        )
        response = urllib_request.urlopen(req).read().decode('utf-8')
        response_data = json.loads(response)
        if response_data.get("status") == "success":
            logger.info(f"VLLM 서비스 시작 성공: {response_data.get('message', 'No message provided')}")

            # 상태 업데이트 및 SSE 브로드캐스트
            updates = {
                "status": "running_vllm",
                "model_name": vllm_config.model_id,
                "max_model_length": vllm_config.max_model_len,
            }
            service._update_instance(instance_id, updates=updates)
            await _broadcast_status_change(instance_id, "running_vllm")

            return {
                "success": True,
                "message": response_data.get("message", "VLLM 서비스가 성공적으로 시작되었습니다"),
                "instance_id": instance_id
            }

    except Exception as e:
        logger.error(f"VLLM 서비스 시작 실패: {e}")
        raise HTTPException(status_code=500, detail="VLLM 서비스 시작 실패")

@router.post("/instances/{instance_id}/vllm-down",
    summary="VLLM 다운",
    description="인스턴스의 VLLM을 다운시킵니다.",
    response_model=Dict[str, Any])
async def vllm_down(request: Request, instance_id: str):
    try:
        service = get_vast_service(request)
        db_instance = service.get_instance_from_db(instance_id)
        port_mappings = db_instance.get_port_mappings_dict()
        controller_url = f"http://{port_mappings.get('12435', {}).get('external_ip')}:{port_mappings.get('12435', {}).get('external_port')}/api/vllm/down"

        # VLLM 다운 요청
        req = urllib_request.Request(
            controller_url,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        response = urllib_request.urlopen(req).read().decode('utf-8')
        response_data = json.loads(response)
        if response_data.get("status") == "success":
            logger.info(f"VLLM 다운 성공: {response_data.get('message', 'No message provided')}")

            service._update_instance(instance_id, updates={
                "status": "running",
                "model_name": "None",
                "max_model_length": 0,
            })

            return {
                "success": True,
                "message": response_data.get("message", "VLLM이 성공적으로 다운되었습니다"),
                "instance_id": instance_id
            }

    except Exception as e:
        logger.error(f"VLLM 다운 실패: {e}")
        raise HTTPException(status_code=500, detail="VLLM 다운 실패")

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
    try:
        # appController의 update_persistent_config 함수를 import
        from controller.appController import update_persistent_config, ConfigUpdateRequest

        try:
            config_composer = get_config_composer(request)
            old_api_base_url = config_composer.get_config_by_name("VLLM_API_BASE_URL").value

            api_base_url_request = ConfigUpdateRequest(value=vllm_config.api_base_url)
            api_base_url_result = await update_persistent_config("VLLM_API_BASE_URL", api_base_url_request, request)

            logger.info(f"VLLM_API_BASE_URL 업데이트: {old_api_base_url} -> {vllm_config.api_base_url}")

        except KeyError:
            logger.warning("VLLM_API_BASE_URL 설정을 찾을 수 없습니다")
            raise HTTPException(status_code=404, detail="VLLM_API_BASE_URL 설정을 찾을 수 없습니다")

        try:
            old_model_name = config_composer.get_config_by_name("VLLM_MODEL_NAME").value

            model_name_request = ConfigUpdateRequest(value=vllm_config.model_name)
            model_name_result = await update_persistent_config("VLLM_MODEL_NAME", model_name_request, request)

            logger.info(f"VLLM_MODEL_NAME 업데이트: {old_model_name} -> {vllm_config.model_name}")

        except KeyError:
            logger.warning("VLLM_MODEL_NAME 설정을 찾을 수 없습니다")
            raise HTTPException(status_code=404, detail="VLLM_MODEL_NAME 설정을 찾을 수 없습니다")
        try:
            provider_request = ConfigUpdateRequest(value='vllm')
            provider_response = await update_persistent_config("DEFAULT_LLM_PROVIDER", provider_request, request)

            logger.info(f"DEFAULT_LLM_PROVIDER 업데이트: {provider_request.value}")

        except KeyError:
            logger.warning("DEFAULT_LLM_PROVIDER 설정을 찾을 수 없습니다")
            raise HTTPException(status_code=404, detail="DEFAULT_LLM_PROVIDER 설정을 찾을 수 없습니다")


        logger.info("Successfully updated VLLM configuration using update_persistent_config")
        logger.info(f"VLLM 설정 업데이트: API Base URL = {api_base_url_result}, 모델명 = {model_name_result}")

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
        logger.error(f"VLLM 설정 업데이트 실패: {e}")
        raise HTTPException(status_code=500, detail=f"VLLM 설정 업데이트 실패: {str(e)}")

@router.post("/instances/vllm-health",
    summary="VLLM 헬스 체크",
    description="지정된 IP와 포트로 VLLM 서비스 헬스 상태를 확인합니다. 프론트엔드에서 안전하게 VLLM 상태를 확인할 수 있습니다.",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "헬스 체크 성공"},
        400: {"description": "잘못된 요청"},
        408: {"description": "타임아웃"},
        500: {"description": "서버 오류"}
    })
async def vllm_health_check(request: Request, health_request: VLLMHealthCheckRequest):
    """VLLM 서비스 헬스 체크"""
    try:
        vllm_ip = health_request.ip
        vllm_port = health_request.port

        if not type(vllm_port) is str:
            vllm_port = str(vllm_port)

        # VLLM 헬스 체크 실행
        health_url = f"http://{vllm_ip}:{vllm_port}/health"

        try:
            import urllib.request
            import urllib.error
            from urllib.request import Request as UrlRequest

            # 10초 타임아웃으로 요청
            req = UrlRequest(health_url, headers={'Content-Type': 'application/json'})

            try:
                response = urllib.request.urlopen(req, timeout=10)
                response_data = response.read().decode('utf-8')

                # 응답 상태 확인
                if response.getcode() == 200:
                    # 응답 내용 파싱 시도
                    health_data = None
                    if response_data and response_data.strip():
                        try:
                            health_data = json.loads(response_data)
                        except json.JSONDecodeError:
                            # JSON이 아닌 경우에도 성공으로 간주
                            health_data = {"status": "ok", "response": response_data}
                    else:
                        # 빈 응답도 성공으로 간주
                        health_data = {"status": "ok", "response": "empty"}

                    logger.info(f"VLLM 헬스 체크 성공: {health_url}")

                    return {
                        "success": True,
                        "vllm_endpoint": {
                            "ip": vllm_ip,
                            "port": vllm_port
                        },
                        "health_url": health_url,
                        "status": "healthy",
                        "message": "VLLM 서비스가 정상 작동 중입니다",
                        "health_data": health_data,
                        "response_code": response.getcode()
                    }
                else:
                    logger.warning(f"VLLM 헬스 체크 실패: HTTP {response.getcode()}")
                    return {
                        "success": False,
                        "vllm_endpoint": {
                            "ip": vllm_ip,
                            "port": vllm_port
                        },
                        "health_url": health_url,
                        "status": "unhealthy",
                        "message": f"VLLM 서비스가 응답하지 않습니다. (상태: {response.getcode()})",
                        "response_code": response.getcode()
                    }

            except urllib.error.HTTPError as http_error:
                logger.error(f"VLLM 헬스 체크 HTTP 오류: {http_error.code}")
                return {
                    "success": False,
                    "vllm_endpoint": {
                        "ip": vllm_ip,
                        "port": vllm_port
                    },
                    "health_url": health_url,
                    "status": "unhealthy",
                    "message": f"VLLM 서비스가 응답하지 않습니다. (상태: {http_error.code})",
                    "error": str(http_error),
                    "response_code": http_error.code
                }

        except Exception as request_error:
            error_message = str(request_error)

            # 타임아웃 오류 처리
            if "timeout" in error_message.lower() or "timed out" in error_message.lower():
                logger.warning(f"VLLM 헬스 체크 타임아웃: {health_url}")
                raise HTTPException(status_code=408, detail="VLLM 헬스 체크 타임아웃 (10초)")

            logger.error(f"VLLM 헬스 체크 요청 실패: {request_error}")
            return {
                "success": False,
                "vllm_endpoint": {
                    "ip": vllm_ip,
                    "port": vllm_port
                },
                "health_url": health_url,
                "status": "error",
                "message": f"VLLM 헬스 체크 실패: {error_message}",
                "error": error_message
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"VLLM 헬스 체크 처리 실패: {e}")
        raise HTTPException(status_code=500, detail=f"VLLM 헬스 체크 처리 실패: {str(e)}")

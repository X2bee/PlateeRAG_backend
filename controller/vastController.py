"""
VastAI 인스턴스 관리 컨트롤러

VastAI 클라우드 GPU 인스턴스의 검색, 생성, 관리 및 모니터링을 위한 RESTful API
"""

from fastapi import APIRouter, HTTPException, Request, BackgroundTasks, Query
from pydantic import BaseModel, Field
import logging
from typing import Dict, Any, Optional, List, Literal
from enum import Enum
import json

from service.vast.vast_service import VastService, auto_run_vllm

router = APIRouter(prefix="/api/vast", tags=["vastAI"])
logger = logging.getLogger("vast-controller")

# ========== Enums ==========
class SortBy(str, Enum):
    price = "price"
    gpu_ram = "gpu_ram"
    num_gpus = "num_gpus"

class VLLMDtype(str, Enum):
    auto = "auto"
    half = "half"
    float16 = "float16"
    bfloat16 = "bfloat16"
    float = "float"
    float32 = "float32"

class ToolCallParser(str, Enum):
    hermes = "hermes"
    mistral = "mistral"
    none = "none"

# ========== Request Models ==========
class OfferSearchRequest(BaseModel):
    """GPU 오퍼 검색 요청"""
    gpu_name: Optional[str] = Field(None, description="GPU 모델명", example="RTX4090")
    max_price: Optional[float] = Field(None, description="최대 시간당 가격 ($)", example=2.0, ge=0)
    min_gpu_ram: Optional[int] = Field(None, description="최소 GPU RAM (GB)", example=16, ge=1)
    num_gpus: Optional[int] = Field(None, description="GPU 개수", example=1, ge=1)
    rentable: Optional[bool] = Field(None, description="렌트 가능 여부", example=True)
    sort_by: SortBy = Field(SortBy.price, description="정렬 기준")
    limit: Optional[int] = Field(20, description="결과 제한 개수", example=10, ge=1, le=100)

class VLLMConfigRequest(BaseModel):
    """VLLM 설정 요청"""
    # 모델 설정
    vllm_model_name: str = Field("Qwen/Qwen3-1.7B", description="사용할 모델명", example="Qwen/Qwen3-1.7B")
    vllm_max_model_len: int = Field(4096, description="최대 모델 길이", example=2048, ge=512, le=32768)

    # 네트워크 설정
    vllm_host_ip: str = Field("0.0.0.0", description="호스트 IP", example="0.0.0.0")
    vllm_port: int = Field(11479, description="VLLM 서비스 포트", example=12434, ge=1024, le=65535)
    vllm_controller_port: int = Field(11480, description="VLLM 컨트롤러 포트", example=12435, ge=1024, le=65535)

    # 성능 설정
    vllm_gpu_memory_utilization: float = Field(0.9, description="GPU 메모리 사용률", example=0.5, ge=0.1, le=1.0)
    vllm_pipeline_parallel_size: int = Field(1, description="파이프라인 병렬 크기", example=1, ge=1)
    vllm_tensor_parallel_size: int = Field(1, description="텐서 병렬 크기", example=1, ge=1)

    # 데이터 타입 및 고급 설정
    vllm_dtype: VLLMDtype = Field(VLLMDtype.auto, description="데이터 타입")
    vllm_tool_call_parser: Optional[ToolCallParser] = Field(None, description="도구 호출 파서")
    vllm_trust_remote_code: bool = Field(True, description="원격 코드 신뢰 여부")
    vllm_enforce_eager: bool = Field(False, description="즉시 실행 강제 여부")
    vllm_max_num_seqs: Optional[int] = Field(None, description="최대 시퀀스 수", ge=1)
    vllm_block_size: int = Field(16, description="블록 크기", example=16, ge=1)
    vllm_swap_space: int = Field(4, description="스왑 공간 (GiB)", example=4, ge=0)
    vllm_disable_log_stats: bool = Field(False, description="로그 통계 비활성화")

class CreateInstanceRequest(BaseModel):
    """인스턴스 생성 요청"""
    offer_id: Optional[str] = Field(None, description="특정 오퍼 ID (없으면 자동 선택)", example="12345")
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

# ========== Response Models ==========
class OfferInfo(BaseModel):
    """GPU 오퍼 정보"""
    id: str = Field(..., description="오퍼 ID")
    gpu_name: str = Field(..., description="GPU 모델명")
    num_gpus: int = Field(..., description="GPU 개수")
    gpu_ram: float = Field(..., description="GPU RAM (GB)")
    dph_total: float = Field(..., description="시간당 총 가격 ($)")
    rentable: bool = Field(..., description="렌트 가능 여부")
    public_ipaddr: Optional[str] = Field(None, description="공개 IP 주소")

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
        config_composer = request.app.state.config_composer
        vast_config = config_composer.get_config_by_category_name("vast")
        db_manager = request.app.state.app_db
        return VastService(vast_config, db_manager)
    except Exception as e:
        logger.error(f"VastService 초기화 실패: {e}")
        raise HTTPException(status_code=500, detail="VastService 초기화 실패")

def generate_vllm_env_vars(config: VLLMConfigRequest, hf_token: Optional[str] = None) -> Dict[str, str]:
    """VLLM 환경변수 생성"""
    env_vars = {
        "VLLM_MODEL_NAME": config.vllm_model_name,
        "VLLM_HOST_IP": config.vllm_host_ip,
        "VLLM_PORT": str(config.vllm_port),
        "VLLM_CONTROLLER_PORT": str(config.vllm_controller_port),
        "VLLM_MAX_MODEL_LEN": str(config.vllm_max_model_len),
        "VLLM_GPU_MEMORY_UTILIZATION": str(config.vllm_gpu_memory_utilization),
        "VLLM_PIPELINE_PARALLEL_SIZE": str(config.vllm_pipeline_parallel_size),
        "VLLM_TENSOR_PARALLEL_SIZE": str(config.vllm_tensor_parallel_size),
        "VLLM_DTYPE": config.vllm_dtype,
        "VLLM_TRUST_REMOTE_CODE": str(config.vllm_trust_remote_code).lower(),
        "VLLM_ENFORCE_EAGER": str(config.vllm_enforce_eager).lower(),
        "VLLM_BLOCK_SIZE": str(config.vllm_block_size),
        "VLLM_SWAP_SPACE": str(config.vllm_swap_space),
        "VLLM_DISABLE_LOG_STATS": str(config.vllm_disable_log_stats).lower()
    }

    if config.vllm_tool_call_parser:
        env_vars["VLLM_TOOL_CALL_PARSER"] = config.vllm_tool_call_parser
    if config.vllm_max_num_seqs:
        env_vars["VLLM_MAX_NUM_SEQS"] = str(config.vllm_max_num_seqs)
    if hf_token:
        env_vars.update({
            "HUGGING_FACE_HUB_TOKEN": hf_token,
            "HF_HUB_TOKEN": hf_token
        })

    return env_vars

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
                if hasattr(service.config, 'configs'):
                    for config_obj in service.config.configs.values():
                        if hasattr(config_obj, 'env_name') and config_obj.env_name == env_name:
                            config_obj.value = value
                            logger.info("VLLM 설정 적용: %s = %s", key, value)

        # 인스턴스 생성
        instance_id = service.create_vllm_instance(
            offer_id=create_request.offer_id,
            template_name=create_request.template_name
        )

        if not instance_id:
            raise HTTPException(status_code=400, detail="인스턴스 생성 실패")

        # 백그라운드 설정
        background_tasks.add_task(service.wait_and_setup_instance, instance_id)

        return {
            "success": True,
            "instance_id": instance_id,
            "template_name": create_request.template_name,
            "message": "인스턴스가 생성되었습니다. 설정이 백그라운드에서 진행됩니다.",
            "status": "creating"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"인스턴스 생성 실패: {e}")
        raise HTTPException(status_code=500, detail="인스턴스 생성 실패")

@router.post("/instances/{instance_id}/setup-vllm",
    summary="VLLM 설정 및 실행",
    description="인스턴스에 VLLM을 설정하고 실행합니다. requirements.txt 설치부터 환경변수 설정, 스크립트 실행까지 자동화됩니다.",
    response_model=Dict[str, Any])
async def setup_vllm(request: Request, instance_id: str, setup_request: SetupVLLMRequest):
    try:
        service = get_vast_service(request)
        results = []

        # 1. 디렉토리 확인
        logger.info(f"📁 디렉토리 확인: {setup_request.script_directory}")
        check_result = service.vast_manager.execute_ssh_command(
            instance_id, f"ls -la {setup_request.script_directory}"
        )
        results.append({"step": "directory_check", "result": check_result})

        if not check_result.get("success"):
            return {
                "success": False,
                "error": f"디렉토리 {setup_request.script_directory}를 찾을 수 없습니다",
                "results": results
            }

        # 2. requirements.txt 설치
        if setup_request.install_requirements:
            logger.info("📦 requirements.txt 확인 및 설치")
            req_check = service.vast_manager.execute_ssh_command(
                instance_id, f"ls {setup_request.script_directory}/requirements.txt 2>/dev/null || echo 'no requirements.txt'"
            )

            if "requirements.txt" in req_check.get("stdout", "") and "no requirements.txt" not in req_check.get("stdout", ""):
                install_result = service.vast_manager.execute_ssh_command(
                    instance_id, f"cd {setup_request.script_directory} && pip3 install -r requirements.txt"
                )
                results.append({"step": "install_requirements", "result": install_result})

        # 3. 환경변수 설정 및 실행
        env_vars = generate_vllm_env_vars(setup_request.vllm_config, setup_request.hf_token)
        if setup_request.additional_env_vars:
            env_vars.update(setup_request.additional_env_vars)

        env_exports = [f"export {key}={value}" for key, value in env_vars.items()]
        env_cmd = " && ".join(env_exports)

        main_py_cmd = f"""cd {setup_request.script_directory} && \\
{env_cmd} && \\
nohup python3 {setup_request.main_script} > {setup_request.log_file} 2>&1 &"""

        main_result = service.vast_manager.execute_ssh_command(instance_id, main_py_cmd)
        results.append({"step": "execute_main", "command": main_py_cmd, "result": main_result})

        # 4. 프로세스 확인
        import time
        time.sleep(2)
        process_check = service.vast_manager.execute_ssh_command(
            instance_id, f"ps aux | grep {setup_request.main_script} | grep -v grep"
        )
        results.append({"step": "process_check", "result": process_check})

        return {
            "success": True,
            "instance_id": instance_id,
            "message": "VLLM 설정 및 실행 완료",
            "config": {
                "script_directory": setup_request.script_directory,
                "main_script": setup_request.main_script,
                "log_file": setup_request.log_file,
                "environment_vars": env_vars
            },
            "results": results
        }

    except Exception as e:
        logger.error(f"VLLM 설정 실패: {e}")
        raise HTTPException(status_code=500, detail="VLLM 설정 실패")

@router.post("/instances/{instance_id}/execute",
    summary="명령어 실행",
    description="인스턴스에서 SSH를 통해 명령어를 실행합니다. 백그라운드 실행과 환경변수 설정을 지원합니다.",
    response_model=CommandExecutionResponse)
async def execute_command(request: Request, instance_id: str, command_request: ExecuteCommandRequest) -> CommandExecutionResponse:
    try:
        service = get_vast_service(request)

        # 명령어 구성
        commands = []
        if command_request.working_directory:
            commands.append(f"cd {command_request.working_directory}")

        if command_request.environment_vars:
            env_exports = [f"export {key}={value}" for key, value in command_request.environment_vars.items()]
            commands.extend(env_exports)

        commands.append(command_request.command)

        final_command = " && ".join(commands)
        if command_request.background:
            final_command = f"nohup bash -c '{final_command}' > /tmp/command_output.log 2>&1 &"

        result = service.vast_manager.execute_ssh_command(instance_id, final_command)

        return CommandExecutionResponse(
            success=result.get("success", False),
            instance_id=instance_id,
            command=command_request.command,
            stdout=result.get("stdout", ""),
            stderr=result.get("stderr", ""),
            background=command_request.background,
            error=result.get("error")
        )
    except Exception as e:
        logger.error(f"명령어 실행 실패: {e}")
        raise HTTPException(status_code=500, detail="명령어 실행 실패")

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

@router.get("/instances/{instance_id}",
    summary="인스턴스 상태 조회",
    description="특정 인스턴스의 상세 상태 정보를 조회합니다. DB 정보, vLLM 상태, 비용 정보 등을 포함합니다.",
    response_model=InstanceStatusResponse,
    responses={500: {"description": "상태 조회 실패"}})
async def get_instance_status(request: Request, instance_id: str) -> InstanceStatusResponse:
    try:
        service = get_vast_service(request)
        enhanced_status = service.get_enhanced_instance_status(instance_id)

        return InstanceStatusResponse(
            instance_id=instance_id,
            status=enhanced_status.get("basic_status", {}).get("status", "unknown"),
            public_ip=enhanced_status.get("basic_status", {}).get("public_ip"),
            urls=enhanced_status.get("basic_status", {}).get("urls", {}),
            port_mappings=enhanced_status.get("basic_status", {}).get("port_mappings", {}),
            gpu_info=enhanced_status.get("db_info", {}).get("gpu_info"),
            cost_per_hour=enhanced_status.get("db_info", {}).get("cost_per_hour"),
            uptime=enhanced_status.get("db_info", {}).get("uptime"),
            vllm_status=enhanced_status.get("vllm_status")
        )
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

@router.get("/instances/{instance_id}/logs",
    summary="로그 조회",
    description="인스턴스의 로그 파일을 조회합니다.",
    response_model=Dict[str, Any])
async def get_logs(request: Request, instance_id: str, log_file: str = "/tmp/vllm.log", lines: int = Query(50, ge=1, le=1000)):
    try:
        service = get_vast_service(request)

        # 보안을 위한 경로 제한
        allowed_paths = ["/tmp/", "/var/log/", "/vllm/vllm-script/"]
        if not any(log_file.startswith(path) for path in allowed_paths):
            log_file = f"/tmp/{log_file}"

        cmd = f"tail -{lines} {log_file} 2>/dev/null || echo 'Log file not found'"
        result = service.vast_manager.execute_ssh_command(instance_id, cmd)

        return {
            "instance_id": instance_id,
            "log_file": log_file,
            "lines_requested": lines,
            "content": result.get("stdout", ""),
            "error": result.get("stderr", ""),
            "success": result.get("success", False)
        }
    except Exception as e:
        logger.error(f"로그 조회 실패: {e}")
        raise HTTPException(status_code=500, detail="로그 조회 실패")

@router.get("/instances/{instance_id}/processes",
    summary="프로세스 상태 조회",
    description="인스턴스에서 실행 중인 프로세스를 조회합니다.",
    response_model=Dict[str, Any])
async def get_processes(request: Request, instance_id: str, process_name: Optional[str] = Query(None, description="특정 프로세스 이름")):
    try:
        service = get_vast_service(request)

        if process_name:
            cmd = f"ps aux | grep {process_name} | grep -v grep"
        else:
            cmd = "ps aux | grep python | grep -v grep"

        result = service.vast_manager.execute_ssh_command(instance_id, cmd)

        processes = []
        if result.get("success") and result.get("stdout"):
            lines = result["stdout"].strip().split('\n')
            for line in lines:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 11:
                        processes.append({
                            "user": parts[0],
                            "pid": parts[1],
                            "cpu": parts[2],
                            "mem": parts[3],
                            "command": " ".join(parts[10:])
                        })

        return {
            "instance_id": instance_id,
            "process_name": process_name,
            "processes": processes,
            "total_processes": len(processes)
        }
    except Exception as e:
        logger.error(f"프로세스 상태 조회 실패: {e}")
        raise HTTPException(status_code=500, detail="프로세스 상태 조회 실패")

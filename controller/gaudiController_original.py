"""
Gaudi VLLM 관리 컨트롤러 (고도화 버전)

로컬 Habana Gaudi HPU 환경에서 VLLM 서비스의 지능형 리소스 관리를 위한 RESTful API
"""

from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
import logging
import subprocess
import psutil
import signal
import os
import json
import asyncio
import time
from typing import Dict, Any, Optional, List, Set, Union
import habana_frameworks.torch.hpu as hpu

router = APIRouter(prefix="/api/gaudi", tags=["gaudi-vllm"])
logger = logging.getLogger("gaudi-controller")

# VLLM 프로세스 및 리소스 관리를 위한 전역 변수
vllm_processes: Dict[str, Dict[str, Any]] = {}  # instance_id -> process_info
hpu_allocation: Dict[int, str] = {}  # hpu_id -> instance_id
hpu_memory_usage: Dict[int, Dict[str, Any]] = {}  # hpu_id -> memory_info

# ========== Request Models ==========
class GaudiVLLMConfigRequest(BaseModel):
    """Gaudi VLLM 설정 요청"""
    model_name: str = Field("x2bee/Polar-14B", description="사용할 모델명")
    max_model_len: int = Field(2048, description="최대 모델 길이", ge=512, le=32768)
    host: str = Field("0.0.0.0", description="호스트 IP")
    port: int = Field(12434, description="VLLM 서비스 포트", ge=1024, le=65535)
    dtype: str = Field("bfloat16", description="데이터 타입")
    tensor_parallel_size: int = Field(1, description="텐서 병렬 크기", ge=1, le=8)
    tool_call_parser: Optional[str] = Field("hermes", description="도구 호출 파서")
    trust_remote_code: bool = Field(True, description="원격 코드 신뢰 여부")
    enable_lora: bool = Field(False, description="LoRA 어댑터 지원")
    gpu_memory_utilization: float = Field(0.9, description="HPU 메모리 사용률", ge=0.1, le=1.0)

class AutoAllocationRequest(BaseModel):
    """자동 HPU 할당 요청"""
    required_hpus: int = Field(1, description="필요한 HPU 개수", ge=1, le=8)
    prefer_consecutive: bool = Field(True, description="연속된 HPU 선호 여부")
    
class ManualAllocationRequest(BaseModel):
    """수동 HPU 할당 요청"""
    device_ids: List[int] = Field(..., description="사용할 HPU 장치 ID 목록", min_items=1, max_items=8)

# ========== 통합 Request Models (422 에러 해결용) ==========
class VLLMAutoStartRequest(BaseModel):
    """VLLM 자동 시작 통합 요청"""
    # VLLM 설정
    model_name: str = Field("x2bee/Polar-14B", description="사용할 모델명")
    max_model_len: int = Field(2048, description="최대 모델 길이", ge=512, le=32768)
    host: str = Field("0.0.0.0", description="호스트 IP")
    port: int = Field(12434, description="VLLM 서비스 포트", ge=1024, le=65535)
    dtype: str = Field("bfloat16", description="데이터 타입")
    tensor_parallel_size: int = Field(1, description="텐서 병렬 크기", ge=1, le=8)
    tool_call_parser: Optional[str] = Field("hermes", description="도구 호출 파서")
    trust_remote_code: bool = Field(True, description="원격 코드 신뢰 여부")
    enable_lora: bool = Field(False, description="LoRA 어댑터 지원")
    gpu_memory_utilization: float = Field(0.9, description="HPU 메모리 사용률", ge=0.1, le=1.0)
    
    # 할당 설정
    required_hpus: int = Field(1, description="필요한 HPU 개수", ge=1, le=8)
    prefer_consecutive: bool = Field(True, description="연속된 HPU 선호 여부")

class VLLMManualStartRequest(BaseModel):
    """VLLM 수동 시작 통합 요청"""
    # VLLM 설정
    model_name: str = Field("x2bee/Polar-14B", description="사용할 모델명")
    max_model_len: int = Field(2048, description="최대 모델 길이", ge=512, le=32768)
    host: str = Field("0.0.0.0", description="호스트 IP")
    port: int = Field(12434, description="VLLM 서비스 포트", ge=1024, le=65535)
    dtype: str = Field("bfloat16", description="데이터 타입")
    tensor_parallel_size: int = Field(1, description="텐서 병렬 크기", ge=1, le=8)
    tool_call_parser: Optional[str] = Field("hermes", description="도구 호출 파서")
    trust_remote_code: bool = Field(True, description="원격 코드 신뢰 여부")
    enable_lora: bool = Field(False, description="LoRA 어댑터 지원")
    gpu_memory_utilization: float = Field(0.9, description="HPU 메모리 사용률", ge=0.1, le=1.0)
    
    # 할당 설정
    device_ids: List[int] = Field(..., description="사용할 HPU 장치 ID 목록", min_items=1, max_items=8)

# ========== Response Models ==========
class HPUResourceStatus(BaseModel):
    """HPU 리소스 상태"""
    total_hpus: int = Field(..., description="전체 HPU 개수")
    available_hpus: int = Field(..., description="사용 가능한 HPU 개수")
    allocated_hpus: Dict[int, str] = Field(..., description="할당된 HPU (hpu_id -> instance_id)")
    hpu_utilization: Dict[int, Dict[str, Any]] = Field(..., description="HPU 사용률 정보")

class VLLMInstanceInfo(BaseModel):
    """VLLM 인스턴스 정보"""
    instance_id: str = Field(..., description="인스턴스 ID")
    status: str = Field(..., description="인스턴스 상태")
    model_name: str = Field(..., description="모델명")
    port: int = Field(..., description="서비스 포트")
    allocated_hpus: List[int] = Field(..., description="할당된 HPU 목록")
    tensor_parallel_size: int = Field(..., description="텐서 병렬 크기")
    pid: Optional[int] = Field(None, description="프로세스 ID")
    memory_usage: Optional[Dict[str, Any]] = Field(None, description="메모리 사용량")
    uptime: Optional[float] = Field(None, description="가동 시간(초)")
    api_url: str = Field(..., description="API 접근 URL")

class ResourceAllocationResponse(BaseModel):
    """리소스 할당 응답"""
    success: bool = Field(..., description="할당 성공 여부")
    allocated_hpus: List[int] = Field(..., description="할당된 HPU 목록")
    message: str = Field(..., description="할당 결과 메시지")
    alternative_suggestion: Optional[List[int]] = Field(None, description="대안 제안")

# ========== Helper Functions ==========
def check_hpu_availability():
    """HPU 가용성 및 상세 정보 확인"""
    try:
        available = hpu.is_available()
        count = hpu.device_count() if available else 0
        device_info = {}
        
        if available and count > 0:
            for i in range(count):
                try:
                    name = hpu.get_device_name(i)
                    
                    # HPU 메모리 정보 가져오기 (타입 변환 필요)
                    try:
                        # 메모리 정보 수집
                        allocated = hpu.memory_allocated(i) if hasattr(hpu, 'memory_allocated') else 0
                        reserved = hpu.memory_reserved(i) if hasattr(hpu, 'memory_reserved') else 0
                        
                        # Utilization 정보 수집
                        utilization_obj = hpu.utilization(i) if hasattr(hpu, 'utilization') else None
                        
                        # C 구조체 객체에서 실제 값을 추출
                        if utilization_obj is not None:
                            if hasattr(utilization_obj, 'gpu'):
                                utilization = float(utilization_obj.gpu)
                            elif hasattr(utilization_obj, 'value'):
                                utilization = float(utilization_obj.value)
                            elif hasattr(utilization_obj, 'memory'):
                                utilization = float(utilization_obj.memory)
                            else:
                                # 구조체의 속성들을 확인
                                logger.debug(f"Utilization object attributes: {dir(utilization_obj)}")
                                utilization = 0.0
                        else:
                            utilization = 0.0
                            
                        memory_info = {
                            "allocated": int(allocated) if allocated is not None else 0,
                            "reserved": int(reserved) if reserved is not None else 0,
                            "utilization": utilization
                        }
                    except Exception as mem_error:
                        logger.warning(f"HPU {i} 메모리 정보 수집 실패: {mem_error}")
                        memory_info = {"allocated": 0, "reserved": 0, "utilization": 0.0}
                    
                    device_info[i] = {
                        "name": str(name),  # 문자열로 명시적 변환
                        "memory": memory_info,
                        "allocated": bool(i in hpu_allocation),  # bool로 명시적 변환
                        "instance_id": hpu_allocation.get(i, None)
                    }
                except Exception as e:
                    logger.warning(f"HPU {i} 정보 수집 실패: {e}")
                    device_info[i] = {
                        "name": f"HPU-{i}: Error ({str(e)})",
                        "memory": {"allocated": 0, "reserved": 0, "utilization": 0.0},
                        "allocated": bool(i in hpu_allocation),
                        "instance_id": hpu_allocation.get(i, None)
                    }
        
        return available, count, device_info
    except Exception as e:
        logger.error(f"HPU 상태 확인 실패: {e}")
        return False, 0, {}

def find_available_hpus(required_count: int, prefer_consecutive: bool = True) -> Optional[List[int]]:
    """사용 가능한 HPU 찾기"""
    _, total_count, device_info = check_hpu_availability()
    
    if total_count < required_count:
        return None
    
    available_hpus = [hpu_id for hpu_id in range(total_count) if hpu_id not in hpu_allocation]
    
    if len(available_hpus) < required_count:
        return None
    
    if prefer_consecutive and required_count > 1:
        # 연속된 HPU 찾기
        for start in range(len(available_hpus) - required_count + 1):
            consecutive = available_hpus[start:start + required_count]
            if all(consecutive[i] + 1 == consecutive[i + 1] for i in range(len(consecutive) - 1)):
                return consecutive
        
        # 연속된 HPU가 없으면 임의로 선택
        logger.info(f"연속된 {required_count}개 HPU를 찾을 수 없어 임의로 선택합니다")
    
    return available_hpus[:required_count]

def allocate_hpus(instance_id: str, hpu_ids: List[int]) -> bool:
    """HPU 할당"""
    # 이미 할당된 HPU 확인
    for hpu_id in hpu_ids:
        if hpu_id in hpu_allocation:
            logger.error(f"HPU {hpu_id}는 이미 인스턴스 {hpu_allocation[hpu_id]}에 할당되었습니다")
            return False
    
    # 할당 수행
    for hpu_id in hpu_ids:
        hpu_allocation[hpu_id] = instance_id
    
    logger.info(f"HPU {hpu_ids} -> 인스턴스 {instance_id} 할당 완료")
    return True

def deallocate_hpus(instance_id: str):
    """HPU 할당 해제"""
    deallocated = []
    for hpu_id, allocated_instance in list(hpu_allocation.items()):
        if allocated_instance == instance_id:
            del hpu_allocation[hpu_id]
            deallocated.append(hpu_id)
    
    if deallocated:
        logger.info(f"HPU {deallocated} 할당 해제 완료 (인스턴스: {instance_id})")
    
    return deallocated

def generate_instance_id(hpu_ids: List[int], model_name: str = "") -> str:
    """인스턴스 ID 생성"""
    hpu_str = "_".join(map(str, sorted(hpu_ids)))
    timestamp = int(time.time() * 1000) % 100000  # 마지막 5자리
    model_short = model_name.split("/")[-1][:8] if model_name else "model"
    return f"vllm_{model_short}_{hpu_str}_{timestamp}"

def build_vllm_command(config: GaudiVLLMConfigRequest, hpu_ids: List[int], instance_id: str) -> List[str]:
    """VLLM 실행 명령어 구성"""
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", config.model_name,
        "--host", config.host,
        "--port", str(config.port),
        "--max-model-len", str(config.max_model_len),
        "--dtype", config.dtype,
        "--device", "hpu",
        "--tensor-parallel-size", str(config.tensor_parallel_size),
        "--gpu-memory-utilization", str(config.gpu_memory_utilization),
    ]
    
    if config.trust_remote_code:
        cmd.append("--trust-remote-code")
    
    if config.tool_call_parser:
        cmd.extend(["--enable-auto-tool-choice", "--tool-call-parser", config.tool_call_parser])
    
    if config.enable_lora:
        cmd.append("--enable-lora")
    
    # 로그 파일 설정
    cmd.extend(["--disable-log-requests"])
    
    return cmd

def get_process_detailed_info(pid: int) -> Dict[str, Any]:
    """상세한 프로세스 정보 조회"""
    try:
        process = psutil.Process(pid)
        return {
            "pid": pid,
            "status": process.status(),
            "cpu_percent": process.cpu_percent(interval=1),
            "memory_info": process.memory_info()._asdict(),
            "memory_percent": process.memory_percent(),
            "create_time": process.create_time(),
            "num_threads": process.num_threads(),
            "cmdline": " ".join(process.cmdline()[:5]) + "..." if len(process.cmdline()) > 5 else " ".join(process.cmdline())
        }
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None

def check_port_available(port: int) -> bool:
    """포트 사용 가능 여부 확인"""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex(('localhost', port))
            return result != 0
    except:
        return False

def get_next_available_port(start_port: int = 12434) -> int:
    """사용 가능한 다음 포트 찾기"""
    port = start_port
    while not check_port_available(port) and port < start_port + 100:
        port += 1
    return port if port < start_port + 100 else None

# ========== Exception Handlers ==========
async def log_request_details(request: Request, extra_info: str = ""):
    """요청 상세 정보 로깅"""
    try:
        body = await request.body()
        logger.info(f"=== 요청 상세 정보 {extra_info} ===")
        logger.info(f"URL: {request.url}")
        logger.info(f"Method: {request.method}")
        logger.info(f"Headers: {dict(request.headers)}")
        logger.info(f"Raw Body: {body.decode()}")
        
        if body:
            try:
                json_data = json.loads(body.decode())
                logger.info(f"Parsed JSON: {json_data}")
            except Exception as json_error:
                logger.error(f"JSON 파싱 실패: {json_error}")
    except Exception as e:
        logger.error(f"요청 로깅 실패: {e}")

# ========== API Endpoints ==========

@router.get("/health", 
    summary="Gaudi HPU 리소스 상태 확인",
    description="HPU 리소스 상태와 VLLM 인스턴스 현황을 상세히 조회합니다.",
    response_model=Dict[str, Any])
async def health_check(request: Request):
    """Gaudi 서비스 상태 확인"""
    try:
        hpu_available, device_count, device_info = check_hpu_availability()
        
        # 실행 중인 VLLM 인스턴스 상태 업데이트
        active_instances = {}
        for instance_id, instance_info in list(vllm_processes.items()):
            process = instance_info["process"]
            if process.poll() is None:  # 프로세스가 실행 중
                proc_info = get_process_detailed_info(process.pid)
                uptime = time.time() - instance_info["start_time"]
                
                active_instances[instance_id] = {
                    "status": "running",
                    "model_name": instance_info["model_name"],
                    "port": instance_info["port"],
                    "allocated_hpus": instance_info["hpu_ids"],
                    "tensor_parallel_size": instance_info["tensor_parallel_size"],
                    "uptime": uptime,
                    "process_info": proc_info,
                    "api_url": f"http://{instance_info['host']}:{instance_info['port']}"
                }
            else:
                # 프로세스가 종료됨 - 리소스 정리
                logger.info(f"인스턴스 {instance_id} 프로세스 종료 감지, 리소스 정리 중")
                deallocate_hpus(instance_id)
                del vllm_processes[instance_id]
        
        available_hpu_count = len([hpu_id for hpu_id in range(device_count) if hpu_id not in hpu_allocation])
        
        return {
            "hpu_status": {
                "available": hpu_available,
                "total_count": device_count,
                "available_count": available_hpu_count,
                "allocated_count": len(hpu_allocation),
                "device_details": device_info,
                "allocation_map": hpu_allocation
            },
            "vllm_instances": {
                "count": len(active_instances),
                "instances": active_instances
            },
            "resource_recommendation": {
                "max_new_instances_single_hpu": available_hpu_count,
                "max_new_instances_dual_hpu": available_hpu_count // 2,
                "suggested_tensor_parallel": min(available_hpu_count, 4) if available_hpu_count > 0 else 0
            }
        }
    except Exception as e:
        logger.error(f"상태 확인 실패: {e}")
        raise HTTPException(status_code=500, detail=f"상태 확인 실패: {str(e)}")

@router.post("/resource/check-allocation",
    summary="HPU 할당 가능성 확인",
    description="요청된 HPU 개수로 할당이 가능한지 확인하고 대안을 제시합니다.",
    response_model=ResourceAllocationResponse)
async def check_allocation_possibility(allocation_request: AutoAllocationRequest):
    """HPU 할당 가능성 확인"""
    try:
        available_hpus = find_available_hpus(
            allocation_request.required_hpus, 
            allocation_request.prefer_consecutive
        )
        
        if available_hpus:
            return ResourceAllocationResponse(
                success=True,
                allocated_hpus=available_hpus,
                message=f"{allocation_request.required_hpus}개 HPU 할당 가능: {available_hpus}"
            )
        else:
            # 대안 제시
            _, total_count, _ = check_hpu_availability()
            available_count = len([i for i in range(total_count) if i not in hpu_allocation])
            
            alternative = None
            if available_count > 0:
                alternative = find_available_hpus(min(available_count, allocation_request.required_hpus), False)
            
            return ResourceAllocationResponse(
                success=False,
                allocated_hpus=[],
                message=f"요청된 {allocation_request.required_hpus}개 HPU 할당 불가. 사용 가능: {available_count}개",
                alternative_suggestion=alternative
            )
            
    except Exception as e:
        logger.error(f"할당 확인 실패: {e}")
        raise HTTPException(status_code=500, detail=f"할당 확인 실패: {str(e)}")

@router.post("/vllm/start-auto",
    summary="VLLM 자동 할당 시작",
    description="사용 가능한 HPU를 자동으로 찾아 VLLM 인스턴스를 시작합니다.",
    response_model=VLLMInstanceInfo)
async def start_vllm_auto_allocation(request: VLLMAutoStartRequest):
    """VLLM 자동 할당 시작 - 통합 요청 모델 사용"""
    try:
        logger.info(f"=== VLLM 자동 시작 요청 ===")
        logger.info(f"모델: {request.model_name}")
        logger.info(f"필요 HPU: {request.required_hpus}")
        logger.info(f"텐서 병렬: {request.tensor_parallel_size}")
        
        # HPU 가용성 확인
        hpu_available, device_count, _ = check_hpu_availability()
        if not hpu_available or device_count == 0:
            raise HTTPException(status_code=400, detail="HPU를 사용할 수 없습니다")
        
        # 텐서 병렬 크기와 요청된 HPU 개수 검증 및 조정
        if request.tensor_parallel_size != request.required_hpus:
            logger.warning(f"텐서 병렬 크기({request.tensor_parallel_size})와 요청 HPU 개수({request.required_hpus})가 다릅니다. 텐서 병렬 크기로 조정합니다.")
            request.required_hpus = request.tensor_parallel_size
        
        # 사용 가능한 HPU 찾기
        available_hpus = find_available_hpus(
            request.required_hpus,
            request.prefer_consecutive
        )
        
        if not available_hpus:
            available_count = len([i for i in range(device_count) if i not in hpu_allocation])
            raise HTTPException(
                status_code=400, 
                detail=f"요청된 {request.required_hpus}개 HPU를 할당할 수 없습니다. 사용 가능: {available_count}개"
            )
        
        # 포트 중복 확인 및 자동 할당
        if not check_port_available(request.port):
            new_port = get_next_available_port(request.port)
            if new_port is None:
                raise HTTPException(status_code=400, detail="사용 가능한 포트를 찾을 수 없습니다")
            logger.info(f"포트 {request.port}가 사용 중이므로 {new_port}로 변경합니다")
            request.port = new_port
        
        # GaudiVLLMConfigRequest 객체 생성
        vllm_config = GaudiVLLMConfigRequest(
            model_name=request.model_name,
            max_model_len=request.max_model_len,
            host=request.host,
            port=request.port,
            dtype=request.dtype,
            tensor_parallel_size=request.tensor_parallel_size,
            tool_call_parser=request.tool_call_parser,
            trust_remote_code=request.trust_remote_code,
            enable_lora=request.enable_lora,
            gpu_memory_utilization=request.gpu_memory_utilization
        )
        
        return await _start_vllm_instance(vllm_config, available_hpus)
        
    except HTTPException:
        raise
    except ValidationError as ve:
        logger.error(f"요청 유효성 검사 실패: {ve}")
        raise HTTPException(status_code=422, detail=f"요청 데이터 오류: {ve}")
    except Exception as e:
        logger.error(f"자동 할당 VLLM 시작 실패: {e}")
        raise HTTPException(status_code=500, detail=f"자동 할당 VLLM 시작 실패: {str(e)}")

@router.post("/vllm/start-manual",
    summary="VLLM 수동 할당 시작",
    description="지정된 HPU 장치에서 VLLM 인스턴스를 시작합니다.",
    response_model=VLLMInstanceInfo)
async def start_vllm_manual_allocation(request: VLLMManualStartRequest):
    """VLLM 수동 할당 시작 - 통합 요청 모델 사용"""
    try:
        logger.info(f"=== VLLM 수동 시작 요청 ===")
        logger.info(f"모델: {request.model_name}")
        logger.info(f"지정 HPU: {request.device_ids}")
        logger.info(f"텐서 병렬: {request.tensor_parallel_size}")
        
        # HPU 가용성 확인
        hpu_available, device_count, _ = check_hpu_availability()
        if not hpu_available or device_count == 0:
            raise HTTPException(status_code=400, detail="HPU를 사용할 수 없습니다")
        
        # 장치 ID 유효성 검사
        for device_id in request.device_ids:
            if device_id >= device_count or device_id < 0:
                raise HTTPException(
                    status_code=400, 
                    detail=f"유효하지 않은 장치 ID: {device_id} (사용 가능: 0-{device_count-1})"
                )
            
            if device_id in hpu_allocation:
                raise HTTPException(
                    status_code=400,
                    detail=f"HPU {device_id}는 이미 인스턴스 {hpu_allocation[device_id]}에 할당되었습니다"
                )
        
        # 텐서 병렬 크기 검증 및 조정
        if request.tensor_parallel_size != len(request.device_ids):
            logger.warning(f"텐서 병렬 크기({request.tensor_parallel_size})와 할당된 HPU 개수({len(request.device_ids)})가 다릅니다. HPU 개수로 조정합니다.")
            request.tensor_parallel_size = len(request.device_ids)
        
        # 포트 중복 확인
        if not check_port_available(request.port):
            raise HTTPException(
                status_code=400, 
                detail=f"포트 {request.port}가 이미 사용 중입니다"
            )
        
        # GaudiVLLMConfigRequest 객체 생성
# GaudiVLLMConfigRequest 객체 생성
        vllm_config = GaudiVLLMConfigRequest(
            model_name=request.model_name,
            max_model_len=request.max_model_len,
            host=request.host,
            port=request.port,
            dtype=request.dtype,
            tensor_parallel_size=request.tensor_parallel_size,
            tool_call_parser=request.tool_call_parser,
            trust_remote_code=request.trust_remote_code,
            enable_lora=request.enable_lora,
            gpu_memory_utilization=request.gpu_memory_utilization
        )
        
        return await _start_vllm_instance(vllm_config, request.device_ids)
        
    except HTTPException:
        raise
    except ValidationError as ve:
        logger.error(f"요청 유효성 검사 실패: {ve}")
        raise HTTPException(status_code=422, detail=f"요청 데이터 오류: {ve}")
    except Exception as e:
        logger.error(f"수동 할당 VLLM 시작 실패: {e}")
        raise HTTPException(status_code=500, detail=f"수동 할당 VLLM 시작 실패: {str(e)}")

def read_process_output(process, instance_id, output_type="stdout"):
    """프로세스 출력을 실시간으로 읽어서 저장"""
    try:
        if output_type == "stdout":
            stream = process.stdout
        else:
            stream = process.stderr
            
        if instance_id not in process_logs:
            process_logs[instance_id] = []
            
        for line in iter(stream.readline, b''):
            if line:
                decoded_line = line.decode('utf-8').strip()
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                log_entry = f"[{timestamp}] [{output_type.upper()}] {decoded_line}"
                
                # 로그 저장 (최대 1000줄 유지)
                process_logs[instance_id].append(log_entry)
                if len(process_logs[instance_id]) > 1000:
                    process_logs[instance_id] = process_logs[instance_id][-1000:]
                
                # 실시간 로깅
                logger.info(f"VLLM[{instance_id}] {decoded_line}")
                
    except Exception as e:
        logger.error(f"로그 읽기 실패 ({instance_id}): {e}")


async def _start_vllm_instance(config: GaudiVLLMConfigRequest, hpu_ids: List[int]) -> VLLMInstanceInfo:
    """VLLM 인스턴스 시작 (내부 함수)"""
    instance_id = generate_instance_id(hpu_ids, config.model_name)
    
    # HPU 할당
    if not allocate_hpus(instance_id, hpu_ids):
        raise HTTPException(status_code=400, detail="HPU 할당 실패")
    
    try:
        # VLLM 명령어 구성
        cmd = build_vllm_command(config, hpu_ids, instance_id)
        
        # 환경 변수 설정
        env = os.environ.copy()
        env["HABANA_VISIBLE_DEVICES"] = ",".join(map(str, hpu_ids))
        env["PT_HPU_ENABLE_LAZY_COLLECTIVES"] = "true"
        env["VLLM_LOGGING_LEVEL"] = "INFO"
        
        logger.info(f"VLLM 인스턴스 시작: {instance_id}")
        logger.info(f"모델: {config.model_name}")
        logger.info(f"HPU: {hpu_ids}")
        logger.info(f"포트: {config.port}")
        logger.info(f"명령어: {' '.join(cmd[:10])}...")
        
        # VLLM 프로세스 시작
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,
            cwd="/workspace/vllm"  # VLLM 작업 디렉토리
        )
        
        # 인스턴스 정보 저장
        start_time = time.time()
        vllm_processes[instance_id] = {
            "process": process,
            "hpu_ids": hpu_ids,
            "model_name": config.model_name,
            "port": config.port,
            "host": config.host,
            "tensor_parallel_size": config.tensor_parallel_size,
            "start_time": start_time,
            "config": config.dict()
        }
        
        # 프로세스 시작 확인 (5초 대기)
        await asyncio.sleep(5)
        
        if process.poll() is not None:
            # 프로세스가 이미 종료됨
            stdout, stderr = process.communicate()
            logger.error(f"VLLM 시작 실패: {stderr.decode()}")
            
            # 리소스 정리
            deallocate_hpus(instance_id)
            if instance_id in vllm_processes:
                del vllm_processes[instance_id]
            
            raise HTTPException(
                status_code=500, 
                detail=f"VLLM 시작 실패: {stderr.decode()[:500]}..."
            )
        
        logger.info(f"VLLM 인스턴스 {instance_id} 시작 성공 (PID: {process.pid})")
        
        return VLLMInstanceInfo(
            instance_id=instance_id,
            status="running",
            model_name=config.model_name,
            port=config.port,
            allocated_hpus=hpu_ids,
            tensor_parallel_size=config.tensor_parallel_size,
            pid=process.pid,
            uptime=0.0,
            api_url=f"http://{config.host}:{config.port}"
        )
        
    except Exception as e:
        # 오류 발생 시 HPU 할당 해제
        deallocate_hpus(instance_id)
        raise e

@router.post("/vllm/{instance_id}/stop",
    summary="VLLM 인스턴스 중지",
    description="지정된 VLLM 인스턴스를 우아하게 중지하고 HPU 리소스를 해제합니다.",
    response_model=Dict[str, Any])
async def stop_vllm_instance(instance_id: str):
    """VLLM 인스턴스 중지"""
    try:
        if instance_id not in vllm_processes:
            raise HTTPException(status_code=404, detail=f"인스턴스 '{instance_id}'를 찾을 수 없습니다")
        
        instance_info = vllm_processes[instance_id]
        process = instance_info["process"]
        
        if process.poll() is not None:
            # 이미 종료된 프로세스
            deallocated_hpus = deallocate_hpus(instance_id)
            del vllm_processes[instance_id]
            return {
                "success": True,
                "message": f"인스턴스 '{instance_id}'가 이미 중지되었습니다",
                "instance_id": instance_id,
                "deallocated_hpus": deallocated_hpus
            }
        
        logger.info(f"VLLM 인스턴스 중지 시작: {instance_id} (PID: {process.pid})")
        
        # 우아한 종료 시도 (SIGTERM)
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            
            # 15초간 종료 대기
            try:
                process.wait(timeout=15)
                logger.info(f"인스턴스 {instance_id} 우아한 종료 완료")
            except subprocess.TimeoutExpired:
                # 강제 종료 (SIGKILL)
                logger.warning(f"우아한 종료 실패, 강제 종료 시행: {instance_id}")
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.wait(timeout=5)
                logger.info(f"인스턴스 {instance_id} 강제 종료 완료")
        except ProcessLookupError:
            # 프로세스가 이미 종료됨
            logger.info(f"프로세스가 이미 종료됨: {instance_id}")
        
        # 리소스 정리
        deallocated_hpus = deallocate_hpus(instance_id)
        del vllm_processes[instance_id]
        
        logger.info(f"VLLM 인스턴스 {instance_id} 중지 및 리소스 정리 완료")
        
        return {
            "success": True,
            "message": f"인스턴스 '{instance_id}'가 성공적으로 중지되었습니다",
            "instance_id": instance_id,
            "deallocated_hpus": deallocated_hpus,
            "freed_resources": len(deallocated_hpus)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"VLLM 인스턴스 중지 실패: {e}")
        raise HTTPException(status_code=500, detail=f"VLLM 인스턴스 중지 실패: {str(e)}")

@router.get("/vllm/instances",
    summary="VLLM 인스턴스 목록 조회",
    description="실행 중인 모든 VLLM 인스턴스의 상세 정보를 조회합니다.",
    response_model=List[VLLMInstanceInfo])
async def list_vllm_instances():
    """VLLM 인스턴스 목록 조회"""
    try:
        instances = []
        
        for instance_id, instance_info in list(vllm_processes.items()):
            process = instance_info["process"]
            
            if process.poll() is None:  # 실행 중
                uptime = time.time() - instance_info["start_time"]
                proc_info = get_process_detailed_info(process.pid)
                
                instances.append(VLLMInstanceInfo(
                    instance_id=instance_id,
                    status="running",
                    model_name=instance_info["model_name"],
                    port=instance_info["port"],
                    allocated_hpus=instance_info["hpu_ids"],
                    tensor_parallel_size=instance_info["tensor_parallel_size"],
                    pid=process.pid,
                    memory_usage=proc_info,
                    uptime=uptime,
                    api_url=f"http://{instance_info['host']}:{instance_info['port']}"
                ))
            else:
                # 종료된 프로세스 정리
                logger.info(f"종료된 인스턴스 정리: {instance_id}")
                deallocate_hpus(instance_id)
                del vllm_processes[instance_id]
        
        return instances
        
    except Exception as e:
        logger.error(f"인스턴스 목록 조회 실패: {e}")
        raise HTTPException(status_code=500, detail="인스턴스 목록 조회 실패")

@router.get("/vllm/{instance_id}/status",
    summary="VLLM 인스턴스 상세 상태",
    description="특정 VLLM 인스턴스의 상세한 상태 정보를 조회합니다.",
    response_model=VLLMInstanceInfo)
async def get_vllm_instance_status(instance_id: str):
    """VLLM 인스턴스 상세 상태 조회"""
    try:
        if instance_id not in vllm_processes:
            raise HTTPException(status_code=404, detail=f"인스턴스 '{instance_id}'를 찾을 수 없습니다")
        
        instance_info = vllm_processes[instance_id]
        process = instance_info["process"]
        
        if process.poll() is not None:
            # 프로세스가 종료됨
            deallocate_hpus(instance_id)
            del vllm_processes[instance_id]
            raise HTTPException(status_code=410, detail=f"인스턴스 '{instance_id}'가 종료되었습니다")
        
        uptime = time.time() - instance_info["start_time"]
        proc_info = get_process_detailed_info(process.pid)
        
        return VLLMInstanceInfo(
            instance_id=instance_id,
            status="running",
            model_name=instance_info["model_name"],
            port=instance_info["port"],
            allocated_hpus=instance_info["hpu_ids"],
            tensor_parallel_size=instance_info["tensor_parallel_size"],
            pid=process.pid,
            memory_usage=proc_info,
            uptime=uptime,
            api_url=f"http://{instance_info['host']}:{instance_info['port']}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"인스턴스 상태 조회 실패: {e}")
        raise HTTPException(status_code=500, detail="인스턴스 상태 조회 실패")

@router.post("/vllm/stop-all",
    summary="모든 VLLM 인스턴스 중지",
    description="실행 중인 모든 VLLM 인스턴스를 일괄 중지하고 모든 HPU 리소스를 해제합니다.",
    response_model=Dict[str, Any])
async def stop_all_vllm_instances():
    """모든 VLLM 인스턴스 중지"""
    try:
        if not vllm_processes:
            return {
                "success": True,
                "message": "중지할 VLLM 인스턴스가 없습니다",
                "stopped_instances": [],
                "failed_instances": [],
                "total_deallocated_hpus": 0
            }
        
        stopped_instances = []
        failed_instances = []
        total_deallocated = 0
        
        # 모든 인스턴스에 대해 중지 시도
        for instance_id in list(vllm_processes.keys()):
            try:
                result = await stop_vllm_instance(instance_id)
                if result["success"]:
                    stopped_instances.append(instance_id)
                    total_deallocated += result.get("freed_resources", 0)
                else:
                    failed_instances.append(instance_id)
            except Exception as e:
                logger.error(f"인스턴스 {instance_id} 중지 실패: {e}")
                failed_instances.append(instance_id)
        
        # 남은 할당 정리 (혹시 모를 경우)
        remaining_allocations = len(hpu_allocation)
        if remaining_allocations > 0:
            logger.warning(f"정리되지 않은 HPU 할당 {remaining_allocations}개 발견, 강제 정리")
            hpu_allocation.clear()
        
        success = len(failed_instances) == 0
        message = f"총 {len(stopped_instances)}개 인스턴스 중지 완료"
        if failed_instances:
            message += f", {len(failed_instances)}개 실패"
        
        return {
            "success": success,
            "message": message,
            "stopped_instances": stopped_instances,
            "failed_instances": failed_instances,
            "total_deallocated_hpus": total_deallocated,
            "remaining_allocations_cleared": remaining_allocations
        }
        
    except Exception as e:
        logger.error(f"전체 인스턴스 중지 실패: {e}")
        raise HTTPException(status_code=500, detail="전체 인스턴스 중지 실패")

@router.get("/resource/available-hpus",
    summary="사용 가능한 HPU 조회",
    description="현재 사용 가능한 HPU 목록과 각 HPU의 상세 정보를 조회합니다.",
    response_model=Dict[str, Any])
async def get_available_hpus():
    """사용 가능한 HPU 조회"""
    try:
        hpu_available, device_count, device_info = check_hpu_availability()
        
        if not hpu_available:
            return {
                "available": False,
                "message": "HPU를 사용할 수 없습니다",
                "available_hpus": [],
                "allocated_hpus": [],
                "recommendations": []
            }
        
        available_hpu_ids = [i for i in range(device_count) if i not in hpu_allocation]
        allocated_hpu_info = {hpu_id: {"instance_id": instance_id, "device_info": device_info.get(hpu_id, {})} 
                             for hpu_id, instance_id in hpu_allocation.items()}
        
        # 추천 사항 생성
        recommendations = []
        if len(available_hpu_ids) >= 2:
            recommendations.append({
                "type": "dual_hpu",
                "description": f"2개 HPU로 텐서 병렬 처리 가능 (추천 조합: {available_hpu_ids[:2]})",
                "hpu_ids": available_hpu_ids[:2]
            })
        
        if len(available_hpu_ids) >= 4:
            recommendations.append({
                "type": "quad_hpu", 
                "description": f"4개 HPU로 고성능 텐서 병렬 처리 가능 (추천 조합: {available_hpu_ids[:4]})",
                "hpu_ids": available_hpu_ids[:4]
            })
        
        if len(available_hpu_ids) == 1:
            recommendations.append({
                "type": "single_hpu",
                "description": f"단일 HPU 사용 가능 (HPU-{available_hpu_ids[0]})",
                "hpu_ids": available_hpu_ids[:1]
            })
        
        return {
            "available": True,
            "total_hpus": device_count,
            "available_hpus": [{"hpu_id": hpu_id, "info": device_info.get(hpu_id, {})} 
                              for hpu_id in available_hpu_ids],
            "allocated_hpus": allocated_hpu_info,
            "available_count": len(available_hpu_ids),
            "allocated_count": len(hpu_allocation),
            "recommendations": recommendations
        }
        
    except Exception as e:
        logger.error(f"사용 가능한 HPU 조회 실패: {e}")
        raise HTTPException(status_code=500, detail="사용 가능한 HPU 조회 실패")

@router.post("/vllm/{instance_id}/health-check",
    summary="VLLM 인스턴스 헬스 체크",
    description="특정 VLLM 인스턴스의 API 서버가 정상 작동하는지 확인합니다.",
    response_model=Dict[str, Any])
async def vllm_instance_health_check(instance_id: str):
    """VLLM 인스턴스 헬스 체크"""
    try:
        if instance_id not in vllm_processes:
            raise HTTPException(status_code=404, detail=f"인스턴스 '{instance_id}'를 찾을 수 없습니다")
        
        instance_info = vllm_processes[instance_id]
        process = instance_info["process"]
        
        # 프로세스 상태 확인
        if process.poll() is not None:
            deallocate_hpus(instance_id)
            del vllm_processes[instance_id]
            return {
                "success": False,
                "status": "process_dead",
                "message": f"인스턴스 '{instance_id}' 프로세스가 종료되었습니다"
            }
        
        # API 헬스 체크
        import urllib.request
        import urllib.error
        
        health_url = f"http://{instance_info['host']}:{instance_info['port']}/health"
        
        logger.info(f"인스턴스 {instance_id} 헬스 체크: {health_url} {instance_info}")
        try:
            req = urllib.request.Request(health_url, headers={'Content-Type': 'application/json'})
            response = urllib.request.urlopen(req, timeout=10)
            
            if response.getcode() == 200:
                return {
                    "success": True,
                    "status": "healthy",
                    "message": "VLLM 인스턴스가 정상 작동 중입니다",
                    "instance_id": instance_id,
                    "api_url": f"http://{instance_info['host']}:{instance_info['port']}",
                    "allocated_hpus": instance_info['hpu_ids'],
                    "uptime": time.time() - instance_info['start_time']
                }
            else:
                return {
                    "success": False,
                    "status": "api_error",
                    "message": f"API 서버가 비정상 응답 (코드: {response.getcode()})",
                    "instance_id": instance_id
                }
                
        except urllib.error.URLError as e:
            return {
                "success": False,
                "status": "api_unavailable", 
                "message": f"API 서버에 접근할 수 없습니다: {str(e)}",
                "instance_id": instance_id,
                "suggestion": "서버가 아직 시작 중이거나 포트가 차단되었을 수 있습니다"
            }
        except Exception as e:
            return {
                "success": False,
                "status": "health_check_error",
                "message": f"헬스 체크 실행 중 오류: {str(e)}",
                "instance_id": instance_id
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"헬스 체크 실패: {e}")
        raise HTTPException(status_code=500, detail=f"헬스 체크 실패: {str(e)}")

# ========== 유틸리티 엔드포인트 ==========

@router.get("/models/recommended",
    summary="추천 모델 목록",
    description="Gaudi HPU에서 잘 작동하는 추천 모델 목록을 제공합니다.",
    response_model=Dict[str, Any])
async def get_recommended_models():
    """추천 모델 목록"""
    recommended_models = {
        "small_models": [
            {
                "name": "microsoft/DialoGPT-medium",
                "description": "대화형 모델, 단일 HPU 적합",
                "recommended_hpus": 1,
                "max_model_len": 1024
            },
            {
                "name": "x2bee/Polar-14B",
                "description": "한국어 특화 모델, 2-4 HPU 권장",
                "recommended_hpus": 2,
                "max_model_len": 2048
            }
        ],
        "large_models": [
            {
                "name": "meta-llama/Llama-2-13b-chat-hf",
                "description": "대화형 13B 모델, 4-8 HPU 권장",
                "recommended_hpus": 4,
                "max_model_len": 4096
            },
            {
                "name": "codellama/CodeLlama-13b-Python-hf",
                "description": "코드 생성 특화, 4-8 HPU 권장",
                "recommended_hpus": 4,
                "max_model_len": 4096
            }
        ],
        "specialized_models": [
            {
                "name": "sentence-transformers/all-MiniLM-L6-v2",
                "description": "임베딩 모델, 단일 HPU 적합",
                "recommended_hpus": 1,
                "max_model_len": 512
            }
        ]
    }
    
    return {
        "recommendations": recommended_models,
        "selection_guide": {
            "single_hpu": "소형 모델이나 테스트용",
            "dual_hpu": "중형 모델이나 적당한 성능",
            "quad_hpu": "대형 모델이나 고성능 요구사항",
            "octa_hpu": "최대 성능이 필요한 초대형 모델"
        },
        "performance_tips": [
            "bfloat16 dtype 사용 권장",
            "tensor_parallel_size는 사용할 HPU 개수와 동일하게 설정",
            "max_model_len은 메모리에 맞게 조정",
            "trust_remote_code=True로 커스텀 모델 지원"
        ]
    }
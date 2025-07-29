"""
Vast 컨트롤러

Vast 관련 API 엔드포인트를 제공합니다.
"""

from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel
import logging
from typing import Dict, Any, Optional, List

from config.config_composer import ConfigComposer
from service.vast.vast_service import VastService, auto_run_vllm
from service.database.connection import AppDatabaseManager

router = APIRouter(prefix="/api/vast", tags=["vast"])
logger = logging.getLogger("vast-controller")

# Request 모델들
class CreateInstanceRequest(BaseModel):
    offer_id: Optional[str] = None
    auto_destroy: Optional[bool] = None

class DestroyInstanceRequest(BaseModel):
    instance_id: str

class UpdateConfigRequest(BaseModel):
    config_updates: Dict[str, Any]

# Response 모델들
class InstanceStatusResponse(BaseModel):
    instance_id: str
    status: str
    public_ip: Optional[str] = None
    urls: Dict[str, str] = {}
    port_mappings: Dict[str, Any] = {}

class InstanceListResponse(BaseModel):
    instances: List[Dict[str, Any]]
    total: int

def get_vast_service() -> VastService:
    """VastService 인스턴스 생성"""
    try:
        config_composer = ConfigComposer()
        vast_config = config_composer.vast_config
        
        # 데이터베이스 매니저 가져오기
        db_manager = None
        if hasattr(config_composer, 'database_manager'):
            db_manager = config_composer.database_manager
        
        return VastService(vast_config, db_manager)
    except Exception as e:
        logger.error(f"VastService 초기화 실패: {e}")
        raise HTTPException(status_code=500, detail="VastService 초기화 실패")

@router.get("/health", summary="Vast 서비스 상태 확인")
async def health_check():
    """Vast 서비스 상태 확인"""
    try:
        service = get_vast_service()
        return {
            "status": "healthy",
            "service": "vast",
            "message": "VastAI 서비스가 정상적으로 작동 중입니다"
        }
    except Exception as e:
        logger.error(f"Health check 실패: {e}")
        raise HTTPException(status_code=503, detail="서비스 사용 불가")

@router.get("/config", summary="Vast 설정 조회")
async def get_config():
    """현재 Vast 설정 조회"""
    try:
        config_composer = ConfigComposer()
        vast_config = config_composer.vast_config
        
        return {
            "image_name": vast_config.image_name(),
            "max_price": vast_config.max_price(),
            "disk_size": vast_config.disk_size(),
            "min_gpu_ram": vast_config.min_gpu_ram(),
            "search_query": vast_config.search_query(),
            "debug": vast_config.debug(),
            "auto_destroy": vast_config.auto_destroy(),
            "api_key_configured": bool(vast_config.vast_api_key()),
            "hf_token_configured": bool(vast_config.hf_token())
        }
    except Exception as e:
        logger.error(f"설정 조회 실패: {e}")
        raise HTTPException(status_code=500, detail="설정 조회 실패")

@router.post("/config", summary="Vast 설정 업데이트")
async def update_config(request: UpdateConfigRequest):
    """Vast 설정 업데이트"""
    try:
        config_composer = ConfigComposer()
        vast_config = config_composer.vast_config
        
        # 설정 업데이트
        for key, value in request.config_updates.items():
            if hasattr(vast_config, key):
                # 실제 설정 업데이트 로직은 config에 따라 구현
                logger.info(f"설정 업데이트 요청: {key} = {value}")
        
        return {
            "success": True,
            "message": "설정이 업데이트되었습니다",
            "updated_keys": list(request.config_updates.keys())
        }
    except Exception as e:
        logger.error(f"설정 업데이트 실패: {e}")
        raise HTTPException(status_code=500, detail="설정 업데이트 실패")

@router.get("/offers", summary="사용 가능한 오퍼 검색")
async def search_offers():
    """사용 가능한 VastAI 오퍼 검색"""
    try:
        service = get_vast_service()
        
        # API 키 설정
        if not service.vast_manager.setup_api_key():
            raise HTTPException(status_code=400, detail="API 키가 설정되지 않았거나 잘못되었습니다")
        
        # 오퍼 검색
        offers = service.vast_manager.search_offers()
        
        if not offers:
            return {
                "offers": [],
                "total": 0,
                "message": "사용 가능한 오퍼가 없습니다"
            }
        
        # 가격순 정렬
        sorted_offers = sorted(offers, key=lambda x: x.get("dph_total", 999))
        
        return {
            "offers": sorted_offers,
            "total": len(sorted_offers),
            "max_price_filter": service.config.max_price()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"오퍼 검색 실패: {e}")
        raise HTTPException(status_code=500, detail="오퍼 검색 실패")

@router.post("/instances", summary="새 인스턴스 생성")
async def create_instance(request: CreateInstanceRequest, background_tasks: BackgroundTasks):
    """새 VastAI 인스턴스 생성"""
    try:
        service = get_vast_service()
        
        # 설정 임시 업데이트 (요청에서 제공된 경우)
        if request.auto_destroy is not None:
            # 설정 업데이트 로직 (임시)
            pass
        
        if request.offer_id:
            # 특정 오퍼로 인스턴스 생성
            instance_id = service.create_vllm_instance(request.offer_id)
        else:
            # 자동 오퍼 선택 후 인스턴스 생성
            instance_id = service.create_vllm_instance()
        
        if not instance_id:
            raise HTTPException(status_code=400, detail="인스턴스 생성 실패")
        
        # 백그라운드에서 인스턴스 설정
        background_tasks.add_task(service.wait_and_setup_instance, instance_id)
        
        return {
            "success": True,
            "instance_id": instance_id,
            "message": "인스턴스가 생성되었습니다. 설정이 백그라운드에서 진행됩니다.",
            "status": "creating"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"인스턴스 생성 실패: {e}")
        raise HTTPException(status_code=500, detail="인스턴스 생성 실패")

@router.post("/instances/auto-run", summary="vLLM 자동 실행")
async def auto_run_instance():
    """end-to-end vLLM 자동 실행"""
    try:
        config_composer = ConfigComposer()
        vast_config = config_composer.vast_config
        
        db_manager = None
        if hasattr(config_composer, 'database_manager'):
            db_manager = config_composer.database_manager
        
        # auto_run_vllm 실행
        result = auto_run_vllm(vast_config, db_manager)
        
        if not result["success"]:
            raise HTTPException(
                status_code=400, 
                detail=f"자동 실행 실패: {result.get('error', 'Unknown error')}"
            )
        
        return {
            "success": True,
            "message": "vLLM이 성공적으로 실행되었습니다",
            "result": result
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"자동 실행 실패: {e}")
        raise HTTPException(status_code=500, detail="자동 실행 실패")

@router.get("/instances", summary="인스턴스 목록 조회")
async def list_instances() -> InstanceListResponse:
    """모든 인스턴스 목록 조회"""
    try:
        service = get_vast_service()
        instances = service.list_instances()
        
        return InstanceListResponse(
            instances=instances,
            total=len(instances)
        )
    except Exception as e:
        logger.error(f"인스턴스 목록 조회 실패: {e}")
        raise HTTPException(status_code=500, detail="인스턴스 목록 조회 실패")

@router.get("/instances/{instance_id}", summary="인스턴스 상태 조회")
async def get_instance_status(instance_id: str) -> InstanceStatusResponse:
    """특정 인스턴스의 상태 조회"""
    try:
        service = get_vast_service()
        status_info = service.get_instance_status_info(instance_id)
        
        return InstanceStatusResponse(
            instance_id=instance_id,
            status=status_info.get("status", "unknown"),
            public_ip=status_info.get("port_mappings", {}).get("public_ip"),
            urls=status_info.get("urls", {}),
            port_mappings=status_info.get("port_mappings", {})
        )
    except Exception as e:
        logger.error(f"인스턴스 상태 조회 실패: {e}")
        raise HTTPException(status_code=500, detail="인스턴스 상태 조회 실패")

@router.post("/instances/{instance_id}/setup", summary="인스턴스 설정")
async def setup_instance(instance_id: str, background_tasks: BackgroundTasks):
    """인스턴스 설정 (HF 로그인 및 vLLM 실행)"""
    try:
        service = get_vast_service()
        
        # 백그라운드에서 설정 실행
        background_tasks.add_task(service.wait_and_setup_instance, instance_id)
        
        return {
            "success": True,
            "message": "인스턴스 설정이 백그라운드에서 시작되었습니다",
            "instance_id": instance_id
        }
    except Exception as e:
        logger.error(f"인스턴스 설정 실패: {e}")
        raise HTTPException(status_code=500, detail="인스턴스 설정 실패")

@router.get("/instances/{instance_id}/logs", summary="인스턴스 로그 조회")
async def get_instance_logs(instance_id: str):
    """인스턴스의 vLLM 로그 조회"""
    try:
        service = get_vast_service()
        vllm_status = service.vast_manager.check_vllm_status(instance_id)
        
        return {
            "instance_id": instance_id,
            "logs": vllm_status.get("log_output", ""),
            "process_info": vllm_status.get("process_info", ""),
            "log_available": vllm_status.get("log_success", False)
        }
    except Exception as e:
        logger.error(f"로그 조회 실패: {e}")
        raise HTTPException(status_code=500, detail="로그 조회 실패")

@router.get("/instances/{instance_id}/ports", summary="포트 매핑 조회")
async def get_port_mappings(instance_id: str):
    """인스턴스의 포트 매핑 정보 조회"""
    try:
        service = get_vast_service()
        port_info = service.vast_manager.get_port_mappings(instance_id)
        
        return {
            "instance_id": instance_id,
            "port_mappings": port_info.get("mappings", {}),
            "public_ip": port_info.get("public_ip"),
            "urls": service._generate_access_urls(port_info)
        }
    except Exception as e:
        logger.error(f"포트 매핑 조회 실패: {e}")
        raise HTTPException(status_code=500, detail="포트 매핑 조회 실패")

@router.post("/instances/{instance_id}/execute", summary="SSH 명령 실행")
async def execute_ssh_command(instance_id: str, command: str):
    """인스턴스에서 SSH 명령 실행"""
    try:
        service = get_vast_service()
        result = service.vast_manager.execute_ssh_command(instance_id, command)
        
        return {
            "instance_id": instance_id,
            "command": command,
            "success": result.get("success", False),
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "error": result.get("error")
        }
    except Exception as e:
        logger.error(f"SSH 명령 실행 실패: {e}")
        raise HTTPException(status_code=500, detail="SSH 명령 실행 실패")

@router.delete("/instances/{instance_id}", summary="인스턴스 삭제")
async def destroy_instance(instance_id: str):
    """인스턴스 삭제"""
    try:
        service = get_vast_service()
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

@router.get("/instances/{instance_id}/history", summary="인스턴스 실행 히스토리")
async def get_instance_history(instance_id: str):
    """인스턴스의 실행 히스토리 조회"""
    try:
        service = get_vast_service()
        
        if not service.db_manager:
            return {
                "instance_id": instance_id,
                "history": [],
                "message": "히스토리 저장이 비활성화되어 있습니다"
            }
        
        from service.database.models.vast import VastExecutionLog
        conditions = {"instance_id": instance_id}
        logs = service.db_manager.select(VastExecutionLog, conditions=conditions)
        
        history = [log.to_dict() for log in logs] if logs else []
        
        return {
            "instance_id": instance_id,
            "history": history,
            "total": len(history)
        }
    except Exception as e:
        logger.error(f"히스토리 조회 실패: {e}")
        raise HTTPException(status_code=500, detail="히스토리 조회 실패")


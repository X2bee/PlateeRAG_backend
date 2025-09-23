"""
Data Manager API 컨트롤러

Data Manager 인스턴스의 생성, 관리, 삭제 및 Huggingface 데이터 관리를 위한 RESTful API
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from service.database.models.user import User
from controller.helper.singletonHelper import get_db_manager, get_data_manager_registry
from controller.helper.controllerHelper import extract_user_id_from_request

router = APIRouter(prefix="", tags=["data-manager"])
logger = logging.getLogger("data-manager-controller")

# ========== Request Models ==========
class GetManagerStatusRequest(BaseModel):
    """매니저 상태 조회 요청"""
    manager_id: str = Field(..., description="매니저 ID")

class DeleteManagerRequest(BaseModel):
    """매니저 삭제 요청"""
    manager_id: str = Field(..., description="매니저 ID")

class SetDatasetRequest(BaseModel):
    """데이터셋 설정 요청"""
    manager_id: str = Field(..., description="매니저 ID")
    dataset: Any = Field(..., description="설정할 데이터셋")

class GetDatasetRequest(BaseModel):
    """데이터셋 조회 요청"""
    manager_id: str = Field(..., description="매니저 ID")

class RemoveDatasetRequest(BaseModel):
    """데이터셋 제거 요청"""
    manager_id: str = Field(..., description="매니저 ID")

class GetDatasetSampleRequest(BaseModel):
    """데이터셋 샘플 조회 요청"""
    manager_id: str = Field(..., description="매니저 ID")
    num_samples: int = Field(10, description="샘플 수", ge=1, le=100)

# ========== Response Models ==========
class ManagerStatusResponse(BaseModel):
    """매니저 상태 응답"""
    manager_id: str = Field(..., description="매니저 ID")
    user_id: str = Field(..., description="사용자 ID")
    user_name: str = Field(..., description="사용자 이름")
    created_at: str = Field(..., description="생성 시간")
    is_active: bool = Field(..., description="활성 상태")
    current_instance_memory_mb: float = Field(..., description="현재 인스턴스 메모리 사용량 (MB)")
    initial_instance_memory_mb: float = Field(..., description="초기 인스턴스 메모리 사용량 (MB)")
    peak_instance_memory_mb: float = Field(..., description="최고 인스턴스 메모리 사용량 (MB)")
    memory_growth_mb: float = Field(..., description="메모리 증가량 (MB)")
    dataset_memory_mb: float = Field(..., description="데이터셋 메모리 사용량 (MB)")
    has_dataset: bool = Field(..., description="데이터셋 보유 여부")
    memory_distribution: Dict[str, float] = Field(..., description="메모리 분포")

class ManagerListResponse(BaseModel):
    """매니저 목록 응답"""
    managers: Dict[str, ManagerStatusResponse] = Field(..., description="매니저 목록")
    total: int = Field(..., description="총 매니저 수")

# ========== Helper Functions ==========
def get_manager_with_auth(registry, manager_id: str, user_id: str):
    """인증된 매니저 가져오기"""
    manager = registry.get_manager(manager_id, user_id)
    if not manager:
        raise HTTPException(
            status_code=404,
            detail=f"매니저 '{manager_id}'를 찾을 수 없거나 접근 권한이 없습니다"
        )
    return manager

# ========== API Endpoints ==========

@router.get("/health",
    summary="서비스 상태 확인",
    description="Data Manager 서비스의 상태를 확인합니다.",
    response_description="서비스 상태 정보")
async def health_check(request: Request):
    """Data Manager 서비스 상태 확인"""
    try:
        registry = get_data_manager_registry(request)
        total_stats = registry.get_total_stats()

        return {
            "status": "healthy",
            "service": "data-manager",
            "message": "Data Manager 서비스가 정상적으로 작동 중입니다",
            "stats": total_stats
        }
    except Exception as e:
        logger.error(f"Health check 실패: {e}")
        raise HTTPException(status_code=503, detail="서비스 사용 불가")

@router.post("/managers",
    summary="데이터 매니저 생성",
    description="새로운 Data Manager 인스턴스를 생성합니다.",
    response_model=Dict[str, Any])
async def create_manager(request: Request) -> Dict[str, Any]:
    """새 데이터 매니저 생성"""
    try:
        # 사용자 ID 추출
        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

        app_db = get_db_manager(request)
        user_db = app_db.find_by_id(User, user_id)
        user_name = user_db.username or user_db.full_name or "Unknown"

        registry = get_data_manager_registry(request)

        # 매니저 생성
        manager_id = registry.create_manager(user_id, user_name)

        logger.info(f"Data Manager {manager_id} 생성됨 (사용자: {user_id})")

        return {
            "success": True,
            "manager_id": manager_id,
            "user_id": user_id,
            "user_name": user_name,
            "message": "Data Manager가 성공적으로 생성되었습니다",
            "created_at": datetime.now().isoformat(),
            "endpoints": {
                "status": f"/api/data-manager/managers/{manager_id}/status",
                "dataset": f"/api/data-manager/managers/{manager_id}/dataset"
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"매니저 생성 실패: {e}")
        raise HTTPException(status_code=500, detail="매니저 생성 실패")

@router.get("/managers",
    summary="데이터 매니저 목록 조회",
    description="사용자의 모든 Data Manager 인스턴스를 조회합니다.",
    response_model=ManagerListResponse)
async def list_managers(request: Request, all_users: bool = False) -> ManagerListResponse:
    """데이터 매니저 목록 조회"""
    try:
        user_id = extract_user_id_from_request(request) if not all_users else None

        registry = get_data_manager_registry(request)
        managers = registry.list_managers(user_id)

        return ManagerListResponse(
            managers={k: ManagerStatusResponse(**v) for k, v in managers.items()},
            total=len(managers)
        )

    except Exception as e:
        logger.error(f"매니저 목록 조회 실패: {e}")
        raise HTTPException(status_code=500, detail="매니저 목록 조회 실패")

@router.post("/managers/status",
    summary="데이터 매니저 상태 조회",
    description="지정된 Data Manager의 상태와 리소스 사용량을 조회합니다.",
    response_model=ManagerStatusResponse)
async def get_manager_status(request: Request, status_request: GetManagerStatusRequest) -> ManagerStatusResponse:
    """매니저 상태 조회"""
    try:
        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

        registry = get_data_manager_registry(request)
        manager = get_manager_with_auth(registry, status_request.manager_id, user_id)

        stats = manager.get_resource_stats()
        return ManagerStatusResponse(**stats)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"매니저 상태 조회 실패: {e}")
        raise HTTPException(status_code=500, detail="매니저 상태 조회 실패")

@router.post("/managers/delete",
    summary="데이터 매니저 삭제",
    description="지정된 Data Manager 인스턴스를 삭제하고 모든 리소스를 정리합니다.",
    response_model=Dict[str, Any])
async def delete_manager(request: Request, delete_request: DeleteManagerRequest) -> Dict[str, Any]:
    """데이터 매니저 삭제"""
    try:
        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

        registry = get_data_manager_registry(request)
        success = registry.remove_manager(delete_request.manager_id, user_id)

        if not success:
            raise HTTPException(status_code=404, detail="매니저를 찾을 수 없거나 삭제 권한이 없습니다")

        logger.info(f"Data Manager {delete_request.manager_id} 삭제됨 (사용자: {user_id})")

        return {
            "success": True,
            "manager_id": delete_request.manager_id,
            "message": "Data Manager가 성공적으로 삭제되었습니다",
            "deleted_at": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"매니저 삭제 실패: {e}")
        raise HTTPException(status_code=500, detail="매니저 삭제 실패")

@router.get("/stats",
    summary="전체 통계 조회",
    description="모든 Data Manager의 통계 정보를 조회합니다.",
    response_model=Dict[str, Any])
async def get_total_stats(request: Request) -> Dict[str, Any]:
    """전체 통계 조회"""
    try:
        registry = get_data_manager_registry(request)
        stats = registry.get_total_stats()

        return {
            "success": True,
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"통계 조회 실패: {e}")
        raise HTTPException(status_code=500, detail="통계 조회 실패")

@router.post("/managers/dataset/sample",
    summary="데이터셋 샘플 조회",
    description="지정된 Data Manager의 데이터셋 샘플을 조회합니다.",
    response_model=Dict[str, Any])
async def get_dataset_sample(request: Request, sample_request: GetDatasetSampleRequest) -> Dict[str, Any]:
    """데이터셋 샘플 조회"""
    try:
        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

        registry = get_data_manager_registry(request)
        manager = get_manager_with_auth(registry, sample_request.manager_id, user_id)
        sample_data = manager.get_dataset_sample(sample_request.num_samples)
        logger.info(f"Dataset sample retrieved for manager {sample_request.manager_id} (samples: {sample_request.num_samples})")

        return sample_data

    except HTTPException:
        raise
    except RuntimeError as e:
        logger.error(f"데이터셋 샘플 조회 실패: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"예상치 못한 오류: {e}")
        raise HTTPException(status_code=500, detail="데이터셋 샘플 조회 중 오류가 발생했습니다")

@router.post("/managers/dataset/remove",
    summary="데이터셋 제거",
    description="지정된 Data Manager의 데이터셋을 제거합니다.",
    response_model=Dict[str, Any])
async def remove_dataset(request: Request, remove_request: RemoveDatasetRequest) -> Dict[str, Any]:
    """데이터셋 제거"""
    try:
        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

        registry = get_data_manager_registry(request)
        manager = get_manager_with_auth(registry, remove_request.manager_id, user_id)

        success = manager.remove_dataset()

        if not success:
            raise HTTPException(status_code=404, detail="제거할 데이터셋이 없습니다")

        logger.info(f"Dataset removed from manager {remove_request.manager_id} (사용자: {user_id})")

        return {
            "success": True,
            "manager_id": remove_request.manager_id,
            "message": "데이터셋이 성공적으로 제거되었습니다",
            "removed_at": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"데이터셋 제거 실패: {e}")
        raise HTTPException(status_code=500, detail="데이터셋 제거 실패")


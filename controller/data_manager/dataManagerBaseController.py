"""
Data Manager API 컨트롤러

Data Manager 인스턴스의 생성, 관리, 삭제 및 Huggingface 데이터 관리를 위한 RESTful API
"""
#/controller/data_manager/dataManagerBaseController.py
from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel, Field
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from service.data_manager.timezone_utils import now_kst_iso
from service.database.models.user import User
from controller.helper.singletonHelper import get_db_manager, get_data_manager_registry
from controller.helper.controllerHelper import extract_user_id_from_request
from pydantic import BaseModel

router = APIRouter(prefix="", tags=["data-manager"])
logger = logging.getLogger("data-manager-controller")

# ========== Request Models ==========
class GetManagerStatusRequest(BaseModel):
    """매니저 상태 조회 요청"""
    manager_id: str = Field(..., description="매니저 ID")

class DeleteManagerRequest(BaseModel):
    """매니저 삭제 요청"""
    manager_id: str = Field(..., description="매니저 ID")
    delete_data: bool = Field(True, description="MinIO/Redis 데이터도 함께 삭제 여부 (기본값: True)")

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

class SwitchDatasetVersionRequest(BaseModel):
    """데이터셋 버전 전환 요청"""
    manager_id: str
    version_number: int

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
            "created_at": now_kst_iso(),
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
    response_model=Dict[str, Any])
async def list_data_managers(request: Request) -> Dict[str, Any]:
    """사용자의 매니저 목록 조회"""
    try:
        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")
        
        user_id = str(user_id)  # 문자열 변환
        registry = get_data_manager_registry(request)
        
        logger.info(f"📋 매니저 목록 조회: user_id={user_id}")
        
        all_managers = []
        
        if registry.redis_manager:
            manager_ids = registry.redis_manager.get_user_managers(user_id)
            logger.info(f"  └─ 발견된 매니저: {len(manager_ids)}개")
            
            for manager_id in manager_ids:
                # 🔧 소유권 체크 강화
                owner = registry.redis_manager.get_manager_owner(manager_id)
                owner_str = str(owner) if owner is not None else None
                
                # 🔍 디버그 로깅 추가
                logger.info(f"  📊 매니저 {manager_id}: owner={owner} (type: {type(owner)}), user_id={user_id} (type: {type(user_id)})")
                
                # ✅ 타입 안전 비교
                if owner_str != user_id:
                    logger.warning(f"  ⚠️ 소유권 불일치: {manager_id} (owner: {owner_str}, expected: {user_id})")
                    continue
                
                # 매니저 정보 조회
                manager = registry.managers.get(manager_id)
                
                if manager:
                    manager_info = manager.get_resource_stats()
                    manager_info['in_memory'] = True
                    manager_info['status'] = 'active'
                else:
                    manager_info = _get_manager_info_from_redis(
                        registry, manager_id, user_id
                    )
                    if manager_info:
                        manager_info['in_memory'] = False
                        manager_info['status'] = 'stored'
                
                if manager_info:
                    all_managers.append(manager_info)
                else:
                    logger.warning(f"  ⚠️ 매니저 정보 조회 실패: {manager_id}")
        
        logger.info(f"  ✅ 최종 매니저 수: {len(all_managers)}개")
        
        return {
            "success": True,
            "user_id": user_id,
            "managers": all_managers,
            "total_count": len(all_managers)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"매니저 목록 조회 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="매니저 목록 조회 실패")

def _get_manager_info_from_redis(registry, manager_id: str, user_id: str) -> Optional[Dict]:
    """Redis에서 매니저 정보 조회"""
    try:
        logger.info(f"  🔍 Redis 정보 조회 시작: {manager_id}")
        
        current_version = registry.redis_manager.get_current_version(manager_id)
        logger.info(f"    └─ current_version: {current_version}")
        
        if current_version == 0:
            logger.warning(f"    └─ ⚠️ 버전이 0입니다")
            return {
                'manager_id': manager_id,
                'user_id': user_id,
                'user_name': 'Unknown',
                'created_at': 'Unknown',
                'is_active': False,
                'current_instance_memory_mb': 0.0,
                'initial_instance_memory_mb': 0.0,
                'peak_instance_memory_mb': 0.0,
                'memory_growth_mb': 0.0,
                'dataset_memory_mb': 0.0,
                'has_dataset': False,
                'memory_distribution': {},
                'dataset_id': None,
                'dataset_rows': 0,
                'dataset_columns': 0,
                'current_version': 0,
                'last_operation': 'created',
                'source_type': None,
                'status': 'empty',
                'storage_location': 'none'
            }
        
        # ⭐ 버전 메타데이터 조회
        version_info = registry.redis_manager.get_version_metadata(
            manager_id, current_version - 1
        )
        
        # ⭐ 디버그 로깅 추가
        logger.info(f"    └─ version_info exists: {version_info is not None}")
        if version_info:
            logger.info(f"    └─ version_info keys: {version_info.keys()}")
            logger.info(f"    └─ dataset_id: {version_info.get('dataset_id')}")
            logger.info(f"    └─ num_rows: {version_info.get('num_rows')}")
        
        if not version_info:
            logger.warning(f"    └─ ⚠️ 버전 메타데이터가 없습니다")
            return {
                'manager_id': manager_id,
                'user_id': user_id,
                'user_name': 'Unknown',
                'created_at': 'Unknown',
                'is_active': False,
                'current_instance_memory_mb': 0.0,
                'initial_instance_memory_mb': 0.0,
                'peak_instance_memory_mb': 0.0,
                'memory_growth_mb': 0.0,
                'dataset_memory_mb': 0.0,
                'has_dataset': False,
                'memory_distribution': {},
                'dataset_id': None,
                'dataset_rows': 0,
                'dataset_columns': 0,
                'current_version': current_version,
                'last_operation': 'unknown',
                'source_type': None,
                'status': 'metadata_missing',
                'storage_location': 'redis'
            }
        
        # ⭐ 소스 정보 조회
        source_history = registry.redis_manager.get_all_source_info(manager_id)
        latest_source = source_history[-1] if source_history else {}
        
        # ⭐ 데이터셋 존재 여부 확인
        has_dataset = (
            version_info.get('num_rows', 0) > 0 or 
            version_info.get('dataset_id') is not None
        )

        logger.info(f"    └─ has_dataset: {has_dataset}")

        # ✨ Manager 상태에서 원본 created_at 가져오기
        manager_state = registry.redis_manager.get_manager_state(manager_id)
        created_at = 'Unknown'
        if manager_state and 'created_at' in manager_state:
            created_at = manager_state['created_at']
        else:
            # Fallback: latest_source의 loaded_at 사용
            created_at = latest_source.get('loaded_at', 'Unknown')
            logger.warning(f"    └─ ⚠️  Manager 상태에 created_at 없음, loaded_at 사용")

        return {
            'manager_id': manager_id,
            'user_id': user_id,
            'user_name': version_info.get('metadata', {}).get('user_name', 'Unknown'),
            'created_at': created_at,
            'is_active': has_dataset,  # ⭐ 데이터 있으면 활성!
            'current_instance_memory_mb': 0.0,
            'initial_instance_memory_mb': 0.0,
            'peak_instance_memory_mb': 0.0,
            'memory_growth_mb': 0.0,
            'dataset_memory_mb': 0.0,
            'has_dataset': has_dataset,
            'memory_distribution': {},
            'dataset_id': version_info.get('dataset_id'),
            'dataset_rows': version_info.get('num_rows', 0),
            'dataset_columns': version_info.get('num_columns', 0),
            'current_version': current_version,
            'last_operation': version_info.get('operation'),
            'source_type': latest_source.get('type'),
            'status': 'stored_in_redis',
            'storage_location': 'redis',
            'in_memory': False
        }
        
    except Exception as e:
        logger.error(f"    └─ ❌ Redis 정보 조회 실패: {manager_id}, {e}", exc_info=True)
        return None

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
    description="지정된 Data Manager 인스턴스를 삭제하고 모든 리소스를 정리합니다. delete_data=True면 MinIO/Redis 데이터도 완전 삭제됩니다.",
    response_model=Dict[str, Any])
async def delete_manager(request: Request, delete_request: DeleteManagerRequest) -> Dict[str, Any]:
    """데이터 매니저 삭제 (MinIO/Redis 데이터 포함)"""
    try:
        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

        registry = get_data_manager_registry(request)
        success = registry.remove_manager(
            delete_request.manager_id,
            user_id,
            delete_data=delete_request.delete_data
        )

        if not success:
            raise HTTPException(status_code=404, detail="매니저를 찾을 수 없거나 삭제 권한이 없습니다")

        deletion_type = "완전 삭제" if delete_request.delete_data else "세션만 제거"
        logger.info(f"Data Manager {delete_request.manager_id} {deletion_type} (사용자: {user_id})")

        return {
            "success": True,
            "manager_id": delete_request.manager_id,
            "message": f"Data Manager가 성공적으로 {deletion_type}되었습니다",
            "deleted_at": now_kst_iso(),
            "deletion_type": deletion_type,
            "data_deleted": delete_request.delete_data
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
            "timestamp": now_kst_iso()
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
            "removed_at": now_kst_iso()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"데이터셋 제거 실패: {e}")
        raise HTTPException(status_code=500, detail="데이터셋 제거 실패")


@router.post("/managers/versions",
    summary="버전 이력 조회",
    description="DataManager의 모든 버전 이력을 조회합니다.",
    response_model=Dict[str, Any])
async def get_version_history(request: Request, status_request: GetManagerStatusRequest) -> Dict[str, Any]:
    """버전 이력 조회"""
    try:
        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

        registry = get_data_manager_registry(request)
        manager = get_manager_with_auth(registry, status_request.manager_id, user_id)

        # 버전 이력 조회
        version_history = manager.get_version_history()
        source_info = manager.redis_manager.get_source_info(manager.dataset_id) if manager.redis_manager and manager.dataset_id else None
        lineage = manager.redis_manager.get_lineage(manager.dataset_id) if manager.redis_manager and manager.dataset_id else None

        return {
            "success": True,
            "manager_id": status_request.manager_id,
            "current_version": manager.current_version,
            "total_versions": len(version_history),
            "source_info": source_info,
            "version_history": version_history,
            "lineage": lineage
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"버전 이력 조회 실패: {e}")
        raise HTTPException(status_code=500, detail="버전 이력 조회 실패")


class RollbackRequest(BaseModel):
    """버전 롤백 요청"""
    manager_id: str = Field(..., description="매니저 ID")
    version: int = Field(..., description="롤백할 버전 번호", ge=0)


@router.post("/managers/rollback",
    summary="버전 롤백",
    description="특정 버전으로 데이터셋을 롤백합니다.",
    response_model=Dict[str, Any])
async def rollback_to_version(request: Request, rollback_request: RollbackRequest) -> Dict[str, Any]:
    """버전 롤백"""
    try:
        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

        registry = get_data_manager_registry(request)
        manager = get_manager_with_auth(registry, rollback_request.manager_id, user_id)
        # 롤백 실행
        result = manager.rollback_to_version(rollback_request.version)

        logger.info(f"버전 롤백 완료: 매니저 {rollback_request.manager_id} → v{rollback_request.version}")

        return {
            "success": True,
            "manager_id": rollback_request.manager_id,
            "message": f"버전 {rollback_request.version}으로 성공적으로 롤백되었습니다",
            "rollback_info": result
        }

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"버전 롤백 실패: {e}")
        raise HTTPException(status_code=500, detail="버전 롤백 실패")


class CompareVersionsRequest(BaseModel):
    """버전 비교 요청"""
    manager_id: str = Field(..., description="매니저 ID")
    version1: int = Field(..., description="비교할 첫 번째 버전")
    version2: int = Field(..., description="비교할 두 번째 버전")


@router.post("/managers/compare-versions",
    summary="버전 비교",
    description="두 버전 간의 차이를 비교합니다.",
    response_model=Dict[str, Any])
async def compare_versions(request: Request, compare_request: CompareVersionsRequest) -> Dict[str, Any]:
    """버전 비교"""
    try:
        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

        registry = get_data_manager_registry(request)
        manager = get_manager_with_auth(registry, compare_request.manager_id, user_id)

        if not manager.redis_manager:
            raise HTTPException(status_code=400, detail="버전 관리 기능이 활성화되지 않았습니다")

        # 버전 정보 조회
        v1_info = manager.redis_manager.get_version_metadata(compare_request.manager_id, compare_request.version1)
        v2_info = manager.redis_manager.get_version_metadata(compare_request.manager_id, compare_request.version2)

        if not v1_info or not v2_info:
            raise HTTPException(status_code=404, detail="요청한 버전을 찾을 수 없습니다")

        # 차이 계산
        comparison = {
            "version1": {
                "version": compare_request.version1,
                "operation": v1_info.get("operation"),
                "timestamp": v1_info.get("timestamp"),
                "rows": v1_info.get("num_rows"),
                "columns": v1_info.get("num_columns"),
                "column_names": v1_info.get("columns"),
                "checksum": v1_info.get("checksum")
            },
            "version2": {
                "version": compare_request.version2,
                "operation": v2_info.get("operation"),
                "timestamp": v2_info.get("timestamp"),
                "rows": v2_info.get("num_rows"),
                "columns": v2_info.get("num_columns"),
                "column_names": v2_info.get("columns"),
                "checksum": v2_info.get("checksum")
            },
            "differences": {
                "rows_diff": v2_info.get("num_rows", 0) - v1_info.get("num_rows", 0),
                "columns_diff": v2_info.get("num_columns", 0) - v1_info.get("num_columns", 0),
                "columns_added": list(set(v2_info.get("columns", [])) - set(v1_info.get("columns", []))),
                "columns_removed": list(set(v1_info.get("columns", [])) - set(v2_info.get("columns", []))),
                "data_changed": v1_info.get("checksum") != v2_info.get("checksum")
            }
        }

        return {
            "success": True,
            "manager_id": compare_request.manager_id,
            "comparison": comparison
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"버전 비교 실패: {e}")
        raise HTTPException(status_code=500, detail="버전 비교 실패")


@router.get("/managers/{manager_id}/minio-versions",
    summary="MinIO 저장된 버전 목록",
    description="MinIO에 저장된 모든 버전 스냅샷 목록을 조회합니다.")
async def list_minio_versions(request: Request, manager_id: str) -> Dict[str, Any]:
    """MinIO 버전 목록 조회"""
    try:
        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

        registry = get_data_manager_registry(request)
        manager = get_manager_with_auth(registry, manager_id, user_id)

        if not manager.minio_storage:
            raise HTTPException(status_code=400, detail="MinIO 스토리지가 활성화되지 않았습니다")

        # MinIO에서 버전 목록 조회
        if not manager.dataset_id:
            raise HTTPException(status_code=404, detail="dataset_id가 없습니다")
        versions = manager.minio_storage.list_versions(manager.dataset_id)

        return {
            "success": True,
            "manager_id": manager_id,
            "total_versions": len(versions),
            "versions": versions
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MinIO 버전 목록 조회 실패: {e}")
    raise HTTPException(status_code=500, detail="버전 목록 조회 실패")@router.post("/managers/dataset-history",

    summary="데이터셋 로드 이력 조회",
    description="매니저의 모든 데이터셋 로드 이력을 조회합니다.",
    response_model=Dict[str, Any])

@router.post("/managers/dataset-history",
    summary="데이터셋 로드 이력 조회",
    description="매니저의 모든 데이터셋 로드 이력을 조회합니다.",
    response_model=Dict[str, Any])
async def get_dataset_load_history(request: Request, status_request: GetManagerStatusRequest) -> Dict[str, Any]:
    """데이터셋 로드 이력 조회"""
    try:
        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

        registry = get_data_manager_registry(request)
        manager = get_manager_with_auth(registry, status_request.manager_id, user_id)

        # Redis에서 소스 이력 조회
        source_history = []
        if manager.redis_manager:
            source_history = manager.redis_manager.get_all_source_info(manager.manager_id) or []

        return {
            "success": True,
            "manager_id": status_request.manager_id,
            "total_loads": manager.dataset_load_count,
            "current_dataset_id": manager.dataset_id,
            "source_history": source_history
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"데이터셋 로드 이력 조회 실패: {e}")
        raise HTTPException(status_code=500, detail="이력 조회 실패")


@router.post("/managers/switch-dataset-version",
    summary="데이터셋 버전 전환",
    description="특정 로드 버전의 데이터셋으로 전환합니다.",
    response_model=Dict[str, Any])
async def switch_dataset_version(
    request: Request, 
    switch_request: SwitchDatasetVersionRequest
) -> Dict[str, Any]:
    """데이터셋 버전 전환 (DataManager.rollback_to_version 사용)"""
    try:
        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

        registry = get_data_manager_registry(request)
        manager = get_manager_with_auth(registry, switch_request.manager_id, user_id)

        # 버전 관리 활성 확인
        if not manager.redis_manager or not manager.minio_storage:
            raise HTTPException(status_code=400, detail="버전 관리가 활성화되지 않았습니다")

        # 클라이언트-측 versionNumber는 1-based로 전달되므로 0-based로 변환
        if not isinstance(switch_request.version_number, int) or switch_request.version_number < 1:
            raise HTTPException(status_code=400, detail="유효한 버전 번호가 필요합니다")

        target_version = switch_request.version_number - 1

        # DataManager.rollback_to_version이 버전 메타를 조회하고 MinIO에서 스냅샷을 로드하여
        # manager.dataset, manager.current_version, manager.viewing_version 등을 갱신함
        result = manager.rollback_to_version(target_version)

        # viewing_version은 UI에서 1-based로 기대하므로 원래 요청값을 기록
        manager.viewing_version = switch_request.version_number

        logger.info(
            f"데이터셋 버전 전환 완료: {manager.manager_id} -> v{switch_request.version_number}"
        )

        return {
            "success": True,
            "manager_id": switch_request.manager_id,
            "switched_to_version": switch_request.version_number,
            "rollback_info": result,
            "message": f"데이터셋 버전 {switch_request.version_number}로 전환되었습니다"
        }

    except HTTPException:
        raise
    except ValueError as e:
        # rollback_to_version에서 존재하지 않는 버전이면 ValueError를 던짐
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"데이터셋 버전 전환 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"버전 전환 실패: {str(e)}")

@router.post("/managers/available-versions",
    summary="사용 가능한 데이터셋 버전 목록",
    description="매니저의 모든 데이터셋 로드 버전 목록을 조회합니다.",
    response_model=Dict[str, Any])
async def get_available_dataset_versions(
    request: Request, 
    status_request: GetManagerStatusRequest
) -> Dict[str, Any]:
    """사용 가능한 데이터셋 버전 목록 조회"""
    try:
        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

        registry = get_data_manager_registry(request)
        manager = get_manager_with_auth(registry, status_request.manager_id, user_id)

        # Redis에서 소스 이력 조회
        source_history = []
        current_viewing_version = getattr(manager, 'viewing_version', manager.dataset_load_count)
        
        if manager.redis_manager and manager.dataset_id:
            source_history = manager.redis_manager.get_all_source_info(manager.dataset_id) or []
        else:
            source_history = []

        # 버전 목록 생성
        versions = []
        for idx, source in enumerate(source_history):
            version_num = idx + 1
            versions.append({
                "version": version_num,
                "dataset_id": source.get('versioned_dataset_id'),
                "source_type": source.get('type'),
                "loaded_at": source.get('loaded_at'),
                "num_rows": source.get('num_rows'),
                "num_columns": source.get('num_columns'),
                "is_current": version_num == current_viewing_version,
                "repo_id": source.get('repo_id') if source.get('type') == 'huggingface' else None,
                "filenames": source.get('filenames') if source.get('type') == 'local' else None
            })

        return {
            "success": True,
            "manager_id": status_request.manager_id,
            "total_versions": len(versions),
            "current_viewing_version": current_viewing_version,
            "versions": versions
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"버전 목록 조회 실패: {e}")
        raise HTTPException(status_code=500, detail="버전 목록 조회 실패")


@router.post("/managers/{manager_id}/load",
    summary="매니저 로드",
    description="저장된 매니저를 메모리에 로드합니다.",
    response_model=Dict[str, Any])
async def load_manager(
    request: Request,
    manager_id: str
) -> Dict[str, Any]:
    """저장된 매니저를 메모리에 로드"""
    try:
        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

        registry = get_data_manager_registry(request)
        
        # get_manager가 자동으로 복원함
        manager = registry.get_manager(manager_id, user_id)
        
        if not manager:
            logger.error(f"매니저 {manager_id}를 찾을 수 없습니다")
            raise HTTPException(status_code=404, detail="매니저를 찾을 수 없습니다")

        return {
            "success": True,
            "message": "매니저가 성공적으로 로드되었습니다",
            "manager": manager.get_resource_stats()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"매니저 로드 실패: {e}")
        raise HTTPException(status_code=500, detail=f"매니저 로드 실패: {str(e)}")


@router.get("/managers/{manager_id}/data",
    summary="매니저 데이터 조회 및 자동 로드",
    description="매니저 데이터 조회 - 메모리에 없으면 자동으로 저장소에서 로드합니다.")
async def get_manager_data(
    request: Request,
    manager_id: str
) -> Dict[str, Any]:
    """매니저 데이터 조회 - 메모리에 없으면 자동 로드"""
    try:
        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

        registry = get_data_manager_registry(request)

        # ✅ get_manager가 자동으로 복원하므로 간단하게 처리
        manager = registry.get_manager(manager_id, user_id)
        
        if not manager:
            logger.error(f"❌ 매니저 {manager_id}를 찾거나 복원할 수 없습니다")
            raise HTTPException(status_code=404, detail="매니저를 찾을 수 없습니다")
        
        if manager.dataset is None:
            logger.warning(f"⚠️  매니저 {manager_id}에 데이터가 없습니다")
            return {
                "data": [],
                "rows": 0,
                "columns": [],
                "source": "no_data"
            }

        # pyarrow.Table을 pandas로 변환하여 반환
        try:
            df = manager.dataset.to_pandas()
            return {
                "data": df.to_dict('records'),
                "rows": manager.dataset.num_rows,
                "columns": manager.dataset.column_names,
                "source": "memory" if manager_id in registry.managers else "restored"
            }
        except Exception as e:
            logger.warning(f"⚠️  데이터 변환 실패: {e}")
            return {
                "data": [],
                "rows": manager.dataset.num_rows if manager.dataset is not None else 0,
                "columns": manager.dataset.column_names if manager.dataset is not None else [],
                "source": "memory" if manager_id in registry.managers else "restored"
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 데이터 로드 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"데이터 로드 실패: {str(e)}")

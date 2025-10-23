"""
Data Manager DB Sync API Controller
자동 DB 동기화 관리 API
"""
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from controller.helper.singletonHelper import get_db_sync_scheduler  # ✨ 변경
from controller.helper.controllerHelper import extract_user_id_from_request, validate_db_config
from service.data_manager.timezone_utils import now_kst_iso

router = APIRouter(prefix="/sync", tags=["data-manager-sync"])
logger = logging.getLogger("data-manager-sync-controller")

def get_app_db(request: Request):
    """Request에서 AppDB 인스턴스 가져오기"""
    if not hasattr(request.app.state, 'app_db'):
        raise HTTPException(status_code=500, detail="AppDB가 초기화되지 않았습니다")
    return request.app.state.app_db

#========== Request Models ==========
class AddDBSyncRequest(BaseModel):
    """DB 자동 동기화 추가 요청"""
    manager_id: str = Field(..., description="매니저 ID")
    db_config: Dict[str, Any] = Field(..., description="DB 연결 설정")
    sync_config: Dict[str, Any] = Field(..., description="동기화 설정")
    
    class Config:
        json_schema_extra = {
            "example": {
                "manager_id": "mgr_abc123",
                "db_config": {
                    "db_type": "postgresql",
                    "host": "localhost",
                    "port": 5432,
                    "database": "myapp",
                    "username": "postgres",
                    "password": "password123"
                },
                "sync_config": {
                    "enabled": True,
                    "schedule_type": "interval",
                    "interval_minutes": 30,
                    "query": "SELECT * FROM users WHERE active = true",
                    "detect_changes": True,
                    "notification_enabled": False,
                    # ✨ MLflow 설정 추가
                    "mlflow_enabled": True,
                    "mlflow_experiment_name": "user_data_sync",
                    "mlflow_tracking_uri": "https://mlflow.example.com"
                }
            }
        }


class RemoveDBSyncRequest(BaseModel):
    """DB 자동 동기화 제거 요청"""
    manager_id: str = Field(..., description="Manager ID")

class ControlDBSyncRequest(BaseModel):
    """DB 자동 동기화 제어 요청 (pause/resume)"""
    manager_id: str = Field(..., description="Manager ID")

class ManualSyncRequest(BaseModel):
    """수동 동기화 요청"""
    manager_id: str = Field(..., description="Manager ID")

class GetSyncStatusRequest(BaseModel):
    """동기화 상태 조회 요청"""
    manager_id: str = Field(..., description="Manager ID")

class UpdateMLflowConfigRequest(BaseModel):
    """MLflow 설정 업데이트 요청"""
    manager_id: str = Field(..., description="매니저 ID")
    mlflow_enabled: bool = Field(..., description="MLflow 자동 업로드 활성화 여부")
    mlflow_experiment_name: Optional[str] = Field(None, description="MLflow 실험 이름")
    mlflow_tracking_uri: Optional[str] = Field(None, description="MLflow Tracking URI")


#========== API Endpoints ==========
@router.post("/add",
summary="DB 자동 동기화 추가",
description="새로운 DB 자동 동기화 설정을 추가합니다 (MLflow 자동 업로드 옵션 포함)",
response_model=Dict[str, Any])
async def add_db_sync(request: Request, sync_request: AddDBSyncRequest) -> Dict[str, Any]:
    """DB 자동 동기화 추가 (MLflow 옵션 포함)"""
    try:
        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")
        
        scheduler = get_db_sync_scheduler(request)
        
        # ✨ sync_config에 MLflow 설정 포함
        success = scheduler.add_db_sync(
            manager_id=sync_request.manager_id,
            user_id=user_id,
            db_config=sync_request.db_config,
            sync_config=sync_request.sync_config  # MLflow 설정 포함됨
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="DB 자동 동기화 추가 실패")
        
        mlflow_status = "MLflow 자동 업로드 활성화" if sync_request.sync_config.get('mlflow_enabled') else ""
        logger.info(f"✅ DB 동기화 추가: manager={sync_request.manager_id} {mlflow_status}")
        
        return {
            'success': True,
            'message': 'DB 자동 동기화가 추가되었습니다',
            'manager_id': sync_request.manager_id,
            'mlflow_enabled': sync_request.sync_config.get('mlflow_enabled', False)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"동기화 추가 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/remove",
summary="DB 자동 동기화 제거",
description="설정된 DB 자동 동기화를 제거합니다",
response_model=Dict[str, Any])
async def remove_db_sync(request: Request, sync_request: RemoveDBSyncRequest) -> Dict[str, Any]:
    """DB 자동 동기화 제거"""
    try:
        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")
        
        # ✨ 스케줄러 가져오기 (변경)
        scheduler = get_db_sync_scheduler(request)
        
        success = scheduler.remove_db_sync(sync_request.manager_id, user_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="동기화 설정을 찾을 수 없습니다")
        
        logger.info(f"✅ DB 동기화 제거: manager={sync_request.manager_id}")
        
        return {
            'success': True,
            'message': 'DB 자동 동기화가 제거되었습니다',
            'manager_id': sync_request.manager_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ DB 동기화 제거 실패: {e}")
        raise HTTPException(status_code=500, detail=f"DB 동기화 제거 실패: {str(e)}")

@router.post("/pause",
summary="DB 자동 동기화 일시 중지",
description="실행 중인 DB 자동 동기화를 일시 중지합니다",
response_model=Dict[str, Any])
async def pause_db_sync(request: Request, sync_request: ControlDBSyncRequest) -> Dict[str, Any]:
    """DB 자동 동기화 일시 중지"""
    try:
        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")
        
        scheduler = get_db_sync_scheduler(request)
        
        success = scheduler.pause_db_sync(sync_request.manager_id, user_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="동기화 설정을 찾을 수 없습니다")
        
        logger.info(f"⏸️  DB 동기화 일시 중지: manager={sync_request.manager_id}")
        
        # ✨ 업데이트된 상태 반환
        updated_status = scheduler.get_sync_status(sync_request.manager_id, user_id)
        
        return {
            'success': True,
            'message': 'DB 자동 동기화가 일시 중지되었습니다',
            'manager_id': sync_request.manager_id,
            'status': updated_status  # ✨ 추가
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"동기화 일시 중지 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/resume",
summary="DB 자동 동기화 재개",
description="일시 중지된 DB 자동 동기화를 재개합니다",
response_model=Dict[str, Any])
async def resume_db_sync(request: Request, sync_request: ControlDBSyncRequest) -> Dict[str, Any]:
    """DB 자동 동기화 재개"""
    try:
        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")
        
        scheduler = get_db_sync_scheduler(request)
        
        success = scheduler.resume_db_sync(sync_request.manager_id, user_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="동기화 설정을 찾을 수 없습니다")
        
        logger.info(f"▶️  DB 동기화 재개: manager={sync_request.manager_id}")
        
        # ✨ 업데이트된 상태 반환
        updated_status = scheduler.get_sync_status(sync_request.manager_id, user_id)
        
        return {
            'success': True,
            'message': 'DB 자동 동기화가 재개되었습니다',
            'manager_id': sync_request.manager_id,
            'status': updated_status  # ✨ 추가
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"동기화 재개 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/status",
summary="DB 동기화 상태 조회",
description="특정 매니저의 DB 동기화 상태를 조회합니다",
response_model=Dict[str, Any])
async def get_sync_status(request: Request, status_request: GetSyncStatusRequest) -> Dict[str, Any]:
    """DB 동기화 상태 조회"""
    try:
        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")
        
        scheduler = get_db_sync_scheduler(request)
        
        status = scheduler.get_sync_status(status_request.manager_id, user_id)
        
        if not status:
            # ✨ 404 대신 success: false로 반환 (프론트에서 처리 용이)
            return {
                'success': False,
                'message': '동기화 설정을 찾을 수 없습니다',
                'status': None
            }
        
        # ✨ 다음 실행 시간 포맷팅 추가
        if status.get('next_run_time'):
            try:
                # ISO 형식으로 변환
                status['next_run_time'] = status['next_run_time']
            except Exception as e:
                logger.warning(f"다음 실행 시간 파싱 실패: {e}")
                status['next_run_time'] = None
        
        return {
            'success': True,
            'status': status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"동기화 상태 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list",
summary="모든 DB 동기화 목록 조회",
description="사용자의 모든 DB 동기화 설정 목록을 조회합니다",
response_model=Dict[str, Any])
async def list_all_syncs(request: Request) -> Dict[str, Any]:
    """모든 DB 동기화 목록 조회"""
    try:
        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")
        
        # ✨ 스케줄러 가져오기 (변경)
        scheduler = get_db_sync_scheduler(request)
        
        syncs = scheduler.list_all_syncs(user_id)
        
        return {
            'success': True,
            'total_count': len(syncs),
            'syncs': syncs
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"동기화 목록 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/trigger",
summary="수동으로 즉시 동기화 실행",
description="스케줄과 관계없이 즉시 동기화를 실행합니다",
response_model=Dict[str, Any])
async def trigger_manual_sync(request: Request, sync_request: ManualSyncRequest) -> Dict[str, Any]:
    """수동으로 즉시 동기화 실행"""
    try:
        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")
        
        scheduler = get_db_sync_scheduler(request)
        
        result = scheduler.trigger_manual_sync(sync_request.manager_id, user_id)

        logger.info(f"✅ 수동 동기화 실행: manager={sync_request.manager_id}")

        # ✨ 실행 결과 반환 (trigger_manual_sync가 이미 올바른 형식으로 반환)
        result['manager_id'] = sync_request.manager_id
        result['timestamp'] = now_kst_iso()
        return result
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"수동 동기화 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"수동 동기화 실패: {str(e)}")

@router.get("/health",
summary="스케줄러 상태 확인",
description="DB 동기화 스케줄러의 상태를 확인합니다")
async def scheduler_health_check(request: Request):
    """스케줄러 상태 확인"""
    try:
        # ✨ 스케줄러 가져오기 (변경) - health check는 예외 발생 안함
        try:
            scheduler = get_db_sync_scheduler(request)
        except HTTPException:
            return {
                "status": "unavailable",
                "message": "스케줄러가 초기화되지 않았습니다"
            }
        
        return {
            "status": "healthy" if scheduler.scheduler.running else "stopped",
            "service": "db-sync-scheduler",
            "running": scheduler.scheduler.running,
            "total_syncs": len(scheduler.sync_configs),
            "active_jobs": len(scheduler.scheduler.get_jobs())
        }
        
    except Exception as e:
        logger.error(f"Health check 실패: {e}")
        raise HTTPException(status_code=503, detail="스케줄러 상태 확인 실패")

@router.post("/update-mlflow-config",
summary="MLflow 설정 업데이트",
description="DB 동기화의 MLflow 자동 업로드 설정을 업데이트합니다",
response_model=Dict[str, Any])
async def update_mlflow_config(request: Request, update_request: UpdateMLflowConfigRequest) -> Dict[str, Any]:
    """MLflow 설정 업데이트"""
    try:
        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")
        
        from service.database.models.db_sync_config import DBSyncConfig
        
        app_db = get_app_db(request)
        
        # DB에서 설정 조회
        db_configs = app_db.find_by_condition(
            DBSyncConfig,
            {'manager_id': update_request.manager_id, 'user_id': user_id},
            limit=1
        )
        
        if not db_configs:
            raise HTTPException(status_code=404, detail="동기화 설정을 찾을 수 없습니다")
        
        db_config = db_configs[0]
        
        # MLflow 활성화 시 실험 이름 필수 체크
        if update_request.mlflow_enabled and not update_request.mlflow_experiment_name:
            raise HTTPException(status_code=400, detail="MLflow 활성화 시 실험 이름은 필수입니다")
        
        # 설정 업데이트
        db_config.mlflow_enabled = update_request.mlflow_enabled
        db_config.mlflow_experiment_name = update_request.mlflow_experiment_name
        db_config.mlflow_tracking_uri = update_request.mlflow_tracking_uri
        db_config.updated_at = now_kst_iso()
        
        app_db.update(db_config)
        
        action = "활성화" if update_request.mlflow_enabled else "비활성화"
        logger.info(f"🔄 MLflow 자동 업로드 {action}: manager={update_request.manager_id}, experiment={update_request.mlflow_experiment_name}")
        
        return {
            'success': True,
            'message': f'MLflow 자동 업로드가 {action}되었습니다',
            'manager_id': update_request.manager_id,
            'mlflow_info': db_config.get_mlflow_info()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MLflow 설정 업데이트 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ✨ MLflow 업로드 이력 조회 엔드포인트
@router.post("/mlflow-upload-history",
summary="MLflow 업로드 이력 조회",
description="특정 매니저의 MLflow 업로드 이력을 조회합니다",
response_model=Dict[str, Any])
async def get_mlflow_upload_history(request: Request, status_request: GetSyncStatusRequest) -> Dict[str, Any]:
    """MLflow 업로드 이력 조회"""
    try:
        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")
        
        from service.database.models.db_sync_config import DBSyncConfig
        
        app_db = get_app_db(request)
        
        db_configs = app_db.find_by_condition(
            DBSyncConfig,
            {'manager_id': status_request.manager_id, 'user_id': user_id},
            limit=1
        )
        
        if not db_configs:
            raise HTTPException(status_code=404, detail="동기화 설정을 찾을 수 없습니다")
        
        db_config = db_configs[0]
        
        return {
            'success': True,
            'manager_id': status_request.manager_id,
            'mlflow_info': db_config.get_mlflow_info()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MLflow 이력 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))
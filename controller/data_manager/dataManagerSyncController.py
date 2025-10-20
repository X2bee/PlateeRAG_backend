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

router = APIRouter(prefix="/sync", tags=["data-manager-sync"])
logger = logging.getLogger("data-manager-sync-controller")

#========== Request Models ==========
class AddDBSyncRequest(BaseModel):
    """DB 자동 동기화 추가 요청"""
    manager_id: str = Field(..., description="Manager ID")
    db_config: Dict[str, Any] = Field(..., description="DB 연결 설정")
    sync_config: Dict[str, Any] = Field(..., description="동기화 설정")

    class Config:
        schema_extra = {
            "example": {
                "manager_id": "mgr_abc123",
                "db_config": {
                    "db_type": "postgresql",
                    "host": "localhost",
                    "port": 5432,
                    "database": "mydb",
                    "username": "user",
                    "password": "password"
                },
                "sync_config": {
                    "enabled": True,
                    "schedule_type": "interval",
                    "interval_minutes": 30,
                    "table_name": "users",
                    "detect_changes": True,
                    "notification_enabled": False
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

#========== API Endpoints ==========
@router.post("/add",
summary="DB 자동 동기화 추가",
description="외부 DB에서 주기적으로 데이터를 가져오는 자동 동기화 설정",
response_model=Dict[str, Any])
async def add_db_sync(request: Request, sync_request: AddDBSyncRequest) -> Dict[str, Any]:
    """DB 자동 동기화 추가"""
    try:
        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")
        
        # DB 설정 검증
        try:
            validate_db_config(sync_request.db_config)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # 동기화 설정 검증
        sync_config = sync_request.sync_config
        
        if sync_config.get('schedule_type') not in ['interval', 'cron']:
            raise HTTPException(
                status_code=400,
                detail="schedule_type은 'interval' 또는 'cron'이어야 합니다"
            )
        
        if sync_config.get('schedule_type') == 'interval':
            if not sync_config.get('interval_minutes'):
                raise HTTPException(
                    status_code=400,
                    detail="interval_minutes가 필요합니다"
                )
            if sync_config.get('interval_minutes') < 1:
                raise HTTPException(
                    status_code=400,
                    detail="interval_minutes는 1 이상이어야 합니다"
                )
        elif sync_config.get('schedule_type') == 'cron':
            if not sync_config.get('cron_expression'):
                raise HTTPException(
                    status_code=400,
                    detail="cron_expression이 필요합니다"
                )
        
        if not sync_config.get('query') and not sync_config.get('table_name'):
            raise HTTPException(
                status_code=400,
                detail="query 또는 table_name 중 하나는 필수입니다"
            )
        
        # ✨ 스케줄러 가져오기 (변경)
        scheduler = get_db_sync_scheduler(request)
        
        # 동기화 추가
        result = scheduler.add_db_sync(
            manager_id=sync_request.manager_id,
            user_id=user_id,
            db_config=sync_request.db_config,
            sync_config=sync_config
        )
        
        logger.info(f"✅ DB 자동 동기화 추가: manager={sync_request.manager_id}, user={user_id}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ DB 동기화 추가 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"DB 동기화 추가 실패: {str(e)}")

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
        
        # ✨ 스케줄러 가져오기 (변경)
        scheduler = get_db_sync_scheduler(request)
        
        success = scheduler.pause_db_sync(sync_request.manager_id, user_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="동기화 설정을 찾을 수 없습니다")
        
        logger.info(f"⏸️  DB 동기화 일시 중지: manager={sync_request.manager_id}")
        
        return {
            'success': True,
            'message': 'DB 자동 동기화가 일시 중지되었습니다',
            'manager_id': sync_request.manager_id
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
        
        # ✨ 스케줄러 가져오기 (변경)
        scheduler = get_db_sync_scheduler(request)
        
        success = scheduler.resume_db_sync(sync_request.manager_id, user_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="동기화 설정을 찾을 수 없습니다")
        
        logger.info(f"▶️  DB 동기화 재개: manager={sync_request.manager_id}")
        
        return {
            'success': True,
            'message': 'DB 자동 동기화가 재개되었습니다',
            'manager_id': sync_request.manager_id
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
        
        # ✨ 스케줄러 가져오기 (변경)
        scheduler = get_db_sync_scheduler(request)
        
        status = scheduler.get_sync_status(status_request.manager_id, user_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="동기화 설정을 찾을 수 없습니다")
        
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
        
        # ✨ 스케줄러 가져오기 (변경)
        scheduler = get_db_sync_scheduler(request)
        
        result = scheduler.trigger_manual_sync(sync_request.manager_id, user_id)
        
        logger.info(f"✅ 수동 동기화 실행: manager={sync_request.manager_id}")
        
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
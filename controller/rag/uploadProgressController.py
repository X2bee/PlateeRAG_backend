"""
업로드 진행 상태 조회 API
"""
from fastapi import APIRouter, HTTPException, Request
from typing import Optional
import logging

from service.retrieval.upload_progress_manager import upload_progress_manager
from controller.helper.controllerHelper import extract_user_id_from_request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/retrieval", tags=["upload-progress"])


@router.get("/upload/progress/{task_id}")
async def get_upload_progress(request: Request, task_id: str):
    """
    업로드 진행 상태 조회

    Args:
        task_id: 업로드 작업 ID

    Returns:
        진행 상태 정보
    """
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    try:
        progress = upload_progress_manager.get_progress(task_id)

        if not progress:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        # 권한 확인 (작업 소유자만 조회 가능) - 타입 변환 후 비교
        if int(progress['user_id']) != int(user_id):
            logger.warning(f"Access denied: progress user_id={progress['user_id']} (type={type(progress['user_id'])}), request user_id={user_id} (type={type(user_id)})")
            raise HTTPException(status_code=403, detail="Access denied")

        # 진행 상태 로그 출력
        logger.info(f"📊 Progress response for task {task_id[:8]}: status={progress['status']}, progress={progress['progress']}%, step={progress.get('current_step', 'N/A')}")

        return progress

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get upload progress for task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get progress: {str(e)}")


@router.get("/upload/progress")
async def get_user_upload_tasks(request: Request):
    """
    사용자의 모든 업로드 작업 조회

    Returns:
        사용자의 모든 업로드 작업 목록
    """
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    try:
        tasks = upload_progress_manager.get_user_tasks(user_id)
        return {"tasks": tasks, "count": len(tasks)}

    except Exception as e:
        logger.error(f"Failed to get user upload tasks for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get tasks: {str(e)}")


@router.post("/upload/progress/{task_id}/cancel")
async def cancel_upload_task(request: Request, task_id: str):
    """
    업로드 작업 취소

    Args:
        task_id: 업로드 작업 ID

    Returns:
        취소 결과
    """
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    try:
        progress = upload_progress_manager.get_progress(task_id)

        if not progress:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        # 권한 확인
        if int(progress['user_id']) != int(user_id):
            raise HTTPException(status_code=403, detail="Access denied")

        # 작업 취소
        success = upload_progress_manager.cancel_task(task_id)

        if not success:
            raise HTTPException(status_code=400, detail="Cannot cancel task (already completed or error)")

        logger.info(f"Upload task {task_id} cancelled by user {user_id}")
        return {"message": "Task cancelled successfully", "task_id": task_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel upload task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel task: {str(e)}")


@router.delete("/upload/progress/{task_id}")
async def delete_upload_task(request: Request, task_id: str):
    """
    업로드 작업 삭제 (완료 또는 에러 상태만 삭제 가능)

    Args:
        task_id: 업로드 작업 ID
    """
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    try:
        progress = upload_progress_manager.get_progress(task_id)

        if not progress:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        # 권한 확인
        if progress['user_id'] != user_id:
            raise HTTPException(status_code=403, detail="Access denied")

        # 진행 중인 작업은 삭제 불가
        if progress['status'] not in ['completed', 'error']:
            raise HTTPException(status_code=400, detail="Cannot delete task in progress")

        upload_progress_manager.delete_task(task_id)

        return {"message": "Task deleted successfully", "task_id": task_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete upload task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete task: {str(e)}")

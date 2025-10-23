"""
업로드 진행 상태를 추적하는 관리자

메모리 기반으로 작동하며, 각 업로드 작업의 진행 상태를 저장합니다.
"""
import time
import uuid
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict
from threading import Lock
import logging

logger = logging.getLogger(__name__)


@dataclass
class UploadProgress:
    """업로드 진행 상태"""
    task_id: str
    user_id: int
    collection_name: str
    status: str  # 'initializing', 'downloading', 'annotating', 'extracting', 'chunking', 'embedding', 'storing', 'completed', 'error'
    progress: float  # 0.0 ~ 100.0
    current_step: str
    total_steps: int
    current_step_number: int
    message: str
    error: Optional[str] = None

    # 세부 정보
    repository_path: Optional[str] = None
    branch: Optional[str] = None
    total_files: int = 0
    processed_files: int = 0
    current_file: Optional[str] = None

    # 타임스탬프
    created_at: float = None
    updated_at: float = None

    # 취소 플래그
    cancelled: bool = False

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.updated_at is None:
            self.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return asdict(self)


class UploadProgressManager:
    """업로드 진행 상태 관리자 (Singleton)"""

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._progress_store: Dict[str, UploadProgress] = {}
        self._store_lock = Lock()
        self._initialized = True
        logger.info("UploadProgressManager initialized")

    def create_task(
        self,
        user_id: int,
        collection_name: str,
        repository_path: Optional[str] = None,
        branch: Optional[str] = None
    ) -> str:
        """새로운 업로드 작업 생성"""
        task_id = str(uuid.uuid4())

        progress = UploadProgress(
            task_id=task_id,
            user_id=user_id,
            collection_name=collection_name,
            status='initializing',
            progress=0.0,
            current_step='Initializing upload...',
            total_steps=7,  # 총 단계 수
            current_step_number=0,
            message='Upload task created',
            repository_path=repository_path,
            branch=branch
        )

        with self._store_lock:
            self._progress_store[task_id] = progress

        logger.info(f"Created upload task: {task_id} for user {user_id}")
        return task_id

    def update_progress(
        self,
        task_id: str,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        current_step: Optional[str] = None,
        current_step_number: Optional[int] = None,
        message: Optional[str] = None,
        total_files: Optional[int] = None,
        processed_files: Optional[int] = None,
        current_file: Optional[str] = None,
        error: Optional[str] = None
    ):
        """진행 상태 업데이트"""
        with self._store_lock:
            if task_id not in self._progress_store:
                logger.warning(f"Task {task_id} not found in progress store")
                return

            progress_obj = self._progress_store[task_id]

            if status is not None:
                progress_obj.status = status
            if progress is not None:
                progress_obj.progress = min(100.0, max(0.0, progress))
            if current_step is not None:
                progress_obj.current_step = current_step
            if current_step_number is not None:
                progress_obj.current_step_number = current_step_number
            if message is not None:
                progress_obj.message = message
            if total_files is not None:
                progress_obj.total_files = total_files
            if processed_files is not None:
                progress_obj.processed_files = processed_files
            if current_file is not None:
                progress_obj.current_file = current_file
            if error is not None:
                progress_obj.error = error
                progress_obj.status = 'error'

            progress_obj.updated_at = time.time()

        logger.info(f"📊 Progress update [{task_id[:8]}]: {status} - {progress}% - {current_step}")

    def get_progress(self, task_id: str) -> Optional[Dict[str, Any]]:
        """진행 상태 조회"""
        with self._store_lock:
            if task_id not in self._progress_store:
                return None
            return self._progress_store[task_id].to_dict()

    def cancel_task(self, task_id: str) -> bool:
        """작업 취소"""
        with self._store_lock:
            if task_id not in self._progress_store:
                logger.warning(f"Task {task_id} not found for cancellation")
                return False

            progress_obj = self._progress_store[task_id]

            # 이미 완료되었거나 에러 상태면 취소 불가
            if progress_obj.status in ['completed', 'error']:
                logger.warning(f"Cannot cancel task {task_id} with status: {progress_obj.status}")
                return False

            progress_obj.cancelled = True
            progress_obj.status = 'error'
            progress_obj.error = 'Upload cancelled by user'
            progress_obj.updated_at = time.time()

            logger.info(f"Task {task_id} cancelled by user")
            return True

    def is_cancelled(self, task_id: str) -> bool:
        """작업 취소 여부 확인"""
        with self._store_lock:
            if task_id not in self._progress_store:
                return False
            return self._progress_store[task_id].cancelled

    def delete_task(self, task_id: str):
        """작업 삭제"""
        with self._store_lock:
            if task_id in self._progress_store:
                del self._progress_store[task_id]
                logger.info(f"Deleted task: {task_id}")

    def cleanup_old_tasks(self, max_age_seconds: int = 3600):
        """오래된 작업 정리 (기본 1시간)"""
        current_time = time.time()

        with self._store_lock:
            tasks_to_delete = []
            for task_id, progress in self._progress_store.items():
                age = current_time - progress.created_at
                if age > max_age_seconds:
                    tasks_to_delete.append(task_id)

            for task_id in tasks_to_delete:
                del self._progress_store[task_id]
                logger.info(f"Cleaned up old task: {task_id}")

        if tasks_to_delete:
            logger.info(f"Cleaned up {len(tasks_to_delete)} old tasks")

    def get_user_tasks(self, user_id: int) -> list:
        """사용자의 모든 작업 조회"""
        with self._store_lock:
            return [
                progress.to_dict()
                for progress in self._progress_store.values()
                if progress.user_id == user_id
            ]


# Singleton 인스턴스
upload_progress_manager = UploadProgressManager()

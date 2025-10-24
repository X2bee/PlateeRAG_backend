"""
SSE 세션 관리자

각 문서 업로드 세션을 격리하고 병렬 처리를 지원합니다.
"""
import asyncio
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import os

logger = logging.getLogger("sse-session-manager")
TIMEZONE = ZoneInfo(os.getenv('TIMEZONE', 'Asia/Seoul'))


class SSESession:
    """개별 SSE 업로드 세션"""

    def __init__(self, session_id: str, user_id: int, collection_name: str, filename: str):
        self.session_id = session_id
        self.user_id = int(user_id)  # 명시적으로 int 타입으로 저장
        self.collection_name = collection_name
        self.filename = filename
        self.created_at = datetime.now(TIMEZONE)
        self.updated_at = datetime.now(TIMEZONE)
        self.status = "initializing"  # initializing, processing, completed, error, cancelled
        self.progress_queue = asyncio.Queue()
        self.task: Optional[asyncio.Task] = None
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None

        logger.debug(f"SSESession created: id={session_id}, user_id={self.user_id} ({type(self.user_id)})")

    def update_status(self, status: str):
        """상태 업데이트"""
        self.status = status
        self.updated_at = datetime.now(TIMEZONE)

    async def put_event(self, event_data: Dict[str, Any]):
        """진행 상황 이벤트 추가"""
        await self.progress_queue.put(event_data)
        self.updated_at = datetime.now(TIMEZONE)

    async def get_event(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """진행 상황 이벤트 가져오기"""
        try:
            event = await asyncio.wait_for(self.progress_queue.get(), timeout=timeout)
            return event
        except asyncio.TimeoutError:
            return None

    def cancel(self):
        """세션 취소"""
        if self.task and not self.task.done():
            self.task.cancel()
        self.update_status("cancelled")

    def is_active(self) -> bool:
        """세션이 활성 상태인지 확인"""
        return self.status in ["initializing", "processing"]

    def is_expired(self, timeout_minutes: int = 60) -> bool:
        """세션이 만료되었는지 확인"""
        return datetime.now(TIMEZONE) - self.updated_at > timedelta(minutes=timeout_minutes)


class SSESessionManager:
    """SSE 세션 관리자 (싱글톤)"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.sessions: Dict[str, SSESession] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._initialized = True
        logger.info("SSESessionManager initialized")

    async def create_session(
        self,
        session_id: str,
        user_id: int,
        collection_name: str,
        filename: str
    ) -> SSESession:
        """새 세션 생성"""
        async with self._lock:
            if session_id in self.sessions:
                # 기존 세션이 있으면 취소
                old_session = self.sessions[session_id]
                old_session.cancel()
                logger.warning(f"Replacing existing session: {session_id}")

            session = SSESession(session_id, user_id, collection_name, filename)
            self.sessions[session_id] = session

            logger.info(f"Created session: {session_id} (user: {user_id}, file: {filename})")
            return session

    async def get_session(self, session_id: str) -> Optional[SSESession]:
        """세션 조회"""
        async with self._lock:
            return self.sessions.get(session_id)

    async def remove_session(self, session_id: str) -> bool:
        """세션 제거"""
        async with self._lock:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                session.cancel()
                del self.sessions[session_id]
                logger.info(f"Removed session: {session_id}")
                return True
            return False

    async def cancel_session(self, session_id: str) -> bool:
        """세션 취소"""
        session = await self.get_session(session_id)
        if session:
            session.cancel()
            logger.info(f"Cancelled session: {session_id}")
            return True
        return False

    async def cleanup_expired_sessions(self, timeout_minutes: int = 60):
        """만료된 세션 정리"""
        async with self._lock:
            expired = []
            for session_id, session in self.sessions.items():
                if session.is_expired(timeout_minutes):
                    expired.append(session_id)

            for session_id in expired:
                session = self.sessions[session_id]
                session.cancel()
                del self.sessions[session_id]
                logger.info(f"Cleaned up expired session: {session_id}")

            if expired:
                logger.info(f"Cleaned up {len(expired)} expired sessions")

    def start_cleanup_task(self, interval_minutes: int = 10):
        """자동 정리 태스크 시작"""
        if self._cleanup_task and not self._cleanup_task.done():
            return

        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(interval_minutes * 60)
                    await self.cleanup_expired_sessions()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in cleanup task: {e}")

        self._cleanup_task = asyncio.create_task(cleanup_loop())
        logger.info(f"Started cleanup task (interval: {interval_minutes} minutes)")

    def stop_cleanup_task(self):
        """자동 정리 태스크 중지"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            logger.info("Stopped cleanup task")

    async def get_active_sessions_count(self) -> int:
        """활성 세션 개수"""
        async with self._lock:
            return sum(1 for s in self.sessions.values() if s.is_active())

    async def get_all_sessions_info(self) -> Dict[str, Any]:
        """모든 세션 정보"""
        async with self._lock:
            return {
                "total": len(self.sessions),
                "active": sum(1 for s in self.sessions.values() if s.is_active()),
                "sessions": {
                    sid: {
                        "user_id": s.user_id,
                        "filename": s.filename,
                        "collection_name": s.collection_name,
                        "status": s.status,
                        "created_at": s.created_at.isoformat(),
                        "updated_at": s.updated_at.isoformat()
                    }
                    for sid, s in self.sessions.items()
                }
            }


# 싱글톤 인스턴스
sse_session_manager = SSESessionManager()

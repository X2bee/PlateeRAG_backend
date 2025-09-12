import logging
import inspect
from pydantic import BaseModel
from typing import Dict, Optional, Any
from fastapi import Request
from service.database.connection import AppDatabaseManager
from service.database.models.backend import BackendLogs

logger = logging.getLogger("backend-logger")

class LogData(BaseModel):
    user_id: Optional[int]
    log_level: str
    message: str
    function_name: Optional[str] = ''
    api_endpoint: Optional[str] = ''
    metadata: Optional[Dict] = {}

class BackendLogger:
    """고도화된 백엔드 로거 클래스"""

    def __init__(self, app_db: AppDatabaseManager, user_id: Optional[int] = None, request: Optional[Request] = None):
        self.app_db = app_db
        self.user_id = user_id
        self.request = request
        self._function_name = None
        self._api_endpoint = None

        # 자동으로 함수명과 API 엔드포인트 추출
        self._extract_context_info()

    def _extract_context_info(self):
        """호출된 컨텍스트에서 API 엔드포인트를 자동 추출"""
        try:
            # Request 객체에서 API 엔드포인트 추출
            if self.request:
                # FastAPI의 경우 request.url.path에서 엔드포인트 추출
                path = self.request.url.path
                # '/api/v1/manager/group/all-groups' -> 'manager/group/all-groups'
                if path.startswith('/api/v1/'):
                    self._api_endpoint = path[8:]  # '/api/v1/' 제거
                elif path.startswith('/'):
                    self._api_endpoint = path[1:]  # '/' 제거
                else:
                    self._api_endpoint = path

        except Exception as e:
            logger.warning(f"Could not extract context info: {str(e)}")

    def _log(self, level: str, message: str, metadata: Optional[Dict] = None,
             function_name: Optional[str] = None, api_endpoint: Optional[str] = None):
        """내부 로깅 메서드"""
        try:
            # 파라미터가 제공되지 않으면 자동 추출된 값 사용
            func_name = function_name or self._function_name or ''
            endpoint = api_endpoint or self._api_endpoint or ''

            log_id = f"LOG__{self.user_id}__{func_name}"
            log_entry = BackendLogs(
                user_id=self.user_id,
                log_id=log_id,
                log_level=level,
                message=message,
                function_name=func_name,
                api_endpoint=endpoint,
                metadata=metadata or {}
            )
            self.app_db.insert(log_entry)
            logger.info(f"Logged backend data with log_id: {log_id}")

        except Exception as e:
            logger.error(f"Error logging backend data: {str(e)}")

    def success(self, message: str, metadata: Optional[Dict] = None,
                function_name: Optional[str] = None, api_endpoint: Optional[str] = None):
        """성공 로그 기록"""
        self._log("INFO", f"SUCCESS: {message}", metadata, function_name, api_endpoint)

    def info(self, message: str, metadata: Optional[Dict] = None,
             function_name: Optional[str] = None, api_endpoint: Optional[str] = None):
        """정보 로그 기록"""
        self._log("INFO", message, metadata, function_name, api_endpoint)

    def warn(self, message: str, metadata: Optional[Dict] = None,
             function_name: Optional[str] = None, api_endpoint: Optional[str] = None):
        """경고 로그 기록"""
        self._log("WARN", message, metadata, function_name, api_endpoint)

    def warning(self, message: str, metadata: Optional[Dict] = None,
                function_name: Optional[str] = None, api_endpoint: Optional[str] = None):
        """경고 로그 기록 (warn의 별칭)"""
        self.warn(message, metadata, function_name, api_endpoint)

    def error(self, message: str, metadata: Optional[Dict] = None,
              function_name: Optional[str] = None, api_endpoint: Optional[str] = None,
              exception: Optional[Exception] = None):
        """에러 로그 기록"""
        error_message = message
        if exception:
            error_message = f"{message}: {str(exception)}"
            if metadata is None:
                metadata = {}
            metadata['exception_type'] = type(exception).__name__
            metadata['exception_details'] = str(exception)

        self._log("ERROR", error_message, metadata, function_name, api_endpoint)

    def debug(self, message: str, metadata: Optional[Dict] = None,
              function_name: Optional[str] = None, api_endpoint: Optional[str] = None):
        """디버그 로그 기록"""
        self._log("DEBUG", message, metadata, function_name, api_endpoint)

def create_logger(app_db: AppDatabaseManager, user_id: Optional[int] = None,
                 request: Optional[Request] = None) -> BackendLogger:
    """백엔드 로거 생성 팩토리 함수"""
    # 호출한 함수의 이름을 자동으로 추출
    caller_function_name = None
    try:
        frame = inspect.currentframe()
        if frame and frame.f_back:
            caller_function_name = frame.f_back.f_code.co_name
    except Exception:
        pass

    logger_instance = BackendLogger(app_db, user_id, request)

    # 자동 추출된 함수명으로 덮어쓰기
    if caller_function_name:
        logger_instance._function_name = caller_function_name

    return logger_instance

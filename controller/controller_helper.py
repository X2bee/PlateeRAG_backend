"""
Controller Helper Functions
컨트롤러에서 공통으로 사용하는 헬퍼 함수들을 정의합니다.
"""

from fastapi import HTTPException, Request
from typing import Optional, Dict, Any
from service.database.models.user import User
import logging

# authController에서 사용하는 함수들을 import
from controller.authController import get_user_by_token

logger = logging.getLogger("controller-helper")


def validate_user_authentication(request: Request) -> Dict[str, Any]:
    """
    요청에서 토큰을 추출하고 검증하여 사용자 정보를 반환합니다.

    Args:
        request: FastAPI Request 객체

    Returns:
        Dict[str, Any]: 검증된 사용자 정보

    Raises:
        HTTPException: 인증 실패 시 발생
    """
    try:
        # Authorization 헤더에서 토큰 추출
        authorization = request.headers.get("Authorization")
        if not authorization:
            raise HTTPException(
                status_code=401,
                detail="Authorization header is missing"
            )

        # Bearer 토큰 형식 확인
        if not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=401,
                detail="Invalid authorization header format. Expected 'Bearer <token>'"
            )

        # 토큰 추출
        token = authorization.replace("Bearer ", "").strip()
        if not token:
            raise HTTPException(
                status_code=401,
                detail="Token is missing in authorization header"
            )

        # 토큰 검증
        user_session = get_user_by_token(request.app.state, token)
        if not user_session:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired token"
            )

        # X-User-ID 헤더 검증 (선택적)
        user_id_header = request.headers.get("X-User-ID")
        if user_id_header and user_id_header != user_session['user_id']:
            raise HTTPException(
                status_code=401,
                detail="User ID in header does not match token"
            )

        logger.info("User authenticated: %s (ID: %s)", user_session['username'], user_session['user_id'])
        return user_session

    except HTTPException:
        # HTTPException은 그대로 재발생
        raise
    except Exception as e:
        logger.error("Authentication validation error: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail="Internal server error during authentication"
        ) from e


def extract_user_id_from_request(request: Request) -> str:
    """
    요청에서 사용자 ID를 추출합니다.
    먼저 토큰을 검증한 후 사용자 ID를 반환합니다.

    Args:
        request: FastAPI Request 객체

    Returns:
        str: 검증된 사용자 ID

    Raises:
        HTTPException: 인증 실패 시 발생
    """
    user_session = validate_user_authentication(request)

    return user_session['user_id']

def extract_token_from_request(request: Request) -> str:
    """
    요청에서 토큰을 추출하고 검증합니다.

    Args:
        request: FastAPI Request 객체

    Returns:
        str: 검증된 토큰

    Raises:
        HTTPException: 인증 실패 시 발생
    """
    # 토큰 검증을 수행하여 유효성 확인
    validate_user_authentication(request)

    # 검증이 성공하면 토큰 반환
    authorization = request.headers.get("Authorization")
    return authorization.replace("Bearer ", "").strip()

def require_admin_access(request: Request) -> Dict[str, Any]:
    """
    관리자 권한이 필요한 API에서 사용하는 함수입니다.
    토큰을 검증하고 관리자 권한을 확인합니다.

    Args:
        request: FastAPI Request 객체

    Returns:
        Dict[str, Any]: 검증된 관리자 사용자 정보

    Raises:
        HTTPException: 인증 실패 또는 권한 부족 시 발생
    """
    app_db = request.app.state.app_db
    user_session = validate_user_authentication(request)
    existing_data = app_db.find_by_condition(
        User,
        {
            "user_id": user_session['user_id'],
        },
        limit=1
    )

    if not existing_data or not existing_data[0].is_admin:
        raise HTTPException(
            status_code=403,
            detail="Admin access required"
        )

    logger.info("Admin access granted: %s (ID: %s)", user_session['username'], user_session['user_id'])
    return user_session


def get_optional_user_info(request: Request) -> Optional[Dict[str, Any]]:
    """
    선택적으로 사용자 정보를 가져옵니다.
    토큰이 없거나 유효하지 않아도 예외를 발생시키지 않습니다.

    Args:
        request: FastAPI Request 객체

    Returns:
        Optional[Dict[str, Any]]: 사용자 정보 (없으면 None)
    """
    try:
        return validate_user_authentication(request)
    except HTTPException:
        return None
    except Exception:
        logger.warning("Optional user info extraction failed")
        return None


def check_token_validity(token: str, app_state) -> bool:
    """
    토큰의 유효성만 검사합니다.

    Args:
        token: 검증할 토큰
        app_state: 애플리케이션 상태

    Returns:
        bool: 토큰 유효성 여부
    """
    try:
        user_session = get_user_by_token(app_state, token)
        return user_session is not None
    except Exception:
        logger.warning("Token validity check failed")
        return False


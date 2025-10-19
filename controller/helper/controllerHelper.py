"""
Controller Helper Functions
컨트롤러에서 공통으로 사용하는 헬퍼 함수들을 정의합니다.
"""

from fastapi import HTTPException, Request
from typing import Optional, Dict, Any
from service.database.models.user import User
import logging
from controller.authController import get_user_by_token
from controller.helper.singletonHelper import get_db_manager
from urllib.parse import urlparse, quote_plus
from sqlalchemy.exc import OperationalError, DatabaseError
import psycopg2

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
    user_session = validate_user_authentication(request)
    return user_session['user_id']

def extract_token_from_request(request: Request) -> str:
    validate_user_authentication(request)
    authorization = request.headers.get("Authorization")
    return authorization.replace("Bearer ", "").strip()

def require_admin_access(request: Request) -> Dict[str, Any]:
    app_db = get_db_manager(request)
    user_session = validate_user_authentication(request)
    existing_data = app_db.find_by_condition(
        User,
        {
            "id": user_session['user_id'],
        },
        limit=1
    )

    if not existing_data or not existing_data[0].is_admin:
        raise HTTPException(
            status_code=403,
            detail="Admin access required"
        )
    return user_session


def get_optional_user_info(request: Request) -> Optional[Dict[str, Any]]:
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

# Helper Functions 섹션에 추가
def parse_connection_url(url: str) -> Dict[str, Any]:
    """
    연결 URL을 파싱하여 db_config 생성
    지원 형식:
    - postgresql://user:pass@host:port/database
    - mysql://user:pass@host:port/database
    - sqlite:///path/to/database.db
    - jdbc:postgresql://host:port/database (JDBC URL도 지원)
    """
    try:
        original_url = url
        
        # JDBC URL 처리 (jdbc: 접두사 제거)
        if url.startswith('jdbc:'):
            url = url.replace('jdbc:', '', 1)
            logger.info(f"JDBC URL 감지, 변환: {url}")
        
        # URL 파싱
        parsed = urlparse(url)
        
        if not parsed.scheme:
            raise ValueError("유효하지 않은 URL 형식입니다. (스키마 누락)")
        
        # DB 타입 결정
        db_type = parsed.scheme.lower()
        if 'postgresql' in db_type or db_type == 'postgres':
            db_type = 'postgresql'
        elif 'mysql' in db_type:
            db_type = 'mysql'
        elif 'sqlite' in db_type:
            db_type = 'sqlite'
        else:
            raise ValueError(f"지원되지 않는 DB 타입: {parsed.scheme}")
        
        # 기본 설정
        config = {
            'db_type': db_type
        }
        
        # SQLite 처리
        if db_type == 'sqlite':
            # sqlite:///path/to/db.sqlite 형식
            if parsed.path:
                config['database'] = parsed.path.lstrip('/')
            else:
                raise ValueError("SQLite 데이터베이스 경로가 필요합니다")
        else:
            # PostgreSQL, MySQL 처리
            if not parsed.hostname:
                raise ValueError("호스트 주소가 필요합니다")
            
            config['host'] = parsed.hostname
            config['port'] = parsed.port if parsed.port else (3306 if db_type == 'mysql' else 5432)
            config['username'] = parsed.username if parsed.username else ''
            config['password'] = parsed.password if parsed.password else ''
            
            # 데이터베이스명 추출
            if parsed.path:
                database = parsed.path.lstrip('/')
                # 쿼리 파라미터 제거
                if '?' in database:
                    database = database.split('?')[0]
                config['database'] = database
            else:
                config['database'] = ''
        
        logger.info(f"URL 파싱 성공: {db_type}://{config.get('username', '')}@{config.get('host', '')}:{config.get('port', '')}/{config.get('database', '')}")
        
        return config
        
    except Exception as e:
        logger.error(f"URL 파싱 실패: {original_url}, error: {str(e)}", exc_info=True)
        raise ValueError(f"URL 파싱 실패: {str(e)}")

def validate_db_config(db_config: Dict[str, Any]) -> None:
    """데이터베이스 설정 검증"""
    if not db_config.get('db_type'):
        raise ValueError("DB 타입이 필요합니다")
    
    if not db_config.get('database'):
        raise ValueError("데이터베이스명이 필요합니다")
    
    # SQLite가 아닌 경우
    if db_config['db_type'] != 'sqlite':
        if not db_config.get('host'):
            raise ValueError("호스트 주소가 필요합니다")
        if not db_config.get('username'):
            raise ValueError("사용자명이 필요합니다")
        if not db_config.get('password'):
            raise ValueError("비밀번호가 필요합니다")
            

def create_db_connection_string(db_config: Dict[str, Any]) -> str:
    """
    DB 설정으로부터 연결 문자열 생성 (특수문자 안전 처리)
    """
    db_type = db_config.get('db_type', 'postgresql').lower()
    
    if db_type == 'sqlite':
        return f"sqlite:///{db_config['database']}"
    
    # username과 password를 URL 인코딩
    username = quote_plus(db_config.get('username', ''))
    password = quote_plus(db_config.get('password', ''))
    host = db_config['host']
    database = db_config['database']
    
    if db_type == 'postgresql':
        port = db_config.get('port', 5432)
        return f"postgresql://{username}:{password}@{host}:{port}/{database}"
    
    elif db_type == 'mysql':
        port = db_config.get('port', 3306)
        return f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
    
    else:
        raise ValueError(f"지원되지 않는 DB 타입: {db_type}")

def parse_db_error(error: Exception, db_config: Dict[str, Any]) -> Dict[str, str]:
    """데이터베이스 에러를 사용자 친화적 메시지로 변환"""
    error_str = str(error).lower()
    db_type = db_config.get('db_type', 'postgresql')
    host = db_config.get('host', 'unknown')
    port = db_config.get('port', 'unknown')
    database = db_config.get('database', 'unknown')
    username = db_config.get('username', 'unknown')
    
    # 연결 거부 (Connection Refused)
    if 'connection refused' in error_str:
        return {
            "error_type": "CONNECTION_REFUSED",
            "user_message": f"데이터베이스 서버에 연결할 수 없습니다.",
            "details": f"서버: {host}:{port}",
            "suggestions": [
                "1. 데이터베이스 서버가 실행 중인지 확인하세요",
                "2. 호스트 주소와 포트 번호가 올바른지 확인하세요",
                "3. 방화벽 설정을 확인하세요",
                f"4. 서버가 {port} 포트에서 연결을 수신하는지 확인하세요"
            ]
        }
    
    # 인증 실패 (Authentication Failed)
    if 'password authentication failed' in error_str or 'access denied' in error_str:
        return {
            "error_type": "AUTHENTICATION_FAILED",
            "user_message": f"데이터베이스 인증에 실패했습니다.",
            "details": f"사용자: {username}, 데이터베이스: {database}",
            "suggestions": [
                "1. 사용자 이름이 올바른지 확인하세요",
                "2. 비밀번호가 올바른지 확인하세요",
                "3. 해당 사용자가 데이터베이스에 접근 권한이 있는지 확인하세요",
                "4. 대소문자를 정확히 입력했는지 확인하세요"
            ]
        }
    
    # 데이터베이스 없음
    if 'database' in error_str and ('does not exist' in error_str or 'unknown database' in error_str):
        return {
            "error_type": "DATABASE_NOT_FOUND",
            "user_message": f"데이터베이스를 찾을 수 없습니다.",
            "details": f"데이터베이스: {database}",
            "suggestions": [
                "1. 데이터베이스 이름의 철자를 확인하세요",
                "2. 데이터베이스가 생성되어 있는지 확인하세요",
                "3. 대소문자를 정확히 입력했는지 확인하세요"
            ]
        }
    
    # 호스트 해석 불가
    if 'could not translate host name' in error_str or 'name or service not known' in error_str:
        return {
            "error_type": "HOST_NOT_FOUND",
            "user_message": f"호스트 주소를 찾을 수 없습니다.",
            "details": f"호스트: {host}",
            "suggestions": [
                "1. 호스트 주소의 철자를 확인하세요",
                "2. DNS 설정을 확인하세요",
                "3. 인터넷 연결을 확인하세요",
                "4. IP 주소로 직접 연결해보세요"
            ]
        }
    
    # 타임아웃
    if 'timeout' in error_str or 'timed out' in error_str:
        return {
            "error_type": "CONNECTION_TIMEOUT",
            "user_message": f"데이터베이스 연결 시간이 초과되었습니다.",
            "details": f"서버: {host}:{port}",
            "suggestions": [
                "1. 네트워크 연결 상태를 확인하세요",
                "2. 서버가 응답하는지 확인하세요",
                "3. 방화벽이나 보안 그룹 설정을 확인하세요",
                "4. VPN 연결이 필요한지 확인하세요"
            ]
        }
    
    # SSL/TLS 관련 오류
    if 'ssl' in error_str or 'certificate' in error_str:
        return {
            "error_type": "SSL_ERROR",
            "user_message": f"SSL/TLS 연결에 문제가 있습니다.",
            "details": "보안 연결 설정 오류",
            "suggestions": [
                "1. SSL 인증서가 유효한지 확인하세요",
                "2. 서버가 SSL 연결을 지원하는지 확인하세요",
                "3. SSL 설정을 비활성화하거나 다른 모드로 시도하세요"
            ]
        }
    
    # 권한 부족
    if 'permission denied' in error_str or 'access denied' in error_str:
        return {
            "error_type": "PERMISSION_DENIED",
            "user_message": f"데이터베이스 접근 권한이 없습니다.",
            "details": f"사용자: {username}",
            "suggestions": [
                "1. 사용자에게 적절한 권한이 부여되었는지 확인하세요",
                "2. 관리자에게 권한 부여를 요청하세요",
                "3. 다른 사용자 계정으로 시도하세요"
            ]
        }
    
    # 알 수 없는 오류
    return {
        "error_type": "UNKNOWN_ERROR",
        "user_message": "데이터베이스 연결 중 오류가 발생했습니다.",
        "details": str(error)[:200],  # 처음 200자만
        "suggestions": [
            "1. 모든 연결 설정을 다시 확인하세요",
            "2. 데이터베이스 타입이 올바른지 확인하세요",
            "3. 관리자에게 문의하세요"
        ]
    }

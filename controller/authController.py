"""
인증 관련 API 컨트롤러
"""
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import Optional
import jwt
import secrets
import logging
import os
import hashlib
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from service.database.models.user import User
from controller.helper.singletonHelper import get_config_composer, get_vector_manager, get_rag_service, get_document_processor, get_db_manager

logger = logging.getLogger("auth-controller")

# 환경변수에서 타임존 가져오기 (기본값: 서울 시간)
TIMEZONE = ZoneInfo(os.getenv('TIMEZONE', 'Asia/Seoul'))

router = APIRouter(prefix="/auth", tags=["Authentication"])

class SignupRequest(BaseModel):
    """회원가입 요청 모델"""
    username: str
    email: str
    password: str
    full_name: Optional[str] = None
    group_name: Optional[str] = "none"
    mobile_phone_number: Optional[str] = None


class SignupResponse(BaseModel):
    """회원가입 응답 모델"""
    success: bool
    message: str
    username: Optional[str] = None

class LoginRequest(BaseModel):
    """로그인 요청 모델"""
    email: str
    password: str

class LoginResponse(BaseModel):
    """로그인 응답 모델"""
    success: bool
    message: str
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    user_id: Optional[int] = None
    username: Optional[str] = None

class LogoutRequest(BaseModel):
    """로그아웃 요청 모델"""
    token: str

class LogoutResponse(BaseModel):
    """로그아웃 응답 모델"""
    success: bool
    message: str

class TokenValidationResponse(BaseModel):
    """토큰 검증 응답 모델"""
    valid: bool
    user_id: Optional[int] = None
    username: Optional[str] = None
    is_admin: Optional[bool] = None

class RefreshTokenRequest(BaseModel):
    """토큰 갱신 요청 모델"""
    refresh_token: str

class RefreshTokenResponse(BaseModel):
    """토큰 갱신 응답 모델"""
    success: bool
    message: str
    access_token: Optional[str] = None
    token_type: str = "bearer"

def generate_token() -> str:
    """세션 토큰 생성"""
    return secrets.token_hex(32)

def generate_sha256_hash(data: str) -> str:
    """SHA256 해시 생성"""
    # 문자열을 bytes로 인코딩
    data_bytes = data.encode('utf-8')

    # SHA256 해시 객체 생성
    hash_obj = hashlib.sha256()

    # 데이터 업데이트
    hash_obj.update(data_bytes)

    # 16진수 해시 값 반환
    hex_hash = hash_obj.hexdigest()

    return hex_hash

# JWT 관련 설정
JWT_SECRET_KEY = "your-secret-key-change-this-in-production"  # 실제 환경에서는 환경변수로 관리
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440
REFRESH_TOKEN_EXPIRE_DAYS = 7

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """JWT Access Token 생성"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(TIMEZONE) + expires_delta
    else:
        expire = datetime.now(TIMEZONE) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt

def create_refresh_token(data: dict):
    """JWT Refresh Token 생성"""
    to_encode = data.copy()
    expire = datetime.now(TIMEZONE) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    """JWT 토큰 검증"""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.PyJWTError:
        return None

def find_user_by_username(app_db, username: str) -> Optional[User]:
    """사용자명으로 사용자 찾기"""
    try:
        db_type = app_db.config_db_manager.db_type

        if db_type == "postgresql":
            query = "SELECT * FROM users WHERE username = %s"
        else:
            query = "SELECT * FROM users WHERE username = ?"

        result = app_db.config_db_manager.execute_query_one(query, (username,))

        if result:
            return User.from_dict(dict(result))
        return None
    except Exception as e:
        logger.error(f"Error finding user by username: {e}")
        return None

def find_user_by_email(app_db, email: str) -> Optional[User]:
    """사용자명으로 사용자 찾기"""
    try:
        db_type = app_db.config_db_manager.db_type

        if db_type == "postgresql":
            query = "SELECT * FROM users WHERE email = %s"
        else:
            query = "SELECT * FROM users WHERE email = ?"

        result = app_db.config_db_manager.execute_query_one(query, (email,))

        if result:
            return User.from_dict(dict(result))
        return None
    except Exception as e:
        logger.error(f"Error finding user by username: {e}")
        return None

def get_user_by_token(app_state, token: str) -> Optional[dict]:
    """JWT 토큰으로 사용자 정보 가져오기"""
    payload = verify_token(token)
    if not payload:
        return None

    user_id = payload.get("sub")
    print(f"User ID from token: {user_id}")
    if not user_id:
        return None

    # app.state에서 사용자 세션 정보 확인 (선택적)
    if hasattr(app_state, 'user_sessions') and str(user_id) in app_state.user_sessions:
        session_info = app_state.user_sessions[str(user_id)]
        return {
            'user_id': str(user_id),
            'username': payload.get('username'),
            'is_admin': payload.get('is_admin', False),
            'login_time': session_info.get('login_time')
        }

    # 토큰에서 직접 정보 반환
    return {
        'user_id': str(user_id),
        'username': payload.get('username'),
        'is_admin': payload.get('is_admin', False),
        'login_time': None
    }

@router.post("/signup", response_model=SignupResponse)
async def signup(request: Request, signup_data: SignupRequest):
    """
    회원가입 API

    Args:
        request: FastAPI Request 객체
        signup_data: 회원가입 데이터 (username, email, password, full_name)

    Returns:
        SignupResponse: 회원가입 결과
    """
    try:
        app_db = get_db_manager(request)
        if not signup_data.username or not signup_data.email or not signup_data.password:
            raise HTTPException(
                status_code=400,
                detail="Username, email, and password are required"
            )

        existing_user_by_username = app_db.find_by_condition(
            User,
            {"username": signup_data.username},
            limit=1
        )
        print('pass')
        if existing_user_by_username:
            raise HTTPException(
                status_code=409,  # Conflict - Username already exists
                detail="Username already exists"
            )

        existing_user_by_email = app_db.find_by_condition(
            User,
            {"email": signup_data.email},
            limit=1
        )
        if existing_user_by_email:
            raise HTTPException(
                status_code=422,  # Unprocessable Entity - Email already exists
                detail="Email already exists"
            )

        preferences = {}
        if signup_data.mobile_phone_number:
            preferences['mobile_phone_number'] = signup_data.mobile_phone_number

        new_user = User(
            username=signup_data.username,
            email=signup_data.email,
            password_hash=signup_data.password,
            full_name=signup_data.full_name,
            is_active=False, # 이제 False로 변경해서 사용자 승인이 되어야 하는 것으로 변경.
            is_admin=False,
            last_login=None,
            user_type="standard",
            group_name=signup_data.group_name,
            preferences=preferences
        )

        # 데이터베이스에 사용자 추가
        insert_result = app_db.insert(new_user)
        if insert_result and insert_result.get("result") == "success":
            username = signup_data.username
        else:
            username = None

        if username:
            logger.info(f"New user created: {signup_data.username} (User Name: {username})")
            return SignupResponse(
                success=True,
                message="User created successfully",
                username=username
            )
        else:
            logger.error(f"Failed to create user: {signup_data.username}")
            raise HTTPException(
                status_code=500,
                detail="Failed to create user"
            )

    except HTTPException:
        # HTTPException은 그대로 재발생
        raise
    except Exception as e:
        logger.error(f"Signup error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.post("/login", response_model=LoginResponse)
async def login(request: Request, login_data: LoginRequest):
    """
    로그인 API

    Args:
        request: FastAPI Request 객체
        login_data: 로그인 데이터 (email, password)

    Returns:
        LoginResponse: 로그인 결과
    """
    try:
        app_db = get_db_manager(request)
        # 입력 데이터 검증
        if not login_data.email or not login_data.password:
            raise HTTPException(
                status_code=400,
                detail="Email and password are required"
            )

        # 사용자 찾기
        user = find_user_by_email(app_db, login_data.email)
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid email or password"
            )

        if user.user_type == "superuser":
            if login_data.password != generate_sha256_hash(user.password_hash):
                raise HTTPException(
                    status_code=401,
                    detail="Invalid username or password"
                )
        else:
            if login_data.password != user.password_hash:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid username or password"
                )

        # 사용자가 활성 상태인지 확인
        if not user.is_active:
            raise HTTPException(
                status_code=403,
                detail="Account is disabled"
            )

        # JWT 토큰 생성
        token_data = {
            "sub": str(user.id),
            "username": user.username,
            "is_admin": user.is_admin
        }

        access_token = create_access_token(token_data)
        refresh_token = create_refresh_token({"sub": str(user.id)})

        # app.state에 사용자 세션 정보 저장 (선택적, JWT가 주요 인증 방식)
        if not hasattr(request.app.state, 'user_sessions'):
            request.app.state.user_sessions = {}
            print("Creating user_sessions in app.state")

        request.app.state.user_sessions[str(user.id)] = {
            'username': user.username,
            'login_time': datetime.now(TIMEZONE),
            'is_admin': user.is_admin,
            'access_token': access_token,
            'refresh_token': refresh_token
        }
        print(f"User session stored in app.state for user ID: {user.id}")

        # # 사용자의 마지막 로그인 시간 업데이트
        user.last_login = datetime.now(TIMEZONE).isoformat()
        app_db.update(user)

        logger.info(f"User logged in: {login_data.email} (ID: {str(user.id)})")


        return LoginResponse(
            success=True,
            message="Login successful",
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            user_id=user.id,
            username=user.username
        )

    except HTTPException:
        # HTTPException은 그대로 재발생
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.post("/logout", response_model=LogoutResponse)
async def logout(request: Request, logout_data: LogoutRequest):
    """
    로그아웃 API

    Args:
        request: FastAPI Request 객체
        logout_data: 로그아웃 데이터 (token)

    Returns:
        LogoutResponse: 로그아웃 결과
    """
    try:
        # JWT 토큰으로 사용자 찾기
        user_session = get_user_by_token(request.app.state, logout_data.token)

        if not user_session:
            raise HTTPException(
                status_code=401,
                detail="Invalid token"
            )

        # 세션에서 사용자 제거 (JWT는 stateless이지만 app.state에서 제거)
        if hasattr(request.app.state, 'user_sessions'):
            user_id = user_session['user_id']
            if user_id in request.app.state.user_sessions:
                del request.app.state.user_sessions[user_id]
                print(f"User session removed from app.state for user ID: {user_id}")

        logger.info(f"User logged out: {user_session['username']} (ID: {user_session['user_id']})")

        return LogoutResponse(
            success=True,
            message="Logout successful"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/validate-token", response_model=TokenValidationResponse)
async def validate_token(request: Request, token: str):
    """
    토큰 검증 API

    Args:
        request: FastAPI Request 객체
        token: 검증할 토큰

    Returns:
        TokenValidationResponse: 토큰 검증 결과
    """
    try:
        user_session = get_user_by_token(request.app.state, token)

        if user_session:
            return TokenValidationResponse(
                valid=True,
                user_id=user_session['user_id'],
                username=user_session['username'],
                is_admin=user_session['is_admin']
            )
        else:
            return TokenValidationResponse(valid=False)

    except Exception as e:
        logger.error(f"Token validation error: {str(e)}")
        return TokenValidationResponse(valid=False)

@router.post("/refresh", response_model=RefreshTokenResponse)
async def refresh_token(request: Request, refresh_data: RefreshTokenRequest):
    """
    토큰 갱신 API

    Args:
        request: FastAPI Request 객체
        refresh_data: refresh token

    Returns:
        RefreshTokenResponse: 새로운 access token
    """
    try:
        # Refresh token 검증
        payload = verify_token(refresh_data.refresh_token)
        if not payload:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired refresh token"
            )

        # 토큰 타입 확인
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=401,
                detail="Invalid token type"
            )

        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Invalid token payload"
            )

        app_db = get_db_manager(request)
        user = app_db.find_by_id(User, int(user_id))
        if not user or not user.is_active:
            raise HTTPException(
                status_code=401,
                detail="User not found or inactive"
            )

        # 새로운 access token 생성
        token_data = {
            "sub": str(user.id),
            "username": user.username,
            "is_admin": user.is_admin
        }

        new_access_token = create_access_token(token_data)

        logger.info(f"Token refreshed for user: {user.username} (ID: {user.id})")

        return RefreshTokenResponse(
            success=True,
            message="Token refreshed successfully",
            access_token=new_access_token,
            token_type="bearer"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/superuser")
async def check_superuser(request: Request):
    try:
        app_db = get_db_manager(request)
        super_users = app_db.find_by_condition(
            User,
            {
                "user_type": "superuser"
            },
            limit=1
        )

        if super_users:
            return {"superuser": True}

        else:
            return {"superuser": False}

    except Exception as e:
        logger.error(f"Token refresh error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

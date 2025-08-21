"""
PDF Document API Controller
CLAUDE.md 지침에 따른 PDF 문서 뷰어 API 엔드포인트 구현
"""
from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import Response, StreamingResponse
from fastapi.security import HTTPBearer
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import aiofiles
import hashlib
import logging
from datetime import datetime
import jwt
from pathlib import Path
import urllib.parse
from controller.authController import verify_token, get_user_by_token
from service.database.models.user import User

logger = logging.getLogger("document-controller")
security = HTTPBearer(auto_error=False)

router = APIRouter(prefix="/api/documents", tags=["Documents"])

# 설정 상수
DOCUMENTS_BASE_DIR = os.getenv("DOCUMENTS_BASE_DIR", "/polarag_backend/downloads")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", str(100 * 1024 * 1024)))  # 100MB
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour

logger.info(f'Document Settingd: DOCUMENTS_BASE_DIR={DOCUMENTS_BASE_DIR}, MAX_FILE_SIZE={MAX_FILE_SIZE}, CACHE_TTL={CACHE_TTL}')

class DocumentRequest(BaseModel):
    """문서 요청 모델"""
    file_path: str
    user_id: Optional[str] = None

class DocumentMetadataResponse(BaseModel):
    """문서 메타데이터 응답 모델"""
    file_name: str
    file_size: int
    created_at: str
    modified_at: str
    page_count: Optional[int] = None
    content_type: str = "application/pdf"
    checksum: str
    permissions: Dict[str, bool]

class DocumentAccessResponse(BaseModel):
    """문서 접근 권한 응답 모델"""
    has_access: bool
    permissions: Dict[str, bool]
    access_level: str

def validate_file_path(file_path: str, base_directory: str) -> str:
    """
    파일 경로 검증 및 sanitize - 디렉터리 순회 공격 방지
    """
    try:
        # 상대 경로 구성 요소 제거
        clean_path = os.path.normpath(file_path)
        
        # 기본 디렉터리를 벗어나지 않도록 보장
        full_path = os.path.join(base_directory, clean_path.lstrip('/'))
        logger.info(f'Fetching document: {full_path}')
        real_path = os.path.realpath(full_path)
        real_base = os.path.realpath(base_directory)
        
        if not real_path.startswith(real_base):
            raise ValueError("Invalid file path")
        
        return real_path
    except Exception as e:
        logger.error(f"File path validation error: {e}")
        raise ValueError("Invalid file path")

async def check_document_access(app_db, user_id: str, file_path: str) -> Dict[str, Any]:
    """
    사용자의 문서 접근 권한 확인
    """
    try:
        # 파일 존재 여부 확인
        safe_path = validate_file_path(file_path, DOCUMENTS_BASE_DIR)
        if not os.path.exists(safe_path):
            return {"has_access": False, "reason": "Document not found"}
        
        # 사용자 정보 가져오기
        user = app_db.find_by_id(User, int(user_id))
        if not user:
            return {"has_access": False, "reason": "User not found"}
        
        if not user.is_active:
            return {"has_access": False, "reason": "User account is disabled"}
        
        # 관리자는 모든 권한
        if user.is_admin:
            return {
                "has_access": True,
                "permissions": {"read": True, "download": True, "share": True},
                "access_level": "full"
            }
        
        # 일반 사용자는 읽기 및 다운로드 권한만
        return {
            "has_access": True,
            "permissions": {"read": True, "download": True, "share": False},
            "access_level": "read-only"
        }
        
    except Exception as e:
        logger.error(f"Error checking document access: {e}")
        return {"has_access": False, "reason": "Access check failed"}

async def get_file_checksum(file_path: str) -> str:
    """
    파일 체크섬 생성 (캐시 무효화용)
    """
    try:
        hash_md5 = hashlib.md5()
        async with aiofiles.open(file_path, "rb") as f:
            async for chunk in iter(lambda: f.read(8192), b""):
                if not chunk:
                    break
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.error(f"Error generating file checksum: {e}")
        return ""

async def get_document_metadata(file_path: str, user_permissions: Dict[str, bool]) -> DocumentMetadataResponse:
    """
    문서 메타데이터 가져오기
    """
    try:
        stat = os.stat(file_path)
        file_name = os.path.basename(file_path)
        checksum = await get_file_checksum(file_path)
        
        return DocumentMetadataResponse(
            file_name=file_name,
            file_size=stat.st_size,
            created_at=datetime.fromtimestamp(stat.st_ctime).isoformat(),
            modified_at=datetime.fromtimestamp(stat.st_mtime).isoformat(),
            page_count=None,  # PDF 페이지 수는 별도 라이브러리 필요
            content_type="application/pdf",
            checksum=checksum,
            permissions=user_permissions
        )
    except Exception as e:
        logger.error(f"Error getting document metadata: {e}")
        raise HTTPException(status_code=500, detail="Failed to get document metadata")

async def stream_file(file_path: str):
    """
    큰 파일을 위한 스트리밍 함수
    """
    try:
        async with aiofiles.open(file_path, 'rb') as file:
            while True:
                chunk = await file.read(8192)  # 8KB 청크
                if not chunk:
                    break
                yield chunk
    except Exception as e:
        logger.error(f"Error streaming file: {e}")
        raise

async def get_user_id_from_request(request: Request, document_request: DocumentRequest, credentials=None) -> str:
    """
    요청에서 사용자 ID 추출 (Regular/Deploy 모드 구분)
    """
    # Deploy 모드: user_id가 요청 본문에 있는 경우
    if document_request.user_id:
        return document_request.user_id
    
    # Regular 모드: JWT 토큰에서 사용자 ID 추출
    if credentials and credentials.credentials:
        payload = verify_token(credentials.credentials)
        if payload:
            return payload.get("sub")
    
    raise HTTPException(status_code=401, detail="Authentication required")

async def log_document_access(user_id: str, file_path: str, action: str):
    """
    문서 접근 로깅
    """
    logger.info(f"User {user_id} performed {action} on {file_path}")

# =============================================================================
# API 엔드포인트
# =============================================================================

@router.post("/fetch")
async def fetch_document(
    request: Request,
    document_request: DocumentRequest,
    credentials = Depends(security)
):
    """
    PDF 문서 바이너리 데이터 가져오기 (Regular Mode)
    """
    try:
        # 사용자 ID 추출
        user_id = await get_user_id_from_request(request, document_request, credentials)
        
        # 파일 경로 검증
        decoded_path = urllib.parse.unquote(document_request.file_path)
        safe_path = validate_file_path(decoded_path, DOCUMENTS_BASE_DIR)
        
        # 접근 권한 확인
        app_db = request.app.state.app_db
        if not app_db:
            raise HTTPException(status_code=500, detail="Database connection not available")
        
        access_check = await check_document_access(app_db, user_id, decoded_path)
        if not access_check["has_access"]:
            raise HTTPException(status_code=403, detail=access_check.get("reason", "Access denied"))
        
        # 파일 존재 여부 확인
        if not os.path.exists(safe_path):
            raise HTTPException(status_code=404, detail="Document not found")
        
        # 파일 크기 확인
        file_size = os.path.getsize(safe_path)
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large")
        
        # 접근 로깅
        await log_document_access(user_id, decoded_path, "fetch")
        
        # 파일 스트리밍으로 반환
        filename = os.path.basename(safe_path)
        # 한글 파일명 인코딩 처리
        try:
            encoded_filename = filename.encode('ascii')
            content_disposition = f"inline; filename={filename}"
        except UnicodeEncodeError:
            # 한글 파일명은 URL 인코딩
            encoded_filename = urllib.parse.quote(filename)
            content_disposition = f"inline; filename*=UTF-8''{encoded_filename}"
        
        return StreamingResponse(
            stream_file(safe_path),
            media_type="application/pdf",
            headers={
                "Content-Disposition": content_disposition,
                "Cache-Control": f"private, max-age={CACHE_TTL}",
                "Content-Length": str(file_size)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/fetch/deploy")
async def fetch_document_deploy(
    request: Request,
    document_request: DocumentRequest
):
    """
    PDF 문서 바이너리 데이터 가져오기 (Deploy Mode - 인증 불필요)
    """
    try:
        # Deploy 모드에서는 user_id가 필수
        if not document_request.user_id:
            raise HTTPException(status_code=400, detail="user_id is required in deploy mode")
        
        user_id = document_request.user_id
        
        # 파일 경로 검증
        decoded_path = urllib.parse.unquote(document_request.file_path)
        safe_path = validate_file_path(decoded_path, DOCUMENTS_BASE_DIR)
        
        # 접근 권한 확인
        app_db = request.app.state.app_db
        if not app_db:
            raise HTTPException(status_code=500, detail="Database connection not available")
        
        access_check = await check_document_access(app_db, user_id, decoded_path)
        if not access_check["has_access"]:
            raise HTTPException(status_code=403, detail=access_check.get("reason", "Access denied"))
        
        # 파일 존재 여부 확인
        if not os.path.exists(safe_path):
            raise HTTPException(status_code=404, detail="Document not found")
        
        # 파일 크기 확인
        file_size = os.path.getsize(safe_path)
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large")
        
        # 접근 로깅
        await log_document_access(user_id, decoded_path, "fetch_deploy")
        
        # 파일 스트리밍으로 반환
        filename = os.path.basename(safe_path)
        # 한글 파일명 인코딩 처리
        try:
            encoded_filename = filename.encode('ascii')
            content_disposition = f"inline; filename={filename}"
        except UnicodeEncodeError:
            # 한글 파일명은 URL 인코딩
            encoded_filename = urllib.parse.quote(filename)
            content_disposition = f"inline; filename*=UTF-8''{encoded_filename}"
        
        return StreamingResponse(
            stream_file(safe_path),
            media_type="application/pdf",
            headers={
                "Content-Disposition": content_disposition,
                "Cache-Control": f"private, max-age={CACHE_TTL}",
                "Content-Length": str(file_size)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document fetch deploy error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/metadata", response_model=DocumentMetadataResponse)
async def get_document_metadata_endpoint(
    request: Request,
    document_request: DocumentRequest,
    credentials = Depends(security)
):
    """
    문서 메타데이터 가져오기 (Regular Mode)
    """
    try:
        # 사용자 ID 추출
        user_id = await get_user_id_from_request(request, document_request, credentials)
        
        # 파일 경로 검증
        decoded_path = urllib.parse.unquote(document_request.file_path)
        safe_path = validate_file_path(decoded_path, DOCUMENTS_BASE_DIR)
        
        # 접근 권한 확인
        app_db = request.app.state.app_db
        if not app_db:
            raise HTTPException(status_code=500, detail="Database connection not available")
        
        access_check = await check_document_access(app_db, user_id, decoded_path)
        if not access_check["has_access"]:
            raise HTTPException(status_code=403, detail=access_check.get("reason", "Access denied"))
        
        # 파일 존재 여부 확인
        if not os.path.exists(safe_path):
            raise HTTPException(status_code=404, detail="Document not found")
        
        # 접근 로깅
        await log_document_access(user_id, decoded_path, "metadata")
        
        # 메타데이터 반환
        return await get_document_metadata(safe_path, access_check["permissions"])
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document metadata error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/metadata/deploy", response_model=DocumentMetadataResponse)
async def get_document_metadata_deploy(
    request: Request,
    document_request: DocumentRequest
):
    """
    문서 메타데이터 가져오기 (Deploy Mode - 인증 불필요)
    """
    try:
        # Deploy 모드에서는 user_id가 필수
        if not document_request.user_id:
            raise HTTPException(status_code=400, detail="user_id is required in deploy mode")
        
        user_id = document_request.user_id
        
        # 파일 경로 
        decoded_path = urllib.parse.unquote(document_request.file_path)
        safe_path = validate_file_path(decoded_path, DOCUMENTS_BASE_DIR)
        
        # 접근 권한 확인
        app_db = request.app.state.app_db
        if not app_db:
            raise HTTPException(status_code=500, detail="Database connection not available")
        
        access_check = await check_document_access(app_db, user_id, decoded_path)
        if not access_check["has_access"]:
            raise HTTPException(status_code=403, detail=access_check.get("reason", "Access denied"))
        
        # 파일 존재 여부 확인
        if not os.path.exists(safe_path):
            raise HTTPException(status_code=404, detail="Document not found")
        
        # 접근 로깅
        await log_document_access(user_id, decoded_path, "metadata_deploy")
        
        # 메타데이터 반환
        return await get_document_metadata(safe_path, access_check["permissions"])
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document metadata deploy error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/access", response_model=DocumentAccessResponse)
async def check_document_access_endpoint(
    request: Request,
    document_request: DocumentRequest,
    credentials = Depends(security)
):
    """
    문서 접근 권한 확인 (Regular Mode)
    """
    try:
        # 사용자 ID 추출
        user_id = await get_user_id_from_request(request, document_request, credentials)
        
        # 접근 권한 확인
        app_db = request.app.state.app_db
        if not app_db:
            raise HTTPException(status_code=500, detail="Database connection not available")
        
        decoded_path = urllib.parse.unquote(document_request.file_path)
        access_check = await check_document_access(app_db, user_id, decoded_path)
        
        # 접근 로깅
        await log_document_access(user_id, decoded_path, "access_check")
        
        return DocumentAccessResponse(
            has_access=access_check["has_access"],
            permissions=access_check.get("permissions", {}),
            access_level=access_check.get("access_level", "none")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document access check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/access/deploy", response_model=DocumentAccessResponse)
async def check_document_access_deploy(
    request: Request,
    document_request: DocumentRequest
):
    """
    문서 접근 권한 확인 (Deploy Mode - 인증 불필요)
    """
    try:
        # Deploy 모드에서는 user_id가 필수
        if not document_request.user_id:
            raise HTTPException(status_code=400, detail="user_id is required in deploy mode")
        
        user_id = document_request.user_id
        
        # 접근 권한 확인
        app_db = request.app.state.app_db
        if not app_db:
            raise HTTPException(status_code=500, detail="Database connection not available")
        

        decoded_path = urllib.parse.unquote(document_request.file_path)
        access_check = await check_document_access(app_db, user_id, decoded_path)
        
        # 접근 로깅
        await log_document_access(user_id, decoded_path, "access_check_deploy")
        
        return DocumentAccessResponse(
            has_access=access_check["has_access"],
            permissions=access_check.get("permissions", {}),
            access_level=access_check.get("access_level", "none")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document access check deploy error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
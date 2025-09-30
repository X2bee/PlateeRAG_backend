"""
MCP Controller - MCP Station 통신 게이트웨이

백엔드와 MCP Station 간의 통신을 담당하는 컨트롤러
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import logging
import httpx
from controller.helper.singletonHelper import get_config_composer, get_db_manager
from controller.helper.controllerHelper import extract_user_id_from_request
from service.database.logger_helper import create_logger

router = APIRouter(
    prefix="/api/mcp",
    tags=["MCP"],
    responses={404: {"description": "Not found"}},
)

logger = logging.getLogger(__name__)

# MCP Station 기본 URL (Docker Compose 네트워크 내부)
MCP_STATION_BASE_URL = "http://mcp_station:20100"


# ========== Request Models ==========
class CreateSessionRequest(BaseModel):
    """MCP 서버 세션 생성 요청"""
    server_type: str = Field(..., description="서버 타입 (python 또는 node)", example="python")
    server_command: str = Field(..., description="실행할 스크립트 경로", example="/app/server.py")
    server_args: Optional[List[str]] = Field(None, description="추가 명령줄 인자")
    env_vars: Optional[Dict[str, str]] = Field(None, description="환경 변수")
    working_dir: Optional[str] = Field(None, description="작업 디렉토리")


class MCPRequestModel(BaseModel):
    """MCP 요청 모델"""
    session_id: str = Field(..., description="대상 세션 ID")
    method: str = Field(..., description="MCP 메서드 (예: tools/list, tools/call)", example="tools/list")
    params: Dict[str, Any] = Field(default_factory=dict, description="메서드 파라미터")


# ========== Response Models ==========
class SessionInfo(BaseModel):
    """세션 정보"""
    session_id: str = Field(..., description="세션 ID")
    server_type: str = Field(..., description="서버 타입")
    status: str = Field(..., description="세션 상태")
    created_at: str = Field(..., description="생성 시간")
    pid: Optional[int] = Field(None, description="프로세스 ID")
    error_message: Optional[str] = Field(None, description="에러 메시지")


class MCPResponse(BaseModel):
    """MCP 응답"""
    success: bool = Field(..., description="성공 여부")
    data: Optional[Any] = Field(None, description="응답 데이터")
    error: Optional[str] = Field(None, description="에러 메시지")


# ========== Helper Functions ==========
async def make_mcp_request(
    method: str,
    endpoint: str,
    data: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """MCP Station에 HTTP 요청을 보내는 헬퍼 함수"""
    url = f"{MCP_STATION_BASE_URL}{endpoint}"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            if method.upper() == "GET":
                response = await client.get(url, params=params)
            elif method.upper() == "POST":
                response = await client.post(url, json=data)
            elif method.upper() == "DELETE":
                response = await client.delete(url)
            else:
                raise ValueError(f"지원하지 않는 HTTP 메서드: {method}")

            response.raise_for_status()

            # 204 No Content 응답 처리
            if response.status_code == 204:
                return {"success": True}

            return response.json()

    except httpx.HTTPStatusError as e:
        logger.error(f"MCP Station HTTP 오류: {e.response.status_code} - {e.response.text}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"MCP Station 요청 실패: {e.response.text}"
        )
    except httpx.RequestError as e:
        logger.error(f"MCP Station 연결 오류: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"MCP Station에 연결할 수 없습니다: {str(e)}"
        )
    except Exception as e:
        logger.error(f"MCP Station 요청 중 예상치 못한 오류: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"MCP Station 요청 실패: {str(e)}"
        )


# ========== API Endpoints ==========

@router.get("/health",
    summary="MCP Station 상태 확인",
    description="MCP Station 서비스의 상태와 연결을 확인합니다.",
    response_description="서비스 상태 정보")
async def health_check(request: Request):
    """MCP Station 헬스체크"""
    user_id = extract_user_id_from_request(request)
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        result = await make_mcp_request("GET", "/")
        backend_log.success("MCP Station 헬스체크 성공", metadata=result)
        return result
    except HTTPException as e:
        backend_log.error("MCP Station 헬스체크 실패", metadata={"status_code": e.status_code, "detail": e.detail})
        raise
    except Exception as e:
        backend_log.error("MCP Station 헬스체크 중 오류 발생", exception=e)
        raise HTTPException(status_code=500, detail="헬스체크 실패")


@router.post("/sessions",
    summary="MCP 세션 생성",
    description="새로운 MCP 서버 세션을 생성합니다.",
    response_model=SessionInfo,
    status_code=201)
async def create_session(request: Request, session_request: CreateSessionRequest):
    """새로운 MCP 서버 세션 생성"""
    user_id = extract_user_id_from_request(request)
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        result = await make_mcp_request(
            "POST",
            "/sessions",
            data=session_request.model_dump()
        )
        backend_log.success(
            "MCP 세션 생성 성공",
            metadata={
                "session_id": result.get("session_id"),
                "server_type": session_request.server_type
            }
        )
        return SessionInfo(**result)
    except HTTPException:
        raise
    except Exception as e:
        backend_log.error("MCP 세션 생성 실패", exception=e)
        raise HTTPException(status_code=500, detail="세션 생성 실패")


@router.get("/sessions",
    summary="모든 세션 목록 조회",
    description="모든 활성 MCP 세션 목록을 조회합니다.",
    response_model=List[SessionInfo])
async def list_sessions(request: Request):
    """모든 활성 세션 목록 조회"""
    user_id = extract_user_id_from_request(request)
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        result = await make_mcp_request("GET", "/sessions")
        backend_log.info("MCP 세션 목록 조회 성공", metadata={"session_count": len(result)})
        return [SessionInfo(**session) for session in result]
    except HTTPException:
        raise
    except Exception as e:
        backend_log.error("MCP 세션 목록 조회 실패", exception=e)
        raise HTTPException(status_code=500, detail="세션 목록 조회 실패")


@router.get("/sessions/{session_id}",
    summary="특정 세션 정보 조회",
    description="특정 세션의 상세 정보를 조회합니다.",
    response_model=SessionInfo)
async def get_session(request: Request, session_id: str):
    """특정 세션 정보 조회"""
    user_id = extract_user_id_from_request(request)
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        result = await make_mcp_request("GET", f"/sessions/{session_id}")
        backend_log.info("MCP 세션 정보 조회 성공", metadata={"session_id": session_id})
        return SessionInfo(**result)
    except HTTPException:
        raise
    except Exception as e:
        backend_log.error("MCP 세션 정보 조회 실패", exception=e, metadata={"session_id": session_id})
        raise HTTPException(status_code=500, detail="세션 정보 조회 실패")


@router.delete("/sessions/{session_id}",
    summary="세션 삭제",
    description="지정된 세션을 삭제하고 관련 프로세스를 종료합니다.",
    status_code=204)
async def delete_session(request: Request, session_id: str):
    """세션 삭제 및 프로세스 종료"""
    user_id = extract_user_id_from_request(request)
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        await make_mcp_request("DELETE", f"/sessions/{session_id}")
        backend_log.success("MCP 세션 삭제 성공", metadata={"session_id": session_id})
        return {"success": True, "message": "세션이 삭제되었습니다"}
    except HTTPException:
        raise
    except Exception as e:
        backend_log.error("MCP 세션 삭제 실패", exception=e, metadata={"session_id": session_id})
        raise HTTPException(status_code=500, detail="세션 삭제 실패")


@router.get("/sessions/{session_id}/tools",
    summary="세션의 도구 목록 조회",
    description="특정 세션에서 사용 가능한 MCP 도구 목록을 조회합니다.")
async def get_session_tools(request: Request, session_id: str):
    """특정 세션의 MCP 도구 목록 조회"""
    user_id = extract_user_id_from_request(request)
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        result = await make_mcp_request("GET", f"/sessions/{session_id}/tools")
        backend_log.info(
            "MCP 도구 목록 조회 성공",
            metadata={
                "session_id": session_id,
                "tool_count": len(result.get("tools", []))
            }
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        backend_log.error("MCP 도구 목록 조회 실패", exception=e, metadata={"session_id": session_id})
        raise HTTPException(status_code=500, detail="도구 목록 조회 실패")


@router.post("/request",
    summary="MCP 요청 실행",
    description="MCP 서버로 요청을 라우팅합니다. tools/list, tools/call, prompts/list 등의 메서드를 지원합니다.",
    response_model=MCPResponse)
async def mcp_request(request: Request, mcp_request: MCPRequestModel):
    """MCP 서버로 요청 라우팅"""
    user_id = extract_user_id_from_request(request)
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        result = await make_mcp_request(
            "POST",
            "/mcp/request",
            data=mcp_request.model_dump()
        )

        if result.get("success"):
            backend_log.success(
                "MCP 요청 성공",
                metadata={
                    "session_id": mcp_request.session_id,
                    "method": mcp_request.method
                }
            )
        else:
            backend_log.warning(
                "MCP 요청 실패",
                metadata={
                    "session_id": mcp_request.session_id,
                    "method": mcp_request.method,
                    "error": result.get("error")
                }
            )

        return MCPResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        backend_log.error(
            "MCP 요청 실행 중 오류",
            exception=e,
            metadata={
                "session_id": mcp_request.session_id,
                "method": mcp_request.method
            }
        )
        raise HTTPException(status_code=500, detail="MCP 요청 실패")


@router.get("/health/detailed",
    summary="상세 헬스체크",
    description="MCP Station의 상세한 상태 정보를 조회합니다.")
async def detailed_health_check(request: Request):
    """상세 헬스체크"""
    user_id = extract_user_id_from_request(request)
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        result = await make_mcp_request("GET", "/health")
        backend_log.info("MCP Station 상세 헬스체크 성공", metadata=result)
        return result
    except HTTPException:
        raise
    except Exception as e:
        backend_log.error("MCP Station 상세 헬스체크 실패", exception=e)
        raise HTTPException(status_code=500, detail="상세 헬스체크 실패")

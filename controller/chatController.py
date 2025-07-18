from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import json
import logging
from datetime import datetime
from service.general_function import create_conversation_function
from service.database.models.executor import ExecutionMeta
from service.database.execution_meta_service import get_or_create_execution_meta, update_execution_meta_count

logger = logging.getLogger("chat-controller")
router = APIRouter(prefix="/api/chat", tags=["chat"])

class ChatNewRequest(BaseModel):
    workflow_name: str
    workflow_id: str
    interaction_id: str
    input_data: Optional[str] = None

class ChatExecutionRequest(BaseModel):
    user_input: str
    interaction_id: str
    workflow_id: Optional[str] = None
    workflow_name: Optional[str] = None
    selected_collection: Optional[str] = None

def get_rag_service(request: Request):
    """RAG 서비스 의존성 주입"""
    if hasattr(request.app.state, 'rag_service') and request.app.state.rag_service:
        return request.app.state.rag_service
    else:
        raise HTTPException(status_code=500, detail="RAG service not available")

@router.post("/new", response_model=Dict[str, Any])
async def chat_new(request: Request, request_body: ChatNewRequest):
    """
    새로운 채팅을 시작하고 DB에 ExecutionMeta를 저장합니다.
    workflow_name과 workflow_id는 반드시 "default_mode"여야 합니다.
    """
    try:
        # workflow_name과 workflow_id 검증
        if request_body.workflow_name != "default_mode":
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid workflow_name: expected 'default_mode', got '{request_body.workflow_name}'"
            )
        
        if request_body.workflow_id != "default_mode":
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid workflow_id: expected 'default_mode', got '{request_body.workflow_id}'"
            )
        
        logger.info(f"Starting new chat session: interaction_id={request_body.interaction_id}")
        
        # 데이터베이스 매니저 가져오기
        if not hasattr(request.app.state, 'app_db') or not request.app.state.app_db:
            raise HTTPException(status_code=500, detail="Database connection not available")
        
        db_manager = request.app.state.app_db
        
        # 설정 컴포저 가져오기
        if not hasattr(request.app.state, 'config_composer'):
            raise HTTPException(status_code=500, detail="Config composer not available")
        
        config_composer = request.app.state.config_composer
        
        # ExecutionMeta 조회 또는 생성
        execution_meta = await get_or_create_execution_meta(
            db_manager,
            request_body.interaction_id,
            request_body.workflow_id,
            request_body.workflow_name,
            first_msg=request_body.input_data if request_body.input_data else "Chat session started"
        )
        
        # 대화 함수 생성
        conversation = create_conversation_function(config_composer, db_manager)
        
        # 첫 번째 대화 실행 (input_data가 있는 경우)
        chat_response = None
        if request_body.input_data:
            chat_result = await conversation(
                user_input=request_body.input_data,
                workflow_id=request_body.workflow_id,
                workflow_name=request_body.workflow_name,
                interaction_id=request_body.interaction_id
            )
            
            if chat_result["status"] == "success":
                chat_response = chat_result["ai_response"]
                
                # ExecutionIO에 채팅 로그 저장
                await _save_chat_execution_io(
                    db_manager,
                    request_body.interaction_id,
                    request_body.workflow_id,
                    request_body.workflow_name,
                    request_body.input_data,
                    chat_response
                )
                
                # ExecutionMeta 업데이트 (interaction_count 증가)
                await update_execution_meta_count(db_manager, execution_meta)
            else:
                raise HTTPException(status_code=500, detail=f"Chat execution failed: {chat_result.get('error_message')}")
        
        return JSONResponse(content={
            "status": "success",
            "message": "New chat session created successfully",
            "interaction_id": request_body.interaction_id,
            "workflow_id": request_body.workflow_id,
            "workflow_name": request_body.workflow_name,
            "execution_meta": {
                "interaction_id": execution_meta.interaction_id,
                "interaction_count": execution_meta.interaction_count + (1 if request_body.input_data else 0),
                "workflow_id": execution_meta.workflow_id,
                "workflow_name": execution_meta.workflow_name
            },
            "chat_response": chat_response,
            "timestamp": datetime.now().isoformat()
        })
        
    except HTTPException:
        raise  # HTTPException은 그대로 전달
    except Exception as e:
        logger.error(f"Error in chat_new: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/execution", response_model=Dict[str, Any])
async def chat_execution(request: Request, request_body: ChatExecutionRequest):
    """
    기존 채팅 세션에서 대화를 계속 진행합니다.
    """
    try:
        logger.info(f"Chat execution: interaction_id={request_body.interaction_id}, user_input={request_body.user_input[:50]}...")
        
        if not hasattr(request.app.state, 'app_db') or not request.app.state.app_db:
            raise HTTPException(status_code=500, detail="Database connection not available")
        
        db_manager = request.app.state.app_db
        
        
        if not hasattr(request.app.state, 'config_composer'):
            raise HTTPException(status_code=500, detail="Config composer not available")
        
        config_composer = request.app.state.config_composer
        
        # 기존 ExecutionMeta 조회
        execution_meta = await _get_execution_meta(
            db_manager,
            request_body.interaction_id,
            request_body.workflow_id or "default_mode"
        )
        
        if not execution_meta:
            raise HTTPException(
                status_code=404, 
                detail=f"Chat session not found for interaction_id: {request_body.interaction_id}"
            )
        
        if request_body.selected_collection:
            selected_collection = request_body.selected_collection
            rag_service = get_rag_service(request)
        else:
            selected_collection = None
            rag_service = None
        
        conversation = create_conversation_function(config_composer, db_manager, rag_service)
        
        # 대화 실행
        chat_result = await conversation(
            user_input=request_body.user_input,
            workflow_id=request_body.workflow_id or execution_meta.workflow_id,
            workflow_name=request_body.workflow_name or execution_meta.workflow_name,
            interaction_id=request_body.interaction_id,
            selected_collection=selected_collection
        )
        
        if chat_result["status"] == "success":
            # ExecutionIO에 채팅 로그 저장
            await _save_chat_execution_io(
                db_manager,
                request_body.interaction_id,
                request_body.workflow_id or execution_meta.workflow_id,
                request_body.workflow_name or execution_meta.workflow_name,
                request_body.user_input,
                chat_result["ai_response"]
            )
            
            # ExecutionMeta 업데이트 (interaction_count 증가)
            await update_execution_meta_count(db_manager, execution_meta)
            
            return JSONResponse(content={
                "status": "success",
                "message": "Chat execution completed successfully",
                "user_input": request_body.user_input,
                "ai_response": chat_result["ai_response"],
                "interaction_id": request_body.interaction_id,
                "session_id": chat_result["session_id"],
                "execution_meta": {
                    "interaction_id": execution_meta.interaction_id,
                    "interaction_count": execution_meta.interaction_count + 1,
                    "workflow_id": execution_meta.workflow_id,
                    "workflow_name": execution_meta.workflow_name
                },
                "timestamp": chat_result["timestamp"]
            })
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"Chat execution failed: {chat_result.get('error_message')}"
            )
        
    except HTTPException:
        raise  # HTTPException은 그대로 전달
    except Exception as e:
        logger.error(f"Error in chat_execution: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


async def _get_execution_meta(db_manager, interaction_id: str, workflow_id: str) -> ExecutionMeta:
    """기존 ExecutionMeta를 조회합니다."""
    try:
        query = """
        SELECT * FROM execution_meta 
        WHERE interaction_id = %s AND workflow_id = %s
        """
        
        # SQLite인 경우 파라미터 플레이스홀더 변경
        if db_manager.config_db_manager.db_type == "sqlite":
            query = query.replace("%s", "?")
        
        result = db_manager.config_db_manager.execute_query(query, (interaction_id, workflow_id))
        
        if result and len(result) > 0:
            row = result[0]
            execution_meta = ExecutionMeta(
                id=row.get('id'),
                interaction_id=row['interaction_id'],
                workflow_id=row['workflow_id'],
                workflow_name=row['workflow_name'],
                interaction_count=row['interaction_count'],
                metadata=json.loads(row['metadata']) if row.get('metadata') else {}
            )
            return execution_meta
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error getting ExecutionMeta: {str(e)}")
        return None


async def _save_chat_execution_io(db_manager, interaction_id: str, workflow_id: str, workflow_name: str, 
                                 user_input: str, ai_response: str):
    """채팅 실행 결과를 ExecutionIO 테이블에 저장합니다."""
    try:
        # 입력 데이터 (사용자 메시지)
        input_data = {
            "node_id": "chat_input",
            "node_name": "Chat Input",
            "inputs": {
                "user_input": user_input
            },
            "result": user_input
        }
        
        # 출력 데이터 (AI 응답)
        output_data = {
            "node_id": "chat_output", 
            "node_name": "Chat Output",
            "inputs": {
                "ai_response": ai_response
            },
            "result": ai_response
        }
        
        # JSON 형태로 변환하여 저장
        input_json = json.dumps(input_data, ensure_ascii=False)
        output_json = json.dumps(output_data, ensure_ascii=False)
        
        # DB 타입에 따른 쿼리 준비
        db_type = db_manager.config_db_manager.db_type
        if db_type == "postgresql":
            query = """
                INSERT INTO execution_io (interaction_id, workflow_id, workflow_name, input_data, output_data, created_at)
                VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            """
        else:  # SQLite
            query = """
                INSERT INTO execution_io (interaction_id, workflow_id, workflow_name, input_data, output_data, created_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """
        
        db_manager.config_db_manager.execute_query(
            query, 
            (interaction_id, workflow_id, workflow_name, input_json, output_json)
        )
        
        logger.info(f"Chat ExecutionIO 데이터 저장 완료: interaction_id={interaction_id}, workflow_id={workflow_id}")
        
    except Exception as e:
        logger.error(f"Chat ExecutionIO 저장 중 오류 발생: {str(e)}")
        # 로그 저장 실패해도 전체 실행은 계속 진행

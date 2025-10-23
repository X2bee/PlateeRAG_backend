import os
import json
import asyncio
import logging

from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from editor.async_workflow_executor import execution_manager, WorkflowCancelledError
from service.database.execution_meta_service import update_execution_meta_count

from controller.helper.singletonHelper import get_db_manager
from controller.workflow.models.requests import WorkflowRequest
from service.database.models.workflow import WorkflowMeta
from controller.workflow.execution_runtime import prepare_workflow_execution, persist_stream_results
from controller.workflow.websocket_support import (
    authenticate_websocket,
    get_db_manager_from_websocket,
    run_websocket_workflow,
    cleanup_websocket_session,
    safe_send_json,
)
from service.database.logger_helper import create_logger

logger = logging.getLogger("workflow-controller")
router = APIRouter(prefix="/deploy", tags=["workflow"])

@router.get("/load/{user_id}/{workflow_id}")
async def load_workflow(request: Request, user_id: str, workflow_id: str):
    """
    특정 workflow를 로드합니다.
    """
    try:
        # downloads_path = os.path.join(os.getcwd(), "downloads")
        # download_path_id = os.path.join(downloads_path, user_id)

        # filename = f"{workflow_id}.json"
        # file_path = os.path.join(download_path_id, filename)

        # if not os.path.exists(file_path):
        #     raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")

        # with open(file_path, 'r', encoding='utf-8') as f:
        #     workflow_data = json.load(f)
        #### DB방식으로 변경중
        app_db = get_db_manager(request)

        workflow_meta = app_db.find_by_condition(WorkflowMeta, {"user_id": user_id, "workflow_name": workflow_id}, limit=1)
        workflow_data = workflow_meta[0].workflow_data if workflow_meta else None
        if isinstance(workflow_data, str):
            workflow_data = json.loads(workflow_data)

        logger.info(f"Workflow loaded successfully: {workflow_id}")
        return JSONResponse(content=workflow_data)

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")
    except Exception as e:
        logger.error(f"Error loading workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load workflow: {str(e)}")

@router.post("/execute/based_id", response_model=Dict[str, Any])
async def execute_workflow_with_id(request: Request, request_body: WorkflowRequest):
    """
    주어진 노드와 엣지 정보로 워크플로우를 실행합니다.
    """
    try:
        app_db = get_db_manager(request)
        login_user_id = str(request_body.user_id) if request_body.user_id is not None else ""

        context = await prepare_workflow_execution(
            app=request.app,
            app_db=app_db,
            login_user_id=login_user_id,
            request_body=request_body,
            request_like=request,
            backend_log=None,
        )

        print("--- 워크플로우 실행 시작 ---")
        print(f"실행 워크플로우: {request_body.workflow_name} ({request_body.workflow_id})")
        print(f"입력 데이터: {request_body.input_data}")

        ## 일반적인 실행(execution)이 아닌 경우, 즉 대화형 실행(conversation execution)인 경우
        ## interaction_id가 "default"가 아닌 경우, 대화형 실행을 위한 메타데이터를 가져오거나 생성
        ## interaction_id가 "default"인 경우, execution_meta는 None으로 설정
        execution_meta = context.execution_meta

        # 비동기 실행 및 결과 수집
        final_outputs = []
        async for output in context.executor.execute_workflow_async():
            final_outputs.append(output)

        ## 대화형 실행인 경우 execution_meta의 값을 가지고, 이 경우에는 대화 count를 증가.
        if execution_meta:
            await update_execution_meta_count(app_db, execution_meta)

        response_data = {"status": "success", "message": "워크플로우 실행 완료", "outputs": final_outputs}

        if execution_meta:
            response_data["execution_meta"] = {
                "interaction_id": execution_meta.interaction_id,
                "interaction_count": execution_meta.interaction_count + 1,
                "workflow_id": execution_meta.workflow_id,
                "workflow_name": execution_meta.workflow_name
            }

        return response_data

    except ValueError as e:
        logging.error(f"Workflow execution error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    finally:
        # 완료된 실행들 정리
        execution_manager.cleanup_completed_executions()

@router.post("/execute/based_id/stream")
async def execute_workflow_with_id_stream(request: Request, request_body: WorkflowRequest):
    """
    주어진 ID를 기반으로 워크플로우를 스트리밍 방식으로 실행합니다.
    """
    user_id = request_body.user_id

    user_id=str(user_id)
    app_db = get_db_manager(request)

    async def stream_generator(async_result_generator, db_manager, user_id, workflow_req):
        """결과 제너레이터를 SSE 형식으로 변환하는 비동기 제너레이터"""
        full_response_chunks = []
        cancelled = False
        try:
            async for chunk in async_result_generator:
                # 클라이언트에 보낼 데이터 형식 정의 (JSON)
                full_response_chunks.append(str(chunk))
                response_chunk = {"type": "data", "content": chunk}
                yield f"data: {json.dumps(response_chunk, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.01) # 짧은 딜레이로 이벤트 스트림 안정화

            end_message = {"type": "end", "message": "Stream finished"}
            yield f"data: {json.dumps(end_message)}\n\n"

        except WorkflowCancelledError:
            cancelled = True
            cancel_message = {
                "type": "cancelled",
                "content": {
                    "workflow_name": workflow_req.workflow_name,
                    "workflow_id": workflow_req.workflow_id,
                    "interaction_id": workflow_req.interaction_id,
                },
            }
            await asyncio.sleep(0)
            yield f"data: {json.dumps(cancel_message)}\n\n"

        except Exception as e:
            logger.error(f"스트리밍 중 오류 발생: {e}", exc_info=True)
            error_message = {"type": "error", "detail": f"스트리밍 중 오류가 발생했습니다: {str(e)}"}
            yield f"data: {json.dumps(error_message)}\n\n"
        finally:
            # ✨ 2. 스트림이 모두 끝난 후, 수집된 내용으로 DB 로그 업데이트
            final_text = "".join(full_response_chunks) if not cancelled else ""
            if not cancelled:
                await persist_stream_results(db_manager, user_id, workflow_req, final_text, backend_log=None)
            execution_manager.cleanup_completed_executions()


    try:
        app_db = get_db_manager(request)
        login_user_id = str(request_body.user_id) if request_body.user_id is not None else ""

        backend_log = None  # deploy 컨트롤러에는 backend logger가 구성되어 있지 않음.

        context = await prepare_workflow_execution(
            app=request.app,
            app_db=app_db,
            login_user_id=login_user_id,
            request_body=request_body,
            request_like=request,
            backend_log=backend_log,
        )

        execution_meta = context.execution_meta

        if execution_meta:
            await update_execution_meta_count(app_db, execution_meta)

        # 비동기 제너레이터 시작 (스트리밍용)
        result_generator = context.executor.execute_workflow_async_streaming()

        # StreamingResponse를 사용하여 SSE 스트림 반환
        return StreamingResponse(
            stream_generator(result_generator, app_db, login_user_id, request_body),
            media_type="text/event-stream"
        )

    except ValueError as e:
        logger.error(f"Workflow execution error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"An unexpected error occurred during workflow setup: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")
    finally:
        # 완료된 실행들 정리
        execution_manager.cleanup_completed_executions()

@router.websocket("/ws")
async def execute_workflow_via_websocket(websocket: WebSocket):
    try:
        auth_context = authenticate_websocket(websocket)
    except ValueError as auth_error:
        await websocket.close(code=1008, reason=str(auth_error))
        return

    user_session = auth_context["user"]
    session_id = auth_context["session_id"]
    login_user_id = str(user_session["user_id"])

    try:
        app_db = get_db_manager_from_websocket(websocket)
    except RuntimeError as db_error:
        await websocket.close(code=1011, reason=str(db_error))
        cleanup_websocket_session(websocket)
        return

    backend_log = create_logger(app_db, login_user_id, request=None)

    await websocket.accept()
    if not hasattr(websocket.state, "active_task"):
        websocket.state.active_task = None
        websocket.state.active_execution_id = None

    await safe_send_json(
        websocket,
        {
            "type": "ready",
            "content": {
                "user_id": login_user_id,
                "session_id": session_id,
                "participants": auth_context.get("participants", []),
                "message": "WebSocket connection established",
            },
        },
    )

    try:
        while True:
            try:
                raw_message = await websocket.receive_json()
            except WebSocketDisconnect:
                logger.info(
                    "WebSocket disconnected by client (deploy controller)",
                    extra={"session_id": session_id, "user_id": login_user_id},
                )
                break
            except Exception as exc:
                logger.error("Failed to decode WebSocket message: %s", exc, exc_info=True)
                try:
                    await safe_send_json(
                        websocket,
                        {"type": "error", "detail": "Invalid message format"},
                    )
                except Exception:
                    pass
                continue

            if not isinstance(raw_message, dict):
                await safe_send_json(
                    websocket,
                    {"type": "error", "detail": "Message must be a JSON object"},
                )
                continue

            message_type = raw_message.get("type", "start")

            if message_type in ("ping", "pong"):
                await safe_send_json(websocket, {"type": "pong"})
                continue

            if message_type == "close":
                await websocket.close(code=1000)
                break

            if message_type not in ("start", "run", "cancel"):
                await safe_send_json(
                    websocket,
                    {"type": "error", "detail": f"Unsupported message type: {message_type}"},
                )
                continue

            payload = raw_message.get("payload") or {
                key: value
                for key, value in raw_message.items()
                if key not in ("type", "payload")
            }

            if not isinstance(payload, dict):
                await safe_send_json(
                    websocket,
                    {"type": "error", "detail": "Payload must be a JSON object"},
                )
                continue

            if message_type == "cancel":
                cancel_reason = raw_message.get("reason")
                cancel_interaction_id = payload.get("interaction_id")
                if not cancel_interaction_id or str(cancel_interaction_id).lower() in ("default", "", "none"):
                    cancel_interaction_id = session_id
                cancel_workflow_id = payload.get("workflow_id")
                if not cancel_workflow_id:
                    await safe_send_json(
                        websocket,
                        {
                            "type": "error",
                            "detail": "workflow_id is required for cancellation",
                        },
                    )
                    continue

                execution_id = f"{cancel_interaction_id}_{cancel_workflow_id}_{login_user_id}"
                cancelled = execution_manager.cancel_execution(execution_id)
                await safe_send_json(
                    websocket,
                    {
                        "type": "cancel_ack",
                        "content": {
                            "workflow_id": cancel_workflow_id,
                            "workflow_name": payload.get("workflow_name"),
                            "interaction_id": cancel_interaction_id,
                            "session_id": session_id,
                            "cancelled": cancelled,
                            "reason": cancel_reason,
                        },
                    },
                )
                if not cancelled:
                    logger.warning(
                        "Cancellation requested but no active execution found (deploy)",
                        extra={
                            "workflow_id": cancel_workflow_id,
                            "interaction_id": cancel_interaction_id,
                            "user_id": login_user_id,
                        },
                    )
                continue

            if message_type in ("start", "run"):
                active_task = getattr(websocket.state, "active_task", None)
                if active_task and not active_task.done():
                    await safe_send_json(
                        websocket,
                        {
                            "type": "error",
                            "detail": "Workflow execution already in progress. Please wait or cancel before starting a new one.",
                        },
                    )
                    continue

                try:
                    request_body = WorkflowRequest(**payload)
                except ValidationError as exc:
                    await safe_send_json(
                        websocket,
                        {
                            "type": "error",
                            "detail": "Invalid workflow request",
                            "errors": exc.errors(),
                        },
                    )
                    continue

                if (
                    not request_body.interaction_id
                    or str(request_body.interaction_id).lower() in ("default", "", "none")
                ) and session_id:
                    request_body.interaction_id = session_id

                execution_id = f"{request_body.interaction_id}_{request_body.workflow_id}_{login_user_id}"

                task = asyncio.create_task(
                    run_websocket_workflow(
                        websocket=websocket,
                        app_db=app_db,
                        login_user_id=login_user_id,
                        request_body=request_body,
                        backend_log=backend_log,
                    )
                )

                websocket.state.active_task = task
                websocket.state.active_execution_id = execution_id

                def _clear_task(done_task: asyncio.Task, logger=logger):
                    try:
                        exception = done_task.exception()
                        if exception:
                            logger.error(
                                "Workflow execution task terminated with exception (deploy)",
                                exc_info=True,
                            )
                    except asyncio.CancelledError:
                        logger.info(
                            "Workflow execution task cancelled by server (deploy)",
                            extra={"session_id": session_id, "user_id": login_user_id},
                        )
                    finally:
                        websocket.state.active_task = None
                        websocket.state.active_execution_id = None

                task.add_done_callback(_clear_task)
                continue
    finally:
        active_execution_id = getattr(websocket.state, "active_execution_id", None)
        active_task = getattr(websocket.state, "active_task", None)
        if active_execution_id:
            execution_manager.cancel_execution(active_execution_id)
        if active_task and not active_task.done():
            try:
                await asyncio.wait_for(active_task, timeout=1.0)
            except Exception:
                active_task.cancel()
        cleanup_websocket_session(websocket)

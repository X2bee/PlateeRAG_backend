"""
워크플로우 실행 관련 엔드포인트들
"""
import asyncio
import json
import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import ValidationError
from controller.helper.controllerHelper import extract_user_id_from_request
from controller.helper.singletonHelper import get_db_manager
from controller.workflow.models.requests import WorkflowRequest
from service.database.execution_meta_service import update_execution_meta_count
from editor.async_workflow_executor import execution_manager, WorkflowCancelledError
from service.database.logger_helper import create_logger
from controller.workflow.execution_runtime import prepare_workflow_execution, persist_stream_results
from controller.workflow.websocket_support import (
    authenticate_websocket,
    get_db_manager_from_websocket,
    run_websocket_workflow,
    cleanup_websocket_session,
    safe_send_json,
    get_session_entry,
)

logger = logging.getLogger("execution-endpoints")
router = APIRouter()



@router.post("/based_id", response_model=Dict[str, Any])
async def execute_workflow_with_id(request: Request, request_body: WorkflowRequest):
    """
    주어진 노드와 엣지 정보로 워크플로우를 실행합니다.
    """
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        backend_log.info("Starting workflow execution",
                        metadata={"workflow_name": request_body.workflow_name,
                                "workflow_id": request_body.workflow_id,
                                "interaction_id": request_body.interaction_id,
                                "input_data_length": len(str(request_body.input_data)) if request_body.input_data else 0})

        context = await prepare_workflow_execution(
            app=request.app,
            app_db=app_db,
            login_user_id=user_id,
            request_body=request_body,
            request_like=request,
            backend_log=backend_log,
        )

        nodes = context.nodes
        edges = context.edges

        print("--- 워크플로우 실행 시작 ---")
        print(f"실행 워크플로우: {request_body.workflow_name} ({request_body.workflow_id})")
        print(f"입력 데이터: {request_body.input_data}")

        # 비동기 실행 및 결과 수집
        final_outputs = []
        try:
            async for output in context.executor.execute_workflow_async():
                final_outputs.append(output)
        except WorkflowCancelledError:
            backend_log.info(
                "Workflow execution cancelled",
                metadata={
                    "workflow_name": request_body.workflow_name,
                    "workflow_id": request_body.workflow_id,
                    "interaction_id": request_body.interaction_id,
                },
            )
            return {
                "status": "cancelled",
                "message": "워크플로우 실행이 취소되었습니다",
                "outputs": [],
            }

        ## 대화형 실행인 경우 count 증가
        if context.execution_meta:
            await update_execution_meta_count(app_db, context.execution_meta)

        response_data = {"status": "success", "message": "워크플로우 실행 완료", "outputs": final_outputs}

        if context.execution_meta:
            response_data["execution_meta"] = {
                "interaction_id": context.execution_meta.interaction_id,
                "interaction_count": context.execution_meta.interaction_count + 1,
                "workflow_id": context.execution_meta.workflow_id,
                "workflow_name": context.execution_meta.workflow_name
            }

        backend_log.success("Workflow execution completed successfully",
                          metadata={"workflow_name": request_body.workflow_name,
                                  "workflow_id": request_body.workflow_id,
                                  "interaction_id": request_body.interaction_id,
                                  "node_count": len(nodes),
                                  "edge_count": len(edges),
                                  "output_count": len(final_outputs),
                                  "is_interactive": context.execution_meta is not None,
                                  "interaction_count": context.execution_meta.interaction_count + 1 if context.execution_meta else 0,
                                  "workflow_source": context.source})

        return response_data

    except ValueError as e:
        backend_log.error("Workflow execution validation error", exception=e,
                         metadata={"workflow_name": request_body.workflow_name,
                                 "workflow_id": request_body.workflow_id})
        logging.error(f"Workflow execution error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        backend_log.error("Workflow execution failed", exception=e,
                         metadata={"workflow_name": request_body.workflow_name,
                                 "workflow_id": request_body.workflow_id,
                                 "interaction_id": request_body.interaction_id})
        logging.error(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    finally:
        # 완료된 실행들 정리
        pass
        execution_manager.cleanup_completed_executions()

@router.post("/based_id/stream")
async def execute_workflow_with_id_stream(request: Request, request_body: WorkflowRequest):
    """
    주어진 ID를 기반으로 워크플로우를 스트리밍 방식으로 실행합니다.
    """
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    backend_log.info("Starting streaming workflow execution with ID",
                    metadata={"workflow_name": request_body.workflow_name,
                            "workflow_id": request_body.workflow_id,
                            "interaction_id": request_body.interaction_id,
                            "input_data_length": len(str(request_body.input_data)) if request_body.input_data else 0})

    async def stream_generator(async_result_generator, db_manager, user_id, workflow_req):
        """결과 제너레이터를 SSE 형식으로 변환하는 비동기 제너레이터"""
        full_response_chunks = []
        chunk_count = 0
        cancelled = False
        try:
            async for chunk in async_result_generator:
                chunk_count += 1
                full_response_chunks.append(str(chunk))
                response_chunk = {"type": "data", "content": chunk}
                yield f"data: {json.dumps(response_chunk, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.01)

            end_message = {"type": "end", "message": "Stream finished"}
            yield f"data: {json.dumps(end_message)}\n\n"

            backend_log.success("Streaming workflow execution completed",
                              metadata={"workflow_name": workflow_req.workflow_name,
                                      "workflow_id": workflow_req.workflow_id,
                                      "interaction_id": workflow_req.interaction_id,
                                      "chunk_count": chunk_count,
                                      "total_output_length": len("".join(full_response_chunks))})

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
            backend_log.info(
                "Streaming workflow execution cancelled",
                metadata={
                    "workflow_name": workflow_req.workflow_name,
                    "workflow_id": workflow_req.workflow_id,
                    "interaction_id": workflow_req.interaction_id,
                    "chunk_count": chunk_count,
                },
            )
        except Exception as e:
            backend_log.error("Streaming workflow execution failed", exception=e,
                            metadata={"workflow_name": workflow_req.workflow_name,
                                    "workflow_id": workflow_req.workflow_id,
                                    "interaction_id": workflow_req.interaction_id,
                                    "chunk_count": chunk_count})
            logger.error(f"스트리밍 중 오류 발생: {e}", exc_info=True)
            error_message = {"type": "error", "detail": f"스트리밍 중 오류가 발생했습니다: {str(e)}"}
            yield f"data: {json.dumps(error_message)}\n\n"
        finally:
            final_text = "".join(full_response_chunks) if not cancelled else ""
            if not cancelled:
                await persist_stream_results(db_manager, user_id, workflow_req, final_text, backend_log)
            execution_manager.cleanup_completed_executions()

    try:
        context = await prepare_workflow_execution(
            app=request.app,
            app_db=app_db,
            login_user_id=user_id,
            request_body=request_body,
            request_like=request,
            backend_log=backend_log,
        )

        if context.execution_meta:
            await update_execution_meta_count(app_db, context.execution_meta)

        backend_log.info("Starting workflow streaming execution",
                        metadata={"workflow_name": request_body.workflow_name,
                                "workflow_id": request_body.workflow_id,
                                "interaction_id": request_body.interaction_id,
                                "node_count": len(context.nodes),
                                "edge_count": len(context.edges),
                                "is_interactive": context.execution_meta is not None,
                                "workflow_source": context.source})

        # 비동기 제너레이터 시작 (스트리밍용)
        result_generator = context.executor.execute_workflow_async_streaming()

        # StreamingResponse를 사용하여 SSE 스트림 반환
        return StreamingResponse(
            stream_generator(result_generator, app_db, user_id, request_body),
            media_type="text/event-stream"
        )

    except ValueError as e:
        backend_log.error("Streaming workflow execution validation error", exception=e,
                         metadata={"workflow_name": request_body.workflow_name,
                                 "workflow_id": request_body.workflow_id})
        logger.error(f"Workflow execution error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        backend_log.error("Streaming workflow execution setup failed", exception=e,
                         metadata={"workflow_name": request_body.workflow_name,
                                 "workflow_id": request_body.workflow_id,
                                 "interaction_id": request_body.interaction_id})
        logger.error(f"An unexpected error occurred during workflow setup: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

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
    connection_id = auth_context.get("connection_id")

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

    if not await safe_send_json(
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
    ):
        cleanup_websocket_session(websocket)
        return

    try:
        while True:
            try:
                raw_message = await websocket.receive_json()
            except WebSocketDisconnect:
                backend_log.info(
                    "WebSocket disconnected by client",
                    metadata={"user_id": login_user_id, "session_id": session_id},
                )
                break
            except Exception as exc:
                backend_log.error(
                    "Failed to decode WebSocket message",
                    exception=exc,
                    metadata={"user_id": login_user_id, "session_id": session_id},
                )
                try:
                    await safe_send_json(
                        websocket,
                        {"type": "error", "detail": "Invalid message format"},
                    )
                except Exception:
                    pass
                continue

            if not isinstance(raw_message, dict):
                if not await safe_send_json(
                    websocket,
                    {"type": "error", "detail": "Message must be a JSON object"},
                ):
                    break
                continue

            message_type = raw_message.get("type", "start")

            session_entry = get_session_entry(websocket.app, session_id) if session_id else None
            if message_type == "mcp_capabilities":
                if session_entry is not None:
                    capabilities = session_entry.setdefault("capabilities", {})
                    capabilities[connection_id] = raw_message.get("payload") or {}
                await safe_send_json(websocket, {"type": "mcp_capabilities_ack"})
                continue

            if message_type in ("mcp_response", "mcp_error"):
                payload = raw_message.get("payload") or {}
                request_id = payload.get("request_id") if isinstance(payload, dict) else None
                if session_entry is not None and request_id:
                    pending = session_entry.setdefault("pending_requests", {})
                    future = pending.pop(request_id, None)
                    if future and not future.done():
                        if message_type == "mcp_response":
                            future.set_result(payload)
                        else:
                            error_msg = payload.get("error") or "MCP request failed"
                            future.set_exception(RuntimeError(error_msg))
                continue

            if message_type in ("ping", "pong"):
                await safe_send_json(websocket, {"type": "pong"})
                continue

            if message_type == "close":
                await websocket.close(code=1000)
                break

            if message_type not in ("start", "run", "cancel"):
                if not await safe_send_json(
                    websocket,
                    {"type": "error", "detail": f"Unsupported message type: {message_type}"},
                ):
                    break
                continue

            payload = raw_message.get("payload") or {
                key: value
                for key, value in raw_message.items()
                if key not in ("type", "payload")
            }

            if not isinstance(payload, dict):
                if not await safe_send_json(
                    websocket,
                    {"type": "error", "detail": "Payload must be a JSON object"},
                ):
                    break
                continue

            if message_type == "cancel":
                cancel_reason = raw_message.get("reason")
                cancel_interaction_id = payload.get("interaction_id")
                if not cancel_interaction_id or str(cancel_interaction_id).lower() in ("default", "", "none"):
                    cancel_interaction_id = session_id
                cancel_workflow_id = payload.get("workflow_id")
                if not cancel_workflow_id:
                    if not await safe_send_json(
                        websocket,
                        {
                            "type": "error",
                            "detail": "workflow_id is required for cancellation",
                        },
                    ):
                        break
                    continue

                execution_id = f"{cancel_interaction_id}_{cancel_workflow_id}_{login_user_id}"
                cancelled = execution_manager.cancel_execution(execution_id)
                if not await safe_send_json(
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
                ):
                    break
                if not cancelled:
                    backend_log.warn(
                        "Cancellation requested but no active execution found",
                        metadata={
                            "workflow_id": cancel_workflow_id,
                            "interaction_id": cancel_interaction_id,
                            "user_id": login_user_id,
                        },
                    )
                continue

            if message_type in ("start", "run"):
                active_task = getattr(websocket.state, "active_task", None)
                if active_task and not active_task.done():
                    if not await safe_send_json(
                        websocket,
                        {
                            "type": "error",
                            "detail": "Workflow execution already in progress. Please wait or cancel before starting a new one.",
                        },
                    ):
                        break
                    continue

                try:
                    request_body = WorkflowRequest(**payload)
                except ValidationError as exc:
                    if not await safe_send_json(
                        websocket,
                        {
                            "type": "error",
                            "detail": "Invalid workflow request",
                            "errors": exc.errors(),
                        },
                    ):
                        break
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

                def _clear_task(done_task: asyncio.Task, backend_log=backend_log):
                    try:
                        exception = done_task.exception()
                        if exception:
                            backend_log.error(
                                "Workflow execution task terminated with exception",
                                exception=exception,
                                metadata={
                                    "user_id": login_user_id,
                                    "session_id": session_id,
                                },
                            )
                    except asyncio.CancelledError:
                        backend_log.info(
                            "Workflow execution task cancelled by server",
                            metadata={"user_id": login_user_id, "session_id": session_id},
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

@router.get("/status")
async def get_all_execution_status(request: Request):
    """
    현재 실행 중인 모든 워크플로우의 상태를 반환합니다.
    """
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        backend_log.info("Retrieving all workflow execution status")

        status_data = execution_manager.get_all_execution_status()

        backend_log.success("Successfully retrieved execution status",
                          metadata={"active_executions": len(status_data)})

        return {
            "active_executions": len(status_data),
            "executions": status_data
        }
    except Exception as e:
        backend_log.error("Failed to retrieve execution status", exception=e)
        logger.error(f"실행 상태 조회 중 오류: {e}")
        raise HTTPException(status_code=500, detail="실행 상태 조회 실패")

@router.get("/status/{execution_id}")
async def get_execution_status(request: Request, execution_id: str):
    """
    특정 워크플로우 실행의 상태를 반환합니다.
    """
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        backend_log.info("Retrieving workflow execution status",
                        metadata={"execution_id": execution_id})

        status = execution_manager.get_execution_status(execution_id)
        if status is None:
            backend_log.warn("Execution not found",
                           metadata={"execution_id": execution_id})
            raise HTTPException(status_code=404, detail="실행을 찾을 수 없습니다")

        backend_log.success("Successfully retrieved execution status",
                          metadata={"execution_id": execution_id,
                                  "status": status.get("status", "unknown")})

        return status
    except HTTPException:
        raise
    except Exception as e:
        backend_log.error("Failed to retrieve execution status", exception=e,
                         metadata={"execution_id": execution_id})
        logger.error(f"실행 상태 조회 중 오류: {e}")
        raise HTTPException(status_code=500, detail="실행 상태 조회 실패")

@router.post("/cleanup")
async def cleanup_completed_executions():
    """
    완료된 워크플로우 실행들을 정리합니다.
    """
    try:
        before_count = len(execution_manager.get_all_execution_status())
        execution_manager.cleanup_completed_executions()
        after_count = len(execution_manager.get_all_execution_status())
        cleaned_count = before_count - after_count

        return {
            "message": f"{cleaned_count}개의 완료된 실행이 정리되었습니다",
            "before_count": before_count,
            "after_count": after_count,
            "cleaned_count": cleaned_count
        }
    except Exception as e:
        logger.error(f"실행 정리 중 오류: {e}")
        raise HTTPException(status_code=500, detail="실행 정리 실패")

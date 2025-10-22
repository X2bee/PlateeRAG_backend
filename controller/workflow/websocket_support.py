from typing import Dict, Any, List
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from controller.authController import get_user_by_token
from controller.workflow.models.requests import WorkflowRequest
from controller.workflow.execution_runtime import (
    prepare_workflow_execution,
    persist_stream_results,
)
from service.database.execution_meta_service import update_execution_meta_count


def authenticate_websocket(websocket: WebSocket) -> Dict[str, Any]:
    authorization = websocket.headers.get("authorization")
    if not authorization:
        raise ValueError("Authorization header is missing")
    if not authorization.startswith("Bearer "):
        raise ValueError("Invalid authorization header format")

    token = authorization.replace("Bearer ", "").strip()
    if not token:
        raise ValueError("Token is missing in authorization header")

    user_session = get_user_by_token(websocket.app.state, token)
    if not user_session:
        raise ValueError("Invalid or expired token")

    user_id_header = websocket.headers.get("x-user-id")
    if user_id_header and str(user_session["user_id"]) != str(user_id_header):
        raise ValueError("User ID in header does not match token")

    return user_session


def get_db_manager_from_websocket(websocket: WebSocket):
    if hasattr(websocket.app.state, "app_db") and websocket.app.state.app_db:
        return websocket.app.state.app_db
    raise RuntimeError("Database connection not available")


async def run_websocket_workflow(
    websocket: WebSocket,
    *,
    app_db,
    login_user_id: str,
    request_body: WorkflowRequest,
    backend_log,
) -> None:
    context = await prepare_workflow_execution(
        app=websocket.app,
        app_db=app_db,
        login_user_id=login_user_id,
        request_body=request_body,
        request_like=None,
        backend_log=backend_log,
    )

    if context.execution_meta:
        await update_execution_meta_count(app_db, context.execution_meta)

    metadata_base = {
        "workflow_name": request_body.workflow_name,
        "workflow_id": request_body.workflow_id,
        "interaction_id": request_body.interaction_id,
        "node_count": len(context.nodes),
        "edge_count": len(context.edges),
        "is_interactive": context.execution_meta is not None,
        "workflow_source": context.source,
    }

    backend_log.info(
        "Starting workflow streaming over WebSocket",
        metadata=metadata_base,
    )

    await websocket.send_json(
        {
            "type": "start",
            "content": {
                "workflow_id": request_body.workflow_id,
                "workflow_name": request_body.workflow_name,
                "interaction_id": request_body.interaction_id,
                "is_interactive": context.execution_meta is not None,
            },
        }
    )

    full_response_chunks: List[str] = []
    chunk_count = 0

    try:
        async for chunk in context.executor.execute_workflow_async_streaming():
            if websocket.application_state == WebSocketState.DISCONNECTED:
                raise WebSocketDisconnect()

            chunk_count += 1
            full_response_chunks.append(str(chunk))
            await websocket.send_json({"type": "data", "content": chunk})

        await websocket.send_json({"type": "end", "message": "Stream finished"})

        backend_log.success(
            "Streaming workflow execution completed over WebSocket",
            metadata={
                **metadata_base,
                "chunk_count": chunk_count,
                "total_output_length": len("".join(full_response_chunks)),
            },
        )
    except WebSocketDisconnect:
        backend_log.warn(
            "WebSocket disconnected during workflow streaming",
            metadata={**metadata_base, "chunk_count": chunk_count},
        )
        raise
    except Exception as exc:
        backend_log.error(
            "WebSocket workflow execution failed",
            exception=exc,
            metadata={**metadata_base, "chunk_count": chunk_count},
        )
        try:
            await websocket.send_json(
                {"type": "error", "detail": f"Streaming error: {str(exc)}"}
            )
        except Exception:
            pass
        raise
    finally:
        final_text = "".join(full_response_chunks)
        await persist_stream_results(app_db, login_user_id, request_body, final_text, backend_log)

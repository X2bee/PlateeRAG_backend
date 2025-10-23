from typing import Dict, Any, List, Optional
import asyncio
import logging
import uuid
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from controller.authController import get_user_by_token
from controller.workflow.models.requests import WorkflowRequest
from controller.workflow.execution_runtime import (
    prepare_workflow_execution,
    persist_stream_results,
)
from service.database.execution_meta_service import update_execution_meta_count
from editor.async_workflow_executor import execution_manager, WorkflowCancelledError


async def safe_send_json(websocket: WebSocket, message: Dict[str, Any]) -> bool:
    """Ensure serialized sends are not interleaved across concurrent tasks.

    Returns False when the websocket is already closed so callers can stop sending.
    """
    lock = getattr(websocket.state, "send_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        websocket.state.send_lock = lock
    async with lock:
        try:
            await websocket.send_json(message)
            return True
        except WebSocketDisconnect:
            logger = logging.getLogger("workflow-websocket")
            logger.debug("WebSocket already disconnected; skipping send", exc_info=False)
            return False
        except Exception as exc:  # pragma: no cover - defensive logging
            logger = logging.getLogger("workflow-websocket")
            logger.warning("Failed to send websocket message gracefully: %s", exc, exc_info=False)
            return False
    return True

SESSION_STORE_ATTR = "ws_sessions"


def _get_ws_session_store(websocket: WebSocket) -> Dict[str, Any]:
    """
    Retrieve (or lazily create) the session store dictionary on app state.
    Structure:
        {
            session_id: {
                "users": Set[str],          # participating user IDs
                "connections": {conn_id: user_id}
            }
        }
    """
    if not hasattr(websocket.app.state, SESSION_STORE_ATTR):
        setattr(websocket.app.state, SESSION_STORE_ATTR, {})
    return getattr(websocket.app.state, SESSION_STORE_ATTR)


def _extract_token_and_user(websocket: WebSocket) -> tuple[str, Optional[str]]:
    """
    Extract token and optional user identifier from headers or query parameters.
    """
    token: Optional[str] = None

    authorization = websocket.headers.get("authorization")
    if authorization and authorization.startswith("Bearer "):
        token = authorization.replace("Bearer ", "").strip()

    if not token:
        token = websocket.query_params.get("token")

    if not token:
        raise ValueError("Authorization token is missing")

    user_candidate = websocket.headers.get("x-user-id") or websocket.query_params.get("user_id")
    return token, user_candidate


def authenticate_websocket(websocket: WebSocket) -> Dict[str, Any]:
    """
    Validate the websocket connection, create or reuse a chat session, and
    return context containing user and session information.
    Multiple users can share the same session.
    """
    token, user_candidate = _extract_token_and_user(websocket)

    user_session = get_user_by_token(websocket.app.state, token)
    if not user_session:
        raise ValueError("Invalid or expired token")

    user_id = str(user_session["user_id"])
    if user_candidate and str(user_candidate) != user_id:
        raise ValueError("User ID does not match token")

    session_store = _get_ws_session_store(websocket)
    session_id = websocket.query_params.get("session_id") or str(uuid.uuid4())
    connection_id = str(uuid.uuid4())

    event_loop = asyncio.get_running_loop()
    session_entry = session_store.setdefault(
        session_id,
        {
            "users": set(),
            "connections": {},
            "websockets": {},
            "pending_requests": {},
            "capabilities": {},
            "loop": event_loop,
        },
    )

    session_entry.setdefault("users", set()).add(user_id)
    session_entry.setdefault("connections", {})[connection_id] = user_id
    session_entry.setdefault("websockets", {})[connection_id] = websocket
    session_entry.setdefault("pending_requests", {})
    session_entry.setdefault("capabilities", {})
    # store event loop reference if not already present
    session_entry.setdefault("loop", event_loop)

    websocket.state.session_id = session_id
    websocket.state.connection_id = connection_id

    return {
        "user": user_session,
        "session_id": session_id,
        "token": token,
        "connection_id": connection_id,
        "participants": list(session_entry["users"]),
        "capabilities": session_entry["capabilities"].get(connection_id, {}),
    }


def cleanup_websocket_session(websocket: WebSocket) -> None:
    """
    Remove the websocket session record when the connection is closed.
    """
    session_id = getattr(websocket.state, "session_id", None)
    connection_id = getattr(websocket.state, "connection_id", None)
    if not session_id or not connection_id:
        return

    session_store = getattr(websocket.app.state, SESSION_STORE_ATTR, None)
    if not isinstance(session_store, dict):
        return

    session_entry = session_store.get(session_id)
    if not session_entry:
        return

    user_id = session_entry["connections"].pop(connection_id, None)
    if "websockets" in session_entry:
        session_entry["websockets"].pop(connection_id, None)
    if "capabilities" in session_entry:
        session_entry["capabilities"].pop(connection_id, None)

    # Recompute active users based on remaining connections
    remaining_users = set(session_entry["connections"].values())
    session_entry["users"] = remaining_users

    if not session_entry["connections"]:
        pending = session_entry.get("pending_requests", {})
        for future in pending.values():
            if future and not future.done():
                future.set_exception(RuntimeError("WebSocket session closed"))
        pending.clear()
        session_store.pop(session_id, None)


def get_db_manager_from_websocket(websocket: WebSocket):
    if hasattr(websocket.app.state, "app_db") and websocket.app.state.app_db:
        return websocket.app.state.app_db
    raise RuntimeError("Database connection not available")


def get_session_entry(app, session_id: str) -> Optional[Dict[str, Any]]:
    session_store = getattr(app.state, SESSION_STORE_ATTR, None)
    if not isinstance(session_store, dict):
        return None
    return session_store.get(session_id)


def _select_websocket_for_session(session_entry: Dict[str, Any], server_name: Optional[str] = None):
    websockets = session_entry.get("websockets", {})
    capabilities = session_entry.get("capabilities", {})

    selected_ws = None
    selected_connection_id = None
    selected_server = server_name

    for conn_id, websocket in websockets.items():
        caps = capabilities.get(conn_id) or {}
        servers = caps.get("servers") or []
        if server_name:
            for server in servers:
                if server.get("name") == server_name or server.get("id") == server_name:
                    selected_ws = websocket
                    selected_connection_id = conn_id
                    selected_server = server.get("name") or server.get("id")
                    break
            if selected_ws:
                break
        else:
            selected_ws = websocket
            selected_connection_id = conn_id
            if servers:
                selected_server = servers[0].get("name") or servers[0].get("id")
            break

    if not selected_ws and websockets and not server_name:
        selected_connection_id, selected_ws = next(iter(websockets.items()))

    if server_name and not selected_ws:
        raise RuntimeError(f"Requested MCP server '{server_name}' is not available for this session")

    return selected_connection_id, selected_ws, selected_server


async def _dispatch_mcp_request(
    session_entry: Dict[str, Any],
    session_id: str,
    payload: Dict[str, Any],
    *,
    server_name: Optional[str] = None,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dictionary")

    connection_id, websocket, resolved_server = _select_websocket_for_session(session_entry, server_name)
    if not websocket:
        raise RuntimeError("No active WebSocket connection available for the session")

    pending_requests: Dict[str, asyncio.Future] = session_entry.setdefault("pending_requests", {})
    request_id = str(uuid.uuid4())
    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()
    pending_requests[request_id] = future

    message_content = {
        "request_id": request_id,
        "session_id": session_id,
        "connection_id": connection_id,
        "payload": payload,
    }
    if resolved_server:
        message_content["server"] = resolved_server

    message = {
        "type": "mcp_request",
        "content": message_content,
    }

    send_success = await safe_send_json(websocket, message)
    if not send_success:
        pending_requests.pop(request_id, None)
        raise RuntimeError("Failed to deliver MCP request to client")

    try:
        response = await asyncio.wait_for(future, timeout=timeout)
        return response
    except asyncio.TimeoutError as exc:
        if not future.done():
            future.set_exception(RuntimeError("Timed out waiting for client MCP response"))
        raise RuntimeError("Timed out waiting for client MCP response") from exc
    finally:
        pending_requests.pop(request_id, None)


def call_client_mcp(
    app,
    session_id: str,
    payload: Dict[str, Any],
    *,
    server_name: Optional[str] = None,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """Synchronously request the connected client to execute an MCP call."""
    session_store = getattr(app.state, SESSION_STORE_ATTR, None)
    if not isinstance(session_store, dict):
        raise RuntimeError("WebSocket session store is not initialized")

    session_entry = session_store.get(session_id)
    if not session_entry:
        raise RuntimeError("Active WebSocket session not found for provided session_id")

    loop = session_entry.get("loop")
    if loop is None:
        raise RuntimeError("Event loop reference missing for WebSocket session")

    if server_name:
        capabilities = session_entry.get("capabilities", {})
        available = False
        for caps in capabilities.values():
            servers = (caps or {}).get("servers") or []
            if any(s.get("name") == server_name or s.get("id") == server_name for s in servers):
                available = True
                break
        if not available:
            raise RuntimeError(f"Client does not advertise MCP server '{server_name}'")

    coro = _dispatch_mcp_request(session_entry, session_id, payload, server_name=server_name, timeout=timeout)
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


async def run_websocket_workflow(
    websocket: WebSocket,
    *,
    app_db,
    login_user_id: str,
    request_body: WorkflowRequest,
    backend_log,
) -> None:
    cancelled = False
    session_id = getattr(websocket.state, "session_id", None)
    if (not request_body.interaction_id or request_body.interaction_id == "default") and session_id:
        request_body.interaction_id = session_id

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

    if not await safe_send_json(
        websocket,
        {
            "type": "start",
            "content": {
                "workflow_id": request_body.workflow_id,
                "workflow_name": request_body.workflow_name,
                "interaction_id": request_body.interaction_id,
                "is_interactive": context.execution_meta is not None,
            },
        },
    ):
        raise WebSocketDisconnect(code=1000)

    full_response_chunks: List[str] = []
    chunk_count = 0

    try:
        async for chunk in context.executor.execute_workflow_async_streaming():
            if websocket.application_state == WebSocketState.DISCONNECTED:
                raise WebSocketDisconnect()

            chunk_count += 1
            full_response_chunks.append(str(chunk))
            if not await safe_send_json(websocket, {"type": "data", "content": chunk}):
                raise WebSocketDisconnect(code=1000)

        if not await safe_send_json(websocket, {"type": "end", "message": "Stream finished"}):
            raise WebSocketDisconnect(code=1000)

        backend_log.success(
            "Streaming workflow execution completed over WebSocket",
            metadata={
                **metadata_base,
                "chunk_count": chunk_count,
                "total_output_length": len("".join(full_response_chunks)),
            },
        )
    except WorkflowCancelledError:
        cancelled = True
        backend_log.info(
            "Workflow execution cancelled over WebSocket",
            metadata={**metadata_base, "chunk_count": chunk_count},
        )
        send_ok = await safe_send_json(
            websocket,
            {
                "type": "cancelled",
                "content": {
                    "workflow_id": request_body.workflow_id,
                    "workflow_name": request_body.workflow_name,
                    "interaction_id": request_body.interaction_id,
                },
            },
        )
        if not send_ok:
            raise WebSocketDisconnect(code=1000)
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
            send_ok = await safe_send_json(
                websocket,
                {"type": "error", "detail": f"Streaming error: {str(exc)}"},
            )
            if not send_ok:
                raise WebSocketDisconnect(code=1000)
        except Exception:
            pass
        raise
    finally:
        final_text = "".join(full_response_chunks) if not cancelled else ""
        if not cancelled:
            await persist_stream_results(app_db, login_user_id, request_body, final_text, backend_log)
        execution_manager.cleanup_completed_executions()

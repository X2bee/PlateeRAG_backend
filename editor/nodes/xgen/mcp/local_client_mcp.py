import logging
from typing import Any, Dict, List, Optional

from editor.node_composer import Node
from langchain_core.tools import tool

from controller.workflow.websocket_support import call_client_mcp

logger = logging.getLogger(__name__)


class LocalClientMCP(Node):
    """Expose client-local MCP tools to the agent just like remote MCP nodes."""

    categoryId = "xgen"
    functionId = "local_client_mcp"
    nodeId = "mcp/LocalClientMCP"
    nodeName = "Local MCP (Client)"
    description = "클라이언트 로컬에서 실행 중인 MCP 서버의 도구를 에이전트 툴로 제공합니다."
    tags = ["mcp", "client", "tool", "agent", "bridge"]

    inputs: List[Dict[str, Any]] = []
    outputs: List[Dict[str, Any]] = [
        {"id": "tools", "name": "Tools", "type": "TOOL"},
    ]

    parameters: List[Dict[str, Any]] = [
        {
            "id": "session_id",
            "name": "Session ID",
            "type": "STR",
            "value": "",
            "optional": True,
            "description": "웹소켓 세션 ID. 비워두면 현재 interaction_id를 사용합니다.",
        },
        {
            "id": "server_name",
            "name": "Server Name",
            "type": "STR",
            "value": "",
            "optional": True,
            "description": "클라이언트가 광고한 MCP 서버 이름.",
        },
        {
            "id": "timeout",
            "name": "Timeout (sec)",
            "type": "FLOAT",
            "value": 30.0,
            "optional": True,
            "description": "클라이언트 응답 대기 시간(초).",
        },
    ]

    def _resolve_session_id(self, provided_session: Optional[str], **kwargs) -> str:
        interaction_id = getattr(self, "interaction_id", None)
        if provided_session:
            return provided_session
        if kwargs.get("interaction_id"):
            return kwargs["interaction_id"]
        if interaction_id:
            return interaction_id
        raise ValueError("session_id (또는 interaction_id) 값을 확인할 수 없습니다.")

    def _request_tools(self, app, session_id: str, server_name: Optional[str], timeout: float) -> List[Dict[str, Any]]:
        payload = {"method": "tools/list"}
        logger.debug("Requesting tool list from client session=%s server=%s", session_id, server_name)
        response = call_client_mcp(
            app,
            session_id=session_id,
            payload=payload,
            server_name=server_name,
            timeout=timeout,
        )
        if not isinstance(response, dict):
            raise RuntimeError("Invalid MCP tool list response from client")

        if response.get("success") is False:
            raise RuntimeError(response.get("error") or "Client MCP tools/list call failed")

        result = response.get("result")
        if isinstance(result, dict) and "tools" in result:
            tools = result["tools"]
        else:
            tools = result

        if not isinstance(tools, list):
            raise RuntimeError("tools/list 응답 형식이 올바르지 않습니다.")

        return tools

    def _build_langchain_tool(
        self,
        app,
        session_id: str,
        server_name: Optional[str],
        timeout: float,
        tool_meta: Dict[str, Any],
    ):
        tool_name = tool_meta.get("name") or tool_meta.get("id")
        if not tool_name:
            raise ValueError("MCP tool metadata에 name이 없습니다.")

        description = tool_meta.get("description", "Client local MCP tool")
        input_schema = tool_meta.get("inputSchema", {})

        logger.debug(
            "Creating local MCP tool wrapper: session=%s server=%s tool=%s",
            session_id,
            server_name,
            tool_name,
        )

        def call_tool(arguments: Dict[str, Any]) -> Any:
            payload = {
                "method": "tools/call",
                "tool": tool_name,
                "arguments": arguments or {},
            }
            return call_client_mcp(
                app,
                session_id=session_id,
                payload=payload,
                server_name=server_name,
                timeout=timeout,
            )

        # wrap with langchain tool decorator to match loader behaviour
        @tool(tool_name, description=description)
        def local_mcp_tool(**kwargs):
            try:
                logger.info(
                    "Invoking local MCP tool '%s' for session '%s' (server=%s) with kwargs=%s",
                    tool_name,
                    session_id,
                    server_name,
                    kwargs,
                )
                response = call_tool(kwargs)
                if isinstance(response, dict):
                    if response.get("success") is False:
                        error_msg = response.get("error") or "Client MCP call failed"
                        raise RuntimeError(error_msg)
                    return response.get("result", response)
                return response
            except Exception as exc:
                logger.error("Error executing local MCP tool %s: %s", tool_name, exc)
                raise

        local_mcp_tool.input_schema = input_schema  # type: ignore[attr-defined]
        return local_mcp_tool

    def execute(
        self,
        session_id: str = "",
        server_name: str = "",
        timeout: float = 30.0,
        *args,
        **kwargs,
    ):
        logger.debug(
            "LocalClientMCP.execute called session_id=%s server_name=%s timeout=%s",
            session_id,
            server_name,
            timeout,
        )

        app = getattr(self, "app", None)
        if app is None:
            raise RuntimeError("Application context is not available for LocalClientMCP node")

        resolved_session_id = self._resolve_session_id(session_id, **kwargs)
        resolved_server_name = server_name or None

        tools_meta = self._request_tools(app, resolved_session_id, resolved_server_name, timeout)

        langchain_tools = []
        for meta in tools_meta:
            try:
                tool_wrapper = self._build_langchain_tool(
                    app=app,
                    session_id=resolved_session_id,
                    server_name=resolved_server_name,
                    timeout=timeout,
                    tool_meta=meta,
                )
                langchain_tools.append(tool_wrapper)
            except Exception as exc:
                logger.error("Failed to create local MCP tool wrapper: %s", exc)
                continue

        if not langchain_tools:
            logger.warning("No usable MCP tools returned by client for session %s", resolved_session_id)

        return langchain_tools

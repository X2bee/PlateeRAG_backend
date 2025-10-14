import logging
import httpx
from typing import Any, Dict, List
from editor.node_composer import Node
from langchain.agents import tool
from fastapi import Request
from editor.utils.helper.async_helper import sync_run_async
from controller.mcpController import MCP_STATION_BASE_URL

logger = logging.getLogger(__name__)

class MCPLoader(Node):
    categoryId = "xgen"
    functionId = "mcp"
    nodeId = "mcp/MCPLoader"
    nodeName = "MCP Tool Loader"
    description = "MCP 서버 세션의 도구들을 Langchain Agent용 Tool로 변환하여 전달"
    tags = ["mcp", "tool", "agent", "setup"]

    inputs = []
    outputs = [
        {"id": "tools", "name": "Tools", "type": "TOOL"},
    ]

    parameters = [
        {
            "id": "session_id",
            "name": "MCP Session",
            "type": "STR",
            "value": "Select Session",
            "required": True,
            "is_api": True,
            "api_name": "api_sessions",
            "options": []
        },
    ]

    async def fetch_sessions(self) -> List[Dict[str, Any]]:
        """MCP Station에서 활성 세션 목록을 가져옵니다."""
        url = f"{MCP_STATION_BASE_URL}/sessions"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch MCP sessions: {e}")
            return []

    async def fetch_session_tools(self, session_id: str) -> List[Dict[str, Any]]:
        """특정 세션의 도구 목록을 가져옵니다."""
        url = f"{MCP_STATION_BASE_URL}/sessions/{session_id}/tools"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                result = response.json()
                return result.get("tools", [])
        except Exception as e:
            logger.error(f"Failed to fetch tools for session {session_id}: {e}")
            return []

    async def call_mcp_tool(self, session_id: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """MCP 도구를 호출합니다."""
        url = f"{MCP_STATION_BASE_URL}/mcp/request"
        payload = {
            "session_id": session_id,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }

        logger.info(f"Calling MCP Station with payload: {payload}")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                result = response.json()

                logger.info(f"MCP Station response: {result}")

                if result.get("success"):
                    data = result.get("data", {})
                    # MCP 응답에서 실제 내용 추출
                    if isinstance(data, dict):
                        # content 배열이 있는 경우
                        if "content" in data:
                            contents = data["content"]
                            if isinstance(contents, list) and len(contents) > 0:
                                # 첫 번째 content의 text 반환
                                first_content = contents[0]
                                if isinstance(first_content, dict) and "text" in first_content:
                                    return first_content["text"]
                        # 직접 결과가 있는 경우
                        return str(data)
                    return str(data)
                else:
                    error_msg = result.get("error", "Unknown error")
                    logger.error(f"MCP tool call failed: {error_msg}")
                    return f"Error: {error_msg}"

        except Exception as e:
            logger.error(f"Failed to call MCP tool {tool_name}: {e}")
            return f"Error calling tool: {str(e)}"

    def api_sessions(self, request: Request) -> List[Dict[str, str]]:
        """API 드롭다운에 표시할 세션 목록을 반환합니다."""
        sessions = sync_run_async(self.fetch_sessions())

        options = []
        for session in sessions:
            session_id = session.get("session_id", "")
            session_name = session.get("session_name", "")
            server_type = session.get("server_type", "")
            status = session.get("status", "")

            # 실행 중인 세션만 포함
            if status == "running":
                label = f"{session_name or session_id[:8]} ({server_type})" if session_name else f"{session_id[:8]}... ({server_type})"
                options.append({
                    "value": session_id,
                    "label": label
                })

        return options

    def execute(self, session_id: str, *args, **kwargs):
        """MCP 세션의 도구들을 Langchain Tool로 변환하여 반환합니다."""

        # 세션의 도구 목록 가져오기
        mcp_tools = sync_run_async(self.fetch_session_tools(session_id))

        if not mcp_tools:
            logger.warning(f"No tools found for session {session_id}")
            return []

        logger.info(f"Found {len(mcp_tools)} tools in session {session_id}")

        # 각 MCP 도구를 Langchain Tool로 변환
        langchain_tools = []

        for mcp_tool in mcp_tools:
            tool_name = mcp_tool.get("name", "unknown_tool")
            tool_description = mcp_tool.get("description", "No description available")
            input_schema = mcp_tool.get("inputSchema", {})

            # 도구별로 클로저를 만들어 고유한 함수 생성
            def create_tool_function(t_name: str, t_session_id: str, t_schema: Dict[str, Any], t_description: str):
                @tool(t_name, description=t_description)
                def mcp_tool_wrapper(**kwargs) -> str:
                    """MCP 도구를 호출하는 래퍼 함수"""
                    try:
                        logger.info(f"Executing MCP tool {t_name} with kwargs: {kwargs}")

                        # 모든 인자를 그대로 전달 (스키마 검증은 MCP 서버에서 수행)
                        arguments = kwargs

                        logger.info(f"Calling MCP tool {t_name} with arguments: {arguments}")

                        # MCP 도구 호출
                        result = sync_run_async(
                            self.call_mcp_tool(t_session_id, t_name, arguments)
                        )

                        logger.info(f"MCP tool {t_name} returned: {result}")
                        return str(result)

                    except Exception as e:
                        logger.error(f"Error executing MCP tool {t_name}: {e}")
                        return f"Error: {str(e)}"

                return mcp_tool_wrapper

            # 도구 생성
            langchain_tool = create_tool_function(tool_name, session_id, input_schema, tool_description)
            langchain_tools.append(langchain_tool)

            logger.info(f"Created Langchain tool for MCP tool: {tool_name}")

        return langchain_tools

"""
MCP Station Node for Workflow Editor

MCP Station의 세션을 선택하고, 해당 세션의 도구들을
LangChain Tool로 변환하여 Agent에게 제공하는 노드
"""
import logging
import httpx
from typing import List, Dict, Any, Optional

# Node 클래스가 있는 경우 import, 없으면 Mock 사용
try:
    from editor.node_composer import Node
except ImportError:
    # Mock Node class for testing
    class Node:
        """Workflow Editor Node의 Mock 구현"""
        categoryId = ""
        functionId = ""
        nodeId = ""
        nodeName = ""
        description = ""
        tags = []
        inputs = []
        outputs = []
        parameters = []

        def execute(self, *args, **kwargs):
            raise NotImplementedError("Subclass must implement execute()")

from langchain.tools import tool, BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MCPStationNode(Node):
    categoryId = "xgen"
    functionId = "mcp"
    nodeId = "mcp/mcp_station"
    nodeName = "MCP Station"
    description = "MCP Station의 세션에 연결하여 MCP 도구들을 Agent에게 제공합니다."
    tags = ["mcp", "tools", "integration"]

    inputs = []
    outputs = [
        {"id": "tools", "name": "Tools", "type": "TOOL"},
    ]

    parameters = [
        {
            "id": "mcp_station_url",
            "name": "MCP Station URL",
            "type": "STRING",
            "value": "http://mcp_station:20100",
            "required": True,
            "description": "MCP Station 서버 URL"
        },
        {
            "id": "session_id",
            "name": "Session ID",
            "type": "STRING",
            "value": "",
            "required": True,
            "description": "사용할 MCP 세션 ID (GET /sessions로 조회 가능)"
        },
    ]

    def execute(
        self,
        mcp_station_url: str = "http://mcp_station:20100",
        session_id: str = "",
        *args,
        **kwargs
    ):
        """
        MCP Station 세션의 도구들을 LangChain Tool로 변환

        Returns:
            List[BaseTool]: Agent에서 사용할 수 있는 도구 목록
        """
        if not session_id:
            raise ValueError("Session ID가 필요합니다. GET /sessions로 세션 목록을 확인하세요.")

        logger.info(f"Connecting to MCP Station: {mcp_station_url}")
        logger.info(f"Using session: {session_id}")

        # MCP 도구 메타데이터 가져오기
        with httpx.Client(timeout=30.0) as client:
            try:
                response = client.get(f"{mcp_station_url}/sessions/{session_id}/tools")
                response.raise_for_status()
                tools_data = response.json()
            except Exception as e:
                logger.error(f"Failed to fetch tools from MCP Station: {e}")
                raise RuntimeError(f"MCP Station 연결 실패: {e}")

        mcp_tools = tools_data.get("tools", [])
        logger.info(f"Found {len(mcp_tools)} MCP tools")

        # LangChain Tool로 변환
        langchain_tools = []
        for mcp_tool in mcp_tools:
            tool_name = mcp_tool["name"]
            tool_description = mcp_tool.get("description", "")
            input_schema = mcp_tool.get("inputSchema", {})

            logger.info(f"Creating LangChain tool: {tool_name}")

            # 동적으로 LangChain Tool 생성
            langchain_tool = self._create_langchain_tool(
                tool_name=tool_name,
                tool_description=tool_description,
                input_schema=input_schema,
                mcp_station_url=mcp_station_url,
                session_id=session_id
            )

            langchain_tools.append(langchain_tool)

        logger.info(f"Successfully created {len(langchain_tools)} LangChain tools")
        return langchain_tools

    def _create_langchain_tool(
        self,
        tool_name: str,
        tool_description: str,
        input_schema: Dict[str, Any],
        mcp_station_url: str,
        session_id: str
    ) -> BaseTool:
        """
        개별 MCP 도구를 LangChain Tool로 변환
        """
        # JSON Schema에서 파라미터 추출
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])

        # 도구 함수 동적 생성
        def create_tool_function():
            """실제 MCP Station으로 요청을 보내는 함수"""

            # 파라미터 문자열 생성 (tool 데코레이터용)
            param_descriptions = []
            for prop_name, prop_info in properties.items():
                prop_desc = prop_info.get("description", "")
                param_descriptions.append(f"{prop_name}: {prop_desc}")

            full_description = f"{tool_description}\n\nParameters:\n" + "\n".join(f"- {d}" for d in param_descriptions)

            @tool(tool_name, return_direct=False)
            def mcp_tool_function(**kwargs) -> str:
                """MCP Station으로 도구 호출 프록시"""
                try:
                    with httpx.Client(timeout=60.0) as client:
                        response = client.post(
                            f"{mcp_station_url}/mcp/request",
                            json={
                                "session_id": session_id,
                                "method": "tools/call",
                                "params": {
                                    "name": tool_name,
                                    "arguments": kwargs
                                }
                            }
                        )
                        response.raise_for_status()
                        result = response.json()

                        if not result.get("success"):
                            error_msg = result.get("error", "Unknown error")
                            logger.error(f"MCP tool call failed: {error_msg}")
                            return f"Error: {error_msg}"

                        # 결과 파싱
                        data = result.get("data", {})

                        # MCP 표준 응답 형식 처리
                        if isinstance(data, dict) and "content" in data:
                            content = data["content"]
                            if isinstance(content, list) and len(content) > 0:
                                return content[0].get("text", str(data))

                        # 일반 응답
                        return str(data)

                except Exception as e:
                    logger.error(f"Error calling MCP tool {tool_name}: {e}")
                    return f"Error calling tool: {str(e)}"

            # Description 업데이트
            mcp_tool_function.description = full_description

            return mcp_tool_function

        return create_tool_function()


class MCPSessionListNode(Node):
    """
    MCP Station의 세션 목록을 조회하는 헬퍼 노드
    Workflow에서 어떤 세션이 있는지 확인할 때 사용
    """
    categoryId = "xgen"
    functionId = "mcp"
    nodeId = "mcp/list_sessions"
    nodeName = "MCP List Sessions"
    description = "MCP Station의 활성 세션 목록을 조회합니다."
    tags = ["mcp", "util"]

    inputs = []
    outputs = [
        {"id": "session_ids", "name": "Session IDs", "type": "LIST"},
    ]

    parameters = [
        {
            "id": "mcp_station_url",
            "name": "MCP Station URL",
            "type": "STRING",
            "value": "http://mcp_station:20100",
            "required": True,
            "description": "MCP Station 서버 URL"
        },
    ]

    def execute(
        self,
        mcp_station_url: str = "http://mcp_station:20100",
        *args,
        **kwargs
    ):
        """
        MCP Station의 세션 목록 반환

        Returns:
            List[Dict]: 세션 정보 목록
        """
        logger.info(f"Fetching sessions from: {mcp_station_url}")

        with httpx.Client(timeout=10.0) as client:
            try:
                response = client.get(f"{mcp_station_url}/sessions")
                response.raise_for_status()
                sessions = response.json()

                logger.info(f"Found {len(sessions)} active sessions")

                # 세션 정보 로깅
                for session in sessions:
                    logger.info(
                        f"Session: {session['session_id']} "
                        f"(Type: {session['server_type']}, Status: {session['status']})"
                    )

                return sessions

            except Exception as e:
                logger.error(f"Failed to fetch sessions: {e}")
                raise RuntimeError(f"세션 목록 조회 실패: {e}")


class MCPCreateSessionNode(Node):
    """
    새로운 MCP 서버 세션을 생성하는 노드
    Workflow 시작 시 필요한 MCP 서버를 자동으로 시작
    """
    categoryId = "xgen"
    functionId = "mcp"
    nodeId = "mcp/create_session"
    nodeName = "MCP Create Session"
    description = "MCP Station에 새로운 MCP 서버 세션을 생성합니다."
    tags = ["mcp", "setup"]

    inputs = []
    outputs = [
        {"id": "session_id", "name": "Session ID", "type": "STRING"},
    ]

    parameters = [
        {
            "id": "mcp_station_url",
            "name": "MCP Station URL",
            "type": "STRING",
            "value": "http://mcp_station:20100",
            "required": True,
            "description": "MCP Station 서버 URL"
        },
        {
            "id": "server_type",
            "name": "Server Type",
            "type": "STRING",
            "value": "node",
            "required": True,
            "description": "MCP 서버 타입 (node 또는 python)"
        },
        {
            "id": "server_command",
            "name": "Server Command",
            "type": "STRING",
            "value": "npx",
            "required": True,
            "description": "실행할 명령어"
        },
        {
            "id": "server_args",
            "name": "Server Args",
            "type": "STRING",
            "value": "-y,@upstash/context7-mcp",
            "required": False,
            "description": "명령어 인자 (쉼표로 구분)"
        },
        {
            "id": "env_vars",
            "name": "Environment Variables",
            "type": "STRING",
            "value": "",
            "required": False,
            "optional": True,
            "description": "환경변수 (JSON 형식, 예: {\"API_KEY\":\"xxx\",\"DEBUG\":\"true\"})"
        },
        {
            "id": "working_dir",
            "name": "Working Directory",
            "type": "STRING",
            "value": "",
            "required": False,
            "optional": True,
            "description": "작업 디렉토리 (선택적)"
        },
    ]

    def execute(
        self,
        mcp_station_url: str = "http://mcp_station:20100",
        server_type: str = "node",
        server_command: str = "npx",
        server_args: str = "-y,@upstash/context7-mcp",
        env_vars: str = "",
        working_dir: str = "",
        *args,
        **kwargs
    ):
        """
        MCP 세션 생성

        Returns:
            str: 생성된 세션 ID
        """
        logger.info(f"Creating new MCP session: {server_type} {server_command}")

        # 인자 파싱
        args_list = [arg.strip() for arg in server_args.split(",") if arg.strip()]

        # 환경변수 파싱 (JSON 문자열 → Dict)
        import json
        env_dict = {}
        if env_vars:
            try:
                env_dict = json.loads(env_vars)
                logger.info(f"Parsed environment variables: {list(env_dict.keys())}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse env_vars as JSON: {e}")
                raise ValueError(f"env_vars must be valid JSON: {e}")

        with httpx.Client(timeout=30.0) as client:
            try:
                # 요청 페이로드 구성
                payload = {
                    "server_type": server_type,
                    "server_command": server_command,
                    "server_args": args_list
                }

                # 환경변수 추가 (있는 경우)
                if env_dict:
                    payload["env_vars"] = env_dict

                # 작업 디렉토리 추가 (있는 경우)
                if working_dir:
                    payload["working_dir"] = working_dir

                logger.info(f"Sending session creation request: {payload}")

                response = client.post(
                    f"{mcp_station_url}/sessions",
                    json=payload
                )
                response.raise_for_status()
                session_info = response.json()

                session_id = session_info["session_id"]
                logger.info(f"Session created: {session_id}")

                # 초기화 대기
                import time
                logger.info("Waiting for MCP server initialization (3 seconds)...")
                time.sleep(3)

                return session_id

            except Exception as e:
                logger.error(f"Failed to create session: {e}")
                raise RuntimeError(f"세션 생성 실패: {e}")

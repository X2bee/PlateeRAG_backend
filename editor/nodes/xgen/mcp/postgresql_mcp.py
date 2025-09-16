import logging
import os
from editor.node_composer import Node
from .postgresql_mcp_helper import PostgreSQLMCPTool

logger = logging.getLogger(__name__)

class PostgreSQLMCP(Node):
    categoryId = "xgen"
    functionId = "mcp"
    nodeId = "mcp/postgresql_mcp"
    nodeName = "PostgreSQL MCP"
    description = "MCP server that provides read-only access to PostgreSQL databases"
    tags = ["mcp", "postgresql", "database", "sql", "read-only"]

    inputs = [
    ]
    outputs = [
        {"id": "tools", "name": "Tools", "type": "TOOL"},
    ]

    parameters = [
        {"id": "host", "name": "Host", "type": "STR", "value": "localhost", "required": True, "description": "PostgreSQL 서버 호스트"},
        {"id": "port", "name": "Port", "type": "INT", "value": 5432, "required": True, "description": "PostgreSQL 서버 포트"},
        {"id": "username", "name": "Username", "type": "STR", "value": "", "required": True, "description": "데이터베이스 사용자명"},
        {"id": "password", "name": "Password", "type": "STR", "value": "", "required": True, "description": "데이터베이스 비밀번호"},
        {"id": "database", "name": "Database", "type": "STR", "value": "", "required": True, "description": "데이터베이스 이름"},
        {"id": "db_prompt", "name": "DB Prompt", "type": "STR", "value": "", "required": False, "optional": True, "expandable": True, "description": "포함할 테이블 목록 (쉼표로 구분)"},
        {"id": "name", "name": "MCP Name", "type": "STR", "value": "postgresql_mcp", "required": False, "optional": True, "description": "해당 MCP 도구의 이름을 지정합니다. postgre 관련 설정을 여러개 사용할 경우 구분하기 위해 설정합니다."},
        {"id": "description", "name": "MCP Description", "type": "STR", "value": "PostgreSQL 데이터베이스에 대한 읽기 전용 접근을 제공합니다.", "required": False, "optional": True, "description": "해당 MCP 도구의 설명을 지정합니다. PostgreSQL MCP 노드를 여러개 사용할 경우 구분하기 위해 설정합니다."},
        {"id": "sample_rows_in_table_info", "name": "Sample Rows", "type": "INT", "value": 3, "required": False, "optional": True, "description": "테이블 정보에 포함할 샘플 행 수"},
    ]

    def execute(self, host: str, port: int = 5432, username: str = 'postgres', password: str = '', database: str = 'postgres', db_prompt: str = '', name: str = 'postgresql_mcp', description: str = 'PostgreSQL 데이터베이스에 대한 읽기 전용 접근을 제공합니다.', *args, **kwargs):
        try:
            include_tables = kwargs.get("include_tables", "")
            sample_rows = kwargs.get("sample_rows_in_table_info", 3)

            # 필수 파라미터 검증
            if not username:
                raise ValueError("사용자명이 필요합니다.")
            if not password:
                raise ValueError("비밀번호가 필요합니다.")
            if not database:
                raise ValueError("데이터베이스 이름이 필요합니다.")

            # PostgreSQL URL 생성
            postgres_url = f"postgresql://{username}:{password}@{host}:{port}/{database}"

            # PostgreSQL MCP 도구 생성
            postgresql_tool = PostgreSQLMCPTool(postgres_url=postgres_url, db_prompt=db_prompt, name=name, description=description)

            logger.info("PostgreSQL MCP 도구가 성공적으로 생성되었습니다. (Host: %s:%s, Database: %s)",
                       host, port, database)
            return postgresql_tool

        except Exception as e:
            logger.error("PostgreSQL MCP 도구 생성 중 오류 발생: %s", str(e))
            raise e

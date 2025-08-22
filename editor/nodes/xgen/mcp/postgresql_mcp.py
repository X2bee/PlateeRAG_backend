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
        {"id": "postgres_url", "name": "PostgreSQL URL", "type": "STR", "value": "", "required": True, "description": "PostgreSQL 데이터베이스 연결 URL입니다. (예: postgresql://user:password@host:port/database)"},
        {"id": "include_tables", "name": "Include Tables", "type": "STR", "value": "", "required": False, "optional": True, "description": "포함할 테이블 목록 (쉼표로 구분)"},
        {"id": "sample_rows_in_table_info", "name": "Sample Rows", "type": "INT", "value": 3, "required": False, "optional": True, "description": "테이블 정보에 포함할 샘플 행 수"},
    ]

    def execute(self, *args, **kwargs):
        try:
            # 파라미터 추출
            postgres_url = kwargs.get("postgres_url", "")
            include_tables = kwargs.get("include_tables", "")
            sample_rows = kwargs.get("sample_rows_in_table_info", 3)

            if not postgres_url:
                raise ValueError("PostgreSQL URL이 필요합니다.")

            # PostgreSQL MCP 도구 생성
            postgresql_tool = PostgreSQLMCPTool(postgres_url=postgres_url)

            logger.info("PostgreSQL MCP 도구가 성공적으로 생성되었습니다. (URL: %s)",
                       postgres_url.split('@')[-1] if '@' in postgres_url else "***")
            return postgresql_tool

        except Exception as e:
            logger.error("PostgreSQL MCP 도구 생성 중 오류 발생: %s", str(e))
            raise e

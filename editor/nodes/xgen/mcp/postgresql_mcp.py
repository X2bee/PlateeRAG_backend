import logging
import os
from editor.node_composer import Node
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class PostgreSQLInput(BaseModel):
    query: str = Field(description="실행할 SQL 쿼리")

class PostgreSQLMCPTool(BaseTool):
    name = "postgresql_mcp"
    description = "PostgreSQL 데이터베이스에 대한 읽기 전용 접근을 제공합니다."
    args_schema: Type[BaseModel] = PostgreSQLInput

    def __init__(self, postgres_url: str):
        super().__init__()
        self.postgres_url = postgres_url

        # SQLDatabase 초기화
        self.db = SQLDatabase.from_uri(postgres_url)

        # QuerySQLDataBaseTool 초기화
        self.sql_tool = QuerySQLDataBaseTool(db=self.db)

    def _run(self, query: str, **kwargs) -> str:
        """PostgreSQL 쿼리를 실행합니다."""
        try:
            # 읽기 전용 쿼리만 허용
            query_upper = query.upper().strip()

            # 위험한 쿼리 방지
            dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'CREATE', 'ALTER', 'TRUNCATE']
            if any(keyword in query_upper for keyword in dangerous_keywords):
                return "보안상의 이유로 읽기 전용 쿼리(SELECT)만 허용됩니다."

            # SELECT 쿼리만 허용
            if not query_upper.startswith('SELECT'):
                return "읽기 전용 접근을 위해 SELECT 쿼리만 허용됩니다."

            # SQL 쿼리 실행
            result = self.sql_tool.run(query)
            return result

        except Exception as e:
            return f"PostgreSQL 쿼리 실행 중 오류 발생: {str(e)}"

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

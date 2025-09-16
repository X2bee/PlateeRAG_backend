from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field

class PostgreSQLInput(BaseModel):
    query: str = Field(description="실행할 SQL 쿼리")

class PostgreSQLMCPTool(BaseTool):
    name: str = "postgresql_mcp"
    description: str = "PostgreSQL 데이터베이스에 대한 읽기 전용 접근을 제공합니다."
    args_schema: Type[BaseModel] = PostgreSQLInput
    postgres_url: str = Field(default="", description="PostgreSQL 연결 URL")
    db: SQLDatabase = Field(description="SQLDatabase 인스턴스")
    sql_tool: QuerySQLDataBaseTool = Field(description="QuerySQLDataBaseTool 인스턴스")
    db_prompt: str = Field(default="", description="데이터베이스 프롬프트")

    def __init__(self, postgres_url: str, db_prompt: str):
        db = SQLDatabase.from_uri(postgres_url)
        sql_tool = QuerySQLDataBaseTool(db=db)

        super().__init__(
            postgres_url=postgres_url,
            db=db,
            sql_tool=sql_tool,
            db_prompt=db_prompt
        )

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
            if self.sql_tool is None:
                return "SQL 도구가 초기화되지 않았습니다."

            result = self.sql_tool.run(query)

            if self.db_prompt and len(self.db_prompt.strip()) > 0:
                result = f"{self.db_prompt}\n\n{result}"
            return result

        except Exception as e:
            return f"PostgreSQL 쿼리 실행 중 오류 발생: {str(e)}"

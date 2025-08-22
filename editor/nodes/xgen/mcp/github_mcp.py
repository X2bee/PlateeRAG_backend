import logging
import os
from editor.node_composer import Node
from langchain_community.tools.github.tool import GitHubAction
from langchain_community.utilities.github import GitHubAPIWrapper
from langchain_core.tools import BaseTool
from typing import Type, List
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class GitHubInput(BaseModel):
    query: str = Field(description="GitHub API에 대한 쿼리 또는 명령")

class GitHubMCPTool(BaseTool):
    name = "github_mcp"
    description = "GitHub API를 사용하여 이슈, PR, 파일, 리포지토리 정보 등을 조회하고 관리합니다."
    args_schema: Type[BaseModel] = GitHubInput

    def __init__(self, access_token: str, repository: str = "", branch: str = "main"):
        super().__init__()
        self.access_token = access_token
        self.repository = repository
        self.branch = branch

        # GitHubAPIWrapper 초기화
        self.github_wrapper = GitHubAPIWrapper(
            github_app_id=None,
            github_app_private_key=None,
            github_repository=repository if repository else None,
            github_branch=branch,
            github_base_branch=branch
        )

        # 다양한 GitHub 액션들 초기화
        self.actions = {
            "get_issues": GitHubAction(api_wrapper=self.github_wrapper, mode="get_issues"),
            "get_issue": GitHubAction(api_wrapper=self.github_wrapper, mode="get_issue"),
            "comment_on_issue": GitHubAction(api_wrapper=self.github_wrapper, mode="comment_on_issue"),
            "create_issue": GitHubAction(api_wrapper=self.github_wrapper, mode="create_issue"),
            "create_pull_request": GitHubAction(api_wrapper=self.github_wrapper, mode="create_pull_request"),
            "read_file": GitHubAction(api_wrapper=self.github_wrapper, mode="read_file"),
            "get_repo_info": GitHubAction(api_wrapper=self.github_wrapper, mode="get_repo_info"),
        }

    def _run(self, query: str, **kwargs) -> str:
        """GitHub API 쿼리를 처리합니다."""
        try:
            # 쿼리 분석하여 적절한 액션 선택
            query_lower = query.lower()

            if "issue" in query_lower and "create" in query_lower:
                return self.actions["create_issue"].run(query)
            elif "issue" in query_lower and "comment" in query_lower:
                return self.actions["comment_on_issue"].run(query)
            elif "issue" in query_lower and ("get" in query_lower or "show" in query_lower):
                if "issues" in query_lower:
                    return self.actions["get_issues"].run(query)
                else:
                    return self.actions["get_issue"].run(query)
            elif "pull request" in query_lower or "pr" in query_lower:
                return self.actions["create_pull_request"].run(query)
            elif "file" in query_lower or "read" in query_lower:
                return self.actions["read_file"].run(query)
            elif "repo" in query_lower or "repository" in query_lower:
                return self.actions["get_repo_info"].run(query)
            else:
                # 기본적으로 이슈 목록 조회
                return self.actions["get_issues"].run(query)

        except Exception as e:
            return f"GitHub API 호출 중 오류 발생: {str(e)}"

class GitHubMCP(Node):
    categoryId = "xgen"
    functionId = "mcp"
    nodeId = "mcp/github_mcp"
    nodeName = "GitHub MCP"
    description = "MCP Server for the GitHub API"
    tags = ["mcp", "github", "version-control", "api", "repository"]

    inputs = [
    ]
    outputs = [
        {"id": "tools", "name": "Tools", "type": "TOOL"},
    ]

    parameters = [
        {"id": "github_access_token", "name": "GitHub Access Token", "type": "STR", "value": "", "required": True, "description": "GitHub API에 접근하기 위한 Personal Access Token입니다."},
        {"id": "github_repository", "name": "GitHub Repository", "type": "STR", "value": "", "required": False, "optional": True, "description": "작업할 GitHub 리포지토리 (형식: owner/repo)"},
        {"id": "github_branch", "name": "GitHub Branch", "type": "STR", "value": "main", "required": False, "optional": True, "description": "작업할 GitHub 브랜치"},
    ]

    def execute(self, *args, **kwargs):
        try:
            # 파라미터 추출
            github_access_token = kwargs.get("github_access_token", "")
            github_repository = kwargs.get("github_repository", "")
            github_branch = kwargs.get("github_branch", "main")

            # GitHub Access Token을 환경 변수로 설정
            os.environ["GITHUB_TOKEN"] = github_access_token
            os.environ["GITHUB_PERSONAL_ACCESS_TOKEN"] = github_access_token

            # GitHub MCP 도구 생성
            github_tool = GitHubMCPTool(
                access_token=github_access_token,
                repository=github_repository,
                branch=github_branch
            )

            logger.info("GitHub MCP 도구가 성공적으로 생성되었습니다. (repository: %s, branch: %s)",
                       github_repository or "지정되지 않음", github_branch)
            return github_tool

        except Exception as e:
            logger.error("GitHub MCP 도구 생성 중 오류 발생: %s", str(e))
            raise e

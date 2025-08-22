import os
from langchain_community.tools.github.tool import GitHubAction
from langchain_community.utilities.github import GitHubAPIWrapper
from langchain_core.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field

class GitHubInput(BaseModel):
    query: str = Field(description="GitHub API에 대한 쿼리 또는 명령")

class GitHubMCPTool(BaseTool):
    name: str = "github_mcp"
    description: str = "GitHub API를 사용하여 이슈, PR, 파일, 리포지토리 정보 등을 조회하고 관리합니다."
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

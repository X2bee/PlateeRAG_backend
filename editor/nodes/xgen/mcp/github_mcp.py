import logging
import os
from editor.node_composer import Node
from .github_mcp_helper import GitHubMCPTool

logger = logging.getLogger(__name__)

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

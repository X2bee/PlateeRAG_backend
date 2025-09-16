import logging
import os
from editor.node_composer import Node
from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
from langchain_community.utilities.github import GitHubAPIWrapper

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
        {"id": "github_app_id", "name": "GitHub App ID", "type": "STR", "value": "", "required": True, "description": "GitHub App ID입니다."},
        {"id": "github_app_private_key", "name": "GitHub App Private Key", "type": "STR", "value": "", "required": True, "expandable": True, "description": "GitHub App Private Key입니다."},
        {"id": "github_repository", "name": "GitHub Repository", "type": "STR", "value": "", "required": True, "description": "작업할 GitHub 리포지토리 (형식: owner/repo)"}
    ]

    def execute(self, *args, **kwargs):
        try:
            # 파라미터 추출
            github_app_id = kwargs.get("github_app_id", "")
            github_app_private_key = kwargs.get("github_app_private_key", "")
            github_repository = kwargs.get("github_repository", "")

            if not github_app_id:
                raise ValueError("GitHub App ID가 필요합니다.")
            if not github_app_private_key:
                raise ValueError("GitHub App Private Key가 필요합니다.")


            os.environ["GITHUB_APP_ID"] = github_app_id
            os.environ["GITHUB_APP_PRIVATE_KEY"] = github_app_private_key
            os.environ["GITHUB_REPOSITORY"] = github_repository

            github = GitHubAPIWrapper()

            toolkit = GitHubToolkit.from_github_api_wrapper(github)
            tools = toolkit.get_tools()
            for tool in tools:
                print(tool.name)
            logger.info("GitHub MCP 도구가 성공적으로 생성되었습니다. (repository: %s)", github_repository or "지정되지 않음")
            return tools

        except Exception as e:
            logger.error("GitHub MCP 도구 생성 중 오류 발생: %s", str(e))
            raise e

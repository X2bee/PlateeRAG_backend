import logging
import os
import requests
from typing import Dict, List, Any
from editor.node_composer import Node
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# LangChain 툴킷을 조건부로 import
try:
    from langchain_community.agent_toolkits.gitlab.toolkit import GitLabToolkit
    from langchain_community.utilities.gitlab import GitLabAPIWrapper
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("langchain_community not available. GitLab MCP will work with limited functionality.")

class GitLabMCP(Node):
    categoryId = "xgen"
    functionId = "mcp"
    nodeId = "mcp/gitlab_mcp"
    nodeName = "GitLab MCP"
    description = "GitLab API 통합 MCP 서버 - GitLab 리포지토리의 이슈, PR, 파일 관리 등 다양한 작업을 수행할 수 있습니다. Self-hosted GitLab 인스턴스도 지원합니다."
    tags = ["mcp", "gitlab", "version-control", "api", "repository", "issue", "merge-request", "file-management"]

    inputs = [
    ]
    outputs = [
        {"id": "tools", "name": "Tools", "type": "TOOL"},
    ]

    parameters = [
        {
            "id": "gitlab_url",
            "name": "GitLab URL",
            "type": "STR",
            "value": "https://gitlab.com",
            "required": False,
            "description": "GitLab 인스턴스 URL (기본값: https://gitlab.com, Self-hosted 지원)"
        },
        {
            "id": "gitlab_personal_access_token",
            "name": "GitLab Personal Access Token",
            "type": "STR",
            "value": "",
            "required": True,
            "expandable": True,
            "description": "GitLab Personal Access Token입니다. (Settings > Access Tokens에서 발급)"
        },
        {
            "id": "gitlab_repository",
            "name": "GitLab Repository",
            "type": "STR",
            "value": "",
            "required": True,
            "description": "작업할 GitLab 리포지토리\n- 형식 1: owner/repo (예: gitlab-org/gitlab)\n- 형식 2: 프로젝트 ID (예: 278964)\n- 프로젝트 내부: 전체 경로 (예: group/subgroup/project)"
        },
        {
            "id": "gitlab_branch",
            "name": "GitLab Branch",
            "type": "STR",
            "value": "main",
            "required": False,
            "description": "작업할 브랜치 (기본값: main)",
            "is_api": True,
            "api_name": "get_branches",
            "dependency": "gitlab_repository",
            "options": []
        }
    ]

    def api_get_branches(self, gitlab_url: str = "https://gitlab.com", gitlab_personal_access_token: str = "", gitlab_repository: str = "") -> List[Dict]:
        """
        GitLab 리포지토리의 브랜치 목록을 조회합니다 (UI에서 호출).

        Args:
            gitlab_url: GitLab 인스턴스 URL
            gitlab_personal_access_token: Personal Access Token
            gitlab_repository: 리포지토리 경로 또는 프로젝트 ID

        Returns:
            [{"label": "main (기본)", "value": "main"}, {"label": "develop", "value": "develop"}, ...]
        """
        if not gitlab_personal_access_token:
            logger.warning("GitLab Personal Access Token이 제공되지 않았습니다.")
            return []

        if not gitlab_repository:
            logger.warning("GitLab Repository가 제공되지 않았습니다.")
            return []

        try:
            result = self._fetch_branches(gitlab_url, gitlab_personal_access_token, gitlab_repository)
            if result.get("success"):
                return result.get("branches", [])
            else:
                logger.error(f"브랜치 조회 실패: {result.get('error')}")
                return []
        except Exception as e:
            logger.error(f"브랜치 조회 중 예외 발생: {e}")
            return []

    def _fetch_branches(self, gitlab_url: str, gitlab_personal_access_token: str, gitlab_repository: str) -> Dict:
        """
        GitLab API를 통해 브랜치 목록을 가져옵니다 (내부 메서드).

        Args:
            gitlab_url: GitLab 인스턴스 URL
            gitlab_personal_access_token: Personal Access Token
            gitlab_repository: 리포지토리 경로 또는 프로젝트 ID

        Returns:
            {"success": True, "branches": [...]} 또는 {"success": False, "error": "..."}
        """
        try:
            # 프로젝트 ID를 URL 인코딩 (owner/repo -> owner%2Frepo)
            import urllib.parse
            project_id = urllib.parse.quote(gitlab_repository, safe='')

            # GitLab API 엔드포인트
            api_url = f"{gitlab_url}/api/v4/projects/{project_id}/repository/branches"

            headers = {
                "PRIVATE-TOKEN": gitlab_personal_access_token
            }

            response = requests.get(api_url, headers=headers, params={"per_page": 100})
            response.raise_for_status()

            branches_data = response.json()
            branch_list = []

            # UI Select 컴포넌트에 맞는 형식으로 변환
            for branch in branches_data:
                branch_name = branch.get("name")
                is_default = branch.get("default", False)
                is_protected = branch.get("protected", False)

                # label에 추가 정보 표시
                label = branch_name
                if is_default:
                    label = f"{branch_name} (기본)"
                elif is_protected:
                    label = f"{branch_name} (보호됨)"

                branch_list.append({
                    "label": label,
                    "value": branch_name,
                    "default": is_default,
                    "protected": is_protected
                })

            # 기본 브랜치를 맨 앞으로 정렬
            branch_list.sort(key=lambda x: (not x["default"], x["value"]))

            logger.info("브랜치 목록 조회 성공: %d개 발견", len(branch_list))
            return {
                "success": True,
                "branches": branch_list,
                "total": len(branch_list)
            }

        except requests.exceptions.HTTPError as e:
            error_message = ""
            if e.response.status_code == 404:
                error_message = f"프로젝트를 찾을 수 없습니다: {gitlab_repository}. 리포지토리 경로 또는 프로젝트 ID를 확인하세요."
                logger.error(error_message)
            elif e.response.status_code == 401:
                error_message = "인증 실패: Personal Access Token이 유효하지 않습니다."
                logger.error(error_message)
            else:
                error_message = f"브랜치 목록 조회 실패: {str(e)}"
                logger.error(error_message)

            return {
                "success": False,
                "error": error_message,
                "branches": []
            }
        except Exception as e:
            error_message = f"브랜치 목록 조회 중 오류 발생: {str(e)}"
            logger.error(error_message)
            return {
                "success": False,
                "error": error_message,
                "branches": []
            }

    def _wrap_gitlab_action(self, gitlab_action) -> StructuredTool:
        """GitLabAction을 LangChain StructuredTool로 변환하여 호환성 문제 해결"""
        mode = getattr(gitlab_action, 'mode', 'unknown')
        original_name = getattr(gitlab_action, 'name', 'Unknown Tool')
        description = getattr(gitlab_action, 'description', 'No description')

        # mode에 따라 적절한 스키마 정의
        if mode == 'get_issues':
            # Get Issues는 입력 파라미터가 필요 없음
            def get_issues_func() -> str:
                return gitlab_action._run(instructions="")

            return StructuredTool.from_function(
                func=get_issues_func,
                name=original_name,
                description=description
            )

        else:
            # 다른 모든 도구는 instructions 파라미터 필요
            class GitLabInstructionsSchema(BaseModel):
                instructions: str = Field(..., description="Instructions for the GitLab action")

            def generic_gitlab_func(instructions: str) -> str:
                try:
                    return gitlab_action._run(instructions=instructions)
                except Exception as e:
                    error_msg = str(e)
                    # 404 에러 처리 - 파일을 찾을 수 없는 경우
                    if "404" in error_msg and mode == 'get_file':
                        return f"""❌ 파일을 찾을 수 없습니다.

오류: {error_msg}

💡 해결 방법:
1. 파일 경로가 정확한지 확인하세요 (대소문자 구분)
2. 파일 경로는 슬래시(/)로 시작하지 않아야 합니다
   - 올바른 예: src/main/java/Example.java
   - 잘못된 예: /src/main/java/Example.java
3. 브랜치가 올바른지 확인하세요
4. 파일이 실제로 레포지토리에 존재하는지 확인하세요

먼저 레포지토리의 파일 구조를 확인한 후 다시 시도해주세요."""
                    else:
                        # 기타 에러는 원래 에러 메시지 반환
                        return f"❌ GitLab 작업 중 오류 발생:\n{error_msg}"

            return StructuredTool.from_function(
                func=generic_gitlab_func,
                name=original_name,
                description=description,
                args_schema=GitLabInstructionsSchema
            )

    def execute(self, *args, **kwargs):
        """GitLab MCP 노드 실행 - LangChain Toolkit 생성 또는 기본 GitLab 도구 반환"""
        try:
            # 파라미터 추출
            gitlab_url = kwargs.get("gitlab_url", "https://gitlab.com")
            gitlab_personal_access_token = kwargs.get("gitlab_personal_access_token", "")
            gitlab_repository = kwargs.get("gitlab_repository", "")
            gitlab_branch = kwargs.get("gitlab_branch", "main")

            logger.info("=" * 80)
            logger.info("🚀 GitLab MCP 노드 실행 시작")
            logger.info(f"  ├─ GitLab URL: {gitlab_url}")
            logger.info(f"  ├─ Repository: {gitlab_repository}")
            logger.info(f"  ├─ Branch: {gitlab_branch}")
            logger.info(f"  └─ Token: {'✓ 제공됨' if gitlab_personal_access_token else '✗ 없음'}")

            if not gitlab_personal_access_token:
                raise ValueError("GitLab Personal Access Token이 필요합니다.")
            if not gitlab_repository:
                raise ValueError("GitLab Repository가 필요합니다.")

            # 환경 변수 설정
            os.environ["GITLAB_URL"] = gitlab_url
            os.environ["GITLAB_PERSONAL_ACCESS_TOKEN"] = gitlab_personal_access_token
            os.environ["GITLAB_REPOSITORY"] = gitlab_repository
            os.environ["GITLAB_BRANCH"] = gitlab_branch

            # LangChain이 사용 가능한 경우 Toolkit 사용
            if LANGCHAIN_AVAILABLE:
                logger.info("📦 LangChain GitLab Toolkit을 사용하여 도구 생성 중...")

                # GitLab API Wrapper 생성
                gitlab = GitLabAPIWrapper()
                logger.info("  └─ GitLab API Wrapper 생성 완료")

                # GitLab Toolkit 생성
                toolkit = GitLabToolkit.from_gitlab_api_wrapper(gitlab)
                gitlab_actions = toolkit.get_tools()

                # GitLabAction들을 StructuredTool로 변환
                wrapped_tools = []
                for action in gitlab_actions:
                    wrapped_tool = self._wrap_gitlab_action(action)
                    wrapped_tools.append(wrapped_tool)

                logger.info(f"✅ GitLab 도구 {len(wrapped_tools)}개 생성 및 변환 완료:")
                for i, tool in enumerate(wrapped_tools, 1):
                    logger.info(f"  {i}. {tool.name}")
                    # 도구 설명의 첫 줄만 출력
                    desc_first_line = tool.description.split('\n')[0].strip() if tool.description else "No description"
                    logger.info(f"     └─ {desc_first_line}")

                logger.info("=" * 80)
                return wrapped_tools
            else:
                # LangChain이 없는 경우 기본 GitLab 도구 반환
                logger.warning("⚠️  LangChain이 설치되지 않아 기본 GitLab 도구를 반환합니다.")
                logger.info("=" * 80)
                return self._create_basic_gitlab_tools(gitlab_url, gitlab_personal_access_token, gitlab_repository, gitlab_branch)

        except Exception as e:
            logger.error("=" * 80)
            logger.error("❌ GitLab MCP 도구 생성 중 오류 발생:")
            logger.error(f"   └─ {str(e)}")
            logger.error("=" * 80)
            raise e

    def _create_basic_gitlab_tools(self, gitlab_url: str, token: str, repository: str, branch: str) -> List[Dict]:
        """LangChain 없이 기본 GitLab API 도구 생성"""
        logger.info("기본 GitLab 도구 생성 중... (repository: %s, branch: %s)", repository, branch)

        # 여기에 필요한 경우 기본 GitLab API 래퍼를 추가할 수 있습니다
        # 현재는 빈 리스트 반환
        return []

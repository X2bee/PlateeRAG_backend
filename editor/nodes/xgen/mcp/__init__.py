# MCP 노드들을 안전하게 import
# 의존성이 없는 노드는 건너뛰고 계속 진행

import logging
logger = logging.getLogger(__name__)

# 필수 노드들
try:
    from .mcp_loader import MCPLoader
except Exception as e:
    logger.warning(f"Failed to load MCPLoader: {e}")

# 선택적 노드들 - 의존성이 있을 경우만 로드
try:
    from .tavily_search_mcp import TavilySearchMCP
except Exception as e:
    logger.debug(f"TavilySearchMCP not available: {e}")

try:
    from .brave_search_mcp import BraveSearchMCP
except Exception as e:
    logger.debug(f"BraveSearchMCP not available: {e}")

try:
    from .epg_daum_mcp import EPGDaumMCP
except Exception as e:
    logger.debug(f"EPGDaumMCP not available: {e}")

try:
    from .epg_naver_mcp import EPGNaverMCP
except Exception as e:
    logger.debug(f"EPGNaverMCP not available: {e}")

try:
    from .github_mcp import GitHubMCP
except Exception as e:
    logger.debug(f"GitHubMCP not available: {e}")

try:
    from .gitlab_mcp import GitLabMCP
except Exception as e:
    logger.debug(f"GitLabMCP not available: {e}")

try:
    from .meta_search_mcp import MetaSearchMCP
except Exception as e:
    logger.debug(f"MetaSearchMCP not available: {e}")

try:
    from .naver_datalab_mcp import NaverDatalabMCP
except Exception as e:
    logger.debug(f"NaverDatalabMCP not available: {e}")

try:
    from .naver_news_mcp import NaverNewsMCP
except Exception as e:
    logger.debug(f"NaverNewsMCP not available: {e}")

try:
    from .postgresql_mcp import PostgreSQLMCP
except Exception as e:
    logger.debug(f"PostgreSQLMCP not available: {e}")

try:
    from .product_search_mcp import ProductSearchMCP
except Exception as e:
    logger.debug(f"ProductSearchMCP not available: {e}")

try:
    from .slack_mcp import SlackMCP
except Exception as e:
    logger.debug(f"SlackMCP not available: {e}")

"""Utilities for fetching and parsing robots.txt."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional
from urllib import robotparser
from urllib.parse import urljoin, urlparse

import requests

DEFAULT_USER_AGENT = "xgen bot/1.0"
REQUEST_TIMEOUT = 5


def check_robots(
    endpoint: str,
    user_agent: Optional[str] = None,
    cache: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Fetch and evaluate robots.txt for the given endpoint."""
    default_agent = user_agent or DEFAULT_USER_AGENT
    parsed = urlparse(endpoint)
    origin = f"{parsed.scheme}://{parsed.netloc}" if parsed.netloc else endpoint

    if parsed.scheme not in {"http", "https"}:
        return {
            "allowed": True,
            "userAgent": default_agent,
            "message": "HTTP/HTTPS URL이 아니므로 robots.txt 검사를 생략합니다.",
        }

    robots_url = urljoin(f"{parsed.scheme}://{parsed.netloc}", "/robots.txt")
    headers = {"User-Agent": default_agent}

    try:
        response = requests.get(robots_url, headers=headers, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        cached = _resolve_cached(cache, origin, endpoint)
        if cached:
            cached_result = deepcopy(cached)
            cached_result["userAgent"] = default_agent
            cached_result.setdefault(
                "message",
                "캐시된 robots.txt 결과를 반환합니다.",
            )
            return cached_result
        return {
            "allowed": True,
            "userAgent": default_agent,
            "message": f"robots.txt 요청에 실패했습니다: {exc}. 기본적으로 허용됩니다.",
        }

    if response.status_code == 404:
        return {
            "allowed": True,
            "userAgent": default_agent,
            "robotsUrl": robots_url,
            "message": "robots.txt를 찾을 수 없습니다. 기본적으로 허용됩니다.",
            "statusCode": response.status_code,
        }

    if response.status_code >= 400:
        fallback = _resolve_cached(cache, origin, endpoint)
        if fallback:
            cached_result = deepcopy(fallback)
            cached_result["userAgent"] = default_agent
            cached_result["statusCode"] = response.status_code
            cached_result.setdefault(
                "message",
                f"robots.txt 요청이 HTTP {response.status_code}로 실패했습니다. 캐시된 정보를 반환합니다.",
            )
            return cached_result
        return {
            "allowed": True,
            "userAgent": default_agent,
            "robotsUrl": robots_url,
            "message": f"robots.txt 요청이 HTTP {response.status_code}로 실패했습니다. 기본적으로 허용됩니다.",
            "statusCode": response.status_code,
        }

    content = response.text or ""
    parser = robotparser.RobotFileParser()
    parser.parse(content.splitlines())

    allowed = parser.can_fetch(default_agent, endpoint)
    crawl_delay = parser.crawl_delay(default_agent) or parser.crawl_delay("*")
    disallowed_paths = _extract_disallowed_rules(parser, default_agent)

    message = (
        "robots.txt 규칙을 확인했습니다. 크롤링이 허용됩니다."
        if allowed
        else "robots.txt 규칙에 따라 해당 경로는 크롤링이 제한됩니다."
    )

    result: Dict[str, Any] = {
        "allowed": allowed,
        "userAgent": default_agent,
        "robotsUrl": robots_url,
        "message": message,
        "statusCode": response.status_code,
    }

    if crawl_delay is not None:
        result["crawlDelay"] = crawl_delay
    if disallowed_paths:
        result["disallowedPaths"] = disallowed_paths

    if cache is not None:
        cache[origin] = deepcopy(result)

    return result


def _resolve_cached(
    cache: Optional[Dict[str, Dict[str, Any]]],
    origin: str,
    endpoint: str,
) -> Optional[Dict[str, Any]]:
    if not cache:
        return None
    return cache.get(origin) or cache.get(endpoint)


def _extract_disallowed_rules(parser: robotparser.RobotFileParser, user_agent: str) -> list[str]:
    """Collect disallowed rule paths for the specified user agent."""
    ua_lower = user_agent.lower()
    entries = parser.entries or []

    entry = _find_entry(entries, ua_lower)
    if entry is None:
        entry = parser.default_entry

    if not entry or not getattr(entry, "rulelines", None):
        return []

    disallowed: list[str] = []
    for line in entry.rulelines:
        if not line.allowance:
            path = line.path or "/"
            disallowed.append(path)
    return disallowed


def _find_entry(entries, ua_lower: str):
    """Find the most appropriate robots entry for the requested agent."""
    for entry in entries:
        if any(agent.lower() == ua_lower for agent in entry.useragents):
            return entry
    for entry in entries:
        if any(agent == "*" for agent in entry.useragents):
            return entry
    return None

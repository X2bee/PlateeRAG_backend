import asyncio
import json
import logging
from typing import Any, AsyncIterator, Dict, Optional, Tuple

import httpx
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

logger = logging.getLogger("vast-proxy-client")


class VastProxyClient:
    """Remote VastAI proxy client.

    Delegates Vast-related operations to a dedicated remote service.
    """

    is_proxy: bool = True

    def __init__(self, config_composer):
        self._vast_config = config_composer.get_config_by_category_name("vast")
        self._synced_context: Optional[Tuple[str, str, str]] = None

        if not self._get_base_url():
            logger.warning("Vast proxy client enabled but base URL is not configured.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_headers(self, extra_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        proxy_token = self._get_proxy_token()
        if proxy_token and (not extra_headers or "Authorization" not in extra_headers):
            headers["Authorization"] = f"Bearer {proxy_token}"
        if extra_headers:
            for key, value in extra_headers.items():
                if value:
                    headers[key] = value
        return headers

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        base_url = self._get_base_url()
        if not base_url:
            raise HTTPException(status_code=500, detail="Vast proxy base URL is not configured")

        url = f"{base_url}{path}"
        request_timeout = timeout if timeout is not None else self._get_timeout()

        try:
            async with httpx.AsyncClient(timeout=request_timeout, headers=self._build_headers(extra_headers)) as client:
                response = await client.request(method=method, url=url, params=params, json=json_body)
        except httpx.RequestError as exc:
            logger.error("Vast proxy request error: %s", exc)
            raise HTTPException(status_code=502, detail="Failed to reach Vast proxy service") from exc

        if response.status_code >= 400:
            detail = self._extract_detail(response)
            logger.warning(
                "Vast proxy returned error %s for %s %s: %s",
                response.status_code,
                method,
                path,
                detail,
            )
            raise HTTPException(status_code=response.status_code, detail=detail)

        if response.headers.get("content-type", "").startswith("application/json") and response.content:
            return response.json()
        return response.content

    @staticmethod
    def _extract_detail(response: httpx.Response) -> Any:
        try:
            payload = response.json()
            if isinstance(payload, dict) and "detail" in payload:
                return payload["detail"]
            return payload
        except json.JSONDecodeError:
            return response.text or "Proxy request failed"

    # ------------------------------------------------------------------
    # Public Vast operations
    # ------------------------------------------------------------------
    async def search_offers(self, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        await self._ensure_api_key()
        return await self._request("POST", "/api/vast/search-offers", json_body=payload, extra_headers=headers)

    async def create_instance(self, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        await self._ensure_api_key()
        return await self._request("POST", "/api/vast/instances", json_body=payload, extra_headers=headers)

    async def health_check(self) -> Dict[str, Any]:
        return await self._request("GET", "/api/vast/health")

    async def list_instances(self, params: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        await self._ensure_api_key()
        return await self._request("GET", "/api/vast/instances", params=params, extra_headers=headers)

    async def get_instance_status(self, instance_id: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        await self._ensure_api_key()
        return await self._request("GET", f"/api/vast/instances/{instance_id}/status", extra_headers=headers)

    async def destroy_instance(self, instance_id: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        await self._ensure_api_key()
        return await self._request("DELETE", f"/api/vast/instances/{instance_id}", extra_headers=headers)

    async def update_ports(self, instance_id: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        await self._ensure_api_key()
        return await self._request("POST", f"/api/vast/instances/{instance_id}/update-ports", extra_headers=headers)

    async def vllm_serve(self, instance_id: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        await self._ensure_api_key()
        return await self._request("POST", f"/api/vast/instances/{instance_id}/vllm-serve", json_body=payload, extra_headers=headers)

    async def vllm_down(self, instance_id: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        await self._ensure_api_key()
        return await self._request("POST", f"/api/vast/instances/{instance_id}/vllm-down", extra_headers=headers)

    async def set_vllm_config(self, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        await self._ensure_api_key()
        return await self._request("PUT", "/api/vast/set-vllm", json_body=payload, extra_headers=headers)

    async def check_vllm_health(self, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        await self._ensure_api_key()
        return await self._request("POST", "/api/vast/instances/vllm-health", json_body=payload, extra_headers=headers)

    async def list_templates(self, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        await self._ensure_api_key()
        return await self._request("GET", "/api/vast/templates", extra_headers=headers)

    async def create_trainer_instance(self, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        await self._ensure_api_key()
        return await self._request("POST", "/api/train/instances", json_body=payload, extra_headers=headers)

    # ------------------------------------------------------------------
    # SSE support
    # ------------------------------------------------------------------
    async def stream_instance_status(self, instance_id: str, headers: Optional[Dict[str, str]] = None) -> StreamingResponse:
        base_url = self._get_base_url()
        if not base_url:
            raise HTTPException(status_code=500, detail="Vast proxy base URL is not configured")

        url = f"{base_url}/api/vast/instances/{instance_id}/status-stream"
        client_headers = self._build_headers(headers)
        # SSE should not enforce strict timeout; keep-alive from remote
        timeout = httpx.Timeout(None, connect=self._get_timeout())

        await self._ensure_api_key()

        async def event_generator() -> AsyncIterator[bytes]:
            async with httpx.AsyncClient(timeout=timeout, headers=client_headers) as client:
                async with client.stream("GET", url) as response:
                    if response.status_code >= 400:
                        detail = self._extract_detail(response)
                        raise HTTPException(status_code=response.status_code, detail=detail)

                    async for chunk in response.aiter_raw():
                        if chunk:
                            yield chunk
                        else:
                            await asyncio.sleep(0.01)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
            },
        )

    async def _ensure_api_key(self) -> None:
        api_key = self._get_api_key()
        if not api_key:
            self._synced_context = None
            return

        proxy_token = self._get_proxy_token()
        context = (api_key, self._get_base_url(), proxy_token)
        if self._synced_context == context:
            return

        extra_headers = {"Authorization": f"Bearer {proxy_token}"} if proxy_token else None

        await self._request(
            "POST",
            "/api/vast/proxy/api-key",
            json_body={"api_key": api_key},
            extra_headers=extra_headers,
        )
        self._synced_context = context

    def _get_base_url(self) -> str:
        config_obj = getattr(self._vast_config, "VAST_PROXY_BASE_URL", None)
        base_url = getattr(config_obj, "value", None) or ""
        return base_url.rstrip("/")

    def _get_timeout(self) -> float:
        config_obj = getattr(self._vast_config, "VAST_PROXY_TIMEOUT", None)
        timeout_value = getattr(config_obj, "value", None)
        return timeout_value or 30

    def _get_proxy_token(self) -> str:
        config_obj = getattr(self._vast_config, "VAST_PROXY_API_TOKEN", None)
        return getattr(config_obj, "value", None) or ""

    def _get_api_key(self) -> Optional[str]:
        config_obj = getattr(self._vast_config, "VAST_API_KEY", None)
        if config_obj is None:
            return None
        return getattr(config_obj, "value", None) or None

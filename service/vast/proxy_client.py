import asyncio
import json
import logging
from typing import Any, AsyncIterator, Dict, Optional

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
        vast_config = config_composer.get_config_by_category_name("vast")
        self._base_url = (vast_config.VAST_PROXY_BASE_URL.value or "").rstrip("/")
        self._timeout = vast_config.VAST_PROXY_TIMEOUT.value or 30
        self._token = vast_config.VAST_PROXY_API_TOKEN.value or ""
        self._api_key = getattr(vast_config, "VAST_API_KEY", None)
        if self._api_key is not None and hasattr(self._api_key, "value"):
            self._api_key = self._api_key.value
        self._api_key_synced = False

        if not self._base_url:
            logger.warning("Vast proxy client enabled but base URL is not configured.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        if not self._base_url:
            raise HTTPException(status_code=500, detail="Vast proxy base URL is not configured")

        url = f"{self._base_url}{path}"
        request_timeout = timeout if timeout is not None else self._timeout

        try:
            async with httpx.AsyncClient(timeout=request_timeout, headers=self._build_headers()) as client:
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
    async def search_offers(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        await self._ensure_api_key()
        return await self._request("POST", "/api/vast/search-offers", json_body=payload)

    async def create_instance(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        await self._ensure_api_key()
        return await self._request("POST", "/api/vast/instances", json_body=payload)

    async def health_check(self) -> Dict[str, Any]:
        return await self._request("GET", "/api/vast/health")

    async def list_instances(self, params: Dict[str, Any]) -> Dict[str, Any]:
        await self._ensure_api_key()
        return await self._request("GET", "/api/vast/instances", params=params)

    async def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        await self._ensure_api_key()
        return await self._request("GET", f"/api/vast/instances/{instance_id}/status")

    async def destroy_instance(self, instance_id: str) -> Dict[str, Any]:
        await self._ensure_api_key()
        return await self._request("DELETE", f"/api/vast/instances/{instance_id}")

    async def update_ports(self, instance_id: str) -> Dict[str, Any]:
        await self._ensure_api_key()
        return await self._request("POST", f"/api/vast/instances/{instance_id}/update-ports")

    async def vllm_serve(self, instance_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        await self._ensure_api_key()
        return await self._request("POST", f"/api/vast/instances/{instance_id}/vllm-serve", json_body=payload)

    async def vllm_down(self, instance_id: str) -> Dict[str, Any]:
        await self._ensure_api_key()
        return await self._request("POST", f"/api/vast/instances/{instance_id}/vllm-down")

    async def set_vllm_config(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        await self._ensure_api_key()
        return await self._request("PUT", "/api/vast/set-vllm", json_body=payload)

    async def check_vllm_health(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        await self._ensure_api_key()
        return await self._request("POST", "/api/vast/instances/vllm-health", json_body=payload)

    async def list_templates(self) -> Dict[str, Any]:
        await self._ensure_api_key()
        return await self._request("GET", "/api/vast/templates")

    async def create_trainer_instance(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        await self._ensure_api_key()
        return await self._request("POST", "/api/train/instances", json_body=payload)

    # ------------------------------------------------------------------
    # SSE support
    # ------------------------------------------------------------------
    async def stream_instance_status(self, instance_id: str) -> StreamingResponse:
        if not self._base_url:
            raise HTTPException(status_code=500, detail="Vast proxy base URL is not configured")

        url = f"{self._base_url}/api/vast/instances/{instance_id}/status-stream"
        headers = self._build_headers()
        # SSE should not enforce strict timeout; keep-alive from remote
        timeout = httpx.Timeout(None, connect=self._timeout)

        await self._ensure_api_key()

        async def event_generator() -> AsyncIterator[bytes]:
            async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
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
        if self._api_key_synced:
            return

        if not self._api_key:
            self._api_key_synced = True
            return

        await self._request("POST", "/api/vast/proxy/api-key", json_body={"api_key": self._api_key})
        self._api_key_synced = True

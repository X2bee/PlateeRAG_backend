"""Distributed locking utilities for scraper execution."""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Dict, Optional, TYPE_CHECKING

try:
    import redis
except ImportError:  # pragma: no cover - redis is an optional dependency
    redis = None  # type: ignore

if TYPE_CHECKING:
    from redis import Redis  # pragma: no cover

logger = logging.getLogger("data-scraper-locks")


class DistributedLockManager:
    """Manage scraper execution locks backed by Redis or in-memory fallbacks."""

    LUA_RELEASE_SCRIPT = """
    if redis.call("get", KEYS[1]) == ARGV[1] then
        return redis.call("del", KEYS[1])
    else
        return 0
    end
    """

    def __init__(self, redis_url: Optional[str] = None, redis_client: Optional["Redis"] = None):
        self._loop = asyncio.get_event_loop()
        self._local_locks: Dict[str, asyncio.Lock] = {}
        self._local_tokens: Dict[str, str] = {}
        self._redis_client = self._create_redis_client(redis_url, redis_client)
        self._redis_release = (
            self._redis_client.register_script(self.LUA_RELEASE_SCRIPT)
            if self._redis_client
            else None
        )

    @staticmethod
    def _create_redis_client(redis_url: Optional[str], redis_client: Optional["Redis"]):
        if redis_client:
            return redis_client
        if not redis_url or not redis:
            if redis_url and not redis:
                logger.warning("redis package not available; falling back to in-memory locks")
            return None
        try:
            client = redis.Redis.from_url(redis_url)
            client.ping()
            logger.info("Data scraper lock manager connected to Redis: %s", redis_url)
            return client
        except Exception as exc:  # pragma: no cover - external dependency
            logger.warning("Failed to connect to Redis at %s (%s); using in-memory locks", redis_url, exc)
            return None

    @property
    def is_distributed(self) -> bool:
        return self._redis_client is not None

    async def acquire(self, key: str, ttl_seconds: int = 600) -> Optional[str]:
        """Attempt to acquire a lock. Returns a token if successful."""
        if self._redis_client:
            return await asyncio.to_thread(self._acquire_redis_lock, key, ttl_seconds)
        return await self._acquire_local_lock(key)

    async def release(self, key: str, token: str) -> None:
        """Release an acquired lock."""
        if self._redis_client:
            await asyncio.to_thread(self._release_redis_lock, key, token)
        else:
            await self._release_local_lock(key, token)

    # ------------------------------------------------------------------
    # Redis-backed locking
    # ------------------------------------------------------------------
    def _acquire_redis_lock(self, key: str, ttl_seconds: int) -> Optional[str]:
        if not self._redis_client:
            return None
        token = uuid.uuid4().hex
        acquired = self._redis_client.set(name=key, value=token, nx=True, ex=ttl_seconds)
        if acquired:
            return token
        return None

    def _release_redis_lock(self, key: str, token: str) -> None:
        if not self._redis_client or not self._redis_release:
            return
        try:
            self._redis_release(keys=[key], args=[token])
        except Exception as exc:  # pragma: no cover - best effort cleanup
            logger.warning("Failed to release Redis lock %s: %s", key, exc)

    # ------------------------------------------------------------------
    # Local fallback locking
    # ------------------------------------------------------------------
    async def _acquire_local_lock(self, key: str) -> Optional[str]:
        lock = self._local_locks.get(key)
        if not lock:
            lock = asyncio.Lock()
            self._local_locks[key] = lock
        token = uuid.uuid4().hex
        acquired = await lock.acquire()
        if acquired:
            self._local_tokens[key] = token
            return token
        return None

    async def _release_local_lock(self, key: str, token: str) -> None:
        lock = self._local_locks.get(key)
        if not lock:
            return
        current_token = self._local_tokens.get(key)
        if current_token == token and lock.locked():
            lock.release()
            self._local_tokens.pop(key, None)


def build_lock_manager(redis_client: Optional["Redis"] = None) -> DistributedLockManager:
    """Factory that pulls Redis configuration from the environment when needed."""
    redis_url = None
    if not redis_client:
        redis_url = os.getenv("SCRAPER_REDIS_URL") or os.getenv("REDIS_URL")
    return DistributedLockManager(redis_url=redis_url, redis_client=redis_client)

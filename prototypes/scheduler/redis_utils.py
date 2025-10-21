"""Light-weight Redis helpers for the scheduler prototype."""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Optional

import redis

logger = logging.getLogger("scheduler-prototype-redis")


class RedisLock:
    """Simple distributed lock using Redis SETNX semantics."""

    LUA_RELEASE = """
    if redis.call("get", KEYS[1]) == ARGV[1] then
        return redis.call("del", KEYS[1])
    else
        return 0
    end
    """

    def __init__(self, client: redis.Redis, key: str, ttl: int = 60):
        self.client = client
        self.key = key
        self.ttl = ttl
        self.token: Optional[str] = None
        self._script = self.client.register_script(self.LUA_RELEASE)

    async def acquire(self) -> bool:
        token = uuid.uuid4().hex
        acquired = await asyncio.to_thread(
            self.client.set, self.key, token, nx=True, ex=self.ttl
        )
        if acquired:
            self.token = token
            return True
        return False

    async def release(self) -> None:
        if not self.token:
            return
        try:
            await asyncio.to_thread(self._script, keys=[self.key], args=[self.token])
        finally:
            self.token = None


def build_redis_client() -> redis.Redis:
    url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    client = redis.Redis.from_url(url)
    client.ping()
    logger.info("Connected to Redis at %s", url)
    return client

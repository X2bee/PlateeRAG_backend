"""Minimal FastAPI app demonstrating APScheduler + Redis locking."""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Dict

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI, HTTPException

from prototypes.scheduler.redis_utils import RedisLock, build_redis_client
from prototypes.scheduler.run_executor import execute_run

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scheduler-prototype")

app = FastAPI(title="Data Scraper Scheduler Prototype")

SCHEDULER_APP_STATE = None


def _run_key(run_id: str) -> str:
    return f"scraper:run:{run_id}"


async def run_job(scraper_id: int, run_id: str, lock_token: str) -> None:
    """Job entry point executed by APScheduler."""
    state = SCHEDULER_APP_STATE
    redis_client = state.redis

    redis_client.hset(
        _run_key(run_id),
        mapping={
            "scraper_id": scraper_id,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    try:
        result = await execute_run(scraper_id, run_id)
        redis_client.hset(
            _run_key(run_id),
            mapping={
                "status": "completed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "duration": result["duration"],
                "items_collected": result["items"],
            },
        )
    except Exception as exc:  # pragma: no cover - demonstration only
        logger.exception("Scheduled run failed: scraper=%s run=%s", scraper_id, run_id)
        redis_client.hset(
            _run_key(run_id),
            mapping={
                "status": "failed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": str(exc),
            },
        )
    finally:
        # Release lock so subsequent runs can be enqueued
        lock = RedisLock(redis_client, f"scraper:{scraper_id}:lock")
        lock.token = lock_token
        await lock.release()


@app.on_event("startup")
async def startup() -> None:
    global SCHEDULER_APP_STATE
    scheduler = AsyncIOScheduler()
    scheduler.start()

    redis_client = build_redis_client()

    app.state.scheduler = scheduler
    app.state.redis = redis_client
    SCHEDULER_APP_STATE = app.state
    logger.info("Scheduler prototype started")


@app.on_event("shutdown")
async def shutdown() -> None:
    scheduler: AsyncIOScheduler = app.state.scheduler
    scheduler.shutdown(wait=False)
    logger.info("Scheduler prototype shutdown complete")


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/runs/manual/{scraper_id}")
async def trigger_manual_run(scraper_id: int) -> Dict[str, str]:
    scheduler: AsyncIOScheduler = app.state.scheduler
    redis_client = app.state.redis

    lock = RedisLock(redis_client, f"scraper:{scraper_id}:lock", ttl=120)
    if not await lock.acquire():
        raise HTTPException(status_code=409, detail="Scraper run already in progress")

    run_id = uuid.uuid4().hex
    redis_client.hset(
        _run_key(run_id),
        mapping={
            "scraper_id": scraper_id,
            "status": "queued",
            "trigger": "manual",
            "enqueued_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    scheduler.add_job(
        run_job,
        trigger="date",
        run_date=datetime.now(timezone.utc),
        args=(scraper_id, run_id, lock.token),
        id=f"prototype-run-{run_id}",
        coalesce=True,
    )

    return {"run_id": run_id, "status": "queued"}

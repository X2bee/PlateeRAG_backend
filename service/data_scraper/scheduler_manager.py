"""APScheduler-based orchestration for scraper runs."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Optional, TYPE_CHECKING
import uuid

from apscheduler.jobstores.base import JobLookupError
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

try:
    from apscheduler.jobstores.redis import RedisJobStore
except ImportError:  # pragma: no cover - redis jobstore optional
    RedisJobStore = None  # type: ignore

if TYPE_CHECKING:
    from redis import Redis  # pragma: no cover

from service.data_scraper.lock_manager import build_lock_manager, DistributedLockManager
from service.data_scraper.repository import ScraperRepository
from service.data_scraper.run_executor import RunExecutor
from service.database.connection import AppDatabaseManager
from service.database.models.data_scraper import ScraperConfig, ScraperRun

logger = logging.getLogger("data-scraper-scheduler")


class DataScraperScheduler:
    """Coordinates scraper run scheduling, locking, and execution."""

    DEFAULT_LOCK_TTL = 900  # 15 minutes

    def __init__(
        self,
        db_manager: AppDatabaseManager,
        *,
        redis_url: Optional[str] = None,
        redis_client: Optional["Redis"] = None,
    ):
        self.db_manager = db_manager
        self.repo = ScraperRepository(db_manager)
        self.lock_manager: DistributedLockManager = (
            build_lock_manager(redis_client=redis_client)
            if redis_url is None
            else DistributedLockManager(redis_url, redis_client)
        )
        self.scheduler = self._build_scheduler(redis_url=redis_url, redis_client=redis_client)
        self.executor = RunExecutor(self.repo)
        self._started = False

    def _build_scheduler(
        self,
        *,
        redis_url: Optional[str],
        redis_client: Optional["Redis"],
    ) -> AsyncIOScheduler:
        jobstores = {}
        if RedisJobStore:
            connection_defined = False
            if redis_url:
                try:
                    jobstores["default"] = RedisJobStore(
                        jobs_key="plateerag:scraper:scheduler:jobs",
                        run_times_key="plateerag:scraper:scheduler:run_times",
                        url=redis_url,
                    )
                    connection_defined = True
                    logger.info("Scraper scheduler using Redis job store URL: %s", redis_url)
                except Exception as exc:  # pragma: no cover - external dependency
                    logger.warning(
                        "Failed to configure Redis job store (%s), reverting to alternative source: %s",
                        redis_url,
                        exc,
                    )
            if not connection_defined and redis_client:
                try:
                    pool_kwargs = getattr(redis_client.connection_pool, "connection_kwargs", {})
                    host = pool_kwargs.get("host", "localhost")
                    port = pool_kwargs.get("port", 6379)
                    db = pool_kwargs.get("db", 0)
                    password = pool_kwargs.get("password")
                    username = pool_kwargs.get("username")

                    jobstore_kwargs = {
                        "jobs_key": "plateerag:scraper:scheduler:jobs",
                        "run_times_key": "plateerag:scraper:scheduler:run_times",
                        "host": host,
                        "port": port,
                        "db": db,
                    }
                    if password:
                        jobstore_kwargs["password"] = password
                    if username:
                        jobstore_kwargs["username"] = username

                    jobstores["default"] = RedisJobStore(**jobstore_kwargs)
                    logger.info(
                        "Scraper scheduler using Redis job store via connection kwargs (host=%s port=%s db=%s)",
                        host,
                        port,
                        db,
                    )
                except Exception as exc:  # pragma: no cover - external dependency
                    logger.warning(
                        "Failed to configure Redis job store from provided client; falling back to in-memory: %s",
                        exc,
                    )

        scheduler = AsyncIOScheduler(jobstores=jobstores or None, timezone=timezone.utc)
        return scheduler

    async def start(self) -> None:
        if self._started:
            return
        self.scheduler.start()
        self._started = True
        await self.refresh_interval_jobs()
        logger.info("Data scraper scheduler started (distributed=%s)", self.lock_manager.is_distributed)

    async def shutdown(self) -> None:
        if not self._started:
            return
        try:
            self.scheduler.shutdown(wait=False)
            logger.info("Data scraper scheduler shut down")
        finally:
            self._started = False

    # ------------------------------------------------------------------
    # Manual run orchestration
    # ------------------------------------------------------------------
    async def enqueue_run(self, scraper: ScraperConfig, run: ScraperRun) -> None:
        """Schedule execution for an existing `scraper_runs` record."""
        if not self._started:
            await self.start()
        job_id = self._run_job_id(run.id)

        self.scheduler.add_job(
            self._execute_run,
            trigger="date",
            run_date=datetime.now(timezone.utc),
            id=job_id,
            args=(run.id,),
            replace_existing=True,
            coalesce=True,
        )
        logger.info("Queued scraper run %s (scraper=%s)", run.id, scraper.id)

    # ------------------------------------------------------------------
    # Interval scheduling
    # ------------------------------------------------------------------
    async def refresh_interval_jobs(self) -> None:
        """Ensure APScheduler jobs match current scraper configurations."""
        configs = self.repo.list_scrapers(limit=1000, offset=0, order_by="id", ascending=True)
        for config in configs:
            self.apply_interval_job(config)

    def apply_interval_job(self, scraper: ScraperConfig) -> None:
        job_id = self._interval_job_id(scraper.id)
        interval = scraper.schedule_interval_minutes
        if interval and interval > 0:
            trigger = IntervalTrigger(minutes=interval)
            self.scheduler.add_job(
                self._schedule_periodic_run,
                trigger=trigger,
                id=job_id,
                args=(scraper.id,),
                replace_existing=True,
                coalesce=True,
                max_instances=1,
            )
            logger.info("Registered interval job for scraper %s (%s min)", scraper.id, interval)
        else:
            try:
                self.scheduler.remove_job(job_id)
                logger.info("Removed interval job for scraper %s", scraper.id)
            except JobLookupError:
                pass

    def remove_interval_job(self, scraper_id: int) -> None:
        job_id = self._interval_job_id(scraper_id)
        try:
            self.scheduler.remove_job(job_id)
            logger.info("Removed interval job for deleted scraper %s", scraper_id)
        except JobLookupError:
            pass

    # ------------------------------------------------------------------
    # Internal scheduling callbacks
    # ------------------------------------------------------------------
    async def _schedule_periodic_run(self, scraper_id: int) -> None:
        scraper = self.repo.get_scraper(scraper_id)
        if not scraper:
            logger.warning("Skipping scheduled run for missing scraper %s", scraper_id)
            return

        run = ScraperRun(
            scraper_id=scraper_id,
            run_uid=uuid.uuid4().hex,
            trigger_type="schedule",
            status="pending",
            project_id=scraper.project_id,
        )
        run_id = self.repo.create_run(run)
        if not run_id:
            logger.error("Failed to persist scheduled run for scraper %s", scraper_id)
            return
        run.id = run_id
        await self.enqueue_run(scraper, run)

    async def _execute_run(self, run_id: int) -> None:
        run = self.repo.get_run(run_id)
        if not run:
            logger.warning("Run %s no longer exists; skipping execution", run_id)
            return

        scraper = self.repo.get_scraper(run.scraper_id)
        if not scraper:
            logger.warning("Scraper %s missing for run %s", run.scraper_id, run_id)
            self.repo.update_run_status(
                run_id,
                "failed",
                error_log={"message": "Scraper configuration missing"},
                finished_at=ScraperRun.now(),
            )
            return

        lock_key = f"scraper:{scraper.id}:lock"
        token = await self.lock_manager.acquire(lock_key, ttl_seconds=self.DEFAULT_LOCK_TTL)
        if not token:
            logger.info("Skipping run %s; lock held for scraper %s", run_id, scraper.id)
            self.repo.update_run_status(
                run_id,
                "failed",
                error_log={
                    "message": "Another run is in progress for this scraper",
                    "code": "lock_unavailable",
                },
                finished_at=ScraperRun.now(),
            )
            self.repo.bump_stats(
                scraper_id=scraper.id,
                success=False,
                duration_seconds=0,
                items_collected=0,
            )
            return

        try:
            await self.executor.execute(scraper, run)
        finally:
            await self.lock_manager.release(lock_key, token)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _interval_job_id(scraper_id: int) -> str:
        return f"scraper-interval-{scraper_id}"

    @staticmethod
    def _run_job_id(run_id: int) -> str:
        return f"scraper-run-{run_id}"


def build_scheduler(
    app_db: AppDatabaseManager,
    redis_client: Optional["Redis"] = None,
) -> DataScraperScheduler:
    """Factory that prefers a provided Redis client, falling back to environment variables."""
    redis_url = None
    if not redis_client:
        redis_url = os.getenv("SCRAPER_REDIS_URL") or os.getenv("REDIS_URL")
    return DataScraperScheduler(app_db, redis_url=redis_url, redis_client=redis_client)

"""Mock run executor used by the scheduler to validate orchestration flow."""
from __future__ import annotations

import asyncio
import logging
import random
from typing import Dict, Optional

from service.database.models.data_scraper import ScraperConfig, ScraperRun
from service.data_scraper.repository import ScraperRepository

logger = logging.getLogger("data-scraper-executor")


class RunExecutor:
    """Simulate scraper executions while updating run state and stats."""

    def __init__(self, repository: ScraperRepository, min_duration: float = 1.0, max_duration: float = 3.0):
        self.repo = repository
        self.min_duration = min_duration
        self.max_duration = max_duration

    async def execute(self, scraper: ScraperConfig, run: ScraperRun) -> Dict[str, Optional[int]]:
        """Simulate a scraper run and update status and statistics."""
        logger.info("Executing scraper run %s for scraper %s", run.id, scraper.id)
        await self._mark_running(run)

        duration_seconds: Optional[int] = None
        items_collected = 0
        try:
            duration_seconds = await self._simulate_workload()
            items_collected = self._mock_items_collected(scraper)

            await self._mark_completed(run, duration_seconds, items_collected)
            self.repo.update_scraper_last_run(scraper.id)
            self.repo.bump_stats(
                scraper_id=scraper.id,
                success=True,
                duration_seconds=duration_seconds,
                items_collected=items_collected,
            )
            logger.info(
                "Scraper run %s finished successfully (duration=%ss, items=%s)",
                run.id,
                duration_seconds,
                items_collected,
            )
            return {"duration": duration_seconds, "items_collected": items_collected}

        except Exception as exc:  # pragma: no cover - unexpected errors
            logger.exception("Scraper run %s failed: %s", run.id, exc)
            await self._mark_failed(run, error=str(exc))
            self.repo.bump_stats(
                scraper_id=scraper.id,
                success=False,
                duration_seconds=duration_seconds or 0,
                items_collected=items_collected,
            )
            return {"duration": duration_seconds, "items_collected": items_collected}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    async def _simulate_workload(self) -> int:
        duration = random.uniform(self.min_duration, self.max_duration)
        await asyncio.sleep(duration)
        return int(duration)

    def _mock_items_collected(self, scraper: ScraperConfig) -> int:
        """Generate a deterministic-ish sample metric based on scraper id."""
        base = (scraper.id or 1) % 7 + 3
        return base

    async def _mark_running(self, run: ScraperRun) -> None:
        run.status = "running"
        run.started_at = ScraperRun.now()
        self.repo.update_run(run)

    async def _mark_completed(self, run: ScraperRun, duration_seconds: int, items_collected: int) -> None:
        run.status = "completed"
        run.finished_at = ScraperRun.now()
        run.duration_seconds = duration_seconds
        run.items_collected = items_collected
        self.repo.update_run(run)

    async def _mark_failed(self, run: ScraperRun, error: str) -> None:
        run.status = "failed"
        run.finished_at = ScraperRun.now()
        run.error_log = {"message": error}
        self.repo.update_run(run)

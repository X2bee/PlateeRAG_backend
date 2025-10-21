"""Mock run executor used by the scheduler prototype."""
from __future__ import annotations

import asyncio
import logging
import random
from typing import Dict

logger = logging.getLogger("scheduler-prototype-executor")


async def execute_run(scraper_id: int, run_id: str) -> Dict[str, int]:
    """Simulate scraping by sleeping for a random duration."""
    logger.info("Executor starting scraper=%s run=%s", scraper_id, run_id)
    duration = random.uniform(0.5, 1.5)
    await asyncio.sleep(duration)
    collected = random.randint(1, 5)
    logger.info(
        "Executor finished scraper=%s run=%s (duration=%.2fs, items=%s)",
        scraper_id,
        run_id,
        duration,
        collected,
    )
    return {"duration": duration, "items": collected}

# Data Scraper Scheduler Prototype

This prototype validates the APScheduler + Redis approach described in `docs/data-scraper-scheduler-prototype.md`.

## Features
- FastAPI app with an `AsyncIOScheduler`.
- Redis-backed lock (`scraper:{id}:lock`) to prevent duplicate executions.
- Run state recorded in `scraper:run:{run_id}` hashes.
- Mock executor that sleeps and records synthetic metrics.

## Requirements
- Python 3.12 (matches project runtime)
- Redis server (local or remote)

Install dependencies:

```bash
poetry install --with dev
```

Set the Redis connection string (defaults to `redis://localhost:6379/0`):

```bash
export REDIS_URL="redis://localhost:6379/0"
```

## Running the Prototype

```bash
poetry run uvicorn prototypes.scheduler.main:app --reload
```

Trigger a manual run:

```bash
curl -X POST http://localhost:8000/runs/manual/1
```

Monitor run state:

```bash
redis-cli HGETALL scraper:run:<run_id>
```

If you send another request while the lock is held you will receive `409 Conflict`, demonstrating duplicate prevention.

## Next Steps
- Extend the executor to call the real Go `newscrawler`.
- Move locking/state helpers into shared utilities for the main backend.
- Integrate interval scheduling tests by adding APScheduler interval jobs.

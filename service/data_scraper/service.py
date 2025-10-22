"""Business logic for data scraper capabilities."""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

from pydantic import BaseModel

from service.database.connection import AppDatabaseManager
from service.database.models.data_scraper import (
    ScraperConfig,
    ScraperRun,
    ScrapedItem,
    ScrapedAsset,
    ScraperStats,
)
from service.data_scraper.repository import ScraperRepository
from service.data_scraper.robots import check_robots as fetch_robots
from service.data_scraper.schemas import (
    ScraperConfigCreate,
    ScraperConfigUpdate,
    ScraperConfigFilterParams,
    ScraperRunRequest,
    ScraperFilterParams,
    ScrapedItemFilterParams,
    Pagination,
)

if TYPE_CHECKING:
    from service.data_scraper.scheduler_manager import DataScraperScheduler

logger = logging.getLogger("data-scraper-service")


class ScraperService:
    """Orchestrates scraper configuration, execution, and ingestion."""

    CONFIG_FIELDS = {
        "name",
        "endpoint",
        "data_source_type",
        "parsing_method",
        "schedule_interval_minutes",
        "max_depth",
        "follow_links",
        "respect_robots",
        "user_agent",
        "headers",
        "authentication",
        "filters",
        "status",
        "project_id",
    }

    CONFIG_FILTER_FIELDS = {
        "status",
        "project_id",
        "data_source_type",
        "scraper_uid",
    }

    RUN_FILTER_FIELDS = {
        "scraper_id",
        "status",
        "trigger_type",
        "project_id",
    }

    ITEM_FILTER_FIELDS = {
        "scraper_id",
        "run_id",
        "content_type",
        "parsing_method",
        "project_id",
    }

    def __init__(self, db: AppDatabaseManager, scheduler: Optional["DataScraperScheduler"] = None):
        self.repo = ScraperRepository(db)
        self.logger = logger
        self.scheduler = scheduler

    # ------------------------------------------------------------------
    # Scraper configuration
    # ------------------------------------------------------------------
    def list_scrapers(
        self,
        filters: Optional[Union[ScraperConfigFilterParams, Dict[str, Any]]] = None,
        pagination: Optional[Pagination] = None,
    ) -> Dict[str, Any]:
        filter_dict = self._normalize_filters(filters)
        pagination = pagination or Pagination()

        allowed_filters = self._filter_allowed(filter_dict, self.CONFIG_FILTER_FIELDS)
        models = self.repo.list_scrapers(
            filters=allowed_filters,
            limit=pagination.limit,
            offset=pagination.offset,
            order_by=pagination.order_by or "updated_at",
            ascending=pagination.ascending,
        )
        items = [self._serialize_model(model) for model in models]
        total = self.repo.count_scrapers(allowed_filters)
        page = None
        if pagination.order_by:  # reuse pagination fields for page calculation on demand
            pass  # placeholder for future sorting options
        current_page = (pagination.offset // pagination.limit + 1) if pagination.limit else 1
        return {
            "items": items,
            "count": len(items),
            "limit": pagination.limit,
            "offset": pagination.offset,
            "total": total,
            "page": current_page,
            "pageSize": pagination.limit,
        }

    def get_scraper(self, scraper_id: int) -> Dict[str, Any]:
        scraper = self.repo.get_scraper(scraper_id)
        if not scraper:
            raise ValueError(f"Scraper {scraper_id} not found")
        return self._serialize_model(scraper)

    def create_scraper(
        self,
        payload: Union[ScraperConfigCreate, Dict[str, Any]],
        created_by: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload_dict = self._payload_to_dict(payload)
        payload_dict.setdefault("scraper_uid", uuid.uuid4().hex)
        if created_by is not None:
            payload_dict["created_by"] = created_by
        config = ScraperConfig(**payload_dict)
        new_id = self.repo.create_scraper(config)
        if new_id is None:
            raise RuntimeError("Failed to persist scraper configuration")
        config.id = new_id
        if self.scheduler:
            self.scheduler.apply_interval_job(config)
        return self._serialize_model(config)

    def update_scraper(
        self,
        scraper_id: int,
        payload: Union[ScraperConfigUpdate, Dict[str, Any]],
    ) -> Dict[str, Any]:
        scraper = self.repo.get_scraper(scraper_id)
        if not scraper:
            raise ValueError(f"Scraper {scraper_id} not found")

        payload_dict = self._payload_to_dict(payload)
        for key, value in payload_dict.items():
            if key in self.CONFIG_FIELDS:
                setattr(scraper, key, value)

        if not self.repo.update_scraper(scraper):
            raise RuntimeError(f"Failed to update scraper {scraper_id}")

        if self.scheduler:
            self.scheduler.apply_interval_job(scraper)
        return self._serialize_model(scraper)

    def delete_scraper(self, scraper_id: int) -> bool:
        deleted = self.repo.delete_scraper(scraper_id)
        if not deleted:
            raise ValueError(f"Scraper {scraper_id} not found or already deleted")
        if self.scheduler:
            self.scheduler.remove_interval_job(scraper_id)
        return True

    # ------------------------------------------------------------------
    # Execution lifecycle
    # ------------------------------------------------------------------
    async def request_run(
        self,
        scraper_id: int,
        payload: Optional[Union[ScraperRunRequest, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        scraper = self.repo.get_scraper(scraper_id)
        if not scraper:
            raise ValueError(f"Scraper {scraper_id} not found")

        payload_dict = self._payload_to_dict(payload) if payload else {}
        trigger_type = payload_dict.get("trigger", "manual")
        triggered_by = payload_dict.get("triggered_by")

        run = ScraperRun(
            scraper_id=scraper_id,
            run_uid=uuid.uuid4().hex,
            trigger_type=trigger_type,
            status="pending",
            project_id=scraper.project_id,
            triggered_by=triggered_by,
        )

        run_id = self.repo.create_run(run)
        if run_id is None:
            self.logger.error("Failed to create run record for scraper %s", scraper_id)
            raise RuntimeError("Failed to create scraper run")
        run.id = run_id

        if self.scheduler:
            await self.scheduler.enqueue_run(scraper, run)
        return self._serialize_model(run)

    def mark_run_started(self, run_id: int) -> Dict[str, Any]:
        run = self.repo.get_run(run_id)
        if not run:
            raise ValueError(f"Run {run_id} not found")
        run.status = "running"
        run.started_at = ScraperRun.now()
        if not self.repo.update_run(run):
            raise RuntimeError(f"Failed to update run {run_id}")
        return self._serialize_model(run)

    def mark_run_completed(
        self,
        run_id: int,
        *,
        items_collected: int,
        items_failed: int = 0,
        duration_seconds: Optional[int] = None,
        error_log: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        run = self.repo.get_run(run_id)
        if not run:
            raise ValueError(f"Run {run_id} not found")
        run.status = "completed"
        run.finished_at = ScraperRun.now()
        run.items_collected = items_collected
        run.items_failed = items_failed
        run.duration_seconds = duration_seconds
        if error_log:
            run.error_log = error_log
        if not self.repo.update_run(run):
            raise RuntimeError(f"Failed to finalize run {run_id}")
        return self._serialize_model(run)

    def mark_run_failed(
        self,
        run_id: int,
        *,
        error_log: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        run = self.repo.get_run(run_id)
        if not run:
            raise ValueError(f"Run {run_id} not found")
        run.status = "failed"
        run.finished_at = ScraperRun.now()
        if error_log:
            run.error_log = error_log
        if not self.repo.update_run(run):
            raise RuntimeError(f"Failed to mark run {run_id} as failed")
        return self._serialize_model(run)

    def list_runs(
        self,
        filters: Optional[Union[ScraperFilterParams, Dict[str, Any]]] = None,
        pagination: Optional[Pagination] = None,
    ) -> Dict[str, Any]:
        filter_dict = self._normalize_filters(filters)
        pagination = pagination or Pagination(limit=50, offset=0, order_by="started_at", ascending=False)
        runs = self.repo.list_runs(
            filters=self._filter_allowed(filter_dict, self.RUN_FILTER_FIELDS),
            limit=pagination.limit,
            offset=pagination.offset,
            order_by=pagination.order_by or "started_at",
            ascending=pagination.ascending,
        )
        items = [self._serialize_model(run) for run in runs]
        return {
            "items": items,
            "count": len(items),
            "limit": pagination.limit,
            "offset": pagination.offset,
        }

    def get_run(self, run_id: int) -> Dict[str, Any]:
        run = self.repo.get_run(run_id)
        if not run:
            raise ValueError(f"Run {run_id} not found")
        return self._serialize_model(run)

    # ------------------------------------------------------------------
    # Data lake
    # ------------------------------------------------------------------
    def record_item(
        self,
        payload: Dict[str, Any],
        assets: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        item = ScrapedItem(**payload)
        item_id = self.repo.create_item(item)
        if item_id is None:
            raise RuntimeError("Failed to create scraped item")
        item.id = item_id

        created_assets: List[Dict[str, Any]] = []
        if assets:
            for asset_payload in assets:
                asset_payload["item_id"] = item_id
                asset = ScrapedAsset(**asset_payload)
                asset_id = self.repo.create_asset(asset)
                if asset_id:
                    asset.id = asset_id
                    created_assets.append(self._serialize_model(asset))

        item_dict = self._serialize_model(item)
        if created_assets:
            item_dict["assets"] = created_assets
        return item_dict

    def list_items(
        self,
        filters: Optional[Union[ScrapedItemFilterParams, Dict[str, Any]]] = None,
        pagination: Optional[Pagination] = None,
    ) -> Dict[str, Any]:
        filter_dict = self._normalize_filters(filters)
        pagination = pagination or Pagination(limit=50, offset=0, order_by="collected_at", ascending=False)
        items = self.repo.list_items(
            filters=self._filter_allowed(filter_dict, self.ITEM_FILTER_FIELDS),
            limit=pagination.limit,
            offset=pagination.offset,
            order_by=pagination.order_by or "collected_at",
            ascending=pagination.ascending,
        )
        serialized = [self._serialize_model(item) for item in items]
        return {
            "items": serialized,
            "count": len(serialized),
            "limit": pagination.limit,
            "offset": pagination.offset,
        }

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def get_stats(self, scraper_id: int) -> Optional[Dict[str, Any]]:
        stats = self.repo.get_stats(scraper_id)
        if not stats:
            return None
        return self._serialize_model(stats)

    def update_stats(self, scraper_id: int, updates: Dict[str, Any]) -> Dict[str, Any]:
        stats = self.repo.get_stats(scraper_id)
        if stats:
            for key, value in updates.items():
                setattr(stats, key, value)
        else:
            stats = ScraperStats(scraper_id=scraper_id, **updates)

        success = self.repo.upsert_stats(stats)
        if not success:
            raise RuntimeError(f"Failed to upsert stats for scraper {scraper_id}")
        return self._serialize_model(stats)

    # ------------------------------------------------------------------
    # Operational helpers
    # ------------------------------------------------------------------
    def check_robots(self, endpoint: str, user_agent: Optional[str]) -> Dict[str, Any]:
        """Fetch and evaluate robots.txt for a scraper endpoint."""
        self.logger.info("Robots check requested for %s", endpoint)
        return fetch_robots(endpoint, user_agent)

    def test_scraper(self, scraper_id: int) -> Dict[str, Any]:
        """Placeholder test run that returns immediate response."""
        scraper = self.repo.get_scraper(scraper_id)
        if not scraper:
            raise ValueError(f"Scraper {scraper_id} not found")
        self.logger.info("Test run requested for scraper %s", scraper.scraper_uid)
        return {
            "scraper_id": scraper_id,
            "status": "queued",
            "message": "Test execution stub. Integrate with executor to run sample crawl.",
        }

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _payload_to_dict(self, payload: Union[BaseModel, Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(payload, BaseModel):
            return payload.dict(exclude_unset=True)
        return {k: v for k, v in payload.items() if v is not None}

    def _normalize_filters(self, filters: Optional[Union[BaseModel, Dict[str, Any]]]) -> Dict[str, Any]:
        if not filters:
            return {}
        if isinstance(filters, BaseModel):
            return filters.dict(exclude_unset=True)
        return {k: v for k, v in filters.items() if v is not None}

    def _filter_allowed(self, data: Dict[str, Any], allowed_keys: set) -> Dict[str, Any]:
        return {k: v for k, v in data.items() if k in allowed_keys}

    def _serialize_model(self, model: Any) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        for key, value in model.__dict__.items():
            if key.startswith("_"):
                continue
            if isinstance(value, datetime):
                data[key] = value.isoformat()
            else:
                data[key] = value
        return data

"""Data scraper related persistent models."""
from __future__ import annotations

from typing import Any, Dict, Optional
import json
from datetime import datetime

from service.database.models.base_model import BaseModel


def _parse_json_field(value: Optional[Any]) -> Optional[Dict[str, Any]]:
    """Helper to ensure JSON fields are consistently stored as dict."""
    if value is None or value == "":
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None
    return None


class ScraperConfig(BaseModel):
    """Configuration metadata for a registered scraper definition."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scraper_uid: str = kwargs.get("scraper_uid", "")
        self.name: str = kwargs.get("name", "")
        self.endpoint: str = kwargs.get("endpoint", "")
        self.data_source_type: str = kwargs.get("data_source_type", "web")
        self.parsing_method: str = kwargs.get("parsing_method", "html")
        self.schedule_interval_minutes: Optional[int] = kwargs.get("schedule_interval_minutes")
        self.max_depth: Optional[int] = kwargs.get("max_depth")
        self.follow_links: bool = kwargs.get("follow_links", False)
        self.respect_robots: bool = kwargs.get("respect_robots", True)
        self.user_agent: Optional[str] = kwargs.get("user_agent")
        self.headers: Optional[Dict[str, Any]] = _parse_json_field(kwargs.get("headers"))
        self.authentication: Optional[Dict[str, Any]] = _parse_json_field(kwargs.get("authentication"))
        self.filters: Optional[Dict[str, Any]] = _parse_json_field(kwargs.get("filters"))
        self.status: str = kwargs.get("status", "idle")
        self.last_run_at: Optional[datetime] = kwargs.get("last_run_at")
        self.project_id: Optional[int] = kwargs.get("project_id")
        self.created_by: Optional[int] = kwargs.get("created_by")

    def get_table_name(self) -> str:
        return "scraper_configs"

    def get_schema(self) -> Dict[str, str]:
        return {
            "scraper_uid": "VARCHAR(64) UNIQUE NOT NULL",
            "name": "VARCHAR(200) NOT NULL",
            "endpoint": "TEXT NOT NULL",
            "data_source_type": "VARCHAR(50) DEFAULT 'web'",
            "parsing_method": "VARCHAR(30) DEFAULT 'html'",
            "schedule_interval_minutes": "INTEGER",
            "max_depth": "INTEGER",
            "follow_links": "BOOLEAN DEFAULT FALSE",
            "respect_robots": "BOOLEAN DEFAULT TRUE",
            "user_agent": "TEXT",
            "headers": "TEXT",
            "authentication": "TEXT",
            "filters": "TEXT",
            "status": "VARCHAR(30) DEFAULT 'idle'",
            "last_run_at": "TIMESTAMP",
            "project_id": "INTEGER REFERENCES group_meta(id) ON DELETE SET NULL",
            "created_by": "INTEGER REFERENCES users(id) ON DELETE SET NULL",
        }


class ScraperRun(BaseModel):
    """Execution record for a scraper run lifecycle."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scraper_id: int = kwargs.get("scraper_id")
        self.run_uid: Optional[str] = kwargs.get("run_uid")
        self.trigger_type: str = kwargs.get("trigger_type", "manual")
        self.status: str = kwargs.get("status", "pending")
        self.started_at: Optional[datetime] = kwargs.get("started_at")
        self.finished_at: Optional[datetime] = kwargs.get("finished_at")
        self.duration_seconds: Optional[int] = kwargs.get("duration_seconds")
        self.items_collected: int = kwargs.get("items_collected", 0)
        self.items_failed: int = kwargs.get("items_failed", 0)
        self.config_snapshot: Optional[Dict[str, Any]] = _parse_json_field(kwargs.get("config_snapshot"))
        self.error_log: Optional[Dict[str, Any]] = _parse_json_field(kwargs.get("error_log"))
        self.project_id: Optional[int] = kwargs.get("project_id")
        self.triggered_by: Optional[int] = kwargs.get("triggered_by")

    def get_table_name(self) -> str:
        return "scraper_runs"

    def get_schema(self) -> Dict[str, str]:
        return {
            "scraper_id": "INTEGER REFERENCES scraper_configs(id) ON DELETE CASCADE",
            "run_uid": "VARCHAR(64)",
            "trigger_type": "VARCHAR(20) DEFAULT 'manual'",
            "status": "VARCHAR(20) DEFAULT 'pending'",
            "started_at": "TIMESTAMP",
            "finished_at": "TIMESTAMP",
            "duration_seconds": "INTEGER",
            "items_collected": "INTEGER DEFAULT 0",
            "items_failed": "INTEGER DEFAULT 0",
            "config_snapshot": "TEXT",
            "error_log": "TEXT",
            "project_id": "INTEGER REFERENCES group_meta(id) ON DELETE SET NULL",
            "triggered_by": "INTEGER REFERENCES users(id) ON DELETE SET NULL",
        }


class ScrapedItem(BaseModel):
    """Normalized item record extracted from crawler outputs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scraper_id: int = kwargs.get("scraper_id")
        self.run_id: int = kwargs.get("run_id")
        self.source_url: Optional[str] = kwargs.get("source_url")
        self.final_url: Optional[str] = kwargs.get("final_url")
        self.title: Optional[str] = kwargs.get("title")
        self.content_type: Optional[str] = kwargs.get("content_type")
        self.raw_path: Optional[str] = kwargs.get("raw_path")
        self.clean_path: Optional[str] = kwargs.get("clean_path")
        self.preview_text: Optional[str] = kwargs.get("preview_text")
        self.metadata: Optional[Dict[str, Any]] = _parse_json_field(kwargs.get("metadata"))
        self.parsing_method: Optional[str] = kwargs.get("parsing_method")
        self.size_bytes: Optional[int] = kwargs.get("size_bytes")
        self.collected_at: Optional[datetime] = kwargs.get("collected_at")
        self.checksum: Optional[str] = kwargs.get("checksum")
        self.project_id: Optional[int] = kwargs.get("project_id")

    def get_table_name(self) -> str:
        return "scraped_items"

    def get_schema(self) -> Dict[str, str]:
        return {
            "scraper_id": "INTEGER REFERENCES scraper_configs(id) ON DELETE CASCADE",
            "run_id": "INTEGER REFERENCES scraper_runs(id) ON DELETE CASCADE",
            "source_url": "TEXT",
            "final_url": "TEXT",
            "title": "TEXT",
            "content_type": "VARCHAR(100)",
            "raw_path": "TEXT",
            "clean_path": "TEXT",
            "preview_text": "TEXT",
            "metadata": "TEXT",
            "parsing_method": "VARCHAR(30)",
            "size_bytes": "BIGINT",
            "collected_at": "TIMESTAMP",
            "checksum": "VARCHAR(128)",
            "project_id": "INTEGER REFERENCES group_meta(id) ON DELETE SET NULL",
        }


class ScrapedAsset(BaseModel):
    """Asset attachment stored alongside a scraped item."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.item_id: int = kwargs.get("item_id")
        self.asset_type: str = kwargs.get("asset_type", "image")
        self.stored_path: str = kwargs.get("stored_path", "")
        self.content_type: Optional[str] = kwargs.get("content_type")
        self.size_bytes: Optional[int] = kwargs.get("size_bytes")
        self.width: Optional[int] = kwargs.get("width")
        self.height: Optional[int] = kwargs.get("height")
        self.metadata: Optional[Dict[str, Any]] = _parse_json_field(kwargs.get("metadata"))

    def get_table_name(self) -> str:
        return "scraped_assets"

    def get_schema(self) -> Dict[str, str]:
        return {
            "item_id": "INTEGER REFERENCES scraped_items(id) ON DELETE CASCADE",
            "asset_type": "VARCHAR(30) NOT NULL",
            "stored_path": "TEXT NOT NULL",
            "content_type": "VARCHAR(100)",
            "size_bytes": "BIGINT",
            "width": "INTEGER",
            "height": "INTEGER",
            "metadata": "TEXT",
        }


class ScraperStats(BaseModel):
    """Denormalized statistics record that can be refreshed periodically."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scraper_id: int = kwargs.get("scraper_id")
        self.total_runs: int = kwargs.get("total_runs", 0)
        self.successful_runs: int = kwargs.get("successful_runs", 0)
        self.failed_runs: int = kwargs.get("failed_runs", 0)
        self.total_items: int = kwargs.get("total_items", 0)
        self.total_size_bytes: int = kwargs.get("total_size_bytes", 0)
        self.avg_duration_seconds: Optional[float] = kwargs.get("avg_duration_seconds")
        self.last_run_at: Optional[datetime] = kwargs.get("last_run_at")

    def get_table_name(self) -> str:
        return "scraper_stats"

    def get_schema(self) -> Dict[str, str]:
        return {
            "scraper_id": "INTEGER REFERENCES scraper_configs(id) ON DELETE CASCADE UNIQUE",
            "total_runs": "INTEGER DEFAULT 0",
            "successful_runs": "INTEGER DEFAULT 0",
            "failed_runs": "INTEGER DEFAULT 0",
            "total_items": "BIGINT DEFAULT 0",
            "total_size_bytes": "BIGINT DEFAULT 0",
            "avg_duration_seconds": "NUMERIC",
            "last_run_at": "TIMESTAMP",
        }

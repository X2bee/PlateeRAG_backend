"""Persistence helpers for data scraper domain."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import logging

from service.database.connection import AppDatabaseManager
from service.database.models.data_scraper import (
    ScraperConfig,
    ScraperRun,
    ScrapedItem,
    ScrapedAsset,
    ScraperStats,
)

logger = logging.getLogger("data-scraper-repository")


class ScraperRepository:
    """Data access layer for scraper-related tables."""

    def __init__(self, db: AppDatabaseManager):
        self.db = db
        self.logger = logger

    # ------------------------------------------------------------------
    # Scraper configuration operations
    # ------------------------------------------------------------------
    def list_scrapers(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 50,
        offset: int = 0,
        order_by: str = "updated_at",
        ascending: bool = False,
    ) -> List[ScraperConfig]:
        manager = self.db.config_db_manager
        db_type = manager.db_type
        table = ScraperConfig().get_table_name()
        filters = filters or {}

        where_clauses: List[str] = []
        values: List[Any] = []

        def add_clause(clause: str, *vals):
            where_clauses.append(clause)
            values.extend(vals)

        for key, value in filters.items():
            if value is None:
                continue
            if key.endswith("__like__"):
                real_key = key.removesuffix("__like__")
                placeholder = "%s" if db_type == "postgresql" else "?"
                column = f"{table}.{real_key}"
                expression = (
                    f"{column} ILIKE {placeholder}"
                    if db_type == "postgresql"
                    else f"{column} LIKE {placeholder}"
                )
                add_clause(expression, f"%{value}%")
            elif key.endswith("__notlike__"):
                real_key = key.removesuffix("__notlike__")
                placeholder = "%s" if db_type == "postgresql" else "?"
                column = f"{table}.{real_key}"
                expression = (
                    f"{column} NOT ILIKE {placeholder}"
                    if db_type == "postgresql"
                    else f"{column} NOT LIKE {placeholder}"
                )
                add_clause(expression, f"%{value}%")
            elif key.endswith("__gte__"):
                real_key = key.removesuffix("__gte__")
                placeholder = "%s" if db_type == "postgresql" else "?"
                column = f"{table}.{real_key}"
                add_clause(f"{column} >= {placeholder}", value)
            elif key.endswith("__lte__"):
                real_key = key.removesuffix("__lte__")
                placeholder = "%s" if db_type == "postgresql" else "?"
                column = f"{table}.{real_key}"
                add_clause(f"{column} <= {placeholder}", value)
            elif key.endswith("__gt__"):
                real_key = key.removesuffix("__gt__")
                placeholder = "%s" if db_type == "postgresql" else "?"
                column = f"{table}.{real_key}"
                add_clause(f"{column} > {placeholder}", value)
            elif key.endswith("__lt__"):
                real_key = key.removesuffix("__lt__")
                placeholder = "%s" if db_type == "postgresql" else "?"
                column = f"{table}.{real_key}"
                add_clause(f"{column} < {placeholder}", value)
            elif key.endswith("__in__"):
                real_key = key.removesuffix("__in__")
                if not isinstance(value, (list, tuple)) or not value:
                    continue
                placeholders = ", ".join(["%s" if db_type == "postgresql" else "?"] * len(value))
                column = f"{table}.{real_key}"
                add_clause(f"{column} IN ({placeholders})", *value)
            elif key.endswith("__notin__"):
                real_key = key.removesuffix("__notin__")
                if not isinstance(value, (list, tuple)) or not value:
                    continue
                placeholders = ", ".join(["%s" if db_type == "postgresql" else "?"] * len(value))
                column = f"{table}.{real_key}"
                add_clause(f"{column} NOT IN ({placeholders})", *value)
            else:
                column = f"{table}.{key}"
                placeholder = "%s" if db_type == "postgresql" else "?"
                add_clause(f"{column} = {placeholder}", value)

        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
        order_clause = f"{table}.{order_by}" if order_by else f"{table}.updated_at"
        order_direction = "ASC" if ascending else "DESC"

        if db_type == "postgresql":
            query = (
                f"SELECT {table}.* FROM {table} "
                f"WHERE {where_clause} "
                f"ORDER BY {order_clause} {order_direction} "
                f"LIMIT %s OFFSET %s"
            )
        else:
            query = (
                f"SELECT {table}.* FROM {table} "
                f"WHERE {where_clause} "
                f"ORDER BY {order_clause} {order_direction} "
                f"LIMIT ? OFFSET ?"
            )

        values_with_paging = list(values) + [limit, offset]
        rows = manager.execute_query(query, tuple(values_with_paging)) or []
        return [ScraperConfig.from_dict(dict(row)) for row in rows]

    def count_scrapers(self, filters: Optional[Dict[str, Any]] = None) -> int:
        manager = self.db.config_db_manager
        db_type = manager.db_type
        table = ScraperConfig().get_table_name()
        where_clauses: List[str] = []
        values: List[Any] = []
        filters = filters or {}

        def add_clause(clause: str, *vals):
            where_clauses.append(clause)
            values.extend(vals)

        for key, value in filters.items():
            if value is None:
                continue
            if key.endswith("__like__"):
                real_key = key.removesuffix("__like__")
                placeholder = "%s" if db_type == "postgresql" else "?"
                column = f"{table}.{real_key}"
                add_clause(
                    f"{column} ILIKE {placeholder}" if db_type == "postgresql" else f"{column} LIKE {placeholder}",
                    f"%{value}%",
                )
            elif key.endswith("__notlike__"):
                real_key = key.removesuffix("__notlike__")
                placeholder = "%s" if db_type == "postgresql" else "?"
                column = f"{table}.{real_key}"
                add_clause(
                    f"{column} NOT ILIKE {placeholder}" if db_type == "postgresql" else f"{column} NOT LIKE {placeholder}",
                    f"%{value}%",
                )
            elif key.endswith("__gte__"):
                real_key = key.removesuffix("__gte__")
                placeholder = "%s" if db_type == "postgresql" else "?"
                column = f"{table}.{real_key}"
                add_clause(f"{column} >= {placeholder}", value)
            elif key.endswith("__lte__"):
                real_key = key.removesuffix("__lte__")
                placeholder = "%s" if db_type == "postgresql" else "?"
                column = f"{table}.{real_key}"
                add_clause(f"{column} <= {placeholder}", value)
            elif key.endswith("__gt__"):
                real_key = key.removesuffix("__gt__")
                placeholder = "%s" if db_type == "postgresql" else "?"
                column = f"{table}.{real_key}"
                add_clause(f"{column} > {placeholder}", value)
            elif key.endswith("__lt__"):
                real_key = key.removesuffix("__lt__")
                placeholder = "%s" if db_type == "postgresql" else "?"
                column = f"{table}.{real_key}"
                add_clause(f"{column} < {placeholder}", value)
            elif key.endswith("__in__"):
                real_key = key.removesuffix("__in__")
                if not isinstance(value, (list, tuple)) or not value:
                    continue
                placeholders = ", ".join(["%s" if db_type == "postgresql" else "?"] * len(value))
                column = f"{table}.{real_key}"
                add_clause(f"{column} IN ({placeholders})", *value)
            elif key.endswith("__notin__"):
                real_key = key.removesuffix("__notin__")
                if not isinstance(value, (list, tuple)) or not value:
                    continue
                placeholders = ", ".join(["%s" if db_type == "postgresql" else "?"] * len(value))
                column = f"{table}.{real_key}"
                add_clause(f"{column} NOT IN ({placeholders})", *value)
            else:
                column = f"{table}.{key}"
                placeholder = "%s" if db_type == "postgresql" else "?"
                add_clause(f"{column} = {placeholder}", value)

        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
        query = f"SELECT COUNT(*) AS count FROM {table} WHERE {where_clause}"
        results = manager.execute_query(query, tuple(values) if values else None)
        if not results:
            return 0
        row = results[0]
        if isinstance(row, dict):
            return int(row.get("count") or next(iter(row.values()), 0) or 0)
        return int(row[0]) if row else 0

    def get_scraper(self, scraper_id: int) -> Optional[ScraperConfig]:
        result = self.db.find_by_condition(
            ScraperConfig,
            conditions={"id": scraper_id},
            limit=1,
        )
        return result[0] if result else None

    def get_scraper_by_uid(self, scraper_uid: str) -> Optional[ScraperConfig]:
        result = self.db.find_by_condition(
            ScraperConfig,
            conditions={"scraper_uid": scraper_uid},
            limit=1,
        )
        return result[0] if result else None

    def create_scraper(self, model: ScraperConfig) -> Optional[int]:
        return_id = self.db.insert(model)
        if return_id and return_id.get("result") == "success":
            return return_id.get("id")
        self.logger.error("Failed to create scraper config: %s", return_id)
        return None

    def update_scraper(self, model: ScraperConfig) -> bool:
        result = self.db.update(model)
        return bool(result and result.get("result") == "success")

    def delete_scraper(self, scraper_id: int) -> bool:
        return self.db.delete(ScraperConfig, scraper_id)

    # ------------------------------------------------------------------
    # Scraper run operations
    # ------------------------------------------------------------------
    def create_run(self, run: ScraperRun) -> Optional[int]:
        return_id = self.db.insert(run)
        if return_id and return_id.get("result") == "success":
            return return_id.get("id")
        self.logger.error("Failed to create scraper run: %s", return_id)
        return None

    def update_run(self, run: ScraperRun) -> bool:
        result = self.db.update(run)
        return bool(result and result.get("result") == "success")

    def update_run_status(self, run_id: int, status: str, **extra_fields) -> bool:
        run = self.get_run(run_id)
        if not run:
            return False
        run.status = status
        for key, value in extra_fields.items():
            setattr(run, key, value)
        return self.update_run(run)

    def list_runs(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 50,
        offset: int = 0,
        order_by: str = "started_at",
        ascending: bool = False,
    ) -> List[ScraperRun]:
        conditions = filters or {}
        return self.db.find_by_condition(
            ScraperRun,
            conditions=conditions,
            limit=limit,
            offset=offset,
            orderby=order_by,
            orderby_asc=ascending,
        )

    def get_run(self, run_id: int) -> Optional[ScraperRun]:
        runs = self.db.find_by_condition(
            ScraperRun,
            conditions={"id": run_id},
            limit=1,
        )
        return runs[0] if runs else None

    # ------------------------------------------------------------------
    # Scraped data operations
    # ------------------------------------------------------------------
    def list_items(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 50,
        offset: int = 0,
        order_by: str = "collected_at",
        ascending: bool = False,
    ) -> List[ScrapedItem]:
        conditions = filters or {}
        return self.db.find_by_condition(
            ScrapedItem,
            conditions=conditions,
            limit=limit,
            offset=offset,
            orderby=order_by,
            orderby_asc=ascending,
        )

    def create_item(self, item: ScrapedItem) -> Optional[int]:
        return_id = self.db.insert(item)
        if return_id and return_id.get("result") == "success":
            return return_id.get("id")
        self.logger.error("Failed to create scraped item: %s", return_id)
        return None

    def create_asset(self, asset: ScrapedAsset) -> Optional[int]:
        return_id = self.db.insert(asset)
        if return_id and return_id.get("result") == "success":
            return return_id.get("id")
        self.logger.error("Failed to create scraped asset: %s", return_id)
        return None

    def list_assets(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[ScrapedAsset]:
        conditions = filters or {}
        return self.db.find_by_condition(
            ScrapedAsset,
            conditions=conditions,
            limit=limit,
            offset=offset,
            orderby="id",
            orderby_asc=False,
        )

    # ------------------------------------------------------------------
    # Statistics operations
    # ------------------------------------------------------------------
    def get_stats(self, scraper_id: int) -> Optional[ScraperStats]:
        stats = self.db.find_by_condition(
            ScraperStats,
            conditions={"scraper_id": scraper_id},
            limit=1,
        )
        return stats[0] if stats else None

    def upsert_stats(self, stats: ScraperStats) -> bool:
        if getattr(stats, "id", None):
            result = self.db.update(stats)
            return bool(result and result.get("result") == "success")

        inserted_id = self.db.insert(stats)
        return bool(inserted_id and inserted_id.get("result") == "success")

    def update_scraper_last_run(self, scraper_id: int) -> None:
        config = self.get_scraper(scraper_id)
        if not config:
            return
        config.last_run_at = ScraperConfig.now()
        self.update_scraper(config)

    def bump_stats(
        self,
        scraper_id: int,
        *,
        success: bool,
        duration_seconds: Optional[int],
        items_collected: int,
    ) -> None:
        stats = self.get_stats(scraper_id)
        if not stats:
            stats = ScraperStats(scraper_id=scraper_id)

        previous_runs = stats.total_runs or 0
        stats.total_runs = previous_runs + 1
        if success:
            stats.successful_runs = (stats.successful_runs or 0) + 1
        else:
            stats.failed_runs = (stats.failed_runs or 0) + 1

        stats.total_items = (stats.total_items or 0) + (items_collected or 0)

        if duration_seconds is not None:
            total_duration = (stats.avg_duration_seconds or 0) * previous_runs
            total_duration += duration_seconds
            stats.avg_duration_seconds = total_duration / stats.total_runs

        stats.last_run_at = ScraperStats.now()
        self.upsert_stats(stats)

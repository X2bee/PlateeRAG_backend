"""Pydantic schemas for data scraper API surfaces."""
from __future__ import annotations

from typing import Any, Dict, Optional, List
from datetime import datetime

from pydantic import BaseModel, Field, ConfigDict


class ScraperConfigBase(BaseModel):
    name: str = Field(..., max_length=200)
    endpoint: str
    data_source_type: str = Field(default="web", max_length=50)
    parsing_method: str = Field(default="html", max_length=30)
    schedule_interval_minutes: Optional[int] = Field(default=None, ge=1)
    max_depth: Optional[int] = Field(default=None, ge=0)
    follow_links: bool = False
    respect_robots: bool = True
    user_agent: Optional[str] = None
    headers: Optional[Dict[str, Any]] = None
    authentication: Optional[Dict[str, Any]] = None
    filters: Optional[Dict[str, Any]] = None
    status: Optional[str] = Field(default="idle", max_length=30)
    project_id: Optional[int] = None


class ScraperConfigCreate(ScraperConfigBase):
    created_by: Optional[int] = None


class ScraperConfigUpdate(BaseModel):
    name: Optional[str] = Field(default=None, max_length=200)
    endpoint: Optional[str] = None
    data_source_type: Optional[str] = Field(default=None, max_length=50)
    parsing_method: Optional[str] = Field(default=None, max_length=30)
    schedule_interval_minutes: Optional[int] = Field(default=None, ge=1)
    max_depth: Optional[int] = Field(default=None, ge=0)
    follow_links: Optional[bool] = None
    respect_robots: Optional[bool] = None
    user_agent: Optional[str] = None
    headers: Optional[Dict[str, Any]] = None
    authentication: Optional[Dict[str, Any]] = None
    filters: Optional[Dict[str, Any]] = None
    status: Optional[str] = Field(default=None, max_length=30)
    project_id: Optional[int] = None


class ScraperRunRequest(BaseModel):
    trigger: str = Field(default="manual", pattern="^(manual|schedule|test)$")
    triggered_by: Optional[int] = None


class ScraperRunResponse(BaseModel):
    run_id: int
    run_uid: str
    status: str
    trigger_type: str
    scraper_id: int
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    items_collected: int
    items_failed: int


class ScraperFilterParams(BaseModel):
    scraper_id: Optional[int] = None
    status: Optional[str] = None
    trigger_type: Optional[str] = None
    project_id: Optional[int] = None


class ScraperConfigFilterParams(BaseModel):
    status: Optional[str] = None
    project_id: Optional[int] = None
    data_source_type: Optional[str] = None
    scraper_uid: Optional[str] = None


class ScrapedItemFilterParams(BaseModel):
    scraper_id: Optional[int] = None
    run_id: Optional[int] = None
    content_type: Optional[str] = None
    parsing_method: Optional[str] = None
    project_id: Optional[int] = None


class Pagination(BaseModel):
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0)
    order_by: Optional[str] = None
    ascending: bool = False


class RobotsCheckRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    endpoint: str
    user_agent: Optional[str] = Field(default=None, alias="userAgent")


class ScraperConfigResponse(ScraperConfigBase):
    id: int
    scraper_uid: str
    last_run_at: Optional[datetime]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]

    class Config:
        orm_mode = True


class ScraperRunDetailResponse(BaseModel):
    id: int
    scraper_id: int
    run_uid: Optional[str]
    trigger_type: str
    status: str
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    duration_seconds: Optional[int]
    items_collected: int
    items_failed: int
    config_snapshot: Optional[Dict[str, Any]]
    error_log: Optional[Dict[str, Any]]
    project_id: Optional[int]
    triggered_by: Optional[int]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]


class ScrapedItemResponse(BaseModel):
    id: int
    scraper_id: int
    run_id: int
    source_url: Optional[str]
    final_url: Optional[str]
    title: Optional[str]
    content_type: Optional[str]
    raw_path: Optional[str]
    clean_path: Optional[str]
    preview_text: Optional[str]
    metadata: Optional[Dict[str, Any]]
    parsing_method: Optional[str]
    size_bytes: Optional[int]
    collected_at: Optional[datetime]
    checksum: Optional[str]
    project_id: Optional[int]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]


class ScraperStatsResponse(BaseModel):
    scraper_id: int
    total_runs: int
    successful_runs: int
    failed_runs: int
    total_items: int
    total_size_bytes: int
    avg_duration_seconds: Optional[float]
    last_run_at: Optional[datetime]

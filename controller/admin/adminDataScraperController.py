"""
Admin data scraper controller

Provides CRUD, execution, and data lake APIs backed by the real scraper service.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Body, HTTPException, Query, Request

from controller.admin.adminBaseController import validate_superuser
from controller.admin.adminHelper import manager_section_access
from controller.helper.singletonHelper import get_db_manager
from service.database.logger_helper import create_logger
from service.data_scraper.service import ScraperService
from service.data_scraper.schemas import (
    ScraperConfigCreate,
    ScraperConfigUpdate,
    ScraperConfigFilterParams,
    ScraperRunRequest,
    ScraperFilterParams,
    ScrapedItemFilterParams,
    RobotsCheckRequest,
    Pagination,
)

router = APIRouter(prefix="/data-scraper", tags=["Admin Data Scraper"])
SECTION_NAME = ["data-scraper"]


async def _get_service_context(request: Request):
    """Validate admin access and prepare service dependencies."""
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(status_code=403, detail="Admin privileges required")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, val_superuser.get("user_id"), request)

    if not manager_section_access(app_db, val_superuser.get("user_id"), SECTION_NAME):
        backend_log.warn(
            "User lacks data scraper permissions",
            metadata={"user_id": val_superuser.get("user_id")},
        )
        raise HTTPException(status_code=403, detail="Data scraper access required")

    scheduler = getattr(request.app.state, "data_scraper_scheduler", None)
    service = ScraperService(app_db, scheduler=scheduler)
    return val_superuser, backend_log, service


@router.get("/scrapers")
async def list_scrapers(
    request: Request,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    page: Optional[int] = Query(None, ge=1),
    page_size: Optional[int] = Query(None, ge=1, le=200),
    status: Optional[str] = Query(None, description="Filter by scraper status"),
    project_id: Optional[int] = Query(None, description="Filter by project/tenant"),
    data_source_type: Optional[str] = Query(None, description="Filter by source type"),
    scraper_uid: Optional[str] = Query(None, description="Filter by scraper UID"),
):
    _, backend_log, service = await _get_service_context(request)
    backend_log.info(
        "Listing scraper configurations",
        metadata={
            "limit": limit,
            "offset": offset,
            "status": status,
            "project_id": project_id,
            "data_source_type": data_source_type,
            "scraper_uid": scraper_uid,
        },
    )
    if page is not None:
        size = page_size or limit
        page = max(page, 1)
        limit = size
        offset = (page - 1) * size

    filters = ScraperConfigFilterParams(
        status=status,
        project_id=project_id,
        data_source_type=data_source_type,
        scraper_uid=scraper_uid,
    )
    pagination = Pagination(limit=limit, offset=offset, order_by="updated_at", ascending=False)
    try:
        result = service.list_scrapers(filters=filters, pagination=pagination)
        backend_log.success(
            "Listed scraper configurations",
            metadata={"count": result.get("count", 0)},
        )
        return result
    except Exception as exc:
        backend_log.error("Failed to list scrapers", exception=exc)
        raise HTTPException(status_code=500, detail="Failed to list scrapers") from exc


@router.post("/scrapers")
async def create_scraper(request: Request, payload: ScraperConfigCreate):
    val_superuser, backend_log, service = await _get_service_context(request)
    backend_log.info(
        "Creating scraper configuration", metadata=payload.model_dump(exclude_none=True)
    )
    try:
        result = service.create_scraper(payload, created_by=val_superuser.get("user_id"))
        backend_log.success(
            "Created scraper configuration",
            metadata={"scraper_id": result.get("id"), "scraper_uid": result.get("scraper_uid")},
        )
        return result
    except Exception as exc:
        backend_log.error("Failed to create scraper", exception=exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/scrapers/{scraper_id}")
async def get_scraper(request: Request, scraper_id: int):
    _, backend_log, service = await _get_service_context(request)
    backend_log.info("Fetching scraper configuration", metadata={"scraper_id": scraper_id})
    try:
        result = service.get_scraper(scraper_id)
        backend_log.success(
            "Fetched scraper configuration",
            metadata={"scraper_id": scraper_id},
        )
        return result
    except ValueError as exc:
        backend_log.warn("Scraper not found", metadata={"scraper_id": scraper_id})
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        backend_log.error("Failed to fetch scraper", exception=exc)
        raise HTTPException(status_code=500, detail="Failed to fetch scraper") from exc


@router.patch("/scrapers/{scraper_id}")
async def update_scraper(request: Request, scraper_id: int, payload: ScraperConfigUpdate):
    _, backend_log, service = await _get_service_context(request)
    backend_log.info(
        "Updating scraper configuration",
        metadata={"scraper_id": scraper_id, **payload.model_dump(exclude_none=True)},
    )
    try:
        result = service.update_scraper(scraper_id, payload)
        backend_log.success(
            "Updated scraper configuration",
            metadata={"scraper_id": scraper_id},
        )
        return result
    except ValueError as exc:
        backend_log.warn("Scraper update target missing", metadata={"scraper_id": scraper_id})
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        backend_log.error("Failed to update scraper", exception=exc)
        raise HTTPException(status_code=500, detail="Failed to update scraper") from exc


@router.delete("/scrapers/{scraper_id}")
async def delete_scraper(request: Request, scraper_id: int):
    _, backend_log, service = await _get_service_context(request)
    backend_log.info("Deleting scraper configuration", metadata={"scraper_id": scraper_id})
    try:
        service.delete_scraper(scraper_id)
        backend_log.success(
            "Deleted scraper configuration",
            metadata={"scraper_id": scraper_id},
        )
        return {"result": "success"}
    except ValueError as exc:
        backend_log.warn("Scraper deletion target missing", metadata={"scraper_id": scraper_id})
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        backend_log.error("Failed to delete scraper", exception=exc)
        raise HTTPException(status_code=500, detail="Failed to delete scraper") from exc


@router.post("/scrapers/{scraper_id}/run")
async def trigger_run(
    request: Request,
    scraper_id: int,
    payload: ScraperRunRequest = Body(default=None),
):
    _, backend_log, service = await _get_service_context(request)
    backend_log.info(
        "Triggering scraper run",
        metadata={
            "scraper_id": scraper_id,
            **(payload.model_dump(exclude_none=True) if payload else {}),
        },
    )
    try:
        result = await service.request_run(scraper_id, payload)
        backend_log.success(
            "Scraper run requested",
            metadata={"scraper_id": scraper_id, "run_id": result.get("id")},
        )
        return result
    except ValueError as exc:
        backend_log.warn("Failed to trigger run - scraper missing", metadata={"scraper_id": scraper_id})
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        backend_log.error("Failed to trigger run", exception=exc)
        raise HTTPException(status_code=500, detail="Failed to trigger run") from exc


@router.post("/scrapers/{scraper_id}/test")
async def test_scraper(request: Request, scraper_id: int):
    _, backend_log, service = await _get_service_context(request)
    backend_log.info("Test run requested", metadata={"scraper_id": scraper_id})
    try:
        result = service.test_scraper(scraper_id)
        backend_log.success("Test run queued", metadata={"scraper_id": scraper_id})
        return result
    except ValueError as exc:
        backend_log.warn("Test run target missing", metadata={"scraper_id": scraper_id})
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        backend_log.error("Failed to queue test run", exception=exc)
        raise HTTPException(status_code=500, detail="Failed to queue test run") from exc


@router.post("/scrapers/robots-check")
async def robots_check(request: Request, payload: RobotsCheckRequest):
    _, backend_log, service = await _get_service_context(request)
    metadata = payload.model_dump(exclude_none=True, by_alias=False)
    backend_log.info(
        "Performing robots.txt check",
        metadata={"endpoint": payload.endpoint, "user_agent": metadata.get("user_agent")},
    )
    try:
        result = service.check_robots(payload.endpoint, payload.user_agent)
        backend_log.success("Robots.txt check completed", metadata={"endpoint": payload.endpoint})
        return result
    except Exception as exc:
        backend_log.error("Failed robots.txt check", exception=exc)
        raise HTTPException(status_code=500, detail="Failed to perform robots check") from exc


@router.get("/runs")
async def list_runs(
    request: Request,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    scraper_id: Optional[int] = Query(None),
    status: Optional[str] = Query(None),
    trigger_type: Optional[str] = Query(None),
    project_id: Optional[int] = Query(None),
):
    _, backend_log, service = await _get_service_context(request)
    backend_log.info(
        "Listing scraper runs",
        metadata={
            "limit": limit,
            "offset": offset,
            "scraper_id": scraper_id,
            "status": status,
            "trigger_type": trigger_type,
            "project_id": project_id,
        },
    )
    filters = ScraperFilterParams(
        scraper_id=scraper_id,
        status=status,
        trigger_type=trigger_type,
        project_id=project_id,
    )
    pagination = Pagination(limit=limit, offset=offset, order_by="started_at", ascending=False)
    try:
        result = service.list_runs(filters=filters, pagination=pagination)
        backend_log.success("Listed scraper runs", metadata={"count": result.get("count", 0)})
        return result
    except Exception as exc:
        backend_log.error("Failed to list runs", exception=exc)
        raise HTTPException(status_code=500, detail="Failed to list runs") from exc


@router.get("/runs/{run_id}")
async def get_run(request: Request, run_id: int):
    _, backend_log, service = await _get_service_context(request)
    backend_log.info("Fetching scraper run", metadata={"run_id": run_id})
    try:
        result = service.get_run(run_id)
        backend_log.success("Fetched scraper run", metadata={"run_id": run_id})
        return result
    except ValueError as exc:
        backend_log.warn("Run not found", metadata={"run_id": run_id})
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        backend_log.error("Failed to fetch run", exception=exc)
        raise HTTPException(status_code=500, detail="Failed to fetch run") from exc


@router.get("/datalake/items")
async def list_datalake_items(
    request: Request,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    scraper_id: Optional[int] = Query(None),
    run_id: Optional[int] = Query(None),
    content_type: Optional[str] = Query(None),
    parsing_method: Optional[str] = Query(None),
    project_id: Optional[int] = Query(None),
):
    _, backend_log, service = await _get_service_context(request)
    backend_log.info(
        "Listing scraped items",
        metadata={
            "limit": limit,
            "offset": offset,
            "scraper_id": scraper_id,
            "run_id": run_id,
            "content_type": content_type,
            "parsing_method": parsing_method,
            "project_id": project_id,
        },
    )
    filters = ScrapedItemFilterParams(
        scraper_id=scraper_id,
        run_id=run_id,
        content_type=content_type,
        parsing_method=parsing_method,
        project_id=project_id,
    )
    pagination = Pagination(limit=limit, offset=offset, order_by="collected_at", ascending=False)
    try:
        result = service.list_items(filters=filters, pagination=pagination)
        backend_log.success("Listed scraped items", metadata={"count": result.get("count", 0)})
        return result
    except Exception as exc:
        backend_log.error("Failed to list scraped items", exception=exc)
        raise HTTPException(status_code=500, detail="Failed to list scraped items") from exc


@router.get("/stats/{scraper_id}")
async def get_scraper_stats(request: Request, scraper_id: int):
    _, backend_log, service = await _get_service_context(request)
    backend_log.info("Fetching scraper stats", metadata={"scraper_id": scraper_id})
    try:
        stats = service.get_stats(scraper_id)
        if not stats:
            backend_log.info("No stats yet for scraper", metadata={"scraper_id": scraper_id})
            return {"scraper_id": scraper_id, "stats": None}
        backend_log.success("Fetched scraper stats", metadata={"scraper_id": scraper_id})
        return stats
    except Exception as exc:
        backend_log.error("Failed to fetch scraper stats", exception=exc)
        raise HTTPException(status_code=500, detail="Failed to fetch scraper stats") from exc


@router.get("/stats/summary")
async def get_scraper_summary(request: Request):
    _, backend_log, service = await _get_service_context(request)
    backend_log.info("Fetching scraper summary metrics")
    try:
        scrapers = service.list_scrapers(pagination=Pagination(limit=500, offset=0))
        items = scrapers.get("items", [])

        total_scrapers = len(items)
        active_scrapers = len([s for s in items if s.get("status") not in {"paused", "idle"}])
        total_items = 0
        total_runs = 0

        for scraper in items:
            stats = service.get_stats(scraper["id"])
            if not stats:
                continue
            total_items += stats.get("total_items", 0) or 0
            total_runs += stats.get("total_runs", 0) or 0

        summary = {
            "total_scrapers": total_scrapers,
            "active_scrapers": active_scrapers,
            "total_items": total_items,
            "total_runs": total_runs,
        }
        backend_log.success("Fetched scraper summary", metadata=summary)
        return summary
    except Exception as exc:
        backend_log.error("Failed to fetch scraper summary", exception=exc)
        raise HTTPException(status_code=500, detail="Failed to fetch scraper summary") from exc


@router.get("/datalake/stats")
async def get_datalake_stats(request: Request):
    """Compatibility endpoint for existing frontend path."""
    return await get_scraper_summary(request)

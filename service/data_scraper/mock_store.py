"""In-memory mock store mirroring the frontend mock scraper data."""
from __future__ import annotations

import random
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from service.data_scraper.robots import check_robots as fetch_robots


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MockScraperStore:
    """Holds mock scraper configurations, stats, and data lake items."""

    def __init__(self) -> None:
        self.scrapers: Dict[str, Dict] = {
            "scraper-001": {
                "id": "scraper-001",
                "name": "Tech News Aggregator",
                "endpoint": "https://news.ycombinator.com",
                "dataSourceType": "web",
                "parsingMethod": "html",
                "interval": 60,
                "maxDepth": 2,
                "followLinks": True,
                "respectRobotsTxt": True,
                "userAgent": "xgen bot/1.0",
                "createdAt": "2025-10-01T08:00:00Z",
                "updatedAt": "2025-10-13T10:30:00Z",
            },
            "scraper-002": {
                "id": "scraper-002",
                "name": "Product API Collector",
                "endpoint": "https://api.example.com/products",
                "dataSourceType": "api",
                "parsingMethod": "json",
                "interval": 30,
                "respectRobotsTxt": False,
                "authentication": {
                    "type": "api-key",
                    "credentials": {"apiKey": "***********"},
                },
                "createdAt": "2025-09-15T14:20:00Z",
                "updatedAt": "2025-10-12T16:45:00Z",
            },
            "scraper-003": {
                "id": "scraper-003",
                "name": "Research Papers Database",
                "endpoint": "postgres://localhost:5432/papers",
                "dataSourceType": "database",
                "parsingMethod": "json",
                "interval": 1440,
                "respectRobotsTxt": False,
                "createdAt": "2025-08-20T11:00:00Z",
                "updatedAt": "2025-10-10T09:15:00Z",
            },
            "scraper-004": {
                "id": "scraper-004",
                "name": "Customer Email Monitor",
                "endpoint": "imap://mail.example.com",
                "dataSourceType": "email",
                "parsingMethod": "html",
                "interval": 15,
                "respectRobotsTxt": False,
                "authentication": {
                    "type": "basic",
                    "credentials": {
                        "username": "monitor@example.com",
                        "password": "***********",
                    },
                },
                "filters": {
                    "includePatterns": ["support@*", "feedback@*"],
                    "contentType": ["text/html", "text/plain"],
                },
                "createdAt": "2025-09-01T07:30:00Z",
                "updatedAt": "2025-10-13T08:00:00Z",
            },
            "scraper-005": {
                "id": "scraper-005",
                "name": "Document Repository",
                "endpoint": "file:///data/documents",
                "dataSourceType": "document",
                "parsingMethod": "raw",
                "interval": 120,
                "respectRobotsTxt": False,
                "filters": {
                    "includePatterns": ["*.pdf", "*.docx", "*.txt"],
                    "excludePatterns": ["*_draft*", "*_temp*"],
                },
                "createdAt": "2025-07-10T13:45:00Z",
                "updatedAt": "2025-10-11T15:20:00Z",
            },
        }

        self.stats: Dict[str, Dict] = {
            "scraper-001": {
                "scraperId": "scraper-001",
                "totalRuns": 487,
                "successfulRuns": 465,
                "failedRuns": 22,
                "totalDataCollected": 12458,
                "totalDataSize": 256_789_123,
                "lastRunAt": "2025-10-13T09:45:00Z",
                "averageRunTime": 45.3,
            },
            "scraper-002": {
                "scraperId": "scraper-002",
                "totalRuns": 976,
                "successfulRuns": 968,
                "failedRuns": 8,
                "totalDataCollected": 45_621,
                "totalDataSize": 89_234_567,
                "lastRunAt": "2025-10-13T10:15:00Z",
                "averageRunTime": 12.7,
            },
            "scraper-003": {
                "scraperId": "scraper-003",
                "totalRuns": 243,
                "successfulRuns": 241,
                "failedRuns": 2,
                "totalDataCollected": 8_934,
                "totalDataSize": 567_890_123,
                "lastRunAt": "2025-10-12T22:00:00Z",
                "averageRunTime": 234.8,
            },
            "scraper-004": {
                "scraperId": "scraper-004",
                "totalRuns": 1_952,
                "successfulRuns": 1_920,
                "failedRuns": 32,
                "totalDataCollected": 3_456,
                "totalDataSize": 23_456_789,
                "lastRunAt": "2025-10-13T10:30:00Z",
                "averageRunTime": 8.4,
            },
            "scraper-005": {
                "scraperId": "scraper-005",
                "totalRuns": 195,
                "successfulRuns": 189,
                "failedRuns": 6,
                "totalDataCollected": 2_341,
                "totalDataSize": 1_234_567_890,
                "lastRunAt": "2025-10-13T06:00:00Z",
                "averageRunTime": 156.2,
            },
        }

        self.scraped_items: List[Dict] = [
            {
                "id": "data-001",
                "scraperId": "scraper-001",
                "url": "https://news.ycombinator.com/item?id=12345",
                "title": "Show HN: New AI Framework for RAG Systems",
                "content": {
                    "headline": "Show HN: New AI Framework for RAG Systems",
                    "author": "techuser123",
                    "points": 342,
                    "comments": 78,
                    "text": "We built a new framework that makes it easier to implement RAG systems...",
                },
                "contentType": "text/html",
                "size": 15234,
                "metadata": {
                    "domain": "news.ycombinator.com",
                    "publishedDate": "2025-10-13T08:30:00Z",
                },
                "parsingMethod": "html",
                "collectedAt": "2025-10-13T09:45:00Z",
                "tags": ["technology", "ai", "rag"],
            },
            {
                "id": "data-002",
                "scraperId": "scraper-002",
                "url": "https://api.example.com/products/prod-123",
                "title": "Product: Premium Widget Pro",
                "content": {
                    "id": "prod-123",
                    "name": "Premium Widget Pro",
                    "category": "Electronics",
                    "price": 299.99,
                    "inStock": True,
                    "description": "High-quality widget with advanced features",
                    "specs": {
                        "weight": "500g",
                        "dimensions": "10x15x3cm",
                        "color": "black",
                    },
                },
                "contentType": "application/json",
                "size": 2345,
                "metadata": {
                    "apiVersion": "v2",
                    "responseTime": 145,
                },
                "parsingMethod": "json",
                "collectedAt": "2025-10-13T10:15:00Z",
                "tags": ["product", "electronics"],
            },
            {
                "id": "data-003",
                "scraperId": "scraper-003",
                "title": "Research Paper: Deep Learning in Healthcare",
                "content": {
                    "title": "Deep Learning in Healthcare: A Comprehensive Review",
                    "authors": ["Dr. Jane Smith", "Dr. John Doe"],
                    "abstract": "This paper reviews the application of deep learning techniques in healthcare...",
                    "keywords": ["deep learning", "healthcare", "medical imaging", "diagnosis"],
                    "year": 2025,
                    "citations": 42,
                },
                "contentType": "application/json",
                "size": 45678,
                "metadata": {
                    "database": "papers_db",
                    "table": "research_papers",
                },
                "parsingMethod": "json",
                "collectedAt": "2025-10-12T22:00:00Z",
                "tags": ["research", "healthcare", "ai"],
            },
            {
                "id": "data-004",
                "scraperId": "scraper-004",
                "title": "Customer Support Email",
                "content": {
                    "from": "customer@example.com",
                    "subject": "Issue with product delivery",
                    "body": "Hello, I ordered product #12345 but have not received it yet...",
                    "receivedAt": "2025-10-13T10:15:00Z",
                    "attachments": [],
                },
                "contentType": "text/plain",
                "size": 1234,
                "metadata": {
                    "mailbox": "support",
                    "priority": "normal",
                    "labels": ["customer-support", "delivery"],
                },
                "parsingMethod": "html",
                "collectedAt": "2025-10-13T10:30:00Z",
                "tags": ["email", "support", "delivery"],
            },
            {
                "id": "data-005",
                "scraperId": "scraper-005",
                "title": "Technical Documentation.pdf",
                "content": "Binary content of PDF file...",
                "contentType": "application/pdf",
                "size": 2_345_678,
                "metadata": {
                    "fileName": "technical_docs_v2.1.pdf",
                    "fileSize": 2_345_678,
                    "createdDate": "2025-10-10T14:30:00Z",
                    "modifiedDate": "2025-10-11T09:15:00Z",
                    "pages": 125,
                },
                "parsingMethod": "raw",
                "collectedAt": "2025-10-11T15:20:00Z",
                "tags": ["document", "technical", "pdf"],
            },
        ]

        self.run_logs: List[Dict] = [
            {
                "id": "run-001",
                "scraperId": "scraper-001",
                "status": "completed",
                "startedAt": "2025-10-13T09:45:00Z",
                "completedAt": "2025-10-13T09:45:45Z",
                "duration": 45,
                "itemsCollected": 23,
                "warnings": ["Rate limit almost reached"],
            },
            {
                "id": "run-002",
                "scraperId": "scraper-002",
                "status": "completed",
                "startedAt": "2025-10-13T10:15:00Z",
                "completedAt": "2025-10-13T10:15:13Z",
                "duration": 13,
                "itemsCollected": 45,
            },
            {
                "id": "run-003",
                "scraperId": "scraper-001",
                "status": "error",
                "startedAt": "2025-10-13T08:45:00Z",
                "completedAt": "2025-10-13T08:45:15Z",
                "duration": 15,
                "itemsCollected": 0,
                "errors": ["Connection timeout", "Failed to fetch robots.txt"],
            },
            {
                "id": "run-004",
                "scraperId": "scraper-003",
                "status": "completed",
                "startedAt": "2025-10-12T22:00:00Z",
                "completedAt": "2025-10-12T22:03:55Z",
                "duration": 235,
                "itemsCollected": 156,
            },
            {
                "id": "run-005",
                "scraperId": "scraper-004",
                "status": "running",
                "startedAt": "2025-10-13T10:30:00Z",
                "itemsCollected": 5,
            },
        ]

        self.robots_results: Dict[str, Dict] = {
            "https://news.ycombinator.com": {
                "allowed": True,
                "userAgent": "xgen bot/1.0",
                "crawlDelay": 1,
                "message": "이 사이트는 크롤링을 허용합니다. 1초의 크롤 지연이 권장됩니다.",
            },
            "https://www.reddit.com": {
                "allowed": False,
                "userAgent": "xgen bot/1.0",
                "disallowedPaths": ["/api/", "/user/"],
                "message": "이 사이트는 특정 경로에 대한 크롤링을 제한합니다.",
            },
            "https://api.example.com": {
                "allowed": True,
                "userAgent": "xgen bot/1.0",
                "message": "API 엔드포인트는 robots.txt 제한이 없습니다.",
            },
            "https://www.github.com": {
                "allowed": True,
                "userAgent": "xgen bot/1.0",
                "crawlDelay": 2,
                "disallowedPaths": ["/search", "/api/graphql"],
                "message": "이 사이트는 대부분의 페이지에 대해 크롤링을 허용합니다. 2초의 크롤 지연이 권장됩니다.",
            },
            "https://forbidden-site.com": {
                "allowed": False,
                "userAgent": "xgen bot/1.0",
                "disallowedPaths": ["/"],
                "message": "이 사이트는 모든 크롤링을 금지합니다.",
            },
        }

        self.parsed_results: Dict[str, Dict] = {
            "json-success": {
                "success": True,
                "data": {
                    "users": [
                        {"id": 1, "name": "John Doe", "email": "john@example.com"},
                        {"id": 2, "name": "Jane Smith", "email": "jane@example.com"},
                    ],
                    "total": 2,
                },
                "method": "json",
                "originalSize": 1234,
                "parsedSize": 1156,
                "parseTime": 5,
            },
            "html-success": {
                "success": True,
                "data": {
                    "title": "Example Page",
                    "headings": ["Introduction", "Features", "Conclusion"],
                    "paragraphs": 15,
                    "links": 42,
                    "images": 8,
                },
                "method": "html",
                "originalSize": 45678,
                "parsedSize": 3456,
                "parseTime": 23,
            },
            "json-error": {
                "success": False,
                "error": "Invalid JSON syntax at line 15: Unexpected token }",
                "method": "json",
                "originalSize": 2345,
                "parsedSize": 0,
                "parseTime": 2,
            },
            "html-error": {
                "success": False,
                "error": "Failed to parse HTML: Malformed document structure",
                "method": "html",
                "originalSize": 12345,
                "parsedSize": 0,
                "parseTime": 8,
            },
        }

    # ------------------------------------------------------------------
    # Scraper operations
    # ------------------------------------------------------------------
    @staticmethod
    def _get_value(payload: Dict, *keys: str, default=None):
        for key in keys:
            if key in payload and payload[key] is not None:
                return payload[key]
        return default

    def list_scrapers(self) -> List[Dict]:
        return [deepcopy(scraper) for scraper in self.scrapers.values()]

    def get_scraper(self, scraper_id: str) -> Dict:
        scraper = self.scrapers.get(scraper_id)
        if not scraper:
            raise KeyError(f"Scraper {scraper_id} not found")
        return deepcopy(scraper)

    def create_scraper(self, payload: Dict) -> Dict:
        scraper_id = f"scraper-{uuid.uuid4().hex[:8]}"
        now = _utc_now_iso()
        scraper = {
            "id": scraper_id,
            "name": self._get_value(payload, "name", default="Untitled Scraper"),
            "endpoint": self._get_value(payload, "endpoint", default=""),
            "dataSourceType": self._get_value(
                payload, "dataSourceType", "data_source_type", default="web"
            ),
            "parsingMethod": self._get_value(
                payload, "parsingMethod", "parsing_method", default="html"
            ),
            "interval": self._get_value(
                payload, "interval", "schedule_interval_minutes", default=60
            ),
            "maxDepth": self._get_value(payload, "maxDepth", "max_depth", default=2),
            "followLinks": self._get_value(payload, "followLinks", "follow_links", default=False),
            "respectRobotsTxt": self._get_value(
                payload, "respectRobotsTxt", "respect_robots_txt", "respect_robots", default=True
            ),
            "userAgent": self._get_value(payload, "userAgent", "user_agent", default="xgen bot/1.0"),
            "headers": self._get_value(payload, "headers"),
            "authentication": self._get_value(payload, "authentication"),
            "filters": self._get_value(payload, "filters"),
            "createdAt": now,
            "updatedAt": now,
        }
        self.scrapers[scraper_id] = scraper
        self.stats[scraper_id] = {
            "scraperId": scraper_id,
            "totalRuns": 0,
            "successfulRuns": 0,
            "failedRuns": 0,
            "totalDataCollected": 0,
            "totalDataSize": 0,
            "lastRunAt": None,
            "averageRunTime": 0,
        }
        return deepcopy(scraper)

    def update_scraper(self, scraper_id: str, payload: Dict) -> Dict:
        scraper = self.scrapers.get(scraper_id)
        if not scraper:
            raise KeyError(f"Scraper {scraper_id} not found")
        updates: Dict[str, Any] = {}
        mapping = {
            "name": ["name"],
            "endpoint": ["endpoint"],
            "dataSourceType": ["dataSourceType", "data_source_type"],
            "parsingMethod": ["parsingMethod", "parsing_method"],
            "interval": ["interval", "schedule_interval_minutes"],
            "maxDepth": ["maxDepth", "max_depth"],
            "followLinks": ["followLinks", "follow_links"],
            "respectRobotsTxt": ["respectRobotsTxt", "respect_robots_txt", "respect_robots"],
            "userAgent": ["userAgent", "user_agent"],
            "headers": ["headers"],
            "authentication": ["authentication"],
            "filters": ["filters"],
        }
        for target_key, source_keys in mapping.items():
            value = self._get_value(payload, *source_keys)
            if value is not None:
                updates[target_key] = value
        scraper.update(updates)
        scraper["updatedAt"] = _utc_now_iso()
        return deepcopy(scraper)

    def delete_scraper(self, scraper_id: str) -> None:
        if scraper_id not in self.scrapers:
            raise KeyError(f"Scraper {scraper_id} not found")
        self.scrapers.pop(scraper_id, None)
        self.stats.pop(scraper_id, None)
        self.scraped_items = [item for item in self.scraped_items if item["scraperId"] != scraper_id]
        self.run_logs = [run for run in self.run_logs if run["scraperId"] != scraper_id]

    # ------------------------------------------------------------------
    # Run & stats operations
    # ------------------------------------------------------------------
    def run_scraper(self, scraper_id: str) -> Dict:
        if scraper_id not in self.scrapers:
            raise KeyError(f"Scraper {scraper_id} not found")
        run_id = f"run-{uuid.uuid4().hex[:8]}"
        collected = random.randint(1, 50)
        duration = random.randint(5, 120)
        run = {
            "id": run_id,
            "scraperId": scraper_id,
            "status": "completed",
            "startedAt": _utc_now_iso(),
            "completedAt": _utc_now_iso(),
            "duration": duration,
            "itemsCollected": collected,
        }
        self.run_logs.insert(0, run)
        stats = self.stats.setdefault(
            scraper_id,
            {
                "scraperId": scraper_id,
                "totalRuns": 0,
                "successfulRuns": 0,
                "failedRuns": 0,
                "totalDataCollected": 0,
                "totalDataSize": 0,
                "lastRunAt": None,
                "averageRunTime": 0,
            },
        )
        stats["totalRuns"] += 1
        stats["successfulRuns"] += 1
        stats["totalDataCollected"] += collected
        stats["totalDataSize"] += collected * 1024  # placeholder
        stats["lastRunAt"] = run["completedAt"]
        if stats["totalRuns"]:
            stats["averageRunTime"] = (
                (stats["averageRunTime"] * (stats["totalRuns"] - 1)) + duration
            ) / stats["totalRuns"]
        return deepcopy(run)

    def test_scraper(self, scraper_id: str) -> Dict:
        if scraper_id not in self.scrapers:
            raise KeyError(f"Scraper {scraper_id} not found")
        return {
            "scraperId": scraper_id,
            "status": "queued",
            "message": "Test execution scheduled (mock response).",
            "requestedAt": _utc_now_iso(),
        }

    def get_scraper_stats(self, scraper_id: str) -> Optional[Dict]:
        stats = self.stats.get(scraper_id)
        if not stats:
            return None
        return deepcopy(stats)

    def list_runs(
        self,
        scraper_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict]:
        runs = self.run_logs
        if scraper_id:
            runs = [run for run in runs if run["scraperId"] == scraper_id]
        if status:
            runs = [run for run in runs if run.get("status") == status]
        return [deepcopy(run) for run in runs]

    def get_run(self, run_id: str) -> Dict:
        for run in self.run_logs:
            if run["id"] == run_id:
                return deepcopy(run)
        raise KeyError(f"Run {run_id} not found")

    # ------------------------------------------------------------------
    # Robots & parsing
    # ------------------------------------------------------------------
    def robots_check(self, endpoint: str, user_agent: Optional[str]) -> Dict:
        result = fetch_robots(endpoint, user_agent, cache=self.robots_results)
        return result

    def parse_data(self, item_id: str, method: str) -> Dict:
        key = f"{method.lower()}-success"
        if random.random() < 0.2:
            key = f"{method.lower()}-error"
        result = self.parsed_results.get(key)
        if result:
            return deepcopy(result)
        return {
            "success": True,
            "data": {"message": f"Parsed {item_id} with {method}"},
            "method": method,
            "originalSize": 0,
            "parsedSize": 0,
            "parseTime": 0,
        }

    # ------------------------------------------------------------------
    # Data lake operations
    # ------------------------------------------------------------------
    def list_items(self, scraper_id: Optional[str] = None) -> List[Dict]:
        items = self.scraped_items
        if scraper_id:
            items = [item for item in items if item["scraperId"] == scraper_id]
        return [deepcopy(item) for item in items]

    def get_item(self, item_id: str) -> Dict:
        for item in self.scraped_items:
            if item["id"] == item_id:
                return deepcopy(item)
        raise KeyError(f"Scraped item {item_id} not found")

    def get_data_lake_stats(self) -> Dict:
        total_size = sum(item.get("size", 0) for item in self.scraped_items)
        stats = {
            "totalItems": len(self.scraped_items),
            "totalSize": total_size,
            "itemsBySourceType": {},
            "itemsByParsingMethod": {},
            "recentItems": 0,
        }
        now = datetime.now(timezone.utc)
        for item in self.scraped_items:
            scraper = self.scrapers.get(item["scraperId"])
            source_type = scraper.get("dataSourceType") if scraper else "unknown"
            stats["itemsBySourceType"].setdefault(source_type, 0)
            stats["itemsBySourceType"][source_type] += 1

            parsing_method = item.get("parsingMethod", "unknown")
            stats["itemsByParsingMethod"].setdefault(parsing_method, 0)
            stats["itemsByParsingMethod"][parsing_method] += 1

            collected_at = item.get("collectedAt")
            try:
                if collected_at:
                    collected_dt = datetime.fromisoformat(collected_at.replace("Z", "+00:00"))
                    if (now - collected_dt).total_seconds() <= 24 * 3600:
                        stats["recentItems"] += 1
            except ValueError:
                continue
        return stats

    def get_summary(self) -> Dict:
        total_scrapers = len(self.scrapers)
        active_scrapers = sum(
            1 for stats in self.stats.values() if stats.get("lastRunAt")
        )
        total_items = sum(stats.get("totalDataCollected", 0) for stats in self.stats.values())
        total_size = sum(stats.get("totalDataSize", 0) for stats in self.stats.values())
        success_rate = 0.0
        total_runs = sum(stats.get("totalRuns", 0) for stats in self.stats.values())
        total_successful = sum(stats.get("successfulRuns", 0) for stats in self.stats.values())
        if total_runs:
            success_rate = round((total_successful / total_runs) * 100, 1)

        return {
            "totalScrapers": total_scrapers,
            "activeScrapers": active_scrapers,
            "totalDataCollected": total_items,
            "totalDataSize": total_size,
            "successRate": success_rate,
        }

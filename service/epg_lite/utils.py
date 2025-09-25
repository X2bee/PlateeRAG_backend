"""경량화된 EPG 유틸리티 모듈"""
import json
import logging
import os
import re
import threading
import time
from datetime import datetime
from functools import wraps
from math import floor
from pathlib import Path
from typing import Callable
import pytz

from bs4 import BeautifulSoup, FeatureNotFound

log = logging.getLogger("EPG_LITE.UTILS")


def get_cache_file_path(filename: str = "test_channels.json") -> str:
    """
    downloads/epg_{YYYY-MM-DD} 구조로 캐시 파일 경로를 생성합니다. (서울 시간 기준)

    Args:
        filename: 캐시 파일명

    Returns:
        str: 생성된 캐시 파일의 전체 경로
    """
    seoul_tz = pytz.timezone('Asia/Seoul')
    today = datetime.now(seoul_tz).strftime("%Y-%m-%d")

    downloads_path = os.path.join(os.getcwd(), "downloads")
    epg_folder = os.path.join(downloads_path, f"epg_{today}")

    Path(epg_folder).mkdir(parents=True, exist_ok=True)
    cache_file_path = os.path.join(epg_folder, filename)

    log.debug("Cache file path created: %s", cache_file_path)
    return cache_file_path


def dump_json(file_path: str, data: dict) -> int:
    """JSON 데이터를 파일에 저장"""
    with open(file_path, "w", encoding="utf-8") as f:
        txt = json.dumps(data, ensure_ascii=False, indent=2)
        # 채널 리스트 압축 형태로 저장
        txt = re.sub(r",\n\s{8}\"", ', "', txt)
        txt = re.sub(r"\s{6}{\s+(.*)\s+}", r"      { \g<1> }", txt)
        return f.write(txt)


class ParserBeautifulSoup(BeautifulSoup):
    """사용 가능한 첫 번째 파서를 선택하는 BeautifulSoup"""

    def insert_before(self, *args):
        pass

    def insert_after(self, *args):
        pass

    def __init__(self, markup, **kwargs):
        for parser in ["lxml", "html.parser"]:
            try:
                super().__init__(markup, parser, **kwargs)
                return
            except FeatureNotFound:
                pass
        raise FeatureNotFound


class RateLimiter:
    """요청 속도 제한기"""

    try:
        now: Callable = time.monotonic
    except AttributeError:
        now: Callable = time.time

    def __init__(self, tps: float = 1.0):
        """
        Args:
            tps: 초당 요청 횟수 (transactions per second)
        """
        if tps <= 0.0:
            raise ValueError("tps must be positive")

        self.max_calls = 1
        self.period = 1 / tps
        self.last_reset = self.now()
        self.num_calls = 0
        self.lock = threading.RLock()

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.lock:
                period_remaining = self.__period_remaining()

                if period_remaining <= 0:
                    self.num_calls = 0
                    self.last_reset = self.now()

                self.num_calls += 1

                if self.num_calls > self.max_calls:
                    self.last_reset = self.now() + period_remaining
                    time.sleep(period_remaining)
                    return func(*args, **kwargs)
            return func(*args, **kwargs)
        return wrapper

    def __period_remaining(self) -> float:
        elapsed = self.now() - self.last_reset
        return self.period - elapsed


class PrefixLogger(logging.LoggerAdapter):
    """로그 접두어 추가 어댑터"""

    def __init__(self, logger, prefix):
        super().__init__(logger, {})
        self.prefix = prefix

    def process(self, msg, kwargs):
        return f"{self.prefix} {msg}", kwargs

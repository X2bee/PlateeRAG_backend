"""경량화된 EPG 핸들러"""
import json
import logging
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from importlib import import_module
from itertools import chain
from typing import List, Dict, Any

import pytz

from .providers import EPGProvider
from .utils import dump_json

log = logging.getLogger("EPG_LITE.HANDLER")


class EPGHandler:
    """경량화된 EPG 핸들러"""

    def __init__(self, cfgs: dict):
        """
        Args:
            cfgs: EPG 프로바이더 설정 딕셔너리
        """
        self.providers: List[EPGProvider] = self.load_providers(cfgs)

    def load_providers(self, cfgs: dict) -> List[EPGProvider]:
        """프로바이더 로드"""
        providers = []

        for name, cfg in cfgs.items():
            if not cfg.get("ENABLED", False):
                continue

            try:
                # 프로바이더 모듈 동적 로드
                m = import_module(f"service.epg_lite.providers.{name.lower()}")
                provider_class = getattr(m, name.upper())
                providers.append(provider_class(cfg))
            except ModuleNotFoundError:
                log.error("No such provider found: '%s'", name)
                sys.exit(1)
            except Exception as e:
                log.error("Failed to load provider '%s': %s", name, e)

        return providers

    def load_channels(self, channelfile: str, parallel: bool = False) -> None:
        """채널 정보 로드"""
        # 캐시된 채널 정보 로드 시도
        try:
            log.debug("Trying to load cached channels from json")
            with open(channelfile, "r", encoding="utf-8") as fp:
                channeljson = json.load(fp)
        except (json.decoder.JSONDecodeError, ValueError, FileNotFoundError) as e:
            log.debug("Failed to load cached channels from json: %s", e)
            channeljson = {}

        # 프로바이더별 채널 로드
        if parallel:
            with ThreadPoolExecutor() as exe:
                for p in self.providers:
                    exe.submit(p.load_svc_channels, channeljson=channeljson)
        else:
            for p in self.providers:
                p.load_svc_channels(channeljson=channeljson)

        # 업데이트된 채널 정보 저장
        if any(p.was_channel_updated for p in self.providers):
            for p in self.providers:
                seoul_tz = pytz.timezone('Asia/Seoul')
                channeljson[p.provider_name.upper()] = {
                    "UPDATED": datetime.now(seoul_tz).isoformat(),
                    "TOTAL": len(p.svc_channels),
                    "CHANNELS": p.svc_channels,
                }
            dump_json(channelfile, channeljson)
            log.info("Channel file was upgraded. You may check the changes here: %s", channelfile)

    def load_req_channels(self):
        """요청된 채널 로드"""
        for p in self.providers:
            p.load_req_channels()

        # 채널 ID 중복 확인
        log.debug("Checking uniqueness of channelid...")
        cids = [c.id for p in self.providers for c in p.req_channels]
        duplicates = {k: v for k, v in Counter(cids).items() if v > 1}
        if duplicates:
            raise AssertionError(f"채널ID 중복: {duplicates}")

    def get_programs(self, parallel: bool = False):
        """프로그램 정보 가져오기"""
        if parallel:
            with ThreadPoolExecutor() as exe:
                for p in self.providers:
                    exe.submit(p.get_programs)
        else:
            for p in self.providers:
                p.get_programs()

    def to_json(self) -> Dict[str, Any]:
        """EPG 데이터를 JSON 형태로 변환"""
        epg_json = {
            "generator-info": {
                "name": "epg_lite"
            },
            "channels": [],
            "programmes": []
        }

        log.debug("Writing channels to JSON...")
        for p in self.providers:
            epg_json["channels"].extend(p.write_channels_json())

        log.debug("Writing programs to JSON...")
        for p in self.providers:
            epg_json["programmes"].extend(p.write_programs_json())

        return epg_json

    @property
    def all_channels(self):
        """모든 프로바이더의 채널에 접근하는 단축키"""
        return chain.from_iterable(p.req_channels for p in self.providers)

    @property
    def all_programs(self):
        """모든 프로바이더의 프로그램에 접근하는 단축키"""
        return chain.from_iterable(ch.programs for ch in self.all_channels)

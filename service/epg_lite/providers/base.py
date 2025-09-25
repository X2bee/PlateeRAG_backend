"""경량화된 EPG 프로바이더 베이스 클래스"""
import json
import logging
import requests
from collections import Counter
from datetime import datetime
from functools import wraps
from typing import List, Dict, Any

import pytz

from ..models import EPGChannel, EPGProgram
from ..utils import RateLimiter, PrefixLogger

log = logging.getLogger("EPG_LITE.PROVIDER")

# 사용자 에이전트
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"


def no_endtime(func):
    """프로그램 종료 시간이 없는 경우 자동으로 설정하는 데코레이터"""
    @wraps(func)
    def wrapped(self: 'EPGProvider', *args, **kwargs):
        func(self, *args, **kwargs)
        for ch in self.req_channels:
            ch.set_etime()
    return wrapped


class EPGProvider:
    """EPG 프로바이더 베이스 클래스"""

    referer: str = None
    tps: float = 1.0
    was_channel_updated: bool = False

    def __init__(self, cfg: dict):
        self.provider_name = self.__class__.__name__
        self.cfg = cfg

        # HTTP 세션 설정
        self.sess = requests.Session()
        self.sess.headers.update({
            "User-Agent": USER_AGENT,
            "Referer": self.referer
        })

        if http_proxy := cfg.get("HTTP_PROXY"):
            self.sess.proxies.update({"http": http_proxy, "https": http_proxy})

        # 요청 속도 제한
        self.request = RateLimiter(tps=self.tps)(self.__request)

        # 채널 데이터
        self.svc_channels: List[dict] = []
        self.req_channels: List[EPGChannel] = []

    def __request(self, url: str, method: str = "GET", **kwargs) -> Any:
        """HTTP 요청 처리"""
        try:
            r = self.sess.request(method=method, url=url, **kwargs)
            try:
                return r.json()
            except (json.decoder.JSONDecodeError, ValueError):
                return r.text
        except requests.exceptions.HTTPError as e:
            log.error("요청 중 에러: %s", e)
        except Exception:
            log.exception("요청 중 예외:")
        return ""

    def load_svc_channels(self, channeljson: dict = None) -> None:
        """서비스 채널 로드"""
        plog = PrefixLogger(log, f"[{self.provider_name:5s}]")

        # 캐시 확인
        try:
            channelinfo = channeljson[self.provider_name.upper()]
            total = channelinfo["TOTAL"]
            channels = channelinfo["CHANNELS"]
            assert total == len(channels), "TOTAL != len(CHANNELS)"

            updated_at = datetime.fromisoformat(channelinfo["UPDATED"])
            seoul_tz = pytz.timezone('Asia/Seoul')

            if (datetime.now(seoul_tz) - updated_at).total_seconds() <= 3600 * 24 * 4:
                self.svc_channels = channels
                plog.info("%03d service channels loaded from cache", len(channels))
                return
            plog.debug("Updating service channels as outdated...")
        except Exception as e:
            plog.debug("Updating service channels as cache broken: %s", e)

        # 새로운 채널 데이터 가져오기
        try:
            channels = self.get_svc_channels()
        except Exception:
            plog.exception("Exception while retrieving service channels:")
        else:
            self.svc_channels = channels
            self.was_channel_updated = True
            plog.info("%03d service channels successfully fetched from server", len(channels))

    def get_svc_channels(self) -> List[dict]:
        """서비스 채널 목록 가져오기 - 하위 클래스에서 구현"""
        raise NotImplementedError("method 'get_svc_channels' must be implemented")

    def load_req_channels(self) -> None:
        """요청된 채널 로드"""
        plog = PrefixLogger(log, f"[{self.provider_name:5s}]")
        my_channels = self.cfg.get("MY_CHANNELS", [])

        if my_channels == "*":
            plog.debug("Overriding all MY_CHANNELS by service channels...")
            my_channels = self.svc_channels

        if not my_channels:
            return

        req_channels = []
        svc_channels = {x["ServiceId"]: x for x in self.svc_channels}

        for my_no, my_ch in enumerate(my_channels):
            if "ServiceId" not in my_ch:
                plog.warning("'ServiceId' Not Found: %s", my_ch)
                continue

            req_ch = svc_channels.pop(my_ch["ServiceId"], None)
            if req_ch is None:
                plog.warning("'ServiceId' Not in Service: %s", my_ch)
                continue

            # 설정 적용
            for _k, _v in my_ch.items():
                if _v:
                    req_ch[_k] = _v

            req_ch["Source"] = self.provider_name
            req_ch.setdefault("No", str(my_no))

            # ID 생성
            if "Id" not in req_ch:
                try:
                    req_ch["Id"] = eval(f"f'{self.cfg['ID_FORMAT']}'", None, req_ch)
                except Exception:
                    req_ch["Id"] = f'{req_ch["ServiceId"]}.{req_ch["Source"].lower()}'

            # 아이콘 설정
            if not self.cfg.get("ADD_CHANNEL_ICON"):
                req_ch.pop("Icon_url", None)

            req_channels.append(EPGChannel.from_dict(**req_ch))

        plog.info("요청 %3d - 불가 %3d = 최종 %3d",
                 len(my_channels), len(my_channels) - len(req_channels), len(req_channels))
        self.req_channels = req_channels

    def get_programs(self) -> None:
        """프로그램 데이터 가져오기 - 하위 클래스에서 구현"""
        raise NotImplementedError("method 'get_programs' must be implemented")

    def write_channels_json(self) -> List[Dict[str, Any]]:
        """채널 정보를 JSON 형태로 반환"""
        channels = []
        for ch in self.req_channels:
            if not ch.programs:
                log.warning("Skip writing as no program entries found for '%s'", ch.id)
                continue
            channels.append(ch.to_json())
        return channels

    def write_programs_json(self) -> List[Dict[str, Any]]:
        """프로그램 정보를 JSON 형태로 반환"""
        programs = []
        for ch in self.req_channels:
            for prog in ch.programs:
                programs.append(prog.to_json(self.cfg))
        return programs

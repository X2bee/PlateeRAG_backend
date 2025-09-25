"""경량화된 NAVER EPG 프로바이더"""
import logging
from datetime import date, datetime, timedelta
from typing import List
from xml.sax.saxutils import unescape
import pytz

from .base import EPGProvider, no_endtime
from ..models import EPGProgram
from ..utils import ParserBeautifulSoup as BeautifulSoup

log = logging.getLogger("EPG_LITE.NAVER")

# 서울 시간 기준으로 오늘 날짜 가져오기
seoul_tz = pytz.timezone('Asia/Seoul')
today = datetime.now(seoul_tz).date()

# 채널 카테고리
CH_CATE = [
    {"name": "지상파", "u1": "100"},
    {"name": "종합 편성", "u1": "500"},
    {"name": "케이블", "u1": "200"},
    {"name": "스카이라이프", "u1": "300"},
    {"name": "해외위성", "u1": "9000"},
    {"name": "라디오", "u1": "400"},
]


class NAVER(EPGProvider):
    """NAVER EPG 프로바이더"""

    referer = "https://m.search.naver.com/search.naver?where=m&query=%ED%8E%B8%EC%84%B1%ED%91%9C"

    def get_svc_channels(self) -> List[dict]:
        """NAVER 서비스 채널 목록 가져오기"""
        svc_channels = []
        url = "https://m.search.naver.com/p/csearch/content/nqapirender.nhn"
        params = {
            "key": "ScheduleChannelList",
            "where": "nexearch",
            "pkid": "66",
            "u1": "CATEGORY_CODE",
        }

        for c in CH_CATE:
            params.update({"u1": c["u1"]})
            data = self.request(url, params=params)

            if data["statusCode"].lower() != "success":
                log.error("유효한 응답이 아닙니다: %s", data["statusCode"])
                continue

            soup = BeautifulSoup(data["dataHtml"])
            for ch in soup.select('li[class="item"]'):
                try:
                    svcid = ch.select("div > div[data-cid]")[0]["data-cid"]
                    name = str(ch.select('div[class="channel_name"] > a')[0].text)
                    svc_channels.append({
                        "Name": name,
                        "ServiceId": svcid,
                        "Category": c["name"],
                    })
                except Exception:
                    pass

        return svc_channels

    @no_endtime
    def get_programs(self) -> None:
        """NAVER EPG 프로그램 데이터 가져오기"""
        url = "https://m.search.naver.com/p/csearch/content/nqapirender.nhn"
        params = {
            "key": "SingleChannelDailySchedule",
            "where": "m",
            "pkid": "66",
            "u1": "SVCID",
            "u2": "EPGDATE"
        }

        for idx, _ch in enumerate(self.req_channels):
            log.info("%03d/%03d %s", idx + 1, len(self.req_channels), _ch)

            for nd in range(int(self.cfg.get("FETCH_LIMIT", 1))):
                day = today + timedelta(days=nd)
                params.update({"u1": _ch.svcid, "u2": day.strftime("%Y%m%d")})
                data = self.request(url, params=params)

                if data["statusCode"].lower() != "success":
                    log.error("유효한 응답이 아닙니다: %s %s", _ch, data["statusCode"])
                    continue

                try:
                    _epgs = self.__epgs_of_day(_ch.id, data, day)
                except Exception:
                    log.exception("프로그램 파싱 중 예외: %s, %s", _ch, day)
                else:
                    _ch.programs.extend(_epgs)

    def __epgs_of_day(self, channelid: str, data: dict, day: datetime) -> List[EPGProgram]:
        """하루치 EPG 프로그램 파싱"""
        _epgs = []
        soup = BeautifulSoup("".join(data["dataHtml"]))

        for row in soup.find_all("li", {"class": "list"}):
            cell = row.find_all("div")
            _epg = EPGProgram(channelid)
            _epg.title = unescape(cell[4].text.strip())
            _epg.stime = datetime.strptime(f"{str(day)} {cell[1].text.strip()}", "%Y-%m-%d %H:%M")

            for span in cell[3].findAll("span"):
                span_txt = span.text.strip()
                if "ico_age" in span["class"]:
                    _epg.rating = int(span_txt.rstrip("세"))
                elif "re" in span["class"]:
                    _epg.rebroadcast = True
                else:
                    _epg.extras = (_epg.extras or []) + [span_txt]

            try:
                _epg.title_sub = cell[5].text.strip()
            except Exception:
                pass

            _epgs.append(_epg)

        return _epgs

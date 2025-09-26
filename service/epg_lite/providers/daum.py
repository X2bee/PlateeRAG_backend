"""경량화된 DAUM EPG 프로바이더"""
import logging
import re
from datetime import datetime, timedelta
from typing import List
from urllib.parse import quote
import pytz

from .base import EPGProvider, no_endtime
from ..models import EPGProgram
from ..utils import ParserBeautifulSoup as BeautifulSoup

log = logging.getLogger("EPG_LITE.DAUM")

# 채널 카테고리
CH_CATE = ["지상파", "종합편성", "케이블", "스카이라이프", "해외위성", "라디오"]


class DAUM(EPGProvider):
    """DAUM EPG 프로바이더"""

    referer = None

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        # 제목 정규식 패턴 컴파일
        self.title_regex = re.compile(
            r"^(?P<title>.*?)\s?([\<\(]?(?P<part>\d{1})부[\>\)]?)?\s?(<(?P<subname1>.*)>)?\s?((?P<epnum>\d+)회)?\s?(<(?P<subname2>.*)>)?$"
        )

    def get_svc_channels(self) -> List[dict]:
        """DAUM 서비스 채널 목록 가져오기"""
        svc_channels = []
        url = "https://search.daum.net/search?DA=B3T&w=tot&rtmaxcoll=B3T&q={}"
        channelsel1 = '#channelNaviLayer > div[class^="layer_tv layer_all"] ul > li'
        channelsel2 = 'div[class="wrap_sub"] > span > a'

        for c in CH_CATE:
            search_url = url.format(f"{c} 편성표")
            data = self.request(search_url)
            soup = BeautifulSoup(data)

            if not soup.find_all(attrs={"disp-attr": "B3T"}):
                continue

            all_channels = [str(x.text.strip()) for x in soup.select(channelsel1)]
            if not all_channels:
                all_channels += [str(x.text.strip()) for x in soup.select(channelsel2)]

            svc_cate = c.replace("스카이라이프", "SKYLIFE")
            for x in all_channels:
                svc_channels.append({
                    "Name": x,
                    "ServiceId": f"{svc_cate} {x}",
                    "Category": c,
                })

        return svc_channels

    @no_endtime
    def get_programs(self) -> None:
        """DAUM EPG 프로그램 데이터 가져오기"""
        url = "https://search.daum.net/search?DA=B3T&w=tot&rtmaxcoll=B3T&q={}"

        for idx, _ch in enumerate(self.req_channels):
            log.info("%03d/%03d %s", idx + 1, len(self.req_channels), _ch)
            search_url = url.format(quote(_ch.svcid + " 편성표"))
            data = self.request(search_url)

            try:
                _epgs = self.__epgs_of_days(_ch.id, data)
            except AssertionError as e:
                log.warning("%s: %s", e, _ch)
            except Exception:
                log.exception("프로그램 파싱 중 예외: %s", _ch)
            else:
                _ch.programs.extend(_epgs)

    def __epgs_of_days(self, channelid: str, data: str) -> List[EPGProgram]:
        """여러 날짜의 EPG 프로그램 파싱"""
        soup = BeautifulSoup(data)
        assert soup.find_all(attrs={"disp-attr": "B3T"}), "EPG 정보가 없거나 없는 채널입니다"

        days = soup.select('div[class="tbl_head head_type2"] > span > span[class="date"]')

        # 현재 서울 시간 기준으로 올바른 날짜 찾기
        seoul_tz = pytz.timezone('Asia/Seoul')
        currdate = datetime.now(seoul_tz)
        current_date_str = currdate.strftime("%m.%d")

        # 오늘 날짜와 일치하는 인덱스 찾기
        start_day_index = 0
        for idx, day_elem in enumerate(days):
            if day_elem.text.strip() == current_date_str:
                start_day_index = idx
                break

        # 첫 번째 날짜를 기준으로 basedate 설정
        basedate = datetime.strptime(days[0].text.strip(), "%m.%d").replace(year=currdate.year)
        basedate = seoul_tz.localize(basedate)
        if (basedate - currdate).days > 0:
            basedate = basedate.replace(year=basedate.year - 1)

        _epgs = []
        # 현재 날짜부터 데이터 가져오기 (최대 FETCH_LIMIT 일수만큼)
        fetch_limit = int(self.cfg.get("FETCH_LIMIT", 1))
        end_day_index = min(start_day_index + fetch_limit, len(days))

        for nd in range(start_day_index, end_day_index):
            hours = soup.select(f'[id="tvProgramListWrap"] > table > tbody > tr > td:nth-of-type({nd+1})')
            assert len(hours) == 24, f"24개의 시간 행이 있어야 합니다: 현재: {len(hours):d}"

            for nh, hour in enumerate(hours):
                for dl in hour.select("dl"):
                    _epg = EPGProgram(channelid)
                    nm = int(dl.select("dt")[0].text.strip())
                    _epg.stime = basedate + timedelta(days=nd, hours=nh, minutes=nm)

                    for atag in dl.select("dd > a"):
                        _epg.title = atag.text.strip()

                    for span in dl.select("dd > span"):
                        class_val = " ".join(span["class"])
                        if class_val == "":
                            _epg.title = span.text.strip()
                        elif "ico_re" in class_val:
                            _epg.rebroadcast = True
                        elif "ico_rate" in class_val:
                            _epg.rating = int(class_val.split("ico_rate")[1].strip())
                        else:
                            _epg.extras = (_epg.extras or []) + [span.text.strip()]

                    # 제목 정규식 매칭
                    if m := self.title_regex.search(_epg.title):
                        _epg.title = m.group("title")
                        _epg.part_num = m.group("part")
                        _epg.ep_num = m.group("epnum")
                        _epg.title_sub = m.group("subname2") or m.group("subname1")
                        if _epg.part_num:
                            _epg.title += f" {_epg.part_num}부"

                    _epgs.append(_epg)

        return _epgs

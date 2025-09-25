"""경량화된 EPG 데이터 모델"""
import re
from dataclasses import dataclass, fields
from datetime import datetime, timedelta
from typing import List, Dict, Any

# 제목 패턴 정규식
PTN_TITLE = re.compile(r"(.*) \(?(\d+부)\)?")
PTN_SPACES = re.compile(r" {2,}")

# 카테고리 매핑
CAT_KO2EN = {
    "교양": "Arts / Culture (without music)",
    "만화": "Cartoons / Puppets",
    "교육": "Education / Science / Factual topics",
    "취미": "Leisure hobbies",
    "드라마": "Movie / Drama",
    "영화": "Movie / Drama",
    "음악": "Music / Ballet / Dance",
    "뉴스": "News / Current affairs",
    "다큐": "Documentary",
    "라이프": "Documentary",
    "시사/다큐": "Documentary",
    "연예": "Show / Game show",
    "스포츠": "Sports",
    "홈쇼핑": "Advertisement / Shopping",
}


@dataclass
class EPGProgram:
    """EPG 프로그램 데이터 모델"""

    channelid: str
    stime: datetime = None
    etime: datetime = None
    title: str = None
    title_sub: str = None
    part_num: str = None
    ep_num: str = None
    categories: List[str] = None
    rebroadcast: bool = False
    rating: int = 0
    desc: str = None
    poster_url: str = None
    cast: List[dict] = None
    crew: List[dict] = None
    extras: List[str] = None
    keywords: List[str] = None

    def sanitize(self) -> None:
        """데이터 정제"""
        for f in fields(self):
            attr = getattr(self, f.name)
            if f.type == List[str] and attr is not None:
                setattr(self, f.name, [x.strip() for x in filter(bool, attr) if x.strip()])
            elif f.type == str:
                setattr(self, f.name, (attr or "").strip())

    def to_json(self, cfg: dict) -> Dict[str, Any]:
        """EPG 프로그램을 JSON 형태로 변환"""
        self.sanitize()

        # 기본 변수 설정
        stime = self.stime.strftime("%Y%m%d%H%M%S +0900") if self.stime else None
        etime = self.etime.strftime("%Y%m%d%H%M%S +0900") if self.etime else None
        title = self.title
        title_sub = self.title_sub
        cast = self.cast or []
        crew = self.crew or []
        categories = self.categories or []
        keywords = self.keywords or []
        episode = self.ep_num
        rebroadcast = "재" if self.rebroadcast else ""
        rating = "전체 관람가" if self.rating == 0 else f"{self.rating}세 이상 관람가"

        # 제목 처리
        if matches := PTN_TITLE.match(title):
            title = matches.group(1).strip()
            title_sub = (matches.group(2) + " " + title_sub).strip()

        processed_title = [
            title or title_sub or "제목 없음",
            f"({episode}회)" if episode and cfg.get("ADD_EPNUM_TO_TITLE") else "",
            f"({rebroadcast})" if rebroadcast and cfg.get("ADD_REBROADCAST_TO_TITLE") else "",
        ]
        processed_title = PTN_SPACES.sub(" ", " ".join(filter(bool, processed_title)))

        # 설명 생성
        desc_parts = []
        if cfg.get("ADD_DESCRIPTION"):
            desc_parts = [
                processed_title,
                f"부제 : {title_sub}" if title_sub else "",
                f"방송 : {rebroadcast}방송" if rebroadcast else "",
                f"회차 : {episode}회" if episode else "",
                f"장르 : {','.join(categories)}" if categories else "",
                f"출연 : {','.join(x['name'] for x in cast)}" if cast else "",
                f"제작 : {','.join(x['name'] for x in crew)}" if crew else "",
                f"등급 : {rating}",
                self.desc,
            ]
            desc_parts = list(filter(bool, desc_parts))

        # JSON 객체 구성
        program_json = {
            "channel": self.channelid,
            "start": stime,
            "stop": etime,
            "title": {
                "text": processed_title,
                "lang": "ko"
            }
        }

        if title_sub:
            program_json["sub-title"] = {
                "text": title_sub,
                "lang": "ko"
            }

        if desc_parts:
            program_json["desc"] = {
                "text": PTN_SPACES.sub(" ", "\n".join(desc_parts)),
                "lang": "ko"
            }

        if categories:
            program_json["categories"] = []
            for cat_ko in categories:
                program_json["categories"].append({"text": cat_ko, "lang": "ko"})
                if cat_en := CAT_KO2EN.get(cat_ko):
                    program_json["categories"].append({"text": cat_en, "lang": "en"})

        if keywords:
            program_json["keywords"] = [{"text": keyword, "lang": "ko"} for keyword in keywords]

        if self.poster_url:
            program_json["icon"] = {"src": self.poster_url}

        if episode:
            if cfg.get("ADD_XMLTV_NS"):
                try:
                    episode_ns = int(episode) - 1
                except ValueError:
                    episode_ns = int(episode.split(",", 1)[0]) - 1
                episode_ns = f"0.{str(episode_ns)}.0/0"
                program_json["episode-num"] = {"text": episode_ns, "system": "xmltv_ns"}
            else:
                program_json["episode-num"] = {"text": episode, "system": "onscreen"}

        if rebroadcast:
            program_json["previously-shown"] = True

        if rating:
            program_json["rating"] = {
                "system": "KMRB",
                "value": rating
            }

        return program_json


@dataclass
class EPGChannel:
    """EPG 채널 데이터 모델"""

    id: str
    src: str
    svcid: str
    name: str
    icon: str = None
    no: str = None
    category: str = None
    programs: List[EPGProgram] = None

    def __post_init__(self):
        if self.programs is None:
            self.programs = []

    @classmethod
    def from_dict(cls, **kwargs):
        """딕셔너리에서 EPGChannel 객체 생성"""
        channel = cls(
            kwargs["Id"],
            kwargs["Source"],
            kwargs["ServiceId"],
            kwargs["Name"]
        )
        channel.icon = kwargs.get("Icon_url")
        channel.no = kwargs.get("No")
        channel.category = kwargs.get("Category")
        return channel

    def __str__(self):
        return f"{self.name} <{self.id}>"

    def set_etime(self) -> None:
        """프로그램 종료 시간 설정"""
        for ind, prog in enumerate(self.programs):
            if prog.etime:
                continue
            try:
                prog.etime = self.programs[ind + 1].stime
            except IndexError:
                prog.etime = (prog.stime + timedelta(days=1)).replace(hour=0, minute=0, second=0)

    def to_json(self) -> Dict[str, Any]:
        """EPG 채널을 JSON 형태로 변환"""
        channel_json = {
            "id": self.id,
            "display-names": [self.name, self.src]
        }

        if self.no:
            channel_json["display-names"].extend([
                f"{self.no}",
                f"{self.no} {self.name}",
                f"{self.no} {self.src}"
            ])

        if self.icon:
            channel_json["icon"] = {"src": self.icon}

        return channel_json

"""
EPG2XML - Electronic Program Guide to XML Converter

이 패키지는 다양한 소스로부터 EPG (Electronic Program Guide) 데이터를 수집하고
XMLTV 형식으로 변환하는 기능을 제공합니다.

지원하는 Provider:
- NAVER: 네이버 편성표
- DAUM: 다음 편성표
- KT, LG, SK: 통신사 EPG
- WAVVE, TVING, SPOTV: OTT 서비스
"""

from .providers import EPGHandler, EPGProvider, EPGProgram, EPGChannel
from .config import Config
from .utils import dump_json

__all__ = [
    "EPGHandler",
    "EPGProvider",
    "EPGProgram",
    "EPGChannel",
    "Config",
    "dump_json"
]

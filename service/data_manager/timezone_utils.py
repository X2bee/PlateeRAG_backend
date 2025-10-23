"""
데이터 매니저용 타임존 유틸리티
한국 시간(KST, Asia/Seoul) 기준으로 시간 관리
"""
import os
from datetime import datetime
from zoneinfo import ZoneInfo

# 한국 시간대 (KST, UTC+9)
KST = ZoneInfo(os.getenv('TIMEZONE', 'Asia/Seoul'))


def now_kst() -> datetime:
    """
    현재 한국 시간 반환

    Returns:
        datetime: 한국 시간대(KST)의 현재 시간
    """
    return datetime.now(KST)


def now_kst_iso() -> str:
    """
    현재 한국 시간을 ISO 형식 문자열로 반환

    Returns:
        str: ISO 형식의 한국 시간 문자열
    """
    return now_kst().isoformat()


def to_kst(dt: datetime) -> datetime:
    """
    datetime 객체를 한국 시간대로 변환

    Args:
        dt: 변환할 datetime 객체

    Returns:
        datetime: 한국 시간대로 변환된 datetime
    """
    if dt.tzinfo is None:
        # naive datetime은 KST로 간주
        return dt.replace(tzinfo=KST)
    else:
        # aware datetime은 KST로 변환
        return dt.astimezone(KST)


def parse_iso_to_kst(iso_string: str) -> datetime:
    """
    ISO 형식 문자열을 한국 시간대의 datetime으로 변환

    Args:
        iso_string: ISO 형식의 시간 문자열

    Returns:
        datetime: 한국 시간대의 datetime 객체
    """
    dt = datetime.fromisoformat(iso_string)
    return to_kst(dt)

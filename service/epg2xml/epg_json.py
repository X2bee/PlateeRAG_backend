import logging
import sys
import os
import json
from datetime import datetime
from pathlib import Path
import pytz

# Add the project root to Python path if running directly
if __name__ == "__main__":
    # Get the project root directory (two levels up from this file)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from service.epg2xml import EPGHandler
from service.epg2xml.utils import get_cache_file_path, dump_json

NAVER_CHANNELS = [
            { "Name": "롯데홈쇼핑", "ServiceId": "815100"},
            { "Name": "현대홈쇼핑", "ServiceId": "815101"},
            { "Name": "CJ온스타일", "ServiceId": "815096"},
            { "Name": "GS SHOP", "ServiceId": "815097"},
            { "Name": "NS홈쇼핑", "ServiceId": "815099"},
            # { "Name": "홈&쇼핑", "ServiceId": "815524"},
            # { "Name": "공영쇼핑", "ServiceId": "9931218"},
            # { "Name": "CJ온스타일플러스", "ServiceId": "18608759"},
            { "Name": "SK stoa", "ServiceId": "19356905"},
            { "Name": "kt알파 쇼핑", "ServiceId": "26445690"},
            # { "Name": "NS Shop+", "ServiceId": "29770878"},
            # 아래는 스카이라이프 편성 채널인데 위와 중복이라 일단 주석 처리
            # { "Name": "롯데홈쇼핑", "ServiceId": "815365"},
            # { "Name": "현대홈쇼핑", "ServiceId": "815366"},
            # { "Name": "CJ온스타일", "ServiceId": "815360"},
            # { "Name": "GS SHOP", "ServiceId": "815362"},
            # { "Name": "NS홈쇼핑", "ServiceId": "815363"},
            # { "Name": "홈&쇼핑", "ServiceId": "815525"},
            # { "Name": "CJ온스타일플러스", "ServiceId": "18608758"},
        ]

DAUM_CHANNELS = [
            { "Name": "롯데홈쇼핑", "ServiceId": "케이블 롯데홈쇼핑"},
            { "Name": "현대홈쇼핑", "ServiceId": "케이블 현대홈쇼핑"},
            { "Name": "CJ온스타일", "ServiceId": "케이블 CJ온스타일"},
            { "Name": "GS SHOP", "ServiceId": "케이블 GS SHOP"},
            # { "Name": "(i)신세계 쇼핑", "ServiceId": "케이블 (i)신세계 쇼핑"},
            # { "Name": "CJ온스타일플러스", "ServiceId": "케이블 CJ온스타일플러스"},
            # { "Name": "GS MY SHOP", "ServiceId": "케이블 GS MY SHOP"},
            # { "Name": "LOTTE OneTV", "ServiceId": "케이블 LOTTE OneTV"},
            # { "Name": "NS Shop+", "ServiceId": "케이블 NS Shop+"},
            { "Name": "NS홈쇼핑", "ServiceId": "케이블 NS홈쇼핑"},
            # { "Name": "SK Stoa 02", "ServiceId": "케이블 SK Stoa 02"},
            # { "Name": "SK Stoa 03", "ServiceId": "케이블 SK Stoa 03"},
            # { "Name": "SK Stoa 04", "ServiceId": "케이블 SK Stoa 04"},
            { "Name": "SK stoa", "ServiceId": "케이블 SK stoa"},
            # { "Name": "W쇼핑", "ServiceId": "케이블 W쇼핑"},
            { "Name": "kt알파 쇼핑", "ServiceId": "케이블 kt알파 쇼핑"},
            # { "Name": "공영쇼핑", "ServiceId": "케이블 공영쇼핑"},
            # { "Name": "쇼핑엔티", "ServiceId": "케이블 쇼핑엔티"},
            # { "Name": "현대홈쇼핑+Shop", "ServiceId": "케이블 현대홈쇼핑+Shop"},
            # { "Name": "홈&쇼핑", "ServiceId": "케이블 홈&쇼핑"},
            # 스카이라이프 채널
            # { "Name": "HD CJ온스타일2", "ServiceId": "SKYLIFE HD CJ온스타일2"},
            # { "Name": "HD GS SHOP", "ServiceId": "SKYLIFE HD GS SHOP"},
            # { "Name": "HD NS홈쇼핑", "ServiceId": "SKYLIFE HD NS홈쇼핑"},
            # { "Name": "HD 롯데홈쇼핑", "ServiceId": "SKYLIFE HD 롯데홈쇼핑"},
            # { "Name": "HD 현대홈쇼핑", "ServiceId": "SKYLIFE HD 현대홈쇼핑"},
        ]

def epg_get_config(provider = "NAVER"):
    """테스트용 설정 생성"""
    if provider == "NAVER":
        return {
            "NAVER": {
                "ENABLED": True,
                "FETCH_LIMIT": 1,  # 하루치만 가져오기
                "ID_FORMAT": "{ServiceId}.naver",
                "ADD_REBROADCAST_TO_TITLE": True,
                "ADD_EPNUM_TO_TITLE": True,
                "ADD_DESCRIPTION": True,
                "ADD_XMLTV_NS": False,
                "GET_MORE_DETAILS": False,
                "ADD_CHANNEL_ICON": True,
                "HTTP_PROXY": None,
                "MY_CHANNELS": NAVER_CHANNELS
            }
        }
    elif provider == "DAUM":
        return {
            "DAUM": {
                "ENABLED": True,
                "FETCH_LIMIT": 1,  # 하루치만 가져오기
                "ID_FORMAT": "{ServiceId}.daum",
                "ADD_REBROADCAST_TO_TITLE": True,
                "ADD_EPNUM_TO_TITLE": True,
                "ADD_DESCRIPTION": True,
                "ADD_XMLTV_NS": False,
                "GET_MORE_DETAILS": False,
                "ADD_CHANNEL_ICON": True,
                "HTTP_PROXY": None,
                "MY_CHANNELS": DAUM_CHANNELS
            }
        }
    else:
        return {
            "NAVER": {
                "ENABLED": True,
                "FETCH_LIMIT": 1,  # 하루치만 가져오기
                "ID_FORMAT": "{ServiceId}.naver",
                "ADD_REBROADCAST_TO_TITLE": True,
                "ADD_EPNUM_TO_TITLE": True,
                "ADD_DESCRIPTION": True,
                "ADD_XMLTV_NS": False,
                "GET_MORE_DETAILS": False,
                "ADD_CHANNEL_ICON": True,
                "HTTP_PROXY": None,
                "MY_CHANNELS": NAVER_CHANNELS
            },
            "DAUM": {
                "ENABLED": True,
                "FETCH_LIMIT": 1,  # 하루치만 가져오기
                "ID_FORMAT": "{ServiceId}.daum",
                "ADD_REBROADCAST_TO_TITLE": True,
                "ADD_EPNUM_TO_TITLE": True,
                "ADD_DESCRIPTION": True,
                "ADD_XMLTV_NS": False,
                "GET_MORE_DETAILS": False,
                "ADD_CHANNEL_ICON": True,
                "HTTP_PROXY": None,
                "MY_CHANNELS": DAUM_CHANNELS
            }
        }

def epg_to_json(provider="NAVER", use_cache=True, target_date=None):
    """
    EPG 데이터를 JSON 형태로 가져오는 함수 (날짜별 캐싱 지원)

    Args:
        provider: EPG 제공자 ("NAVER", "DAUM", "BOTH")
        use_cache: 캐시 사용 여부 (기본값: True)
        target_date: 대상 날짜 (YYYY-MM-DD 형식, None이면 오늘 날짜)

    Returns:
        dict: EPG 데이터 JSON
    """
    # 대상 날짜 설정 (서울 시간 기준)
    if target_date is None:
        seoul_tz = pytz.timezone('Asia/Seoul')
        target_date = datetime.now(seoul_tz).strftime("%Y-%m-%d")

    # 캐시 파일 경로 생성
    cache_filename = f"epg_{provider.lower()}_{target_date}.json"
    cache_file_path = get_cache_file_path(cache_filename)

    # 캐시 파일이 존재하고 use_cache가 True인 경우 캐시에서 로드
    if use_cache and os.path.exists(cache_file_path):
        try:
            logging.info(f"Loading EPG data from cache: {cache_file_path}")
            with open(cache_file_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
            return result
        except Exception as e:
            logging.warning(f"Failed to load cache file {cache_file_path}: {e}")
            logging.info("Proceeding to fetch fresh data...")

    # 캐시가 없거나 사용하지 않는 경우 새로 데이터 가져오기
    logging.info(f"Fetching fresh EPG data for provider: {provider}, date: {target_date}")
    configs = epg_get_config(provider)
    handler = EPGHandler(configs)

    # 채널 파일 로드
    channel_file = get_cache_file_path("test_channels.json")
    handler.load_channels(channel_file, parallel=False)
    handler.load_req_channels()
    handler.get_programs(parallel=False)

    # JSON 결과 생성
    result = handler.to_json()

    # 결과를 캐시 파일에 저장
    try:
        logging.info(f"Saving EPG data to cache: {cache_file_path}")
        dump_json(cache_file_path, result)
    except Exception as e:
        logging.error(f"Failed to save cache file {cache_file_path}: {e}")

    return result

def get_cached_epg_files(provider=None):
    """
    캐시된 EPG 파일 목록을 반환하는 함수

    Args:
        provider: 특정 제공자로 필터링 (None이면 모든 제공자)

    Returns:
        list: 캐시된 파일 정보 리스트 [{filename, provider, date, path}, ...]
    """
    downloads_path = os.path.join(os.getcwd(), "downloads")
    cached_files = []

    # downloads 폴더의 모든 epg_ 폴더를 검사
    for folder_name in os.listdir(downloads_path):
        if folder_name.startswith("epg_"):
            folder_path = os.path.join(downloads_path, folder_name)
            if os.path.isdir(folder_path):
                # 폴더 내의 EPG JSON 파일들을 검사
                for filename in os.listdir(folder_path):
                    if filename.startswith("epg_") and filename.endswith(".json"):
                        # 파일명에서 정보 추출 (예: epg_naver_2025-09-24.json)
                        parts = filename.replace(".json", "").split("_")
                        if len(parts) >= 3:
                            file_provider = parts[1].upper()
                            file_date = parts[2]

                            # 제공자 필터링
                            if provider is None or file_provider == provider.upper():
                                cached_files.append({
                                    "filename": filename,
                                    "provider": file_provider,
                                    "date": file_date,
                                    "path": os.path.join(folder_path, filename),
                                    "folder": folder_name
                                })

    # 날짜순으로 정렬
    cached_files.sort(key=lambda x: x["date"], reverse=True)
    return cached_files

def clear_old_cache(days_to_keep=7):
    """
    오래된 캐시 파일들을 정리하는 함수

    Args:
        days_to_keep: 보관할 일수 (기본값: 7일)
    """
    from datetime import datetime, timedelta

    seoul_tz = pytz.timezone('Asia/Seoul')
    cutoff_date = datetime.now(seoul_tz) - timedelta(days=days_to_keep)
    cutoff_str = cutoff_date.strftime("%Y-%m-%d")

    downloads_path = os.path.join(os.getcwd(), "downloads")
    removed_count = 0

    for folder_name in os.listdir(downloads_path):
        if folder_name.startswith("epg_"):
            folder_date = folder_name.replace("epg_", "")
            if folder_date < cutoff_str:
                folder_path = os.path.join(downloads_path, folder_name)
                try:
                    import shutil
                    shutil.rmtree(folder_path)
                    logging.info(f"Removed old cache folder: {folder_name}")
                    removed_count += 1
                except Exception as e:
                    logging.error(f"Failed to remove cache folder {folder_name}: {e}")

    logging.info(f"Cache cleanup completed. Removed {removed_count} old cache folders.")
    return removed_count

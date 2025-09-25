import sys
import logging
from pathlib import Path

# 현재 디렉토리를 Python path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from service.epg2xml_backup import EPGHandler
from service.epg2xml_backup.config import setup_root_logger
from service.epg2xml_backup.utils import get_cache_file_path


def setup_test_config():
    """테스트용 설정 생성"""
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
            "MY_CHANNELS": [
                { "Name": "롯데홈쇼핑", "ServiceId": "815100"},
                { "Name": "현대홈쇼핑", "ServiceId": "815101"},
                { "Name": "CJ온스타일", "ServiceId": "815096"},
                { "Name": "GS SHOP", "ServiceId": "815097"},
                { "Name": "NS홈쇼핑", "ServiceId": "815099"},
                { "Name": "홈&쇼핑", "ServiceId": "815524"},
                { "Name": "공영쇼핑", "ServiceId": "9931218"},
                { "Name": "CJ온스타일플러스", "ServiceId": "18608759"},
                { "Name": "SK stoa", "ServiceId": "19356905"},
                { "Name": "kt알파 쇼핑", "ServiceId": "26445690"},
                { "Name": "NS Shop+", "ServiceId": "29770878"},
                # 아래는 스카이라이프 편성 채널인데 위와 중복이라 일단 주석 처리
                # { "Name": "롯데홈쇼핑", "ServiceId": "815365"},
                # { "Name": "현대홈쇼핑", "ServiceId": "815366"},
                # { "Name": "CJ온스타일", "ServiceId": "815360"},
                # { "Name": "GS SHOP", "ServiceId": "815362"},
                # { "Name": "NS홈쇼핑", "ServiceId": "815363"},
                # { "Name": "홈&쇼핑", "ServiceId": "815525"},
                # { "Name": "CJ온스타일플러스", "ServiceId": "18608758"},
            ]
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
            "MY_CHANNELS": [
                { "Name": "롯데홈쇼핑", "ServiceId": "케이블 롯데홈쇼핑"},
                { "Name": "현대홈쇼핑", "ServiceId": "케이블 현대홈쇼핑"},
                { "Name": "CJ온스타일", "ServiceId": "케이블 CJ온스타일"},
                { "Name": "GS SHOP", "ServiceId": "케이블 GS SHOP"},
                { "Name": "(i)신세계 쇼핑", "ServiceId": "케이블 (i)신세계 쇼핑"},
                { "Name": "CJ온스타일플러스", "ServiceId": "케이블 CJ온스타일플러스"},
                { "Name": "GS MY SHOP", "ServiceId": "케이블 GS MY SHOP"},
                { "Name": "LOTTE OneTV", "ServiceId": "케이블 LOTTE OneTV"},
                { "Name": "NS Shop+", "ServiceId": "케이블 NS Shop+"},
                { "Name": "NS홈쇼핑", "ServiceId": "케이블 NS홈쇼핑"},
                { "Name": "SK Stoa 02", "ServiceId": "케이블 SK Stoa 02"},
                { "Name": "SK Stoa 03", "ServiceId": "케이블 SK Stoa 03"},
                { "Name": "SK Stoa 04", "ServiceId": "케이블 SK Stoa 04"},
                { "Name": "SK stoa", "ServiceId": "케이블 SK stoa"},
                { "Name": "W쇼핑", "ServiceId": "케이블 W쇼핑"},
                { "Name": "kt알파 쇼핑", "ServiceId": "케이블 kt알파 쇼핑"},
                { "Name": "공영쇼핑", "ServiceId": "케이블 공영쇼핑"},
                { "Name": "쇼핑엔티", "ServiceId": "케이블 쇼핑엔티"},
                { "Name": "현대홈쇼핑+Shop", "ServiceId": "케이블 현대홈쇼핑+Shop"},
                { "Name": "홈&쇼핑", "ServiceId": "케이블 홈&쇼핑"},
                # 스카이라이프 채널
                # { "Name": "HD CJ온스타일2", "ServiceId": "SKYLIFE HD CJ온스타일2"},
                # { "Name": "HD GS SHOP", "ServiceId": "SKYLIFE HD GS SHOP"},
                # { "Name": "HD NS홈쇼핑", "ServiceId": "SKYLIFE HD NS홈쇼핑"},
                # { "Name": "HD 롯데홈쇼핑", "ServiceId": "SKYLIFE HD 롯데홈쇼핑"},
                # { "Name": "HD 현대홈쇼핑", "ServiceId": "SKYLIFE HD 현대홈쇼핑"},
            ]
        }
    }


def test_json_output():
    configs = setup_test_config()
    handler = EPGHandler(configs)
    channel_file = get_cache_file_path("test_channels.json")
    handler.load_channels(channel_file, parallel=False)
    handler.load_req_channels()
    handler.get_programs(parallel=False)
    handler.print_json()


def main():
    """메인 함수"""
    # 로깅 설정
    setup_root_logger(level=logging.INFO)
    try:
        # 3. JSON 출력 테스트
        test_json_output()

    except KeyboardInterrupt:
        print("\n테스트가 중단되었습니다.")
    except Exception as e:
        print(f"\n테스트 중 에러 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

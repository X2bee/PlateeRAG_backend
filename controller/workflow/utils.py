import re
from fastapi import HTTPException, Request

def extract_collection_name(collection_full_name: str) -> str:
    """
    컬렉션 이름에서 UUID 부분을 제거하고 실제 이름만 추출합니다.

    예: '장하렴연구_3a6a552d-d277-490d-9f3c-cead80d651f7' -> '장하렴연구'

    Args:
        collection_full_name: UUID가 포함된 전체 컬렉션 이름

    Returns:
        UUID 부분이 제거된 깨끗한 컬렉션 이름
    """
    # UUID 패턴: 8-4-4-4-12 형태의 16진수 문자열
    uuid_pattern = r'_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'

    # UUID 부분을 제거하고 앞의 이름만 반환
    clean_name = re.sub(uuid_pattern, '', collection_full_name, flags=re.IGNORECASE)

    return clean_name

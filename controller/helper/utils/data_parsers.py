"""
데이터 파싱 관련 유틸리티 함수들
"""
import re
import logging

logger = logging.getLogger("data-parsers")

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

def parse_input_data(input_data_str: str) -> str:
    """
    input_data에서 실제 입력 텍스트만 파싱합니다.

    Args:
        input_data_str: 파싱할 입력 데이터 문자열

    Returns:
        파싱된 입력 텍스트
    """
    if not input_data_str or not isinstance(input_data_str, str):
        return input_data_str

    # "Input: " 패턴으로 시작하는지 확인
    if input_data_str.startswith("Input: "):
        # "Input: " 이후의 텍스트 추출
        after_input = input_data_str[7:]  # "Input: " 길이만큼 자르기

        # "\n\nparameters:" 또는 "\n\nAdditional Parameters:" 패턴 찾기
        patterns = ["\n\nparameters:", "\n\nAdditional Parameters:", "\n\nValidation Error:"]

        for pattern in patterns:
            if pattern in after_input:
                # 패턴 앞까지의 텍스트 반환
                return after_input.split(pattern)[0].strip()

        # 패턴이 없으면 전체 텍스트 반환 (Input: 이후)
        return after_input.strip()

    # "Input: " 패턴이 없으면 원본 반환
    return input_data_str

def clean_llm_output(actual_output: str) -> str:
    """
    LLM 출력에서 불필요한 태그들을 제거합니다.
    
    Args:
        actual_output: 정리할 LLM 출력
    
    Returns:
        정리된 출력 텍스트
    """
    if not actual_output:
        return actual_output
    
    # <think> 태그 제거
    if '<think>' in actual_output and '</think>' in actual_output:
        actual_output = re.sub(r'<think>.*?</think>', '', actual_output, flags=re.DOTALL).strip()

    # [Cite.{{}}] 태그 제거
    if '[Cite.' in actual_output and '}}]' in actual_output:
        actual_output = re.sub(r'\[Cite\.\s*\{\{.*?\}\}\]', '', actual_output, flags=re.DOTALL).strip()

    # <TOOLUSELOG> 태그 제거
    if '<TOOLUSELOG>' in actual_output and '</TOOLUSELOG>' in actual_output:
        actual_output = re.sub(r'<TOOLUSELOG>.*?</TOOLUSELOG>', '', actual_output, flags=re.DOTALL).strip()

    # <TOOLOUTPUTLOG> 태그 제거
    if '<TOOLOUTPUTLOG>' in actual_output and '</TOOLOUTPUTLOG>' in actual_output:
        actual_output = re.sub(r'<TOOLOUTPUTLOG>.*?</TOOLOUTPUTLOG>', '', actual_output, flags=re.DOTALL).strip()

    return actual_output

def safe_round_float(value, decimal_places=4):
    """
    Decimal 타입을 float로 변환하는 헬퍼 함수
    """
    if value is None:
        return None
    try:
        # Decimal, float, int, str 모든 타입을 float로 변환
        if hasattr(value, '__float__'):  # Decimal 포함
            return round(float(value), decimal_places)
        elif isinstance(value, (int, float)):
            return round(float(value), decimal_places)
        elif isinstance(value, str):
            return round(float(value), decimal_places)
        else:
            return float(value) if value else 0.0
    except (ValueError, TypeError):
        return 0.0
# your_package/document_processor/ocr.py
import base64
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("document-processor")

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    LANGCHAIN_OPENAI_AVAILABLE = True
except ImportError:
    LANGCHAIN_OPENAI_AVAILABLE = False

from .config import is_image_text_enabled


def parse_batch_ocr_response(response_text: str, expected_count: int) -> List[str]:
    """배치 OCR 응답을 이미지별로 분할"""
    try:
        pattern = r'=== 이미지 (\d+) ===\s*(.*?)(?=\s*=== 이미지 \d+ ===|\s*$)'
        matches = re.findall(pattern, response_text, re.DOTALL)
        results: List[str] = []

        if matches and len(matches) >= expected_count:
            for i in range(expected_count):
                _, content = matches[i]
                results.append(content.strip())
        else:
            logger.warning("Pattern matching failed, using simple split")
            parts = re.split(r'=== 이미지 \d+ ===', response_text)
            for i in range(expected_count):
                if i + 1 < len(parts):
                    results.append(parts[i + 1].strip())
                else:
                    results.append("[이미지 분할 실패]")

        while len(results) < expected_count:
            results.append("[이미지 처리 실패]")

        return results[:expected_count]
    except Exception:
        # 실패 시 동일한 응답을 모든 이미지에 적용
        return [response_text for _ in range(expected_count)]


async def convert_image_to_text(image_path: str, current_config: Dict[str, Any]) -> str:
    """이미지를 텍스트로 변환 (단건) - 원문 프롬프트 완전 포함"""
    if not is_image_text_enabled(current_config, LANGCHAIN_OPENAI_AVAILABLE):
        return "[이미지 파일: 이미지-텍스트 변환이 설정되지 않았습니다]"

    try:
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode('utf-8')

        provider = current_config.get('provider', 'openai')
        api_key = current_config.get('api_key', '')
        base_url = current_config.get('base_url', 'https://api.openai.com/v1')
        model = current_config.get('model', 'gpt-4-vision-preview')
        temperature = current_config.get('temperature', 0.7)

        if provider in ('openai', 'vllm'):
            llm = ChatOpenAI(
                model=model,
                openai_api_key=api_key or 'dummy',
                base_url=base_url,
                temperature=temperature
            )
        else:
            logger.error(f"Unsupported image-text provider: {provider}")
            return f"[이미지 파일: 지원하지 않는 프로바이더 - {provider}]"

        # ── 원문 단건 OCR 프롬프트 (규칙/예시 포함) ─────────────────────────
        prompt = """이 이미지를 정확한 텍스트로 변환해주세요. 다음 규칙을 철저히 지켜주세요:

                    1. **표 구조 보존**: 표가 있다면 정확한 행과 열 구조를 유지하고, 마크다운 표 형식으로 변환해주세요 또한 병합된 셀의 경우, 각각 해당 사항에 모두 넣어주세요.
                    2. **레이아웃 유지**: 원본의 레이아웃, 들여쓰기, 줄바꿈을 최대한 보존해주세요
                    3. **정확한 텍스트**: 모든 문자, 숫자, 기호를 정확히 인식해주세요
                    4. **구조 정보**: 제목, 부제목, 목록, 단락 구분을 명확히 표현해주세요
                    5. **특수 형식**: 날짜, 금액, 주소, 전화번호 등의 형식을 정확히 유지해주세요
                    6. **슬라이드 구조**: 슬라이드 제목, 내용, 차트/그래프 설명을 구분해주세요
                    7. **섹션 구분**: MarkDown 형식을 철저히 준수해서 섹션 구분을 철저히 나눠주세요.

                    8. **표 구분**: 각 이미지 내에서 표는 [표 구분]으로 명확히 구분하고, 표의 제목, 설명, 표 본체, 텍스트 변환을 모두 포함해주세요
                    9. **언어*** : 한국어/영어가 아닌 문자를 포함하지 않도록 주의해주세요. 한자의 경우 한국어로 넣어주세요.
                    **섹션 구분 예시:**
                    - 서로 다른 주제나 단락 사이 
                    - 표와 다른 내용 사이
                    - 각 표는 제목부터 텍스트 설명까지 하나의 [표 구분]을 이룹니다.
                    - 단 상품의 제목 과 설명 / 표의 제목 및 설명 등 같은 내용은 하나의 색션을 구성해야합니다

                    **출력 형식 예시:**
                    # 제목
                    [섹션 구분]
                    본문 내용이 여기에...

                    [섹션 구분]
                    ## 소제목
                    다른 섹션의 내용...

                    [표 구분]
                    ### 상가건물 지역별 임대 현황
                    출처: 부동산 통계청 (2024년 3분기)
                    ※ 단위: 면적(㎡), 임대료(만원/월), 보증금(만원)
                    
                    | 구분 | 지역명 | 평균임대료(만원) | 보증금(만원) | 면적(㎡) | 업종분류 | 계약현황 |
                    |------|--------|------------------|--------------|----------|----------|----------|
                    | 상가 | 강남구 | 450 | 8,500 | 65.2 | 음식점 | 계약완료 |
                    | 상가 | 서초구 | 320 | 6,200 | 48.7 | 의류 | 협의중 |
                    | 상가 | 마포구 | 280 | 4,800 | 52.3 | 카페 | 계약완료 |
                    
                    **표 내용 완전 텍스트 변환**: 이 표는 상가건물 지역별 임대 현황을 나타내는 표로, 부동산 통계청에서 발표한 2024년 3분기 자료입니다. 면적은 제곱미터, 임대료는 월 단위 만원, 보증금은 만원 단위로 표시되어 있습니다. 총 3개 지역의 상가 정보가 포함되어 있습니다. 첫 번째는 강남구 상가로 평균임대료가 450만원, 보증금이 8,500만원이며, 면적은 65.2㎡이고 음식점 업종으로 계약이 완료된 상태입니다. 두 번째는 서초구 상가로 평균임대료가 320만원, 보증금이 6,200만원이며, 면적은 48.7㎡이고 의류 업종으로 현재 협의중인 상태입니다. 세 번째는 마포구 상가로 평균임대료가 280만원, 보증금이 4,800만원이며, 면적은 52.3㎡이고 카페 업종으로 계약이 완료된 상태입니다.

                    반드시 한국어 및 영어로 된 텍스트만 출력하고, 추가 설명은 하지 마세요."""
        # ────────────────────────────────────────────────────────────────

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        )
        response = await llm.ainvoke([message])
        logger.info(f"Successfully converted image to text using {provider}: {Path(image_path).name}")
        return response.content

    except Exception as e:
        logger.error(f"Error converting image to text {image_path}: {e}")
        return f"[이미지 파일: 텍스트 변환 중 오류 발생 - {str(e)}]"


async def convert_image_to_text_with_reference(image_path: str, reference_text: str, current_config: Dict[str, Any]) -> str:
    """이미지를 텍스트/HTML로 변환 (단건, 참고 텍스트 활용) - 원문 프롬프트 완전 포함"""
    if not is_image_text_enabled(current_config, LANGCHAIN_OPENAI_AVAILABLE):
        return "[이미지 파일: 이미지-텍스트 변환이 설정되지 않았습니다]"

    try:
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode('utf-8')

        provider = current_config.get('provider', 'openai')
        api_key = current_config.get('api_key', '')
        base_url = current_config.get('base_url', 'https://api.openai.com/v1')
        model = current_config.get('model', 'gpt-4-vision-preview')
        temperature = current_config.get('temperature', 0.7)

        if provider in ('openai', 'vllm'):
            llm = ChatOpenAI(
                model=model,
                openai_api_key=api_key or 'dummy',
                base_url=base_url,
                temperature=temperature
            )
        else:
            logger.error(f"Unsupported image-text provider: {provider}")
            return f"[이미지 파일: 지원하지 않는 프로바이더 - {provider}]"

        # 참고 텍스트 유무에 따라 원문 두 가지 프롬프트를 그대로 사용
        if reference_text and reference_text.strip():
            prompt = f"""이 이미지를 정확한 HTML 텍스트로 변환해주세요. 

                    **🔥 중요: 기계적 파싱 참고 텍스트 활용**
                    아래는 같은 페이지에서 기계적으로 추출된 텍스트입니다. 이를 참고하여 OCR 정확도를 높여주세요:
                    {reference_text}

                    **HTML 변환 규칙:**
                    1. **참고 텍스트 활용**: 위의 참고 텍스트를 활용하여 누락된 단어나 부정확한 인식을 보완해주세요
                    2. **HTML 구조 보존**: 문서의 구조를 semantic HTML로 정확히 표현해주세요
                    3. **표 구조**: 표가 있다면 `<table>`, `<thead>`, `<tbody>`, `<tr>`, `<th>`, `<td>` 태그를 사용하여 정확한 구조로 변환해주세요
                    4. **병합된 셀**: `colspan`, `rowspan` 속성을 사용하여 병합된 셀을 정확히 표현해주세요
                    5. **레이아웃 유지**: 원본의 레이아웃과 계층 구조를 HTML로 보존해주세요
                    6. **정확한 텍스트**: 모든 문자, 숫자, 기호를 정확히 인식해주세요
                    7. **구조화**: 제목(`<h1>-<h6>`), 단락(`<p>`), 목록(`<ul>`, `<ol>`, `<li>`) 등을 적절히 사용해주세요
                    8. **특수 형식**: 날짜, 금액, 주소, 전화번호 등의 형식을 정확히 유지해주세요
                    9. **섹션 구분**: `<section>`, `<div>`, `<article>` 등을 사용하여 논리적 구분을 명확히 해주세요
                    10. **표 구분**: 각 표는 `<div class="table-section">` 으로 감싸서 명확히 구분해주세요
                    11. **언어**: 한국어/영어가 아닌 문자를 포함하지 않도록 주의해주세요. 한자의 경우 한국어로 변환해주세요

                    **HTML 출력 형식 예시:**
                    ```html
                    <!DOCTYPE html>
                    <html lang="ko">
                    <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>문서 제목</title>
                    </head>
                    <body>
                    <header>
                        <h1>제목</h1>
                    </header>
                    
                    <main>
                        <section>
                            <p>본문 내용이 여기에...</p>
                        </section>
                        
                        <section>
                            <h2>소제목</h2>
                            <p>다른 섹션의 내용...</p>
                        </section>
                        
                        <div class="table-section">
                            <h3>상가건물 지역별 임대 현황</h3>
                            <p class="source">출처: 부동산 통계청 (2024년 3분기)</p>
                            <p class="note">※ 단위: 면적(㎡), 임대료(만원/월), 보증금(만원)</p>
                            
                            <table border="1">
                                <thead>
                                    <tr>
                                        <th>구분</th>
                                        <th>지역명</th>
                                        <th>평균임대료(만원)</th>
                                        <th>보증금(만원)</th>
                                        <th>면적(㎡)</th>
                                        <th>업종분류</th>
                                        <th>계약현황</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr>
                                        <td>상가</td>
                                        <td>강남구</td>
                                        <td>450</td>
                                        <td>8,500</td>
                                        <td>65.2</td>
                                        <td>음식점</td>
                                        <td>계약완료</td>
                                    </tr>
                                    <tr>
                                        <td>상가</td>
                                        <td>서초구</td>
                                        <td>320</td>
                                        <td>6,200</td>
                                        <td>48.7</td>
                                        <td>의류</td>
                                        <td>협의중</td>
                                    </tr>
                                    <tr>
                                        <td>상가</td>
                                        <td>마포구</td>
                                        <td>280</td>
                                        <td>4,800</td>
                                        <td>52.3</td>
                                        <td>카페</td>
                                        <td>계약완료</td>
                                    </tr>
                                </tbody>
                            </table>
                            
                            <div class="table-description">
                                <p><strong>표 내용 완전 텍스트 변환</strong>: 이 표는 상가건물 지역별 임대 현황을 나타내는 표로, 부동산 통계청에서 발표한 2024년 3분기 자료입니다. 면적은 제곱미터, 임대료는 월 단위 만원, 보증금은 만원 단위로 표시되어 있습니다. 총 3개 지역의 상가 정보가 포함되어 있습니다. 첫 번째는 강남구 상가로 평균임대료가 450만원, 보증금이 8,500만원이며, 면적은 65.2㎡이고 음식점 업종으로 계약이 완료된 상태입니다. 두 번째는 서초구 상가로 평균임대료가 320만원, 보증금이 6,200만원이며, 면적은 48.7㎡이고 의류 업종으로 현재 협의중인 상태입니다. 세 번째는 마포구 상가로 평균임대료가 280만원, 보증금이 4,800만원이며, 면적은 52.3㎡이고 카페 업종으로 계약이 완료된 상태입니다.</p>
                            </div>
                        </div>
                    </main>
                    </body>
                    </html>"""
        else:
            prompt = f"""이 이미지를 정확한 HTML 텍스트로 변환해주세요. 

                    **🔥 중요: 기계적 파싱 참고 텍스트 활용**
                    아래는 같은 페이지에서 기계적으로 추출된 텍스트입니다. 이를 참고하여 OCR 정확도를 높여주세요:
                    {reference_text}

                    **HTML 변환 규칙:**
                    1. **참고 텍스트 활용**: 위의 참고 텍스트를 활용하여 누락된 단어나 부정확한 인식을 보완해주세요
                    2. **HTML 구조 보존**: 문서의 구조를 semantic HTML로 정확히 표현해주세요
                    3. **표 구조**: 표가 있다면 `<table>`, `<thead>`, `<tbody>`, `<tr>`, `<th>`, `<td>` 태그를 사용하여 정확한 구조로 변환해주세요
                    4. **병합된 셀**: `colspan`, `rowspan` 속성을 사용하여 병합된 셀을 정확히 표현해주세요
                    5. **레이아웃 유지**: 원본의 레이아웃과 계층 구조를 HTML로 보존해주세요
                    6. **정확한 텍스트**: 모든 문자, 숫자, 기호를 정확히 인식해주세요
                    7. **섹션 구분**: <!-- [section] -->를 의미론적 구분자로 사용하고 이를 철저히 삽입해서 섹션 구분을 철저히 나눠주세요.
                    8. **특수 형식**: 날짜, 금액, 주소, 전화번호 등의 형식을 정확히 유지해주세요
                    9. **표 구분**: 각 표는 `<div class="table-section">` 으로 감싸서 명확히 구분해주세요
                    10. **언어**: 한국어/영어가 아닌 문자를 포함하지 않도록 주의해주세요. 한자의 경우 한국어로 변환해주세요

                    **섹션 구분 예시:**
                    - 서로 다른 주제나 단락 사이 
                    - 표와 다른 내용 사이
                    - 각 표는 제목부터 텍스트 설명까지 하나의 [표 구분]을 이룹니다.
                    - 단 상품의 제목과 설명 / 표의 제목 및 설명 등 같은 내용은 하나의 섹션을 구성해야합니다

                    **HTML 출력 형식 예시:**
                    ```html
                    <!DOCTYPE html>
                    <html lang="ko">
                    <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>문서 제목</title>
                    </head>
                    <body>
                    <header>
                        <h1>제목</h1>
                    </header>
                    
                    <!-- [section] -->
                    <main>
                        <section>
                            <p>본문 내용이 여기에...</p>
                        </section>
                        
                        <!-- [section] -->
                        <section>
                            <h2>소제목</h2>
                            <p>다른 섹션의 내용...</p>
                        </section>
                        
                        <!-- [section] -->
                        <div class="table-section">
                            <h3>상가건물 지역별 임대 현황</h3>
                            <p class="source">출처: 부동산 통계청 (2024년 3분기)</p>
                            <p class="note">※ 단위: 면적(㎡), 임대료(만원/월), 보증금(만원)</p>
                            
                            <table border="1">
                                <thead>
                                    <tr>
                                        <th>구분</th>
                                        <th>지역명</th>
                                        <th>평균임대료(만원)</th>
                                        <th>보증금(만원)</th>
                                        <th>면적(㎡)</th>
                                        <th>업종분류</th>
                                        <th>계약현황</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr>
                                        <td>상가</td>
                                        <td>강남구</td>
                                        <td>450</td>
                                        <td>8,500</td>
                                        <td>65.2</td>
                                        <td>음식점</td>
                                        <td>계약완료</td>
                                    </tr>
                                    <tr>
                                        <td>상가</td>
                                        <td>서초구</td>
                                        <td>320</td>
                                        <td>6,200</td>
                                        <td>48.7</td>
                                        <td>의류</td>
                                        <td>협의중</td>
                                    </tr>
                                    <tr>
                                        <td>상가</td>
                                        <td>마포구</td>
                                        <td>280</td>
                                        <td>4,800</td>
                                        <td>52.3</td>
                                        <td>카페</td>
                                        <td>계약완료</td>
                                    </tr>
                                </tbody>
                            </table>
                            
                            <div class="table-description">
                                <p><strong>표 내용 완전 텍스트 변환</strong>: 이 표는 상가건물 지역별 임대 현황을 나타내는 표로, 부동산 통계청에서 발표한 2024년 3분기 자료입니다. 면적은 제곱미터, 임대료는 월 단위 만원, 보증금은 만원 단위로 표시되어 있습니다. 총 3개 지역의 상가 정보가 포함되어 있습니다. 첫 번째는 강남구 상가로 평균임대료가 450만원, 보증금이 8,500만원이며, 면적은 65.2㎡이고 음식점 업종으로 계약이 완료된 상태입니다. 두 번째는 서초구 상가로 평균임대료가 320만원, 보증금이 6,200만원이며, 면적은 48.7㎡이고 의류 업종으로 현재 협의중인 상태입니다. 세 번째는 마포구 상가로 평균임대료가 280만원, 보증금이 4,800만원이며, 면적은 52.3㎡이고 카페 업종으로 계약이 완료된 상태입니다.</p>
                            </div>
                        </div>
                    </main>
                    </body>
                    </html>
                    ```
                    => 이미지 내용과 이에 대한 해석을 제외한 어떠한 추가 설명도 하지 마세요. 또한 표에 대한 완전 텍스트 변환은 표의 내용 및 주변 메타데이터이외 어떤 내용도 포함하지 마세요.
                    """

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        )
        response = await llm.ainvoke([message])
        return response.content

    except Exception as e:
        logger.error(f"Error converting image to text {image_path}: {e}")
        return f"[이미지 파일: 텍스트 변환 중 오류 발생 - {str(e)}]"


async def convert_multiple_images_to_text(image_paths: List[str], config: Dict[str, Any]) -> List[str]:
    """여러 이미지를 한번에 OCR 처리 (배치) - 원문 멀티 이미지 프롬프트 완전 포함"""
    try:
        if not is_image_text_enabled(config, LANGCHAIN_OPENAI_AVAILABLE):
            return ["[이미지 파일: 이미지-텍스트 변환이 설정되지 않았습니다]" for _ in image_paths]

        provider = config.get('provider', 'openai')
        api_key = config.get('api_key', '')
        base_url = config.get('base_url', 'https://api.openai.com/v1')
        model = config.get('model', 'gpt-4-vision-preview')
        temperature = config.get('temperature', 0.7)

        if provider not in ('openai', 'vllm'):
            return [f"[이미지 파일: 지원하지 않는 프로바이더 - {provider}]" for _ in image_paths]

        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage

        # 모든 이미지를 base64로 준비
        images_b64: List[str] = []
        for p in image_paths:
            with open(p, "rb") as f:
                images_b64.append(base64.b64encode(f.read()).decode('utf-8'))

        llm = ChatOpenAI(
            model=model,
            openai_api_key=api_key or 'dummy',
            base_url=base_url,
            temperature=temperature
        )

        # ── 원문 멀티 이미지 프롬프트 (표 연속성 규칙 포함) ────────────────
        prompt = f"""다음 {len(image_paths)}개의 이미지를 각각 정확한 텍스트로 변환해주세요. 

                **🔥 중요: 연속된 페이지의 표 처리 규칙**
                   이 이미지들은 연속된 문서 페이지입니다. 표가 여러 페이지에 걸쳐 나뉘어져 있을 수 있으므로 다음 규칙을 반드시 지켜주세요:
                   
                   1. **표 연속성 인식**: 이전 페이지에서 시작된 표가 다음 페이지에서 계속되는 경우를 인식해주세요
                   2. **시기 정보 보존**: 담보취득 시기(예: "2018.09.18일부터 담보취득분")가 이전 페이지에 있고 다음 페이지에 해당 데이터가 있는 경우, 시기 정보를 각 행에 포함해주세요
                   3. **병합된 셀 완전 복원**: 병합된 셀의 내용이 여러 행에 적용되는 경우, 모든 해당 행에 완전한 정보를 표시해주세요
                   4. **지역명 완전 표기**: 지역명이 생략되거나 빈 칸으로 표시된 경우, 문맥을 고려하여 완전한 지역명을 추가해주세요

                   **중요한 규칙:**
                   1. 각 이미지의 결과를 명확히 구분해주세요
                   2. 다음 형식으로 응답해주세요:

                   === 이미지 1 ===
                   [첫 번째 이미지의 텍스트 내용]

                   === 이미지 2 ===
                   [두 번째 이미지의 텍스트 내용]

                   === 이미지 3 ===
                   [세 번째 이미지의 텍스트 내용]

                   **변환 규칙:**
                   1. **표 구조 보존**: 표가 있다면 정확한 행과 열 구조를 유지하고, 마크다운 표 형식으로 변환해주세요 또한 병합된 셀의 경우, 각각 해당 사항에 모두 넣어주세요.
                   2. **레이아웃 유지**: 원본의 레이아웃, 들여쓰기, 줄바꿈을 최대한 보존해주세요
                   3. **정확한 텍스트**: 모든 문자, 숫자, 기호를 정확히 인식해주세요
                   4. **구조 정보**: 제목, 부제목, 목록, 단락 구분을 명확히 표현해주세요
                   5. **특수 형식**: 날짜, 금액, 주소, 전화번호 등의 형식을 정확히 유지해주세요
                   6. **슬라이드 구조**: 슬라이드 제목, 내용, 차트/그래프 설명을 구분해주세요
                   7. **섹션 구분**: MarkDown 형식을 철저히 준수해서 섹션 구분을 철저히 나눠주세요.
                   8. **표 구분**: 각 이미지 내에서 표는 [표 구분]으로 명확히 구분하고, 표의 제목, 설명, 표 본체, 텍스트 변환을 모두 포함해주세요
                   9. **언어*** : 한국어/영어가 아닌 문자를 포함하지 않도록 주의해주세요. 한자의 경우 한국어로 넣어주세요.
                   **섹션 구분 예시:**
                   - 서로 다른 주제나 단락 사이 
                   - 표와 다른 내용 사이
                   - 각 표는 제목부터 텍스트 설명까지 하나의 [표 구분]을 이룹니다.
                   - 단 상품의 제목 과 설명 / 표의 제목 및 설명 등 같은 내용은 하나의 색션을 구성해야합니다

                   **출력 형식 예시:**
                   === 이미지 1 ===
                   # 제목
                   [섹션 구분]
                   본문 내용이 여기에...

                   [섹션 구분]
                   ## 소제목
                   다른 섹션의 내용...

                   [표 구분]
                   ### 지역별 상가 운영 현황 (2024년 기준)
                   ※ 단위: 면적(㎡), 임대료(만원), 보증금(만원)
                   
                   | 지역 | 매장명 | 면적(㎡) | 월임대료(만원) | 보증금(만원) | 업종 | 운영상태 |
                   |------|--------|----------|----------------|--------------|------|----------|
                   | 강남구 | 카페A | 45.2 | 350 | 5,000 | 카페 | 운영중 |
                   | 강남구 | 식당B | 82.5 | 520 | 8,000 | 한식 | 운영중 |
                   | 서초구 | 의류C | 38.7 | 280 | 4,500 | 의류 | 임시휴업 |
                   | 서초구 | 편의점D | 25.3 | 180 | 2,500 | 편의점 | 운영중 |
                   | 마포구 | 학원E | 95.8 | 420 | 6,000 | 교육 | 운영중 |
                   
                   **표 내용 완전 텍스트 변환**: 이 표는 지역별 상가 운영 현황을 나타내는 표로, 2024년 기준으로 작성되었으며 면적은 제곱미터, 임대료와 보증금은 만원 단위로 표시되어 있습니다. 총 5개의 상가 정보가 포함되어 있습니다. 첫 번째 상가는 강남구에 위치한 카페A로 면적은 45.2㎡이며, 월임대료는 350만원, 보증금은 5,000만원이고 카페 업종으로 현재 운영중입니다. 두 번째 상가는 같은 강남구에 위치한 식당B로 면적은 82.5㎡이며, 월임대료는 520만원, 보증금은 8,000만원이고 한식 업종으로 현재 운영중입니다. 세 번째 상가는 서초구에 위치한 의류C로 면적은 38.7㎡이며, 월임대료는 280만원, 보증금은 4,500만원이고 의류 업종으로 현재 임시휴업 상태입니다. 네 번째 상가는 같은 서초구에 위치한 편의점D로 면적은 25.3㎡이며, 월임대료는 180만원, 보증금은 2,500만원이고 편의점 업종으로 현재 운영중입니다. 다섯 번째 상가는 마포구에 위치한 학원E로 면적은 95.8㎡이며, 월임대료는 420만원, 보증금은 6,000만원이고 교육 업종으로 현재 운영중입니다. 
                   
                   표 전체를 분석하면, 지역별로는 강남구 2개, 서초구 2개, 마포구 1개 상가가 분포되어 있습니다. 면적 범위는 25.3㎡에서 95.8㎡까지이며, 평균 면적은 약 57.5㎡입니다. 월임대료는 180만원에서 520만원까지의 범위를 보이며, 평균 월임대료는 350만원입니다. 보증금은 2,500만원에서 8,000만원까지의 범위이며, 평균 보증금은 5,200만원입니다. 업종별로는 카페, 한식, 의류, 편의점, 교육으로 다양하게 구성되어 있으며, 운영상태는 4개 상가가 운영중이고 1개 상가가 임시휴업 상태입니다.

                   텍스트만 출력하고, 추가 설명은 하지 마세요."""
        # ─────────────────────────────────────────────────────────────

        content = [{"type": "text", "text": prompt}]
        for b64 in images_b64:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

        resp = await llm.ainvoke([HumanMessage(content=content)])
        return parse_batch_ocr_response(resp.content, len(image_paths))

    except Exception as e:
        logger.error(f"Error in batch OCR: {e}")
        # 실패 시 개별 처리 폴백
        out: List[str] = []
        for p in image_paths:
            out.append(await convert_image_to_text(p, config))
        return out


async def convert_multiple_images_to_text_with_reference(image_paths: List[str], references: List[str], config: Dict[str, Any]) -> List[str]:
    """여러 이미지를 한번에 OCR 처리 (배치 + 참고 텍스트) - 원문 프롬프트 완전 포함"""
    try:
        if not is_image_text_enabled(config, LANGCHAIN_OPENAI_AVAILABLE):
            return ["[이미지 파일: 이미지-텍스트 변환이 설정되지 않았습니다]" for _ in image_paths]

        provider = config.get('provider', 'openai')
        api_key = config.get('api_key', '')
        base_url = config.get('base_url', 'https://api.openai.com/v1')
        model = config.get('model', 'gpt-4-vision-preview')
        temperature = config.get('temperature', 0.7)

        if provider not in ('openai', 'vllm'):
            return [f"[이미지 파일: 지원하지 않는 프로바이더 - {provider}]" for _ in image_paths]

        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage

        images_b64: List[str] = []
        for p in image_paths:
            with open(p, "rb") as f:
                images_b64.append(base64.b64encode(f.read()).decode('utf-8'))

        llm = ChatOpenAI(
            model=model,
            openai_api_key=api_key or 'dummy',
            base_url=base_url,
            temperature=temperature
        )

        # 참고 텍스트 섹션 구성
        reference_info = "\n**🔥 기계적 파싱 참고 텍스트:**\n"
        for i, ref in enumerate(references or []):
            if ref and str(ref).strip():
                reference_info += f"\n--- 이미지 {i+1} 참고 텍스트 ---\n{ref}\n"
            else:
                reference_info += f"\n--- 이미지 {i+1} 참고 텍스트 ---\n[기계적 파싱으로 추출된 텍스트 없음]\n"

        # ── 원문 멀티 이미지 + 참고 텍스트 프롬프트 ───────────────────────
        prompt = f"""다음 {len(image_paths)}개의 이미지를 각각 정확한 텍스트로 변환해주세요.

                {reference_info}
                위의 텍스트는 해당 이미지로부터 텍스트를 파싱한 결과물입니다. 참고하여 아래의 규칙을 준수하여 응답하시오.
                    **🔥 중요: 연속된 페이지의 표 처리 규칙**
                   이 이미지들은 연속된 문서 페이지입니다. 표가 여러 페이지에 걸쳐 나뉘어져 있을 수 있으므로 다음 규칙을 반드시 지켜주세요:
                   
                   1. **표 연속성 인식**: 이전 페이지에서 시작된 표가 다음 페이지에서 계속되는 경우를 인식해주세요
                   2. **시기 정보 보존**: 담보취득 시기(예: "2018.09.18일부터 담보취득분")가 이전 페이지에 있고 다음 페이지에 해당 데이터가 있는 경우, 시기 정보를 각 행에 포함해주세요
                   3. **병합된 셀 완전 복원**: 병합된 셀의 내용이 여러 행에 적용되는 경우, 모든 해당 행에 완전한 정보를 표시해주세요
                   4. **지역명 완전 표기**: 지역명이 생략되거나 빈 칸으로 표시된 경우, 문맥을 고려하여 완전한 지역명을 추가해주세요

                   **중요한 규칙:**
                   1. 각 이미지의 결과를 명확히 구분해주세요
                   2. 다음 형식으로 응답해주세요:

                   === 이미지 1 ===
                   [첫 번째 이미지의 텍스트 내용]

                   === 이미지 2 ===
                   [두 번째 이미지의 텍스트 내용]

                   === 이미지 3 ===
                   [세 번째 이미지의 텍스트 내용]

                   **변환 규칙:**
                   1. **표 구조 보존**: 표가 있다면 정확한 행과 열 구조를 유지하고, 마크다운 표 형식으로 변환해주세요 또한 병합된 셀의 경우, 각각 해당 사항에 모두 넣어주세요.
                    => 표의 머지된 부분을 정확하게 고려해서 표 아래에 모든 내용을 텍스트로 변환하여 적어주세요                   
                    2. **레이아웃 유지**: 원본의 레이아웃, 들여쓰기, 줄바꿈을 최대한 보존해주세요
                   3. **정확한 텍스트**: 모든 문자, 숫자, 기호를 정확히 인식해주세요
                   4. **구조 정보**: 제목, 부제목, 목록, 단락 구분을 명확히 표현해주세요
                   5. **특수 형식**: 날짜, 금액, 주소, 전화번호 등의 형식을 정확히 유지해주세요
                   6. **슬라이드 구조**: 슬라이드 제목, 내용, 차트/그래프 설명을 구분해주세요
                   7. **섹션 구분**: MarkDown 형식을 철저히 준수해서 섹션 구분을 철저히 나눠주세요.
                   8. **표 구분**: 각 이미지 내에서 표는 [표 구분]으로 명확히 구분하고, 표의 제목, 설명, 표 본체, 텍스트 변환을 모두 포함해주세요
                   9. **언어*** : 한국어/영어가 아닌 문자를 포함하지 않도록 주의해주세요. 한자의 경우 한국어로 넣어주세요.
                   **섹션 구분 예시:**
                   - 서로 다른 주제나 단락 사이 
                   - 표와 다른 내용 사이
                   - 각 표는 제목부터 텍스트 설명까지 하나의 [표 구분]을 이룹니다.
                   - 단 상품의 제목 과 설명 / 표의 제목 및 설명 등 같은 내용은 하나의 색션을 구성해야합니다

                   **출력 형식 예시:**
                   === 이미지 1 ===
                   # 제목
                   [섹션 구분]
                   본문 내용이 여기에...

                   [섹션 구분]
                   ## 소제목
                   다른 섹션의 내용...

                   [표 구분]
                   ### 지역별 상가 운영 현황 (2024년 기준)
                   ※ 단위: 면적(㎡), 임대료(만원), 보증금(만원)
                   
                   | 지역 | 매장명 | 면적(㎡) | 월임대료(만원) | 보증금(만원) | 업종 | 운영상태 |
                   |------|--------|----------|----------------|--------------|------|----------|
                   | 강남구 | 카페A | 45.2 | 350 | 5,000 | 카페 | 운영중 |
                   | 강남구 | 식당B | 82.5 | 520 | 8,000 | 한식 | 운영중 |
                   | 서초구 | 의류C | 38.7 | 280 | 4,500 | 의류 | 임시휴업 |
                   | 서초구 | 편의점D | 25.3 | 180 | 2,500 | 편의점 | 운영중 |
                   | 마포구 | 학원E | 95.8 | 420 | 6,000 | 교육 | 운영중 |
                   
                   **표 내용 완전 텍스트 변환**: 이 표는 지역별 상가 운영 현황을 나타내는 표로, 2024년 기준으로 작성되었으며 면적은 제곱미터, 임대료와 보증금은 만원 단위로 표시되어 있습니다. 총 5개의 상가 정보가 포함되어 있습니다. 첫 번째 상가는 강남구에 위치한 카페A로 면적은 45.2㎡이며, 월임대료는 350만원, 보증금은 5,000만원이고 카페 업종으로 현재 운영중입니다. 두 번째 상가는 같은 강남구에 위치한 식당B로 면적은 82.5㎡이며, 월임대료는 520만원, 보증금은 8,000만원이고 한식 업종으로 현재 운영중입니다. 세 번째 상가는 서초구에 위치한 의류C로 면적은 38.7㎡이며, 월임대료는 280만원, 보증금은 4,500만원이고 의류 업종으로 현재 임시휴업 상태입니다. 네 번째 상가는 같은 서초구에 위치한 편의점D로 면적은 25.3㎡이며, 월임대료는 180만원, 보증금은 2,500만원이고 편의점 업종으로 현재 운영중입니다. 다섯 번째 상가는 마포구에 위치한 학원E로 면적은 95.8㎡이며, 월임대료는 420만원, 보증금은 6,000만원이고 교육 업종으로 현재 운영중입니다. 
                   
                   표 전체를 분석하면, 지역별로는 강남구 2개, 서초구 2개, 마포구 1개 상가가 분포되어 있습니다. 면적 범위는 25.3㎡에서 95.8㎡까지이며, 평균 면적은 약 57.5㎡입니다. 월임대료는 180만원에서 520만원까지의 범위를 보이며, 평균 월임대료는 350만원입니다. 보증금은 2,500만원에서 8,000만원까지의 범위이며, 평균 보증금은 5,200만원입니다. 업종별로는 카페, 한식, 의류, 편의점, 교육으로 다양하게 구성되어 있으며, 운영상태는 4개 상가가 운영중이고 1개 상가가 임시휴업 상태입니다.

                   반드시 한국어 및 영어로 된 텍스트만 출력하고, 추가 설명은 하지 마세요."""
        # ─────────────────────────────────────────────────────────────

        content = [{"type": "text", "text": prompt}]
        for b64 in images_b64:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

        resp = await llm.ainvoke([HumanMessage(content=content)])
        return parse_batch_ocr_response(resp.content, len(image_paths))

    except Exception as e:
        logger.error(f"Error in batch OCR with reference: {e}")
        # 실패 시 개별 처리 폴백
        out: List[str] = []
        for i, p in enumerate(image_paths):
            ref = references[i] if i < len(references) else ""
            out.append(await convert_image_to_text_with_reference(p, ref, config))
        return out


async def convert_images_to_text_batch(image_paths: List[str], config: Dict[str, Any], batch_size: int = 1) -> List[str]:
    """여러 이미지를 배치 크기로 쪼개서 OCR (프롬프트는 내부에서 공통 사용)"""
    batch_size = max(1, min(batch_size, 10))
    results: List[str] = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        if len(batch) == 1:
            results.append(await convert_image_to_text(batch[0], config))
        else:
            results.extend(await convert_multiple_images_to_text(batch, config))
    return results


async def convert_images_to_text_batch_with_reference(
    image_paths: List[str],
    references: List[str],
    config: Dict[str, Any],
    batch_size: int = 1
) -> List[str]:
    """여러 이미지를 배치 크기로 쪼개서 OCR (참고 텍스트 포함)"""
    batch_size = max(1, min(batch_size, 10))
    results: List[str] = []
    for i in range(0, len(image_paths), batch_size):
        b_paths = image_paths[i:i + batch_size]
        b_refs = references[i:i + batch_size] if i + batch_size <= len(references) else references[i:]
        if len(b_paths) == 1:
            ref = b_refs[0] if b_refs else ""
            results.append(await convert_image_to_text_with_reference(b_paths[0], ref, config))
        else:
            results.extend(await convert_multiple_images_to_text_with_reference(b_paths, b_refs, config))
    return results

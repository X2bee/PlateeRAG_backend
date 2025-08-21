# your_package/document_processor/ocr.py
import base64
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("document-processor")

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    LANGCHAIN_OPENAI_AVAILABLE = True
except ImportError:
    LANGCHAIN_OPENAI_AVAILABLE = False

from .config import is_image_text_enabled


# ========================
# Common helpers (DRY)
# ========================

def _ocr_enabled(cfg: Dict[str, Any]) -> bool:
    return is_image_text_enabled(cfg, LANGCHAIN_OPENAI_AVAILABLE)

def _build_llm(cfg: Dict[str, Any], *, temperature_override: Optional[float] = None) -> Tuple[Optional[Any], Optional[str]]:
    provider = cfg.get("provider", "openai")
    if provider not in ("openai", "vllm"):
        return None, f"[이미지 파일: 지원하지 않는 프로바이더 - {provider}]"

    if not LANGCHAIN_OPENAI_AVAILABLE:
        return None, "[이미지 파일: langchain_openai 미설치]"

    model = cfg.get("model", "gpt-4-vision-preview")
    api_key = cfg.get("api_key", "")
    base_url = cfg.get("base_url", "https://api.openai.com/v1")
    temperature = cfg.get("temperature", 0.7 if temperature_override is None else temperature_override)
    try:
        llm = ChatOpenAI(
            model=model,
            openai_api_key=api_key or "dummy",
            base_url=base_url,
            temperature=temperature,
        )
        return llm, None
    except Exception as e:
        logger.error(f"LLM 생성 실패: {e}")
        return None, f"[LLM 생성 실패: {str(e)}]"

def _b64_from_file(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def _b64_from_files(paths: List[str]) -> List[str]:
    return [_b64_from_file(p) for p in paths]

async def _ainvoke_images(llm: Any, prompt: str, images_b64: List[str]) -> str:
    content = [{"type": "text", "text": prompt}]
    for b64 in images_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    resp = await llm.ainvoke([HumanMessage(content=content)])
    return resp.content

def _provider_or_error(cfg: Dict[str, Any]) -> Optional[str]:
    provider = cfg.get("provider", "openai")
    if provider not in ("openai", "vllm"):
        return f"[이미지 파일: 지원하지 않는 프로바이더 - {provider}]"
    return None


# ========================
# Image Merging Function
# ========================

def merge_images_vertically(image_paths: List[str], max_width: int = 2000) -> str:
    """여러 이미지를 세로로 합쳐서 하나의 이미지로 만들기"""
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("PIL(Pillow)이 필요합니다: pip install Pillow")
    
    if not image_paths:
        raise ValueError("이미지 경로가 비어있습니다")
    
    images = []
    for path in image_paths:
        img = Image.open(path)
        # 너무 큰 이미지는 리사이즈
        if img.width > max_width:
            ratio = max_width / img.width
            new_height = int(img.height * ratio)
            img = img.resize((max_width, new_height), Image.LANCZOS)
        images.append(img)
    
    # 최대 너비 계산
    max_img_width = max(img.width for img in images)
    total_height = sum(img.height for img in images)
    
    # 합성 이미지 생성 (흰색 배경)
    merged = Image.new('RGB', (max_img_width, total_height), 'white')
    
    # 이미지들을 세로로 붙이기
    y_offset = 0
    for img in images:
        # 중앙 정렬
        x_offset = (max_img_width - img.width) // 2
        merged.paste(img, (x_offset, y_offset))
        y_offset += img.height
    
    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(suffix='_merged.png', delete=False) as temp_file:
        merged.save(temp_file.name, 'PNG', quality=95)
        return temp_file.name


# ========================
# Prompt Builders (단일 소스)
# ========================

def _prompt_single_text() -> str:
    return (
        "이 이미지를 정확한 텍스트로 변환해주세요. 다음 규칙을 철저히 지켜주세요:\n\n"
        "1. **표 구조 보존**: 표가 있다면 정확한 행과 열 구조를 유지하고, 마크다운 표 형식으로 변환해주세요 또한 병합된 셀의 경우, 각각 해당 사항에 모두 넣어주세요.\n"
        "2. **레이아웃 유지**: 원본의 레이아웃, 들여쓰기, 줄바꿈을 최대한 보존해주세요\n"
        "3. **정확한 텍스트**: 모든 문자, 숫자, 기호를 정확히 인식해주세요\n"
        "4. **구조 정보**: 제목, 부제목, 목록, 단락 구분을 명확히 표현해주세요\n"
        "5. **특수 형식**: 날짜, 금액, 주소, 전화번호 등의 형식을 정확히 유지해주세요\n"
        "6. **슬라이드 구조**: 슬라이드 제목, 내용, 차트/그래프 설명을 구분해주세요\n"
        "7. **섹션 구분**: MarkDown 형식을 철저히 준수해서 섹션 구분을 철저히 나눠주세요.\n"
        "8. **표 구분**: 각 이미지 내에서 표는 [표 구분]으로 명확히 구분하고, 표의 제목, 설명, 표 본체, 텍스트 변환을 모두 포함해주세요\n"
        "9. **언어*** : 한국어/영어가 아닌 문자를 포함하지 않도록 주의해주세요. 한자의 경우 한국어로 넣어주세요.\n\n"
        "반드시 한국어 및 영어로 된 텍스트만 출력하고, 추가 설명은 하지 마세요.\n\n"
        "**출력 예시:**\n"
        "# 계약서\n"
        "[섹션 구분]\n"
        "본 계약은 2024년 5월 2일 체결되었습니다.\n\n"
        "[표 구분]\n"
        "### 지급 조건\n"
        "| 항목 | 금액 | 지급일 |\n"
        "|------|------|--------|\n"
        "| 계약금 | 500만원 | 2024-05-10 |\n"
        "| 중도금 | 1,000만원 | 2024-06-15 |\n"
        "| 잔금 | 1,500만원 | 2024-07-30 |\n"
    )

def _prompt_single_html_with_ref(reference_text: str, use_comment_section_marker: bool) -> str:
    # use_comment_section_marker=True 일 때는 <!-- [section] --> 표기 버전
    section_rule = (
        "7. **섹션 구분**: <!-- [section] -->를 의미론적 구분자로 사용하고 이를 철저히 삽입해서 섹션 구분을 철저히 나눠주세요.\n"
        if use_comment_section_marker else
        "7. **구조화**: 제목(`<h1>-<h6>`), 단락(`<p>`), 목록(`<ul>`, `<ol>`, `<li>`) 등을 적절히 사용해주세요\n"
    )
    return (
        "이 이미지를 정확한 HTML 텍스트로 변환해주세요.\n\n"
        "**🔥 중요: 기계적 파싱 참고 텍스트 활용**\n"
        "아래는 같은 페이지에서 기계적으로 추출된 텍스트입니다. 이를 참고하여 OCR 정확도를 높여주세요:\n"
        f"{reference_text}\n\n"
        "**HTML 변환 규칙:**\n"
        "1. **참고 텍스트 활용**: 위의 참고 텍스트를 활용하여 누락된 단어나 부정확한 인식을 보완해주세요\n"
        "2. **HTML 구조 보존**: 문서의 구조를 semantic HTML로 정확히 표현해주세요\n"
        "3. **표 구조**: 표가 있다면 `<table>`, `<thead>`, `<tbody>`, `<tr>`, `<th>`, `<td>` 태그를 사용하여 정확한 구조로 변환해주세요\n"
        "4. **병합된 셀**: `colspan`, `rowspan` 속성을 사용하여 병합된 셀을 정확히 표현해주세요\n"
        "5. **레이아웃 유지**: 원본의 레이아웃과 계층 구조를 HTML로 보존해주세요\n"
        "6. **정확한 텍스트**: 모든 문자, 숫자, 기호를 정확히 인식해주세요\n"
        f"{section_rule}"
        "8. **특수 형식**: 날짜, 금액, 주소, 전화번호 등의 형식을 정확히 유지해주세요\n"
        "9. **표 구분**: 각 표는 `<div class=\"table-section\">` 으로 감싸서 명확히 구분해주세요\n"
        "10. **언어**: 한국어/영어가 아닌 문자를 포함하지 않도록 주의해주세요. 한자의 경우 한국어로 변환해주세요\n\n"
        "HTML만 출력하고 추가 설명은 하지 마세요."
        "**출력 예시:**\n"
        "<!DOCTYPE html>\n"
        "<html lang='ko'>\n"
        "<head><meta charset='UTF-8'></head>\n"
        "<body>\n"
        "<header><h1>임대차 계약서</h1></header>\n"
        "<main>\n"
        "<section>\n"
        "   <p>본 계약은 2024년 5월 2일 체결되었습니다.</p>\n"
        "</section>\n"
        "<div class='table-section'>\n"
        "   <h3>임대 조건</h3>\n"
        "   <table border='1'>\n"
        "      <tr><th>항목</th><th>금액</th><th>지급일</th></tr>\n"
        "      <tr><td>계약금</td><td>500만원</td><td>2024-05-10</td></tr>\n"
        "   </table>\n"
        "</div>\n"
        "</main>\n"
        "</body>\n"
        "</html>\n"
    )

def _prompt_multi_text(num: int) -> str:
    return (
        f"다음 {num}개의 이미지를 각각 정확한 텍스트로 변환해주세요.\n\n"
        "**🔥 중요: 연속된 페이지의 표 처리 규칙**\n"
        "1. **표 연속성 인식**\n"
        "2. **시기 정보 보존**\n"
        "3. **병합된 셀 완전 복원**\n"
        "4. **지역명 완전 표기**\n\n"
        "**중요한 규칙:**\n"
        "각 이미지의 결과를 아래와 같이 구분:\n\n"
        "=== 이미지 1 ===\n[첫 번째 이미지의 텍스트]\n\n"
        "=== 이미지 2 ===\n[두 번째 이미지의 텍스트]\n\n"
        "…\n\n"
        "**변환 규칙:**\n"
        "1. 표는 마크다운 표로, 병합 내용은 각 행에 반영\n"
        "2. 레이아웃 보존\n"
        "3. 정확한 문자 인식\n"
        "4. 구조 정보 명확화\n"
        "5. 특수 형식 유지\n"
        "6. 슬라이드 구조 구분\n"
        "7. 섹션 구분(마크다운)\n"
        "8. [표 구분] 블록 포함\n"
        "9. 한국어/영어만 사용\n\n"
        "텍스트만 출력하고, 추가 설명은 하지 마세요."
        "**출력 예시:**\n"
        "=== 이미지 1 ===\n"
        "# 회의록\n"
        "[섹션 구분]\n"
        "- 일시: 2024-07-10\n"
        "- 참석자: 김철수, 이영희\n\n"
        "[표 구분]\n"
        "### 안건별 의결 현황\n"
        "| 안건 | 결과 |\n"
        "|------|------|\n"
        "| 예산 승인 | 가결 |\n\n"
        "=== 이미지 2 ===\n"
        "## 부록\n"
        "- 추가 자료는 별도 첨부"
    )

def _prompt_multi_text_with_ref(num: int, references: List[str]) -> str:
    ref_block = "\n**🔥 기계적 파싱 참고 텍스트:**\n" + "\n".join(
        [f"--- 이미지 {i+1} 참고 텍스트 ---\n{(r if (r and str(r).strip()) else '[기계적 파싱으로 추출된 텍스트 없음]')}" for i, r in enumerate(references or [])]
    )
    return (
        f"다음 {num}개의 이미지를 각각 정확한 텍스트로 변환해주세요.\n\n"
        f"{ref_block}\n\n"
        "**🔥 중요: 연속된 페이지의 표 처리 규칙**\n"
        "1. 표 연속성 인식\n2. 시기 정보 보존\n3. 병합된 셀 완전 복원\n4. 지역명 완전 표기\n\n"
        "각 이미지 결과는 `=== 이미지 N ===` 구획으로 나눠서 출력.\n\n"
        "변환 규칙은 단일 멀티 프롬프트와 동일하며, 표 병합 내용은 표 아래에 텍스트로도 완전 기재.\n\n"
        "반드시 한국어/영어만 출력, 추가 설명 금지."
        
    )

def _prompt_pdf_single_md(html_reference: str) -> str:
    
    return (
        "이 PDF 페이지 이미지를 정확한 마크다운으로 변환해주세요.\n\n"
        "**HTML 참고 텍스트:**\n"
        f"{html_reference}\n"
        "**변환 규칙:**\n"
        "1. HTML 참고로 누락/불분명 텍스트 보완\n"
        "2. 이미지 고유 정보(병합, 차트/그래프 설명, 색상/강조/레이아웃, 폰트 계층) 추가\n"
        "3. 완전한 마크다운 문법 사용(#, 표, 목록, 강조, 코드블록)\n"
        "4. 표 완전 복원(병합 반영), 표 제목/설명 포함"
        "반드시 표의 내용은 행과 열 각 셀의 정확한 위치를 준수하여 주세요.\n"
        "5. 논리적 섹션으로 구분\n"
        "6. 한국어/영어만 사용, 한자는 한국어로\n\n"
        "7. 빈셀이나 병합된 셀은 의미적으로 처리하여 셀을 채워 주세요. 예를 들어 A/B셀에 모두 병합된 셀이있고 'A'가 들어가야 함다면 두 셀 모두에 'A'를 넣어주세요.\n"
        "8. 표에 대한 설명은 반드시 표마다 주석으로 추가해주세요. 단 표 내용에 대한 자세한 설명이어야며, 가공하기 위한 과정에 대한 내용이어선 안됩니다."
        "마크다운만 출력하고 추가 설명은 하지 마세요."
        "**출력 예시:**\n"
        "# 보고서 요약\n"
        "- 작성일: 2024-06-01\n\n"
        "## 1. 개요\n"
        "이 문서는 분기별 매출 실적을 정리한 보고서입니다.\n\n"
        "## 2. 매출 현황\n"
        "| 분기 | 매출액(억원) | 증감률 |\n"
        "|------|-------------|--------|\n"
        "| Q1 | 120 | +5% |\n"
        "| Q2 | 135 | +12% |\n\n"
        "**주석) 표 설명**: Q2 매출은 135이며 전분기 대비 12% 상승하였으며 Q1은 120이었으며 전분기 대비 5% 증가"
    )

def _prompt_pdf_multi_md(num: int, html_reference: str) -> str:
    return (
        f"다음 {num}개의 PDF 페이지 이미지를 각각 정확한 마크다운으로 변환해주세요.\n\n"
        "**HTML 참고 텍스트:**\n"
        f"{html_reference}\n\n"
        "**변환 규칙:**\n"
        "1. HTML 참고로 각 이미지의 불분명 텍스트 보완\n"
        "2. 각 이미지에서만 확인 가능한 정보 추가\n"
        "3. 페이지 연속성(표/내용 이어짐) 고려\n"
        "4. 완전한 마크다운 문법\n"
        "5. 각 페이지를 논리적 섹션으로 구분\n\n"
        "6. 빈셀이나 병합된 셀은 의미적으로 처리하여 셀을 채워 주세요. 예를 들어 A/B셀에 모두 병합된 셀이있고 'A'가 들어가야 함다면 두 셀 모두에 'A'를 넣어주세요.\n"
        "7. 표에 대한 설명은 주석으로 추가해주세요. 단 표 내용에 대한 자세한 설명이어야며, 가공하기 위한 과정에 대한 내용이어선 안됩니다."
        "반드시 표의 내용은 행과 열 각 셀의 정확한 위치를 준수하여 주세요."
        "각 결과는 아래 형태로 구분해서 출력:\n"
        "=== 이미지 1 ===\n[마크다운]\n\n=== 이미지 2 ===\n[마크다운]\n\n"
        "추가 설명 없이 마크다운만 출력하세요."
        "**출력 형식 예시:**\n"
        "## 페이지 1\n"
        "# 여신 결재권한 기준표\n\n"
        "| 구분 | 세부분류 | 신용등급 | 금액한도 | 영업점장 | 심사역협의회 | 여신협의회 | 비고 |\n"
        "|------|----------|----------|----------|----------|-------------|------------|------|\n"
        "| 신용대출 | 일반신용 | BBB- | $100이하 | 전결 | - | - | - |\n"
        "| 신용대출 | 일반신용 | BB+ | $50이하 | 전결 | - | - | - |\n\n"
        "**주석) 표 설명**: 이 표는 신용등급별 여신 결재권한을 정의한 것으로, BBB- 등급의 경우 100달러 이하는 영업점장 전결 가능하며...\n\n"
    )

def _prompt_merged_pages_md(html_reference: str, num_pages: int) -> str:
    return (
        f"이 이미지는 PDF 페이지를 세로로 합친 것입니다. "
        "각 페이지를 구분하여 정확한 마크다운으로 변환해주세요.\n\n"
        "**HTML 참고 텍스트:**\n"
        f"{html_reference}\n\n"
        "**변환 규칙:**\n"
        "1. HTML 참고로 각 이미지의 불분명 텍스트 보완\n"
        "2. 각 이미지에서만 확인 가능한 정보 추가\n"
        "3. 페이지 연속성(표/내용 이어짐) 고려\n"
        "4. 완전한 마크다운 문법\n"
        "5. 각 페이지를 논리적 섹션으로 구분\n\n"
        "6. 빈셀이나 병합된 셀은 의미적으로 처리하여 셀을 채워 주세요. 예를 들어 A/B셀에 모두 병합된 셀이있고 'A'가 들어가야 함다면 두 셀 모두에 'A'를 넣어주세요.\n"
        "7. 표에 대한 설명은 주석으로 추가해주세요. 단 표 내용에 대한 자세한 설명이어야며, 가공하기 위한 과정에 대한 내용이어선 안됩니다."
        "반드시 표의 내용은 행과 열 각 셀의 정확한 위치를 준수하여 주세요."
        "각 결과는 아래 형태로 구분해서 출력:\n"
        "=== 이미지 1 ===\n[마크다운]\n\n=== 이미지 2 ===\n[마크다운]\n\n"
        "추가 설명 없이 마크다운만 출력하세요."
        "**출력 형식 예시:**\n"
        "## 페이지 1\n"
        "# 여신 결재권한 기준표\n\n"
        "| 구분 | 세부분류 | 신용등급 | 금액한도 | 영업점장 | 심사역협의회 | 여신협의회 | 비고 |\n"
        "|------|----------|----------|----------|----------|-------------|------------|------|\n"
        "| 신용대출 | 일반신용 | BBB- | $100이하 | 전결 | - | - | - |\n"
        "| 신용대출 | 일반신용 | BB+ | $50이하 | 전결 | - | - | - |\n\n"
        "**주석) 표 설명**: 이 표는 신용등급별 여신 결재권한을 정의한 것으로, BBB- 등급의 경우 100달러 이하는 영업점장 전결 가능하며...\n\n"
    )


# ========================
# Parsing helper (원본 유지)
# ========================

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
        return [response_text for _ in range(expected_count)]


# ========================
# Public APIs (동일 시그니처, 내부 DRY)
# ========================

async def convert_image_to_text(image_path: str, current_config: Dict[str, Any]) -> str:
    if not _ocr_enabled(current_config):
        return "[이미지 파일: 이미지-텍스트 변환이 설정되지 않았습니다]"

    prov_err = _provider_or_error(current_config)
    if prov_err:
        return prov_err

    llm, err = _build_llm(current_config)
    if err:
        return err

    try:
        b64 = _b64_from_file(image_path)
        prompt = _prompt_single_text()
        content = await _ainvoke_images(llm, prompt, [b64])
        logger.info(f"Successfully converted image to text: {Path(image_path).name}")
        return content
    except Exception as e:
        logger.error(f"Error converting image to text {image_path}: {e}")
        return f"[이미지 파일: 텍스트 변환 중 오류 발생 - {str(e)}]"


async def convert_image_to_text_with_reference(image_path: str, reference_text: str, current_config: Dict[str, Any]) -> str:
    if not _ocr_enabled(current_config):
        return "[이미지 파일: 이미지-텍스트 변환이 설정되지 않았습니다]"

    prov_err = _provider_or_error(current_config)
    if prov_err:
        return prov_err

    # 동일 동작: reference_text 유무에 따라 다른 프롬프트를 사용하던 기존 로직을 간소화
    use_comment_marker = not bool(reference_text and reference_text.strip())  # 참이면 <!-- [section] --> 버전
    llm, err = _build_llm(current_config)
    if err:
        return err

    try:
        b64 = _b64_from_file(image_path)
        prompt = _prompt_single_html_with_ref(reference_text or "[참고텍스트 없음]", use_comment_marker)
        return await _ainvoke_images(llm, prompt, [b64])
    except Exception as e:
        logger.error(f"Error converting image to HTML {image_path}: {e}")
        return f"[이미지 파일: 텍스트 변환 중 오류 발생 - {str(e)}]"


async def convert_multiple_images_to_text(image_paths: List[str], config: Dict[str, Any]) -> List[str]:
    if not _ocr_enabled(config):
        return ["[이미지 파일: 이미지-텍스트 변환이 설정되지 않았습니다]" for _ in image_paths]

    prov_err = _provider_or_error(config)
    if prov_err:
        return [prov_err for _ in image_paths]

    try:
        images_b64 = _b64_from_files(image_paths)
        llm, err = _build_llm(config)
        if err:
            return [err for _ in image_paths]

        prompt = _prompt_multi_text(len(image_paths))
        resp = await _ainvoke_images(llm, prompt, images_b64)
        return parse_batch_ocr_response(resp, len(image_paths))
    except Exception as e:
        logger.error(f"Error in batch OCR: {e}")
        # 폴백: 단건 처리
        out: List[str] = []
        for p in image_paths:
            out.append(await convert_image_to_text(p, config))
        return out


async def convert_multiple_images_to_text_with_reference(image_paths: List[str], references: List[str], config: Dict[str, Any]) -> List[str]:
    if not _ocr_enabled(config):
        return ["[이미지 파일: 이미지-텍스트 변환이 설정되지 않았습니다]" for _ in image_paths]

    prov_err = _provider_or_error(config)
    if prov_err:
        return [prov_err for _ in image_paths]

    try:
        images_b64 = _b64_from_files(image_paths)
        llm, err = _build_llm(config)
        if err:
            return [err for _ in image_paths]

        prompt = _prompt_multi_text_with_ref(len(image_paths), references or [])
        resp = await _ainvoke_images(llm, prompt, images_b64)
        return parse_batch_ocr_response(resp, len(image_paths))
    except Exception as e:
        logger.error(f"Error in batch OCR with reference: {e}")
        # 폴백: 단건+참고
        out: List[str] = []
        for i, p in enumerate(image_paths):
            ref = references[i] if i < len(references) else ""
            out.append(await convert_image_to_text_with_reference(p, ref, config))
        return out


async def convert_images_to_text_batch(image_paths: List[str], config: Dict[str, Any], batch_size: int = 1) -> List[str]:
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


# ========================
# PDF → Markdown (수정된 메인 함수)
# ========================

async def convert_pdf_to_markdown_with_html_reference(pdf_path: str, html_reference: str, current_config: Dict[str, Any]) -> str:
    if not _ocr_enabled(current_config):
        return "[PDF 변환 실패: OCR 설정 없음]"

    try:
        from pdf2image import convert_from_path
        from PIL import Image
    except Exception as e:
        logger.error(f"pdf2image 또는 PIL 미설치 또는 오류: {e}")
        return "[PDF 처리 실패: pdf2image와 PIL이 필요합니다]"

    try:
        images = convert_from_path(pdf_path, dpi=300)
        image_paths: List[str] = []
        for i, image in enumerate(images):
            with tempfile.NamedTemporaryFile(suffix=f'_page_{i+1}.png', delete=False) as temp_img:
                image.save(temp_img.name, 'PNG')
                image_paths.append(temp_img.name)

        if not image_paths:
            return "[PDF 처리 실패: 이미지 추출 불가]"

        try:
            # 3페이지 이하면 수직으로 합쳐서 처리
            if len(image_paths) <= 3:
                logger.info(f"페이지 수({len(image_paths)})가 3 이하이므로 이미지를 합쳐서 처리합니다.")
                merged_path = None
                try:
                    merged_path = merge_images_vertically(image_paths)
                    
                    # 합쳐진 이미지로 OCR 처리
                    llm, err = _build_llm(current_config, temperature_override=0.1)
                    if err:
                        return err
                    
                    b64 = _b64_from_file(merged_path)
                    prompt = _prompt_merged_pages_md(html_reference, len(image_paths))

                    logger.info(f"합쳐진 이미지 OCR 시작: {prompt}페이지")
                    result = await _ainvoke_images(llm, prompt, [b64])
                    
                    logger.info(f"합쳐진 이미지 OCR 결과: {result}")
                    logger.info(f"합쳐진 이미지 OCR 완료: {len(image_paths)}페이지")
                    return result if result and not str(result).startswith("[이미지 파일:") else "[마크다운 변환 실패]"
                    
                except Exception as e:
                    logger.error(f"이미지 합치기 실패, 개별 처리로 폴백: {e}")
                    # 실패시 기존 방식으로 폴백
                    batch_size = current_config.get('batch_size', 2)
                    if batch_size == 1:
                        # batch_size가 1일 때만 배치 처리
                        page_texts = await convert_pdf_images_to_markdown_batch(
                            image_paths, html_reference, current_config, batch_size
                        )
                    else:
                        # batch_size가 1이 아니면 개별 처리
                        page_texts = []
                        for path in image_paths:
                            text = await convert_single_pdf_image_to_markdown_with_html(path, html_reference, current_config)
                            page_texts.append(text)
                    
                    all_md = []
                    for i, text in enumerate(page_texts):
                        if text and not str(text).startswith("[이미지 파일:"):
                            all_md.append(f"\n## 페이지 {i+1}\n\n{text}\n")
                    return "".join(all_md) if "".join(all_md).strip() else "[마크다운 변환 실패]"
                finally:
                    # 합쳐진 이미지 삭제
                    if merged_path:
                        try:
                            os.unlink(merged_path)
                        except Exception:
                            pass
            else:
                # 4페이지 이상이면서 batch_size가 1일 때만 배치 처리
                batch_size = current_config.get('batch_size', 2)
                logger.info(f"페이지 수({len(image_paths)})가 3 초과입니다. batch_size: {batch_size}")
                
                if batch_size == 1:
                    # batch_size가 1일 때만 배치 처리
                    logger.info("batch_size가 1이므로 배치 처리합니다.")
                    page_texts = await convert_pdf_images_to_markdown_batch(
                        image_paths, html_reference, current_config, batch_size
                    )
                else:
                    # batch_size가 1이 아니면 개별 처리
                    logger.info("batch_size가 1이 아니므로 개별 처리합니다.")
                    page_texts = []
                    for path in image_paths:
                        text = await convert_single_pdf_image_to_markdown_with_html(path, html_reference, current_config)
                        page_texts.append(text)
                
                all_md = []
                for i, text in enumerate(page_texts):
                    if text and not str(text).startswith("[이미지 파일:"):
                        all_md.append(f"\n## 페이지 {i+1}\n\n{text}\n")
                return "".join(all_md) if "".join(all_md).strip() else "[마크다운 변환 실패]"
                
        finally:
            # 원본 페이지 이미지들 삭제
            for p in image_paths:
                try:
                    os.unlink(p)
                except Exception:
                    pass
    except Exception as e:
        logger.error(f"PDF to markdown error: {e}")
        return f"[PDF 마크다운 변환 오류: {str(e)}]"


async def convert_pdf_images_to_markdown_batch(image_paths: List[str], html_reference: str, config: Dict[str, Any], batch_size: int = 1) -> List[str]:
    batch_size = max(1, min(batch_size, 10))
    results: List[str] = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        if len(batch) == 1:
            results.append(await convert_single_pdf_image_to_markdown_with_html(batch[0], html_reference, config))
        else:
            results.extend(await convert_multiple_pdf_images_to_markdown_with_html(batch, html_reference, config))
    return results


async def convert_single_pdf_image_to_markdown_with_html(image_path: str, html_reference: str, config: Dict[str, Any]) -> str:
    if not _ocr_enabled(config):
        return "[이미지 변환 실패: OCR 설정 없음]"

    prov_err = _provider_or_error(config)
    if prov_err:
        return prov_err

    llm, err = _build_llm(config, temperature_override=0.1)
    if err:
        return err

    try:
        b64 = _b64_from_file(image_path)
        prompt = _prompt_pdf_single_md(html_reference)
        return await _ainvoke_images(llm, prompt, [b64])
    except Exception as e:
        logger.error(f"Single image markdown conversion error: {e}")
        return f"[이미지 마크다운 변환 오류: {str(e)}]"


async def convert_multiple_pdf_images_to_markdown_with_html(image_paths: List[str], html_reference: str, config: Dict[str, Any]) -> List[str]:
    if not _ocr_enabled(config):
        return ["[배치 변환 실패: OCR 설정 없음]" for _ in image_paths]

    prov_err = _provider_or_error(config)
    if prov_err:
        return [prov_err for _ in image_paths]

    llm, err = _build_llm(config, temperature_override=0.1)
    if err:
        return [err for _ in image_paths]

    try:
        images_b64 = _b64_from_files(image_paths)
        prompt = _prompt_pdf_multi_md(len(image_paths), html_reference)
        resp = await _ainvoke_images(llm, prompt, images_b64)
        return parse_batch_ocr_response(resp, len(image_paths))
    except Exception as e:
        logger.error(f"Batch markdown conversion error: {e}")
        return [f"[배치 마크다운 변환 오류: {str(e)}]" for _ in image_paths]

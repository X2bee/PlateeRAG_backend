# your_package/document_processor/chunking.py
import logging, re
from typing import Any, Dict, List
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

logger = logging.getLogger("document-processor")

LANGCHAIN_CODE_LANGUAGE_MAP = {
    'py': Language.PYTHON, 'js': Language.JS, 'ts': Language.TS,
    'java': Language.JAVA, 'cpp': Language.CPP, 'c': Language.CPP,
    'cs': Language.CSHARP, 'go': Language.GO, 'rs': Language.RUST,
    'php': Language.PHP, 'rb': Language.RUBY, 'swift': Language.SWIFT,
    'kt': Language.KOTLIN, 'scala': Language.SCALA,
    'html': Language.HTML, 'jsx': Language.JS, 'tsx': Language.TS,
}

def split_text_preserving_html_blocks(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if not text or not text.strip():
        logger.warning("Empty text provided for chunking")
        return [""]

    html_chunks = []
    html_code_pattern = r'```html\s*(.*?)\s*```'
    table_section_pattern = r'<div class="table-section">(.*?)</div>'
    combined_pattern = f'({html_code_pattern}|{table_section_pattern})'

    matches = list(re.finditer(combined_pattern, text, re.DOTALL))
    if matches:
        current_pos = 0
        for m in matches:
            s, e = m.span()
            before = text[current_pos:s].strip()
            if before:
                html_chunks.append(('text', before))
            html_chunks.append(('html', text[s:e]))
            current_pos = e
        after = text[current_pos:].strip()
        if after:
            html_chunks.append(('text', after))
    else:
        html_chunks = [('text', text)]

    final_chunks: List[str] = []
    for kind, content in html_chunks:
        if kind == 'html':
            final_chunks.append(content)
            continue

        section_markers = ["[색션 구분]", "[표 구분]", "[섹션 구분]",
                           "<!-- [섹션 구분] -->", "<!-- [표 구분] -->", "<!-- [section] -->",
                           '<div class="table-section">']
        if any(m in content for m in section_markers):
            pattern = r'(\[색션 구분\]|\[표 구분\]|\[섹션 구분\]|<!-- \[섹션 구분\] -->|<!-- \[표 구분\] -->|<!-- \[section\] -->|<div class="table-section">)'
            parts = re.split(pattern, content)
            majors = [p.strip() for p in parts if p.strip() and not any(m in p for m in section_markers)]

            merged: List[str] = []
            current = ""
            prev_tail = ""
            for sec in majors:
                pot = (current + "\n\n" + sec) if current else sec
                if len(pot) <= chunk_size:
                    current = pot
                else:
                    if current:
                        prev_tail = current[-chunk_overlap:] if len(current) > chunk_overlap else current
                        merged.append(current)
                        current = (prev_tail + "\n\n" + sec) if prev_tail else sec
                    else:
                        current = sec
            if current:
                if prev_tail and len(merged) > 0 and not current.startswith(prev_tail):
                    current = prev_tail + "\n\n" + current
                merged.append(current)

            for sec in merged:
                if len(sec) > chunk_size * 2:
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                        length_function=len, separators=["\n\n", "\n", " ", ""]
                    )
                    final_chunks.extend(splitter.split_text(sec))
                else:
                    final_chunks.append(sec)
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                length_function=len, separators=["\n\n", "\n", " ", ""]
            )
            final_chunks.extend(splitter.split_text(content))

    # 빈 문자열이나 페이지 마커만 있는 청크 제거
    cleaned_chunks = []
    for c in final_chunks:
        if not c.strip():
            continue
        
        # 기존 === 형태와 새로운 XML 태그 형태 모두 처리
        page_marker_patterns = [
            r"===\s*(페이지|슬라이드)\s*\d+(\s*\(OCR[+참고]*\))?\s*===",  # 기존 === 형태
            r"<(페이지\s*번호|슬라이드\s*번호)>\s*\d+\s*(?:\(OCR\))?\s*</(페이지\s*번호|슬라이드\s*번호)>"  # XML 태그 형태
        ]
        
        # 페이지/슬라이드 마커만 있는 청크인지 확인
        is_page_marker_only = False
        for pattern in page_marker_patterns:
            if re.fullmatch(pattern, c.strip()):
                is_page_marker_only = True
                break
        
        if not is_page_marker_only:
            cleaned_chunks.append(c)
        
    logger.info(f"Final text split into {len(cleaned_chunks)} chunks (after cleaning)")
    return cleaned_chunks
    
def reconstruct_text_from_chunks(chunks: List[str], chunk_overlap: int) -> str:
    if not chunks: return ""
    if len(chunks) == 1: return chunks[0]
    out = chunks[0]
    for i in range(1, len(chunks)):
        prev = chunks[i-1]; cur = chunks[i]
        ov = find_overlap_length(prev, cur, chunk_overlap)
        out += cur[ov:] if ov > 0 else cur
    return out

def find_overlap_length(c1: str, c2: str, max_overlap: int) -> int:
    max_check = min(len(c1), len(c2), max_overlap)
    for ov in range(max_check, 0, -1):
        if c1[-ov:] == c2[:ov]:
            return ov
    return 0

def chunk_code_text(text: str, file_type: str, chunk_size: int = 1500, chunk_overlap: int = 300) -> List[str]:
    if not text or not text.strip():
        return [""]
    lang = LANGCHAIN_CODE_LANGUAGE_MAP.get(file_type.lower())
    if lang:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=lang, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            length_function=len, separators=["\n\n", "\n", " ", ""]
        )
    chunks = splitter.split_text(text)
    logger.info(f"Code text split into {len(chunks)} chunks (size: {chunk_size}, overlap: {chunk_overlap})")
    return chunks

def estimate_chunks_count(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> int:
    if not text: return 0
    if len(text) <= chunk_size: return 1
    eff = chunk_size - chunk_overlap
    return max(1, (len(text) - chunk_overlap) // eff + 1)

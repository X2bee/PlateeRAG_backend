# your_package/document_processor/utils.py
import logging, re, bisect
from typing import Any, Dict, List, Optional

logger = logging.getLogger("document-processor")

def clean_text(text: Optional[str]) -> str:
   if not text:
       return ""
   text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
   return text.strip()

def clean_code_text(text: str) -> str:
    if not text:
        return ""
    text = text.rstrip().replace('\t', '    ')
    return text

def is_text_quality_sufficient(text: Optional[str], min_chars: int = 500, min_word_ratio: float = 0.6) -> bool:
    try:
        if not text or len(text) < min_chars:
            return False
        word_chars = re.findall(r"[\w\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF]", text)
        ratio = len(word_chars) / max(1, len(text))
        return ratio >= min_word_ratio
    except Exception:
        return False

def find_chunk_position(chunk: str, full_text: str, start_pos: int = 0) -> int:
    try:
        pos = full_text.find(chunk, start_pos)
        if pos != -1:
            return pos
        lines = chunk.strip().split('\n')
        if lines and len(lines[0]) >= 10:
            first_line = lines[0].strip()
            pos = full_text.find(first_line, start_pos)
            if pos != -1:
                chunk_start = full_text.find(chunk[:50] if len(chunk) > 50 else chunk, pos)
                return chunk_start if chunk_start != -1 else pos
        if len(chunk.strip()) >= 10:
            start = chunk.strip()[:50]
            pos = full_text.find(start, start_pos)
            if pos != -1:
                return pos
        return -1
    except Exception:
        return -1

def build_line_starts(text: str) -> List[int]:
    try:
        starts = [0]
        for i, ch in enumerate(text):
            if ch == '\n' and i + 1 < len(text):
                starts.append(i + 1)
        return starts
    except Exception:
        return [0]

def pos_to_line(pos: int, line_starts: List[int]) -> int:
    try:
        if pos < 0:
            return 1
        idx = bisect.bisect_right(line_starts, pos) - 1
        return max(1, idx + 1)
    except Exception:
        return 1

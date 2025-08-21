# your_package/document_processor/pdf_handler.py
import logging, asyncio, os, tempfile
from typing import Any, Dict, List, Optional
from pathlib import Path

import PyPDF2
from .utils import clean_text, is_text_quality_sufficient
from .ocr import convert_images_to_text_batch_with_reference
from .config import is_image_text_enabled

logger = logging.getLogger("document-processor")

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except Exception:
    PYMUPDF_AVAILABLE = False

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text, extract_pages
    from pdfminer.layout import LTTextContainer, LAParams
    PDFMINER_AVAILABLE = True
except Exception:
    PDFMINER_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except Exception:
    PDFPLUMBER_AVAILABLE = False

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except Exception:
    PDF2IMAGE_AVAILABLE = False

async def extract_text_pages_for_reference(file_path: str) -> List[str]:
    try:
        if PYMUPDF_AVAILABLE:
            doc = fitz.open(file_path)
            pages = []
            for i in range(len(doc)):
                p = doc.load_page(i)
                pages.append((p.get_text("text") or "").strip())
            doc.close()
            return pages
        # fallback
        out = []
        with open(file_path, 'rb') as f:
            r = PyPDF2.PdfReader(f)
            for p in r.pages:
                t = p.extract_text()
                out.append((t or "").strip())
        return out
    except Exception:
        return []

def extract_text_from_pdf_fitz(file_path: str) -> str:
    doc = fitz.open(file_path)
    all_text = ""
    for i in range(len(doc)):
        p = doc.load_page(i)
        t = p.get_text("text")
        all_text += f"\n=== 페이지 {i+1} ===\n{t}"
        if not str(t).endswith("\n"):
            all_text += "\n"
    doc.close()
    return all_text

def extract_text_from_pdf_layout(file_path: str) -> Optional[str]:
    try:
        all_text = ""
        laparams = LAParams()
        for page_num, page_layout in enumerate(extract_pages(file_path, laparams=laparams)):
            lines = []
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    try:
                        text_block = element.get_text()
                    except Exception:
                        text_block = ""
                    if text_block and text_block.strip():
                        for ln in text_block.rstrip('\n').splitlines():
                            lines.append(ln)
            if lines:
                all_text += f"\n=== 페이지 {page_num+1} ===\n"
                all_text += "\n".join(lines) + "\n"
        return all_text if all_text.strip() else None
    except Exception:
        return None

async def extract_pages_pdfminer(file_path: str, num_pages: int, max_workers: int = 4) -> List[Optional[str]]:
    from pdfminer.high_level import extract_text as _extract_text
    async def _single(pn: int) -> Optional[str]:
        try:
            return _extract_text(file_path, page_numbers={pn})
        except Exception:
            return None
    results: List[Optional[str]] = [None]*num_pages
    for start in range(0, num_pages, max_workers):
        batch = list(range(start, min(start+max_workers, num_pages)))
        done = await asyncio.gather(*[asyncio.to_thread(_single, p) for p in batch])
        for idx, val in zip(batch, done):
            results[idx] = val
    return results

async def extract_text_from_pdf_fallback(file_path: str) -> str:
    text = ""
    with open(file_path, 'rb') as f:
        r = PyPDF2.PdfReader(f)
        for i, page in enumerate(r.pages):
            t = page.extract_text()
            if t:
                text += f"\n=== 페이지 {i+1} ===\n{t}\n"
    return clean_text(text)

async def extract_text_from_pdf_via_ocr(file_path: str, current_config: Dict[str, Any]) -> str:
    if not PDF2IMAGE_AVAILABLE:
        return "[PDF 파일: pdf2image 라이브러리가 필요합니다]"
    return extract_text_from_pdf_fallback(file_path)
    '''
    extracted_refs = await extract_text_pages_for_reference(file_path)
    images = convert_from_path(file_path, dpi=300)

    temp_files: List[str] = []
    try:
        for img in images:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tf:
                img.save(tf.name, 'PNG')
                temp_files.append(tf.name)
        batch_size = current_config.get('batch_size', 1)
        page_texts = await convert_images_to_text_batch_with_reference(
            temp_files, extracted_refs, current_config, batch_size
        )
        all_text = ""
        for i, t in enumerate(page_texts):
            if not str(t).startswith("[이미지 파일:"):
                all_text += f"\n=== 페이지 {i+1} (OCR+참고) ===\n{t}\n"
        return clean_text(all_text) if all_text.strip() else await extract_text_from_pdf_fallback(file_path)
    finally:
        for p in temp_files:
            try: os.unlink(p)
            except: pass'''
            
async def extract_text_from_pdf(file_path: str, current_config: Dict[str, Any]) -> str:
    provider = current_config.get('provider', 'no_model')
    logger.info(f"🔄 Real-time PDF processing with provider: {provider}")

    if provider == 'no_model':
        if PYMUPDF_AVAILABLE:
            try:
                fitz_text = extract_text_from_pdf_fitz(file_path)
                if fitz_text and fitz_text.strip():
                    if not is_text_quality_sufficient(fitz_text):
                        if PDFPLUMBER_AVAILABLE:
                            try:
                                with pdfplumber.open(file_path) as pdf:
                                    pages = [p.extract_text() or "" for p in pdf.pages]
                                pb_text = "\n".join(pages).strip()
                                if pb_text and is_text_quality_sufficient(pb_text):
                                    return clean_text(pb_text)
                            except Exception:
                                pass
                        if PDFMINER_AVAILABLE:
                            try:
                                layout_text = await asyncio.to_thread(extract_text_from_pdf_layout, file_path)
                                if layout_text and is_text_quality_sufficient(layout_text):
                                    return clean_text(layout_text)
                            except Exception:
                                pass
                    if is_text_quality_sufficient(fitz_text):
                        return fitz_text
            except Exception:
                pass

        if PDFMINER_AVAILABLE:
            try:
                layout_text = await asyncio.to_thread(extract_text_from_pdf_layout, file_path)
                if layout_text and layout_text.strip():
                    cleaned = clean_text(layout_text)
                    if cleaned.strip():
                        return cleaned

                # per-page
                try:
                    r = PyPDF2.PdfReader(file_path)
                    num_pages = len(r.pages)
                except Exception:
                    num_pages = None

                if num_pages and num_pages > 0:
                    pts = await extract_pages_pdfminer(file_path, num_pages, max_workers=2)
                    combined = ""
                    for i, t in enumerate(pts):
                        if t and t.strip():
                            combined += f"\n=== 페이지 {i+1} ===\n{t}\n"
                    cleaned = clean_text(combined)
                    if cleaned.strip():
                        return cleaned

                text = pdfminer_extract_text(file_path)
                cleaned = clean_text(text)
                if len(cleaned) > 100:
                    return cleaned
            except Exception:
                pass

        if PYMUPDF_AVAILABLE:
            try:
                fitz_text = await asyncio.to_thread(extract_text_from_pdf_fitz, file_path)
                if fitz_text and fitz_text.strip():
                    return fitz_text
            except Exception:
                pass
        return await extract_text_from_pdf_fallback(file_path)

    # OCR 모드
    return await extract_text_from_pdf_via_ocr(file_path, current_config)

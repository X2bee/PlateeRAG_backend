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
    """참조용 페이지별 텍스트 추출"""
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
    """PyMuPDF로 텍스트 추출"""
    doc = fitz.open(file_path)
    all_text = ""
    for i in range(len(doc)):
        p = doc.load_page(i)
        t = p.get_text("text")
        
        # 텍스트 앞부분에서 페이지 번호 제거
        page_num = str(i + 1)
        if t.strip().startswith(page_num):
            # 페이지 번호 다음에 공백이나 개행이 있는지 확인하고 제거
            remaining_text = t.strip()[len(page_num):].lstrip()
            t = remaining_text
        
        all_text += f"\n<페이지 번호> {i+1} </페이지 번호>\n{t}"
        if not str(t).endswith("\n"):
            all_text += "\n"
    doc.close()
    return all_text

def extract_text_from_pdf_layout(file_path: str) -> Optional[str]:
    """pdfminer layout 분석으로 텍스트 추출"""
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
                all_text += f"\n<페이지 번호> {page_num+1} </페이지 번호>\n"
                all_text += "\n".join(lines) + "\n"
        return all_text if all_text.strip() else None
    except Exception:
        return None

async def extract_pages_pdfminer(file_path: str, num_pages: int, max_workers: int = 4) -> List[Optional[str]]:
    """pdfminer로 페이지별 병렬 처리"""
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
    """PyPDF2 fallback 처리"""
    text = ""
    with open(file_path, 'rb') as f:
        r = PyPDF2.PdfReader(f)
        for i, page in enumerate(r.pages):
            t = page.extract_text()
            if t:
                text += f"\n<페이지 번호> {i+1} </페이지 번호>\n{t}\n"
    return clean_text(text)

async def _extract_pdf_text_only(file_path: str) -> str:
    """기계적 텍스트 추출만 사용 (OCR 없이)"""
    logger.info("PDF: Using text-only extraction methods")
    
    # PyMuPDF 우선 시도
    if PYMUPDF_AVAILABLE:
        try:
            fitz_text = extract_text_from_pdf_fitz(file_path)
            if fitz_text and fitz_text.strip():
                if is_text_quality_sufficient(fitz_text):
                    logger.info("PDF: Successful extraction with PyMuPDF")
                    return fitz_text
        except Exception:
            pass

    # pdfplumber 시도
    if PDFPLUMBER_AVAILABLE:
        try:
            with pdfplumber.open(file_path) as pdf:
                all_text = ""
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        all_text += f"\n<페이지 번호> {i+1} </페이지 번호>\n{text}\n"
            if all_text.strip() and is_text_quality_sufficient(all_text):
                logger.info("PDF: Successful extraction with pdfplumber")
                return clean_text(all_text)
        except Exception:
            pass

    # pdfminer layout 시도
    if PDFMINER_AVAILABLE:
        try:
            layout_text = await asyncio.to_thread(extract_text_from_pdf_layout, file_path)
            if layout_text and layout_text.strip():
                cleaned = clean_text(layout_text)
                if cleaned.strip() and is_text_quality_sufficient(cleaned):
                    logger.info("PDF: Successful extraction with pdfminer layout")
                    return cleaned
        except Exception:
            pass

    # pdfminer 전체 추출 시도
    if PDFMINER_AVAILABLE:
        try:
            text = pdfminer_extract_text(file_path)
            cleaned = clean_text(text)
            if len(cleaned) > 100:
                logger.info("PDF: Successful extraction with pdfminer")
                return cleaned
        except Exception:
            pass

    # 최후 수단: PyPDF2
    logger.warning("PDF: Falling back to PyPDF2")
    return await extract_text_from_pdf_fallback(file_path)

async def extract_text_from_pdf_via_ocr(file_path: str, current_config: Dict[str, Any]) -> str:
    """OCR 기반 텍스트 추출"""
    if not is_image_text_enabled(current_config, True):
        logger.warning("PDF: OCR requested but not enabled, falling back to text extraction")
        return await _extract_pdf_text_only(file_path)
    
    if not PDF2IMAGE_AVAILABLE:
        logger.error("PDF: OCR requested but pdf2image not available")
        return await _extract_pdf_text_only(file_path)
    
    logger.info("PDF: Starting OCR processing")
    
    try:
        # 참조용 텍스트 추출
        extracted_refs = await extract_text_pages_for_reference(file_path)
        
        # PDF를 이미지로 변환
        images = convert_from_path(file_path, dpi=300)
        
        temp_files: List[str] = []
        try:
            # 임시 이미지 파일 생성
            for img in images:
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tf:
                    img.save(tf.name, 'PNG')
                    temp_files.append(tf.name)
            
            # OCR 처리
            batch_size = current_config.get('batch_size', 1)
            page_texts = await convert_images_to_text_batch_with_reference(
                temp_files, extracted_refs, current_config, batch_size
            )
            
            # 결과 조합
            all_text = ""
            for i, t in enumerate(page_texts):
                if not str(t).startswith("[이미지 파일:"):
                    all_text += f"\n<페이지 번호> {i+1} (OCR) </페이지 번호>\n{t}\n"
            
            if all_text.strip():
                logger.info(f"PDF: OCR processing completed for {len(page_texts)} pages")
                return clean_text(all_text)
            else:
                logger.warning("PDF: OCR failed, falling back to text extraction")
                return await _extract_pdf_text_only(file_path)
                
        finally:
            # 임시 파일 정리
            for p in temp_files:
                try: 
                    os.unlink(p)
                except: 
                    pass
                    
    except Exception as e:
        logger.error(f"PDF: OCR processing failed: {e}, falling back to text extraction")
        return await _extract_pdf_text_only(file_path)

async def _extract_pdf_default(file_path: str, current_config: Dict[str, Any]) -> str:
    """기존 PDF 처리 로직 (자동 선택)"""
    provider = current_config.get('provider', 'no_model')
    
    if provider == 'no_model':
        # PyMuPDF 우선 시도
        if PYMUPDF_AVAILABLE:
            try:
                fitz_text = extract_text_from_pdf_fitz(file_path)
                if fitz_text and fitz_text.strip():
                    # 텍스트 품질 검사
                    if not is_text_quality_sufficient(fitz_text):
                        # pdfplumber로 재시도
                        if PDFPLUMBER_AVAILABLE:
                            try:
                                with pdfplumber.open(file_path) as pdf:
                                    pages = [p.extract_text() or "" for p in pdf.pages]
                                pb_text = "\n".join(pages).strip()
                                if pb_text and is_text_quality_sufficient(pb_text):
                                    return clean_text(pb_text)
                            except Exception:
                                pass
                        
                        # pdfminer layout으로 재시도
                        if PDFMINER_AVAILABLE:
                            try:
                                layout_text = await asyncio.to_thread(extract_text_from_pdf_layout, file_path)
                                if layout_text and is_text_quality_sufficient(layout_text):
                                    return clean_text(layout_text)
                            except Exception:
                                pass
                    
                    # fitz 결과가 충분한 품질이면 사용
                    if is_text_quality_sufficient(fitz_text):
                        return fitz_text
            except Exception:
                pass

        # pdfminer 시도
        if PDFMINER_AVAILABLE:
            try:
                # layout 방식 우선
                layout_text = await asyncio.to_thread(extract_text_from_pdf_layout, file_path)
                if layout_text and layout_text.strip():
                    cleaned = clean_text(layout_text)
                    if cleaned.strip():
                        return cleaned

                # 페이지별 처리
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
                            combined += f"\n<페이지 번호> {i+1} </페이지 번호>\n{t}\n"
                    cleaned = clean_text(combined)
                    if cleaned.strip():
                        return cleaned

                # 전체 문서 추출
                text = pdfminer_extract_text(file_path)
                cleaned = clean_text(text)
                if len(cleaned) > 100:
                    return cleaned
            except Exception:
                pass

        # PyMuPDF 재시도 (위에서 실패했을 경우)
        if PYMUPDF_AVAILABLE:
            try:
                fitz_text = await asyncio.to_thread(extract_text_from_pdf_fitz, file_path)
                if fitz_text and fitz_text.strip():
                    return fitz_text
            except Exception:
                pass
        
        # 최후 수단: PyPDF2 fallback
        return await extract_text_from_pdf_fallback(file_path)

    else:
        # OCR 모드 (provider가 no_model이 아닌 경우)
        return await extract_text_from_pdf_via_ocr(file_path, current_config)

async def extract_text_from_pdf(file_path: str, current_config: Dict[str, Any], process_type: str = "default") -> str:
    """PDF 텍스트 추출 메인 함수"""
    provider = current_config.get('provider', 'no_model')
    logger.info(f"Real-time PDF processing with provider: {provider}, process_type: {process_type}")

    if process_type == "text":
        # 기계적 텍스트 추출 (OCR 없이)
        logger.info("PDF text extraction processing requested")
        return await _extract_pdf_text_only(file_path)
    
    elif process_type == "ocr":
        # OCR 강제 사용
        logger.info("PDF OCR processing requested")
        return await extract_text_from_pdf_via_ocr(file_path, current_config)
    
    else:  # process_type == "default"
        # 기존 자동 선택 로직 유지
        return await _extract_pdf_default(file_path, current_config)
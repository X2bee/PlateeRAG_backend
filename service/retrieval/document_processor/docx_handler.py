# your_package/document_processor/docx_handler.py
import logging, os, tempfile, asyncio
from typing import Any, Dict, List, Set
from pathlib import Path
from docx import Document

from .utils import clean_text
from .ocr import convert_images_to_text_batch
from .config import is_image_text_enabled

logger = logging.getLogger("document-processor")

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except Exception:
    PDF2IMAGE_AVAILABLE = False

try:
    from docx2pdf import convert as docx_to_pdf_convert
    DOCX2PDF_AVAILABLE = True
except Exception:
    DOCX2PDF_AVAILABLE = False

def _has_page_break_element(element) -> bool:
    try:
        nsmap = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        if element.findall('.//w:br[@w:type="page"]', nsmap): return True
        if element.findall('.//w:lastRenderedPageBreak', nsmap): return True
        return False
    except Exception:
        return False

def _paragraph_has_page_break(paragraph) -> bool:
    try:
        for run in paragraph.runs:
            if run.element.findall('.//w:br[@w:type="page"]',
                                   {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}):
                return True
        if hasattr(paragraph, '_element'):
            return _has_page_break_element(paragraph._element)
        return False
    except Exception:
        return False

def _extract_paragraph_text(para_element, doc) -> str:
    try:
        text = ""
        nsmap = doc.element.nsmap if hasattr(doc.element, 'nsmap') else {}
        for run in para_element.findall('.//w:r', nsmap):
            for t in run.findall('.//w:t', nsmap):
                if t.text: text += t.text
            for drawing in run.findall('.//w:drawing', nsmap):
                text += " [이미지] "
                try:
                    for desc in drawing.findall('.//wp:docPr', nsmap):
                        if desc.get('descr'): text += f"[설명: {desc.get('descr')}] "
                        elif desc.get('name'): text += f"[이름: {desc.get('name')}] "
                except:
                    pass
        return text
    except Exception:
        try:
            ts = para_element.findall('.//w:t', {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'})
            return ''.join([e.text or '' for e in ts])
        except:
            return para_element.text or ""

def _extract_table_text_xml(table_element) -> str:
    try:
        ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        out = ""
        for row in table_element.findall('.//w:tr', ns):
            row_text = []
            for cell in row.findall('.//w:tc', ns):
                cell_text = ""
                for t in cell.findall('.//w:t', ns):
                    if t.text: cell_text += t.text
                row_text.append(cell_text.strip())
            if any(c.strip() for c in row_text):
                out += " | ".join(row_text) + "\n"
        return out
    except Exception:
        return ""

def _extract_simple_table_text(table) -> str:
    try:
        out = ""
        for row in table.rows:
            row_text = []
            for cell in row.cells:
                row_text.append(cell.text.strip())
            if any(c.strip() for c in row_text):
                out += " | ".join(row_text) + "\n"
        return out
    except Exception:
        return ""

def _is_similar_table_text(t1: str, t2: str, threshold: float = 0.8) -> bool:
    import re
    if not t1 or not t2: return False
    a = re.sub(r'\s+', ' ', t1.strip()); b = re.sub(r'\s+', ' ', t2.strip())
    if a == b: return True
    l1, l2 = len(a), len(b)
    if min(l1, l2)/max(l1, l2) < 0.5: return False
    s = a if l1 < l2 else b
    L = b if l1 < l2 else a
    ratio = len(set(s.split()) & set(L.split())) / len(set(s.split()))
    return ratio >= threshold

async def convert_docx_to_images(file_path: str) -> List[str]:
    temp_files: List[str] = []
    try:
        if DOCX2PDF_AVAILABLE and PDF2IMAGE_AVAILABLE:
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tp:
                docx_to_pdf_convert(file_path, tp.name)
                images = convert_from_path(tp.name, dpi=300)
                for im in images:
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as ti:
                        im.save(ti.name, 'PNG')
                        temp_files.append(ti.name)
                os.unlink(tp.name)
            return temp_files
        # LibreOffice CLI fallback
        elif PDF2IMAGE_AVAILABLE:
            import subprocess
            with tempfile.TemporaryDirectory() as td:
                try:
                    subprocess.run(['libreoffice','--headless','--convert-to','pdf','--outdir', td, file_path],
                                   check=True, capture_output=True)
                    pdf_path = os.path.join(td, Path(file_path).stem + '.pdf')
                    if os.path.exists(pdf_path):
                        images = convert_from_path(pdf_path, dpi=300)
                        for im in images:
                            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as ti:
                                im.save(ti.name, 'PNG')
                                temp_files.append(ti.name)
                        return temp_files
                except Exception:
                    pass
        return []
    except Exception:
        for p in temp_files:
            try: os.unlink(p)
            except: pass
        return []

async def extract_text_from_docx_via_ocr(file_path: str, current_config: Dict[str, Any]) -> str:
    if not is_image_text_enabled(current_config, True):
        return await extract_text_from_docx_fallback(file_path)
    images = await convert_docx_to_images(file_path)
    if not images:
        return await extract_text_from_docx_fallback(file_path)
    try:
        batch_size = current_config.get('batch_size', 1)
        page_texts = await convert_images_to_text_batch(images, current_config, batch_size)
        all_text = ""
        for i, t in enumerate(page_texts):
            if not str(t).startswith("[이미지 파일:"):
                all_text += f"\n=== 페이지 {i+1} (OCR) ===\n{t}\n"
        return clean_text(all_text) if all_text.strip() else await extract_text_from_docx_fallback(file_path)
    finally:
        for p in images:
            try: os.unlink(p)
            except: pass

async def extract_text_from_docx_fallback(file_path: str) -> str:
    doc = Document(file_path)
    text = ""
    processed: Set[str] = set()
    current_page = 1
    try:
        for element in doc.element.body:
            if element.tag.endswith('p'):
                if _has_page_break_element(element):
                    current_page += 1
                    text += f"\n=== 페이지 {current_page} ===\n"
                para_text = _extract_paragraph_text(element, doc)
                if para_text.strip():
                    text += para_text + "\n"
            elif element.tag.endswith('tbl'):
                t = _extract_table_text_xml(element)
                if t.strip():
                    text += "\n=== 표 ===\n" + t + "\n=== 표 끝 ===\n\n"
                    processed.add(t)
    except Exception:
        text = ""
        current_page = 1
        for p in doc.paragraphs:
            if _paragraph_has_page_break(p):
                current_page += 1
                text += f"\n=== 페이지 {current_page} ===\n"
            if p.text.strip():
                text += p.text + "\n"

    for i, table in enumerate(doc.tables):
        t = _extract_simple_table_text(table)
        if t.strip() and not any(_is_similar_table_text(t, x) for x in processed):
            text += f"\n=== 표 {i+1} ===\n{t}\n=== 표 끝 ===\n\n"
            processed.add(t)

    if current_page > 1 and not text.startswith("=== 페이지 1 ==="):
        text = f"=== 페이지 1 ===\n{text}"
    return clean_text(text)

async def extract_text_from_docx(file_path: str, current_config: Dict[str, Any]) -> str:
    provider = current_config.get('provider', 'no_model')
    logger.info(f"🔄 Real-time DOCX processing with provider: {provider}")
    if provider == 'no_model':
        return await extract_text_from_docx_fallback(file_path)
    return await extract_text_from_docx_via_ocr(file_path, current_config)

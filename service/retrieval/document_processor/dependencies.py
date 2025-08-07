"""
Document Processor 의존성 검사 및 관리
"""

import logging

logger = logging.getLogger("document-processor")

# 의존성 체크
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from pdfminer.high_level import extract_text
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
    logger.info("✅ pdf2image available")
except ImportError:
    PDF2IMAGE_AVAILABLE = False

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    LANGCHAIN_OPENAI_AVAILABLE = True
except ImportError:
    LANGCHAIN_OPENAI_AVAILABLE = False

try:
    from docx2pdf import convert as docx_to_pdf_convert
    DOCX2PDF_AVAILABLE = True
    logger.info("✅ docx2pdf available")
except ImportError:
    DOCX2PDF_AVAILABLE = False

try:
    from pptx import Presentation
    PYTHON_PPTX_AVAILABLE = True
    logger.info("✅ python-pptx available")
except ImportError:
    PYTHON_PPTX_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont
    from io import BytesIO
    PIL_AVAILABLE = True
    logger.info("✅ PIL available")
except ImportError:
    PIL_AVAILABLE = False

def log_dependency_warnings():
    """의존성 경고 로그 출력"""
    if not PDFMINER_AVAILABLE:
        logger.warning("pdfminer not available. Using PyPDF2 fallback.")
    if not PDF2IMAGE_AVAILABLE:
        logger.warning("pdf2image not available. OCR disabled.")
    if not DOCX2PDF_AVAILABLE and not PIL_AVAILABLE:
        logger.warning("docx2pdf and PIL not available. DOCX/PPT OCR disabled.")
    if not LANGCHAIN_OPENAI_AVAILABLE:
        logger.warning("langchain_openai not available. Image processing disabled.")
# your_package/document_processor/document_processor.py
import logging, os, re, bisect, asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from .utils import (clean_text, clean_code_text, find_chunk_position, build_line_starts,
                    pos_to_line)
from .chunking import (split_text_preserving_html_blocks, reconstruct_text_from_chunks,
                       find_overlap_length, chunk_code_text, estimate_chunks_count)
from .pdf_handler import extract_text_from_pdf
from .docx_handler import extract_text_from_docx
from .ppt_handler import extract_text_from_ppt
from .table_data_handler import extract_text_from_excel, extract_text_from_csv
from .text_handler import extract_text_from_text_file

logger = logging.getLogger("document-processor")

class DocumentProcessor:
    """
    원본 클래스의 공개 메서드/시그니처/동작을 그대로 유지.
    내부 구현은 모듈로 분리(위임).
    """
    def __init__(self, config_composer = None):
        self.config_composer = config_composer

        # 타입 세트(원본과 동일)
        self.document_types = ['pdf', 'docx', 'doc', 'pptx', 'ppt']
        self.text_types = ['txt','md','markdown','rtf']
        self.code_types = ['py','js','ts','java','cpp','c','h','cs','go','rs',
                           'php','rb','swift','kt','scala','dart','r','sql',
                           'html','css','jsx','tsx','vue','svelte']
        self.config_types = ['json','yaml','yml','xml','toml','ini','cfg','conf','properties','env']
        self.data_types   = ['csv','tsv','xlsx','xls']
        self.script_types = ['sh','bat','ps1','zsh','fish']
        self.log_types    = ['log']
        self.web_types    = ['htm','xhtml']
        self.image_types  = ['jpg','jpeg','png','gif','bmp','webp']

        self.supported_types = ( self.document_types + self.text_types + self.code_types +
                                 self.config_types + self.data_types + self.script_types +
                                 self.log_types + self.web_types + self.image_types )

        # 가용 라이브러리 체크(원본과 동일한 경고/동작)
        try:
            from openpyxl import load_workbook  # noqa
            OPENPYXL_AVAILABLE = True
        except Exception:
            OPENPYXL_AVAILABLE = False
        try:
            import xlrd  # noqa
            XLRD_AVAILABLE = True
        except Exception:
            XLRD_AVAILABLE = False
        try:
            from langchain_openai import ChatOpenAI  # noqa
            LANGCHAIN_OPENAI_AVAILABLE = True
        except Exception:
            LANGCHAIN_OPENAI_AVAILABLE = False
        try:
            from pdfminer.high_level import extract_text  # noqa
            PDFMINER_AVAILABLE = True
        except Exception:
            PDFMINER_AVAILABLE = False
        try:
            from pdf2image import convert_from_path  # noqa
            PDF2IMAGE_AVAILABLE = True
        except Exception:
            PDF2IMAGE_AVAILABLE = False
        try:
            from docx2pdf import convert as docx_to_pdf_convert  # noqa
            DOCX2PDF_AVAILABLE = True
        except Exception:
            DOCX2PDF_AVAILABLE = False
        try:
            from pptx import Presentation  # noqa
            PYTHON_PPTX_AVAILABLE = True
        except Exception:
            PYTHON_PPTX_AVAILABLE = False
        try:
            from PIL import Image  # noqa
            PIL_AVAILABLE = True
        except Exception:
            PIL_AVAILABLE = False

        if not OPENPYXL_AVAILABLE and not XLRD_AVAILABLE:
            self.supported_types = [t for t in self.supported_types if t not in ['xlsx','xls']]
            logger.warning("openpyxl and xlrd not available. Excel processing disabled.")
        if not LANGCHAIN_OPENAI_AVAILABLE:
            self.supported_types = [t for t in self.supported_types if t not in self.image_types]
            logger.warning("langchain_openai not available. Image processing disabled.")
        if not PDFMINER_AVAILABLE:
            logger.warning("pdfminer not available. Using PyPDF2 fallback.")
        if not PDF2IMAGE_AVAILABLE:
            logger.warning("pdf2image not available. OCR disabled.")
        if not DOCX2PDF_AVAILABLE and not PIL_AVAILABLE:
            logger.warning("docx2pdf and PIL not available. DOCX/PPT OCR disabled.")

        self.encodings = ['utf-8','utf-8-sig','cp949','euc-kr','latin-1','ascii']

    # === 기존 private 메서드 대체 ===
    def _get_current_image_text_config(self) -> Dict[str, Any]:
        if self.config_composer:
            document_processor_config = self.config_composer.get_config_by_category_name("document-processor")
            provider = document_processor_config.DOCUMENT_PROCESSOR_IMAGE_TEXT_MODEL_PROVIDER.value

            if provider == "openai":
                config = {
                    'provider': str(document_processor_config.DOCUMENT_PROCESSOR_IMAGE_TEXT_MODEL_PROVIDER.value).lower(),
                    'base_url': str(document_processor_config.DOCUMENT_PROCESSOR_OPENAI_IMAGE_TEXT_BASE_URL.value),
                    'api_key': str(self.config_composer.get_config_by_name("OPENAI_API_KEY").value),
                    'model': str(document_processor_config.DOCUMENT_PROCESSOR_OPENAI_IMAGE_TEXT_MODEL_NAME.value),
                    'temperature': float(document_processor_config.DOCUMENT_PROCESSOR_IMAGE_TEXT_TEMPERATURE.value),
                    'batch_size': int(document_processor_config.DOCUMENT_PROCESSOR_IMAGE_TEXT_BATCH_SIZE.value),
                }

            elif provider == "vllm":
                config = {
                    'provider': str(document_processor_config.DOCUMENT_PROCESSOR_IMAGE_TEXT_MODEL_PROVIDER.value).lower(),
                    'base_url': str(document_processor_config.DOCUMENT_PROCESSOR_VLLM_IMAGE_TEXT_BASE_URL.value),
                    'api_key': str(document_processor_config.DOCUMENT_PROCESSOR_VLLM_IMAGE_TEXT_API_KEY.value),
                    'model': str(document_processor_config.DOCUMENT_PROCESSOR_VLLM_IMAGE_TEXT_MODEL_NAME.value),
                    'temperature': float(document_processor_config.DOCUMENT_PROCESSOR_IMAGE_TEXT_TEMPERATURE.value),
                    'batch_size': int(document_processor_config.DOCUMENT_PROCESSOR_IMAGE_TEXT_BATCH_SIZE.value),
                }

            elif provider == "no_model":
                config = {
                    'provider': str(document_processor_config.DOCUMENT_PROCESSOR_IMAGE_TEXT_MODEL_PROVIDER.value).lower(),
                    'base_url': "",
                    'api_key': "",
                    'model': "",
                    'temperature': 0,
                    'batch_size': 0
                }

            else:
                raise ValueError(f"Unsupported IMAGE_TEXT_MODEL_PROVIDER: {provider}")
        else:
            raise ValueError("Config composer not provided")

        return config

    # def _is_image_text_enabled(self, config: Dict[str, Any]) -> bool:
    #     try:
    #         from langchain_openai import ChatOpenAI  # noqa
    #         langchain_ok = True
    #     except Exception:
    #         langchain_ok = False
    #     return is_image_text_enabled(config, langchain_ok)

    # === 공개 API들: 원본과 동일 시그니처/동작 ===

    def get_supported_types(self) -> List[str]:
        return self.supported_types.copy()

    def get_file_category(self, file_type: str) -> str:
        ft = file_type.lower()
        if ft in self.document_types: return 'document'
        if ft in self.text_types:     return 'text'
        if ft in self.code_types:     return 'code'
        if ft in self.config_types:   return 'config'
        if ft in self.data_types:     return 'data'
        if ft in self.script_types:   return 'script'
        if ft in self.log_types:      return 'log'
        if ft in self.web_types:      return 'web'
        if ft in self.image_types:    return 'image'
        return 'unknown'

    def clean_text(self, text: str) -> str:
        return clean_text(text)

    def _is_text_quality_sufficient(self, text: Optional[str], min_chars: int = 500, min_word_ratio: float = 0.6) -> bool:
        from .utils import is_text_quality_sufficient
        return is_text_quality_sufficient(text, min_chars, min_word_ratio)

    def clean_code_text(self, text: str, file_type: str) -> str:
        return clean_code_text(text)

    async def extract_text_from_file(self, file_path: str, file_extension: str, process_type: str) -> str:
        category = self.get_file_category(file_extension)
        logger.info(f"Extracting text from {file_extension} file ({category} category): {file_path}")
        cfg = self._get_current_image_text_config()

        if file_extension == 'pdf':
            return await extract_text_from_pdf(file_path, cfg, process_type)
        elif file_extension in ['docx', 'doc']:
            return await extract_text_from_docx(file_path, cfg, process_type)
        elif file_extension in ['pptx', 'ppt']:
            return await extract_text_from_ppt(file_path, cfg)
        elif file_extension in ['xlsx', 'xls']:
            return await extract_text_from_excel(file_path)
        elif file_extension in ['csv','tsv']:
            # tsv도 csv reader로 안전하게 처리(원본 동일)
            return await extract_text_from_csv(file_path)
        elif file_extension in self.image_types:
            # 이미지 파일은 단건 OCR(원본과 동일 동작: 내부에서 config 검사)
            from .ocr import convert_image_to_text
            return await convert_image_to_text(file_path, cfg)
        elif file_extension in (self.text_types + self.code_types + self.config_types +
                                self.script_types + self.log_types + self.web_types):
            is_code = file_extension in self.code_types
            return await extract_text_from_text_file(file_path, file_extension, self.encodings, is_code=is_code)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    # === 청킹 관련 ===
    def chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        return split_text_preserving_html_blocks(text, chunk_size, chunk_overlap)

    def _reconstruct_text_from_chunks(self, chunks: List[str], chunk_overlap: int) -> str:
        return reconstruct_text_from_chunks(chunks, chunk_overlap)

    def _find_overlap_length(self, chunk1: str, chunk2: str, max_overlap: int) -> int:
        return find_overlap_length(chunk1, chunk2, max_overlap)

    def chunk_code_text(self, text: str, file_type: str, chunk_size: int = 1500, chunk_overlap: int = 300) -> List[str]:
        return chunk_code_text(text, file_type, chunk_size, chunk_overlap)

    def estimate_chunks_count(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> int:
        return estimate_chunks_count(text, chunk_size, chunk_overlap)

    # === 라인/페이지 매핑(원본 로직과 동일 효과) ===
    def _extract_page_mapping(self, text: str, file_extension: str) -> List[Dict[str, Any]]:
        try:
            page_mapping: List[Dict[str, Any]] = []
            if file_extension in ['pdf','ppt','pptx','doc','docx']:
                patterns = [
                    r'=== 페이지 (\d+) ===',
                    r'=== 페이지 (\d+) \(OCR\) ===',
                    r'=== 페이지 (\d+) \(OCR\+참고\) ===',
                    r'=== 슬라이드 (\d+) ===',
                    r'=== 슬라이드 (\d+) \(OCR\) ===',
                    r'<페이지\s*번호>\s*(\d+)\s*</페이지\s*번호>',
                    r'<페이지\s*번호>\s*(\d+)\s*\(OCR\)\s*</페이지\s*번호>',
                    r'<페이지\s*번호>\s*(\d+)\s*\(OCR\+참고\)\s*</페이지\s*번호>',
                    r'<슬라이드\s*번호>\s*(\d+)\s*</슬라이드\s*번호>',
                    r'<슬라이드\s*번호>\s*(\d+)\s*\(OCR\)\s*</슬라이드\s*번호>',
                ]
                for pat in patterns:
                    matches = list(re.finditer(pat, text))
                    if matches:
                        for i, m in enumerate(matches):
                            page_num = int(m.group(1))
                            start = m.end()
                            end = matches[i+1].start() if i+1 < len(matches) else len(text)
                            page_mapping.append({"page_num": page_num, "start_pos": start, "end_pos": end})
                        page_mapping.sort(key=lambda x: x["page_num"])
                        break
                if not page_mapping and file_extension in ['doc','docx']:
                    chars_per_page = 1500
                    L = len(text)
                    if L > chars_per_page:
                        est = (L + chars_per_page - 1)//chars_per_page
                        for pn in range(1, est+1):
                            s = (pn-1)*chars_per_page
                            e = min(pn*chars_per_page, L)
                            page_mapping.append({"page_num": pn, "start_pos": s, "end_pos": e})
                if not page_mapping:
                    page_mapping = [{"page_num":1,"start_pos":0,"end_pos":len(text)}]
            elif file_extension in ['xlsx','xls']:
                matches = list(re.finditer(r'=== 시트: ([^=]+) ===', text))
                if matches:
                    for i, m in enumerate(matches):
                        s = m.end()
                        e = matches[i+1].start() if i+1 < len(matches) else len(text)
                        page_mapping.append({"page_num": i+1, "start_pos": s, "end_pos": e,
                                             "sheet_name": m.group(1).strip()})
                else:
                    page_mapping = [{"page_num":1,"start_pos":0,"end_pos":len(text)}]
            else:
                lines = text.split('\n')
                lpp = 1000
                if len(lines) > lpp:
                    pc = (len(lines) + lpp - 1)//lpp
                    cur = 0
                    for pn in range(1, pc+1):
                        sline = (pn-1)*lpp
                        eline = min(pn*lpp, len(lines))
                        page_text = '\n'.join(lines[sline:eline])
                        s = cur; e = cur + len(page_text)
                        page_mapping.append({"page_num": pn, "start_pos": s, "end_pos": e})
                        cur = e + 1
                else:
                    page_mapping = [{"page_num":1,"start_pos":0,"end_pos":len(text)}]
            return page_mapping
        except Exception:
            return [{"page_num":1,"start_pos":0,"end_pos":len(text)}]

    def _find_line_index_by_pos(self, pos: int, line_table: List[Dict[str, int]]) -> int:
        try:
            if not line_table:
                return 0
            starts = [l["start"] for l in line_table]
            idx = bisect.bisect_right(starts, pos) - 1
            return 0 if idx < 0 else min(idx, len(line_table)-1)
        except Exception:
            return 0

    def _build_line_offset_table(self, text: str, file_extension: str) -> List[Dict[str, int]]:
        try:
            lines = text.split('\n')
            table: List[Dict[str, int]] = []
            pos = 0
            page_mapping = self._extract_page_mapping(text, file_extension)
            def _page_for_pos(p: int) -> int:
                for info in page_mapping:
                    if info["start_pos"] <= p < info["end_pos"]:
                        return info["page_num"]
                return 1
            for i, line in enumerate(lines):
                start = pos
                end = pos + len(line)
                mid = start + max(0, (end-start)//2)
                page = _page_for_pos(mid)
                table.append({"line_num": i+1, "start": start, "end": end, "page": page})
                pos = end + 1
            return table
        except Exception:
            return [{"line_num":1,"start":0,"end":len(text),"page":1}]

    def chunk_text_with_metadata(self, text: str, file_extension: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
        chunks = self.chunk_text(text, chunk_size, chunk_overlap)
        reconstructed = self._reconstruct_text_from_chunks(chunks, chunk_overlap)
        line_table = self._build_line_offset_table(reconstructed, file_extension)

        out: List[Dict[str, Any]] = []
        cur = 0
        for idx, ch in enumerate(chunks):
            start = cur
            end = cur + len(ch) - 1
            sidx = self._find_line_index_by_pos(start, line_table)
            eidx = self._find_line_index_by_pos(end, line_table)
            line_start = line_table[sidx]["line_num"]
            line_end = line_table[eidx]["line_num"]
            page_number = line_table[sidx].get("page", 1)
            out.append({
                "text": ch,
                "page_number": page_number,
                "line_start": line_start,
                "line_end": line_end,
                "global_start": start,
                "global_end": end,
                "chunk_index": idx
            })
            cur += len(ch)
            if idx < len(chunks) - 1:
                ov = find_overlap_length(ch, chunks[idx+1], chunk_overlap)
                cur -= ov
        logger.info(f"Created {len(out)} chunks with metadata using reconstructed text")
        return out

    def validate_file_format(self, file_path: str) -> tuple[bool, str]:
        try:
            ext = Path(file_path).suffix[1:].lower()
            return (ext in self.supported_types, ext)
        except Exception:
            return (False, "")

    def get_file_info(self, file_path: str) -> Dict[str, str]:
        try:
            ext = Path(file_path).suffix[1:].lower()
            cat = self.get_file_category(ext)
            ok = ext in self.supported_types
            return {'extension': ext, 'category': cat, 'supported': str(ok)}
        except Exception:
            return {'extension':'unknown','category':'unknown','supported':'false'}

    # def get_current_config_status(self) -> Dict[str, Any]:
    #     try:
    #         cfg = self._get_current_image_text_config()
    #         try:
    #             from langchain_openai import ChatOpenAI  # noqa
    #             langchain_ok = True
    #         except Exception:
    #             langchain_ok = False
    #         status = {
    #             "provider": cfg.get('provider','unknown'),
    #             "ocr_enabled": is_image_text_enabled(cfg, langchain_ok),
    #             "base_url": cfg.get('base_url','unknown'),
    #             "model": cfg.get('model','unknown'),
    #             "temperature": cfg.get('temperature','unknown'),
    #             "batch_size": cfg.get('batch_size',1),
    #             "langchain_available": langchain_ok,
    #         }
    #         # 라이브러리 가용성 요약(간단)
    #         try:
    #             from pdf2image import convert_from_path  # noqa
    #             status["pdf2image_available"] = True
    #         except Exception:
    #             status["pdf2image_available"] = False
    #         try:
    #             from docx2pdf import convert as docx_to_pdf_convert  # noqa
    #             status["docx2pdf_available"] = True
    #         except Exception:
    #             status["docx2pdf_available"] = False
    #         try:
    #             from pptx import Presentation  # noqa
    #             status["python_pptx_available"] = True
    #         except Exception:
    #             status["python_pptx_available"] = False
    #         try:
    #             from PIL import Image  # noqa
    #             status["pil_available"] = True
    #         except Exception:
    #             status["pil_available"] = False
    #         return status
    #     except Exception as e:
    #         return {"error": str(e)}

    def test(self):
        try:
            cfg = self._get_current_image_text_config()
            logger.info(f"🔍 Test - Current provider: {cfg.get('provider','no_model')}")
            logger.info(f"🔍 Test - Current config: {cfg}")
            try:
                from langchain_openai import ChatOpenAI  # noqa
                langchain_ok = True
            except Exception:
                langchain_ok = False
        except Exception as e:
            logger.error(f"Error in test method: {e}")
            raise

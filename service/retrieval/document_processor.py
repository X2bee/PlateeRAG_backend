"""
문서 처리 모듈

이 모듈은 다양한 형식의 문서에서 텍스트를 추출하고, 
텍스트를 벡터화에 적합한 청크로 분할하는 기능을 제공합니다.
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Set
import aiofiles
import PyPDF2
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

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

logger = logging.getLogger("document-processor")

# Langchain에서 지원하는 언어별 코드 분할 매핑
LANGCHAIN_CODE_LANGUAGE_MAP = {
    'py': Language.PYTHON,
    'js': Language.JS,
    'ts': Language.TS,
    'java': Language.JAVA,
    'cpp': Language.CPP,
    'c': Language.CPP,
    'cs': Language.CSHARP,
    'go': Language.GO,
    'rs': Language.RUST,
    'php': Language.PHP,
    'rb': Language.RUBY,
    'swift': Language.SWIFT,
    'kt': Language.KOTLIN,
    'scala': Language.SCALA,
    'html': Language.HTML,
    'jsx': Language.JS,
    'tsx': Language.TS,
    # vue, svelte 등은 언어 지원 범위에 따라 fallback 처리 필요
}

class DocumentProcessor:
    """문서 처리를 담당하는 클래스"""
    
    def __init__(self):
        """DocumentProcessor 초기화"""
        # 지원하는 파일 타입들을 카테고리별로 정의
        self.document_types = ['pdf', 'docx', 'doc']
        self.text_types = ['txt', 'md', 'markdown', 'rtf']
        self.code_types = [
            'py', 'js', 'ts', 'java', 'cpp', 'c', 'h', 'cs', 'go', 'rs', 
            'php', 'rb', 'swift', 'kt', 'scala', 'dart', 'r', 'sql', 
            'html', 'css', 'jsx', 'tsx', 'vue', 'svelte'
        ]
        self.config_types = [
            'json', 'yaml', 'yml', 'xml', 'toml', 'ini', 'cfg', 'conf',
            'properties', 'env'
        ]
        self.data_types = ['csv', 'tsv', 'xlsx', 'xls']
        self.script_types = ['sh', 'bat', 'ps1', 'zsh', 'fish']
        self.log_types = ['log']
        self.web_types = ['htm', 'xhtml']
        
        # 모든 지원하는 타입들을 하나의 리스트로 통합
        self.supported_types = (
            self.document_types + self.text_types + self.code_types + 
            self.config_types + self.data_types + self.script_types + 
            self.log_types + self.web_types
        )
        
        # pandas가 없으면 Excel 파일 형식 제거
        if not PANDAS_AVAILABLE:
            self.supported_types = [t for t in self.supported_types if t not in ['xlsx', 'xls']]
        
        # 인코딩 시도 순서 (일반적인 순서대로)
        self.encodings = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr', 'latin-1', 'ascii']
        
        if not PDFMINER_AVAILABLE:
            logger.warning("pdfminer not available. PDF processing will use fallback PyPDF2 method.")
        if not PANDAS_AVAILABLE:
            logger.warning("pandas not available. Excel file processing (.xlsx, .xls) will not be supported.")
    
    def get_supported_types(self) -> List[str]:
        """지원하는 파일 형식 목록 반환"""
        return self.supported_types.copy()
    
    def get_file_category(self, file_type: str) -> str:
        """파일 타입의 카테고리 반환"""
        file_type = file_type.lower()
        if file_type in self.document_types:
            return 'document'
        elif file_type in self.text_types:
            return 'text'
        elif file_type in self.code_types:
            return 'code'
        elif file_type in self.config_types:
            return 'config'
        elif file_type in self.data_types:
            return 'data'
        elif file_type in self.script_types:
            return 'script'
        elif file_type in self.log_types:
            return 'log'
        elif file_type in self.web_types:
            return 'web'
        else:
            return 'unknown'
    
    def clean_text(self, text):
        """텍스트 정리"""
        if not text:
            return ""
        # 연속된 공백을 하나로 통합
        text = re.sub(r'\s+', ' ', text)
        # 연속된 줄바꿈을 두 개로 제한
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        return text.strip()
    
    def clean_code_text(self, text: str, file_type: str) -> str:
        """코드 텍스트 정리 (코드의 구조를 보존)"""
        if not text:
            return ""
        
        # 코드 파일의 경우 들여쓰기와 줄바꿈을 보존
        # 다만 파일 끝의 과도한 공백은 제거
        text = text.rstrip()
        
        # 탭을 4개의 스페이스로 변환 (일관성을 위해)
        text = text.replace('\t', '    ')
        
        return text
    
    async def extract_text_from_file(self, file_path: str, file_type: str = None) -> str:
        """파일에서 텍스트 추출
        
        Args:
            file_path: 파일 경로
            file_type: 파일 형식 (없으면 확장자에서 추출)
            
        Returns:
            추출된 텍스트
            
        Raises:
            ValueError: 지원하지 않는 파일 형식
            Exception: 파일 처리 오류
        """
        from pathlib import Path
        import subprocess
        import os
        
        if file_type is None:
            file_type = Path(file_path).suffix[1:].lower()
        
        file_type = file_type.lower()
        
        if file_type not in self.supported_types:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        try:
            category = self.get_file_category(file_type)
            
            # .doc 파일은 libreoffice로 .docx로 변환 후 파싱 (항상 변환 시도)
            if file_type == 'doc':
                doc_path = Path(file_path)
                docx_path = doc_path.with_suffix('.docx')
                logger.info(f"[INFO] .doc 파일을 .docx로 변환 중: {doc_path.name}")
                result = subprocess.run([
                    "libreoffice", "--headless", "--convert-to", "docx", str(doc_path),
                    "--outdir", str(doc_path.parent)
                ], capture_output=True)
                if result.returncode != 0:
                    logger.error(f"[ERROR] .doc -> .docx 변환 실패: {result.stderr.decode()}")
                    raise Exception(f".doc 파일을 .docx로 변환하는데 실패했습니다: {result.stderr.decode()}")
                else:
                    logger.info(f"[SUCCESS] 변환 완료: {docx_path.name}")
                # 변환된 .docx 파일로 파싱
                return await self._extract_text_from_docx(str(docx_path))
            
            if file_type == 'pdf':
                return await self._extract_text_from_pdf(file_path)
            elif file_type == 'docx':
                return await self._extract_text_from_docx(file_path)
            elif file_type in ['xlsx', 'xls']:
                if not PANDAS_AVAILABLE:
                    raise ValueError(f"Excel file processing requires pandas but it's not available: {file_type}")
                return await self._extract_text_from_excel(file_path)
            elif category in ['text', 'code', 'config', 'data', 'script', 'log', 'web']:
                return await self._extract_text_from_text_file(file_path, file_type)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        except Exception as e:
            logger.error(f"Failed to extract text from {file_path}: {e}")
            raise
    
    async def _extract_text_from_pdf(self, file_path: str) -> str:
        """PDF 파일에서 텍스트 추출 (기본 pdfminer 사용)"""
        try:
            if PDFMINER_AVAILABLE:
                logger.info(f"Using basic pdfminer for {file_path}")
                text = extract_text(file_path)
                # 추출된 텍스트 정리
                cleaned_text = self.clean_text(text)
                return cleaned_text
            else:
                # fallback to PyPDF2
                logger.info(f"Using fallback PDF processing for {file_path}")
                return await self._extract_text_from_pdf_fallback(file_path)
                
        except Exception as e:
            logger.error(f"PDF processing failed, trying fallback: {e}")
            # fallback to PyPDF2 if pdfminer fails
            return await self._extract_text_from_pdf_fallback(file_path)
    
    async def _extract_text_from_pdf_fallback(self, file_path: str) -> str:
        """PDF 파일에서 텍스트 추출 (PyPDF2 fallback)"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n=== 페이지 {page_num + 1} ===\n"
                        text += page_text + "\n"
            return self.clean_text(text)
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
            raise
    
    async def _extract_text_from_docx(self, file_path: str) -> str:
        """DOCX 파일에서 텍스트 추출 (표, 이미지 포함)"""
        try:
            doc = Document(file_path)
            text = ""
            processed_tables = set()  # 중복 처리 방지
            
            # 방법 1: 문서의 모든 요소를 순서대로 처리 (고급 방법)
            try:
                for element in doc.element.body:
                    if element.tag.endswith('p'):  # 단락(paragraph)
                        para_text = self._extract_paragraph_text(element, doc)
                        if para_text.strip():
                            text += para_text + "\n"
                            
                    elif element.tag.endswith('tbl'):  # 표(table)
                        table_text = self._extract_table_text(element)
                        if table_text.strip():
                            text += "\n=== 표 ===\n" + table_text + "\n=== 표 끝 ===\n\n"
                            processed_tables.add(table_text)
                
                logger.info("Successfully used advanced DOCX parsing method")
            except Exception as e:
                logger.warning(f"Advanced parsing failed, falling back to simple method: {e}")
                # Fallback: 간단한 방법으로 모든 단락 추출
                text = ""
                for paragraph in doc.paragraphs:
                    if paragraph.text.strip():
                        text += paragraph.text + "\n"
            
            # 방법 2: 모든 표를 확실히 추출 (중복 제거)
            for i, table in enumerate(doc.tables):
                table_text = self._extract_simple_table_text(table)
                if table_text.strip():
                    # 이미 처리된 표인지 확인 (간단한 비교)
                    is_duplicate = any(
                        self._is_similar_table_text(table_text, processed) 
                        for processed in processed_tables
                    )
                    
                    if not is_duplicate:
                        text += f"\n=== 표 {i+1} ===\n" + table_text + "\n=== 표 끝 ===\n\n"
                        processed_tables.add(table_text)
            
            logger.info(f"Extracted {len(processed_tables)} tables from DOCX")
            return self.clean_text(text)
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {e}")
            raise
    
    def _extract_paragraph_text(self, para_element, doc) -> str:
        """단락에서 텍스트와 이미지 정보 추출"""
        try:
            text = ""
            nsmap = doc.element.nsmap if hasattr(doc.element, 'nsmap') else {}
            
            for run in para_element.findall('.//w:r', nsmap):
                # 텍스트 추출
                for text_elem in run.findall('.//w:t', nsmap):
                    if text_elem.text:
                        text += text_elem.text
                
                # 이미지 정보 추출
                for drawing in run.findall('.//w:drawing', nsmap):
                    # 이미지가 있다는 표시 추가
                    text += " [이미지] "
                    
                    # 이미지 설명 텍스트 찾기 (가능한 경우)
                    try:
                        for desc in drawing.findall('.//wp:docPr', nsmap):
                            if desc.get('descr'):
                                text += f"[설명: {desc.get('descr')}] "
                            elif desc.get('name'):
                                text += f"[이름: {desc.get('name')}] "
                    except:
                        pass  # 이미지 메타데이터 추출 실패해도 계속
            
            return text
        except Exception as e:
            logger.debug(f"Error extracting paragraph text: {e}")
            # Fallback 1: 더 간단한 방법
            try:
                text_elements = para_element.findall('.//w:t', {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'})
                return ''.join([elem.text or '' for elem in text_elements])
            except:
                # Fallback 2: 최종 방법
                try:
                    return para_element.text or ""
                except:
                    return ""
    
    def _extract_table_text(self, table_element) -> str:
        """표 요소에서 텍스트 추출"""
        try:
            text = ""
            # namespace 정의
            ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            
            # XML에서 직접 표 데이터 추출
            for row in table_element.findall('.//w:tr', ns):
                row_text = []
                for cell in row.findall('.//w:tc', ns):
                    cell_text = ""
                    for text_elem in cell.findall('.//w:t', ns):
                        if text_elem.text:
                            cell_text += text_elem.text
                    row_text.append(cell_text.strip())
                
                if any(cell.strip() for cell in row_text):  # 빈 행이 아닌 경우만
                    text += " | ".join(row_text) + "\n"
            
            return text
        except Exception as e:
            logger.debug(f"Error extracting table text: {e}")
            return ""
    
    def _extract_simple_table_text(self, table) -> str:
        """python-docx Table 객체에서 텍스트 추출"""
        try:
            text = ""
            for row in table.rows:
                row_text = []
                for cell in row.cells:
                    cell_text = cell.text.strip()
                    row_text.append(cell_text)
                
                if any(cell.strip() for cell in row_text):  # 빈 행이 아닌 경우만
                    text += " | ".join(row_text) + "\n"
            
            return text
        except Exception as e:
            logger.debug(f"Error extracting simple table text: {e}")
            return ""
    
    def _is_similar_table_text(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """두 표 텍스트가 유사한지 확인 (중복 제거용)"""
        try:
            if not text1 or not text2:
                return False
            
            # 공백 정규화
            text1_clean = re.sub(r'\s+', ' ', text1.strip())
            text2_clean = re.sub(r'\s+', ' ', text2.strip())
            
            # 완전히 동일한 경우
            if text1_clean == text2_clean:
                return True
            
            # 길이가 매우 다른 경우
            len1, len2 = len(text1_clean), len(text2_clean)
            if min(len1, len2) / max(len1, len2) < 0.5:
                return False
            
            # 간단한 유사도 검사 (공통 부분 비율)
            shorter = text1_clean if len1 < len2 else text2_clean
            longer = text2_clean if len1 < len2 else text1_clean
            
            common_ratio = len(set(shorter.split()) & set(longer.split())) / len(set(shorter.split()))
            return common_ratio >= threshold
            
        except Exception:
            return False
    
    async def _extract_text_from_excel(self, file_path: str) -> str:
        """Excel 파일에서 텍스트 추출"""
        if not PANDAS_AVAILABLE:
            raise Exception("pandas is required for Excel file processing but is not available")
        
        try:
            # Excel 파일 읽기 (모든 시트)
            excel_file = pd.ExcelFile(file_path)
            text = ""
            
            for sheet_name in excel_file.sheet_names:
                logger.info(f"Processing sheet: {sheet_name}")
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # 시트 이름 추가
                text += f"\n=== 시트: {sheet_name} ===\n"
                
                # 컬럼 헤더 추가
                if not df.empty:
                    text += "컬럼: " + ", ".join(str(col) for col in df.columns) + "\n\n"
                    
                    # 데이터 행들을 텍스트로 변환
                    for index, row in df.iterrows():
                        row_text = " | ".join(str(value) for value in row.values if pd.notna(value))
                        if row_text.strip():  # 빈 행 제외
                            text += row_text + "\n"
                    text += "\n"
            
            return self.clean_text(text)
        except Exception as e:
            logger.error(f"Error extracting text from Excel {file_path}: {e}")
            raise
    
    async def _extract_text_from_text_file(self, file_path: str, file_type: str) -> str:
        """텍스트 기반 파일에서 텍스트 추출 (다양한 인코딩 시도)"""
        category = self.get_file_category(file_type)
        
        for encoding in self.encodings:
            try:
                async with aiofiles.open(file_path, 'r', encoding=encoding) as file:
                    text = await file.read()
                
                logger.info(f"Successfully read {file_path} with {encoding} encoding")
                
                # 파일 카테고리에 따라 다른 정리 방식 적용
                if category == 'code':
                    return self.clean_code_text(text, file_type)
                else:
                    return self.clean_text(text)
                    
            except UnicodeDecodeError:
                logger.debug(f"Failed to read {file_path} with {encoding} encoding, trying next...")
                continue
            except Exception as e:
                logger.error(f"Error reading file {file_path} with {encoding} encoding: {e}")
                continue
        
        # 모든 인코딩이 실패한 경우
        raise Exception(f"Could not read file {file_path} with any supported encoding")
    
    def chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """텍스트를 청크로 분할
        
        Args:
            text: 분할할 텍스트
            chunk_size: 청크 크기
            chunk_overlap: 청크 간 중복 크기
            
        Returns:
            분할된 텍스트 청크 리스트
            
        Raises:
            Exception: 텍스트 분할 오류
        """
        try:
            if not text or not text.strip():
                logger.warning("Empty text provided for chunking")
                return [""]
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = text_splitter.split_text(text)
            logger.info(f"Text split into {len(chunks)} chunks (size: {chunk_size}, overlap: {chunk_overlap})")
            return chunks
        except Exception as e:
            logger.error(f"Error chunking text: {e}")
            raise
    
    def chunk_code_text(self, text: str, file_type: str, chunk_size: int = 1500, chunk_overlap: int = 300) -> List[str]:
        """코드 텍스트를 청크로 분할 (언어별 구문 구조를 고려한 분할)
        
        Args:
            text: 분할할 코드 텍스트
            file_type: 파일 형식
            chunk_size: 청크 크기 (코드는 좀 더 큰 청크 사용)
            chunk_overlap: 청크 간 중복 크기
            
        Returns:
            분할된 텍스트 청크 리스트
        """
        try:
            if not text or not text.strip():
                logger.warning("Empty code text provided for chunking")
                return [""]
            
            lang = LANGCHAIN_CODE_LANGUAGE_MAP.get(file_type.lower())

            if lang:
                text_splitter = RecursiveCharacterTextSplitter.from_language(
                    language=lang,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                logger.info(f"Using language-specific splitter for {file_type} ({lang})")
            else:
                # fallback: 기존 방식 사용
                logger.info(f"No language-specific splitter for {file_type}, using fallback.")
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                    separators=["\n\n", "\n", " ", ""]
                )

            chunks = text_splitter.split_text(text)
            logger.info(f"Code text split into {len(chunks)} chunks (size: {chunk_size}, overlap: {chunk_overlap})")
            return chunks
        except Exception as e:
            logger.error(f"Error chunking code text: {e}")
            raise
    
    def validate_file_format(self, file_path: str) -> tuple[bool, str]:
        """파일 형식 유효성 검사
        
        Args:
            file_path: 파일 경로
            
        Returns:
            (유효성, 파일형식) 튜플
        """
        try:
            file_extension = Path(file_path).suffix[1:].lower()
            is_valid = file_extension in self.supported_types
            return is_valid, file_extension
        except Exception:
            return False, ""
    
    def estimate_chunks_count(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> int:
        """텍스트에서 생성될 청크 수 추정
        
        Args:
            text: 대상 텍스트
            chunk_size: 청크 크기
            chunk_overlap: 청크 간 중복 크기
            
        Returns:
            예상 청크 수
        """
        if not text:
            return 0
        
        text_length = len(text)
        if text_length <= chunk_size:
            return 1
        
        # 간단한 추정 공식
        effective_chunk_size = chunk_size - chunk_overlap
        estimated_chunks = (text_length - chunk_overlap) // effective_chunk_size + 1
        return max(1, estimated_chunks)
    
    def get_file_info(self, file_path: str) -> Dict[str, str]:
        """파일 정보 반환
        
        Args:
            file_path: 파일 경로
            
        Returns:
            파일 정보 딕셔너리 (extension, category, supported)
        """
        try:
            file_extension = Path(file_path).suffix[1:].lower()
            category = self.get_file_category(file_extension)
            is_supported = file_extension in self.supported_types
            
            return {
                'extension': file_extension,
                'category': category,
                'supported': str(is_supported)
            }
        except Exception:
            return {
                'extension': 'unknown',
                'category': 'unknown', 
                'supported': 'false'
            } 
"""
문서 처리 모듈

이 모듈은 다양한 형식의 문서에서 텍스트를 추출하고, 
텍스트를 벡터화에 적합한 청크로 분할하는 기능을 제공합니다.
"""

import logging
import re
from pathlib import Path
from typing import List
import aiofiles
import PyPDF2
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from pdfminer.high_level import extract_text
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False

logger = logging.getLogger("document-processor")

class DocumentProcessor:
    """문서 처리를 담당하는 클래스"""
    
    def __init__(self):
        """DocumentProcessor 초기화"""
        self.supported_types = ['pdf', 'docx', 'doc', 'txt']
        if not PDFMINER_AVAILABLE:
            logger.warning("pdfminer not available. PDF processing will use fallback PyPDF2 method.")
    
    def get_supported_types(self) -> List[str]:
        """지원하는 파일 형식 목록 반환"""
        return self.supported_types.copy()
    
    def clean_text(self, text):
        """텍스트 정리"""
        if not text:
            return ""
        # 연속된 공백을 하나로 통합
        text = re.sub(r'\s+', ' ', text)
        # 연속된 줄바꿈을 두 개로 제한
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        return text.strip()
    
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
        if file_type is None:
            file_type = Path(file_path).suffix[1:].lower()
        
        file_type = file_type.lower()
        
        if file_type not in self.supported_types:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        try:
            if file_type == 'pdf':
                return await self._extract_text_from_pdf(file_path)
            elif file_type in ['docx', 'doc']:
                return await self._extract_text_from_docx(file_path)
            elif file_type == 'txt':
                return await self._extract_text_from_txt(file_path)
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
        """DOCX 파일에서 텍스트 추출"""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return self.clean_text(text)
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {e}")
            raise
    
    async def _extract_text_from_txt(self, file_path: str) -> str:
        """TXT 파일에서 텍스트 추출"""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
                text = await file.read()
            return self.clean_text(text)
        except UnicodeDecodeError:
            # UTF-8로 읽기 실패 시 다른 인코딩 시도
            try:
                async with aiofiles.open(file_path, 'r', encoding='cp949') as file:
                    text = await file.read()
                return self.clean_text(text)
            except Exception as e:
                logger.error(f"Error reading TXT file with cp949 encoding {file_path}: {e}")
                raise
        except Exception as e:
            logger.error(f"Error extracting text from TXT {file_path}: {e}")
            raise
    
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
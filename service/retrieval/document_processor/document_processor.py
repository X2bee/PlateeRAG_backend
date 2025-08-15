"""
문서 처리 모듈 - 메인 클래스

이 모듈은 다양한 형식의 문서에서 텍스트를 추출하고, 
텍스트를 벡터화에 적합한 청크로 분할하는 기능을 제공합니다.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Any

from .constants import (
    DOCUMENT_TYPES, TEXT_TYPES, CODE_TYPES, CONFIG_TYPES, 
    DATA_TYPES, SCRIPT_TYPES, LOG_TYPES, WEB_TYPES, IMAGE_TYPES
)
from .dependencies import (
    OPENPYXL_AVAILABLE, XLRD_AVAILABLE, LANGCHAIN_OPENAI_AVAILABLE, 
    log_dependency_warnings
)
from .config_manager import ConfigManager
from .text_utils import TextUtils
from .ocr_processor import OCRProcessor
from .extractors import DocumentExtractor

logger = logging.getLogger("document-processor")

class DocumentProcessor:
    """문서 처리를 담당하는 클래스"""
    
    def __init__(self, collection_config=None):
        # 파일 타입 분류
        self.document_types = DOCUMENT_TYPES
        self.text_types = TEXT_TYPES
        self.code_types = CODE_TYPES
        self.config_types = CONFIG_TYPES
        self.data_types = DATA_TYPES
        self.script_types = SCRIPT_TYPES
        self.log_types = LOG_TYPES
        self.web_types = WEB_TYPES
        self.image_types = IMAGE_TYPES
        
        # 지원되는 파일 타입 목록 생성
        self.supported_types = (
            self.document_types + self.text_types + self.code_types +
            self.config_types + self.data_types + self.script_types +
            self.log_types + self.web_types + self.image_types
        )
        
        self.collection_config = collection_config
        
        # 의존성에 따른 지원 타입 조정
        if not XLRD_AVAILABLE or not OPENPYXL_AVAILABLE:
            self.supported_types = [t for t in self.supported_types if t not in ['xlsx','xls']]
        if not LANGCHAIN_OPENAI_AVAILABLE:
            self.supported_types = [t for t in self.supported_types if t not in self.image_types]
            logger.warning("langchain_openai not available. Image processing disabled.")
        
        # 의존성 경고 로그
        log_dependency_warnings()
        
        # 컴포넌트 초기화
        self.config_manager = ConfigManager()
        self.ocr_processor = OCRProcessor(self.config_manager)
        self.extractor = DocumentExtractor(self.ocr_processor, self.config_manager)

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
        elif file_type in self.image_types:
            return 'image'
        else:
            return 'unknown'
    
    def clean_text(self, text):
        """텍스트 정리"""
        return TextUtils.clean_text(text)
    
    def clean_code_text(self, text: str, file_type: str) -> str:
        """코드 텍스트 정리 (코드의 구조를 보존)"""
        return TextUtils.clean_code_text(text, file_type)
    
    async def extract_text_from_file(self, file_path: str, file_extension: str) -> str:
        """파일에서 텍스트 추출 (파일 형식에 따라 적절한 메서드 호출)"""
        try:
            category = self.get_file_category(file_extension)
            logger.info(f"Extracting text from {file_extension} file ({category} category): {file_path}")
           
            # 파일 형식별 텍스트 추출
            if file_extension == 'pdf':
                return await self.extractor.extract_text_from_pdf(file_path)
            elif file_extension in ['docx', 'doc']:
                return await self.extractor.extract_text_from_docx(file_path)
            elif file_extension in ['pptx', 'ppt']:
                return await self.extractor.extract_text_from_ppt(file_path)
            elif file_extension in ['xlsx', 'xls']:
                return await self.extractor.extract_text_from_excel(file_path)
            elif file_extension in self.image_types:
                return await self.ocr_processor.convert_image_to_text(file_path)
            elif file_extension in (self.text_types + self.code_types + self.config_types + 
                                self.script_types + self.log_types + self.web_types):
                return await self.extractor.extract_text_from_text_file(file_path, file_extension)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
        except Exception as e:
            logger.error(f"Failed to extract text from {file_path}: {e}")
            raise

    def chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """텍스트를 청크로 분할"""
        return TextUtils.chunk_text(text, chunk_size, chunk_overlap)
    
    def chunk_code_text(self, text: str, file_type: str, chunk_size: int = 1500, chunk_overlap: int = 300) -> List[str]:
        """코드 텍스트를 청크로 분할 (언어별 구문 구조를 고려한 분할)"""
        return TextUtils.chunk_code_text(text, file_type, chunk_size, chunk_overlap)
    
    def validate_file_format(self, file_path: str) -> tuple[bool, str]:
        """파일 형식 유효성 검사"""
        try:
            logger.info(f"Validating file format for: {self.supported_types}")
            file_extension = Path(file_path).suffix[1:].lower()
            is_valid = file_extension in self.supported_types
            return is_valid, file_extension
        except Exception:
            return False, ""
    
    def estimate_chunks_count(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> int:
        """텍스트에서 생성될 청크 수 추정"""
        return TextUtils.estimate_chunks_count(text, chunk_size, chunk_overlap)
    
    def get_file_info(self, file_path: str) -> Dict[str, str]:
        """파일 정보 반환"""
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

    def get_current_config_status(self) -> Dict[str, Any]:
        """현재 설정 상태 반환 (디버깅용)"""
        try:
            current_config = self.config_manager.get_current_image_text_config()
            from .dependencies import (
                LANGCHAIN_OPENAI_AVAILABLE, PDF2IMAGE_AVAILABLE,
                DOCX2PDF_AVAILABLE, PYTHON_PPTX_AVAILABLE, PIL_AVAILABLE
            )
            
            return {
                "provider": current_config.get('provider', 'unknown'),
                "ocr_enabled": self.config_manager.is_image_text_enabled(current_config),
                "base_url": current_config.get('base_url', 'unknown'),
                "model": current_config.get('model', 'unknown'),
                "temperature": current_config.get('temperature', 'unknown'),
                "batch_size": current_config.get('batch_size', 1),
                "langchain_available": LANGCHAIN_OPENAI_AVAILABLE,
                "pdf2image_available": PDF2IMAGE_AVAILABLE,
                "docx2pdf_available": DOCX2PDF_AVAILABLE,
                "python_pptx_available": PYTHON_PPTX_AVAILABLE,
                "pil_available": PIL_AVAILABLE
            }
        except Exception as e:
            return {"error": str(e)}

    def test(self):
        """설정 테스트 메서드 (디버깅용)"""
        try:
            current_config = self.config_manager.get_current_image_text_config()
            provider = current_config.get('provider', 'no_model')
            
            logger.info(f"🔍 Test - Current provider: {provider}")
            logger.info(f"🔍 Test - Current config: {current_config}")
            logger.info(f"🔍 Test - OCR enabled: {self.config_manager.is_image_text_enabled(current_config)}")
            
        except Exception as e:
            logger.error(f"Error in test method: {e}")
            raise
"""
문서 처리 모듈

이 모듈은 다양한 형식의 문서에서 텍스트를 추출하고, 
텍스트를 벡터화에 적합한 청크로 분할하는 기능을 제공합니다.
"""

import logging
import re
import base64
from pathlib import Path
from typing import List, Dict, Optional, Any
import aiofiles
import PyPDF2
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from service.llm.llm_service import LLMService

import os
import csv
try:
    from openpyxl import load_workbook
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

try:
    import xlrd
    XLRD_AVAILABLE = True
except ImportError:
    XLRD_AVAILABLE = False

try:
    from pdfminer.high_level import extract_text
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
    print("✅ pdf2image available")
except ImportError:
    PDF2IMAGE_AVAILABLE = False

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    LANGCHAIN_OPENAI_AVAILABLE = True
except ImportError:
    LANGCHAIN_OPENAI_AVAILABLE = False

# DOCX to Image 변환을 위한 라이브러리 체크
try:
    from docx2pdf import convert as docx_to_pdf_convert
    DOCX2PDF_AVAILABLE = True
    print("✅ docx2pdf available")
except ImportError:
    DOCX2PDF_AVAILABLE = False

# PPT 처리를 위한 라이브러리 체크
try:
    from pptx import Presentation
    PYTHON_PPTX_AVAILABLE = True
    print("✅ python-pptx available")
except ImportError:
    PYTHON_PPTX_AVAILABLE = False

# 대안: python-docx + PIL을 이용한 방법
try:
    from PIL import Image, ImageDraw, ImageFont
    from io import BytesIO
    PIL_AVAILABLE = True
    print("✅ PIL available")
except ImportError:
    PIL_AVAILABLE = False

logger = logging.getLogger("document-processor")

LANGCHAIN_CODE_LANGUAGE_MAP = {
    'py': Language.PYTHON, 'js': Language.JS, 'ts': Language.TS,
    'java': Language.JAVA, 'cpp': Language.CPP, 'c': Language.CPP,
    'cs': Language.CSHARP, 'go': Language.GO, 'rs': Language.RUST,
    'php': Language.PHP, 'rb': Language.RUBY, 'swift': Language.SWIFT,
    'kt': Language.KOTLIN, 'scala': Language.SCALA,
    'html': Language.HTML, 'jsx': Language.JS, 'tsx': Language.TS,
}

class DocumentProcessor:
    """문서 처리를 담당하는 클래스"""
    def __init__(self, collection_config=None):
        # 🔥 PPT 형식 추가
        self.document_types = ['pdf', 'docx', 'doc', 'pptx', 'ppt']
        self.text_types = ['txt', 'md', 'markdown', 'rtf']
        self.code_types = ['py','js','ts','java','cpp','c','h','cs','go','rs',
                          'php','rb','swift','kt','scala','dart','r','sql',
                          'html','css','jsx','tsx','vue','svelte']
        self.config_types = ['json','yaml','yml','xml','toml','ini','cfg','conf','properties','env']
        self.data_types   = ['csv','tsv','xlsx','xls']
        self.script_types = ['sh','bat','ps1','zsh','fish']
        self.log_types    = ['log']
        self.web_types    = ['htm','xhtml']
        # 🔥 JPG 형식 추가
        self.image_types  = ['jpg','jpeg','png','gif','bmp','webp']
        self.supported_types = (
            self.document_types + self.text_types + self.code_types +
            self.config_types + self.data_types + self.script_types +
            self.log_types + self.web_types + self.image_types
        )
        self.collection_config = collection_config
        
        # 변경
        if not OPENPYXL_AVAILABLE and not XLRD_AVAILABLE:
            # openpyxl도 없고 xlrd도 없으면 Excel 지원 제거
            self.supported_types = [t for t in self.supported_types if t not in ['xlsx', 'xls']]
        if not LANGCHAIN_OPENAI_AVAILABLE:
            self.supported_types = [t for t in self.supported_types if t not in self.image_types]
            logger.warning("langchain_openai not available. Image processing disabled.")
        
        # 🔥 PPT는 LibreOffice로도 처리 가능하므로 지원 목록에서 제거하지 않음
        # if not PYTHON_PPTX_AVAILABLE:
        #     self.supported_types = [t for t in self.supported_types if t not in ['pptx', 'ppt']]
        #     logger.warning("python-pptx not available. PPT processing disabled.")
            
        self.encodings = ['utf-8','utf-8-sig','cp949','euc-kr','latin-1','ascii']
        if not PDFMINER_AVAILABLE:
            logger.warning("pdfminer not available. Using PyPDF2 fallback.")
        if not PDF2IMAGE_AVAILABLE:
            logger.warning("pdf2image not available. OCR disabled.")
        if not DOCX2PDF_AVAILABLE and not PIL_AVAILABLE:
            logger.warning("docx2pdf and PIL not available. DOCX/PPT OCR disabled.")

    def _get_current_image_text_config(self) -> Dict[str, Any]:
        """실시간으로 현재 IMAGE_TEXT 설정 가져오기"""
        try:
            from main import app
            if hasattr(app.state, 'config_composer'):
                collection_config = app.state.config_composer.get_config_by_category_name("collection")
                
                # 🔥 get_env_value() 대신 직접 .value 접근
                if hasattr(collection_config, 'IMAGE_TEXT_MODEL_PROVIDER'):
                    provider_obj = getattr(collection_config, 'IMAGE_TEXT_MODEL_PROVIDER')
                    base_url_obj = getattr(collection_config, 'IMAGE_TEXT_BASE_URL')
                    api_key_obj = getattr(collection_config, 'IMAGE_TEXT_API_KEY')
                    model_obj = getattr(collection_config, 'IMAGE_TEXT_MODEL_NAME')
                    temp_obj = getattr(collection_config, 'IMAGE_TEXT_TEMPERATURE')
                    # 🔥 배치 처리 설정 추가
                    batch_size_obj = getattr(collection_config, 'IMAGE_TEXT_BATCH_SIZE', None)
                    
                    # PersistentConfig 객체에서 실제 값 추출
                    config = {
                        'provider': str(provider_obj.value if hasattr(provider_obj, 'value') else provider_obj).lower(),
                        'base_url': str(base_url_obj.value if hasattr(base_url_obj, 'value') else base_url_obj),
                        'api_key': str(api_key_obj.value if hasattr(api_key_obj, 'value') else api_key_obj),
                        'model': str(model_obj.value if hasattr(model_obj, 'value') else model_obj),
                        'temperature': float(temp_obj.value if hasattr(temp_obj, 'value') else temp_obj),
                        # 🔥 배치 크기 설정 (기본값: 1, 최대: 5)
                        'batch_size': int(batch_size_obj.value if batch_size_obj and hasattr(batch_size_obj, 'value') else 1)
                    }
                    
                    logger.info(f"🔄 Direct value access config: {config}")
                    return config
            
        except Exception as e:
            logger.error(f"🔍 Error in _get_current_image_text_config: {e}")
            import traceback
            logger.error(f"🔍 Traceback: {traceback.format_exc()}")
        
        # fallback
        logger.warning("🔍 Using fallback config")
        return {'provider': 'no_model', 'batch_size': 1}

    def _is_image_text_enabled(self, config: Dict[str, Any]) -> bool:
        """설정에 따라 OCR이 활성화되어 있는지 확인"""
        provider = config.get('provider', 'no_model')
        if provider in ('openai', 'vllm'):
            # OCR 사용 가능한 프로바이더인지 확인
            if not LANGCHAIN_OPENAI_AVAILABLE:
                logger.warning("langchain_openai not available for OCR")
                return False
            return True
        return False

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
    
    # 🔥 배치 OCR 처리 메서드들
    async def _convert_images_to_text_batch(self, image_paths: List[str], batch_size: int = 1) -> List[str]:
        """여러 이미지를 배치로 텍스트 변환 (실시간 설정 사용)"""
        current_config = self._get_current_image_text_config()
        
        if not self._is_image_text_enabled(current_config):
            return ["[이미지 파일: 이미지-텍스트 변환이 설정되지 않았습니다]" for _ in image_paths]
        
        # 배치 크기 제한 (1-5)
        batch_size = max(1, min(batch_size, 10))
        
        results = []
        total_batches = (len(image_paths) + batch_size - 1) // batch_size
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches} with {len(batch_paths)} images")
            
            if len(batch_paths) == 1:
                # 단일 이미지는 기존 방식 사용
                result = await self._convert_image_to_text(batch_paths[0])
                results.append(result)
            else:
                # 여러 이미지는 배치 처리
                batch_results = await self._convert_multiple_images_to_text(batch_paths, current_config)
                results.extend(batch_results)
        
        return results

    async def _convert_multiple_images_to_text(self, image_paths: List[str], config: Dict[str, Any]) -> List[str]:
        """여러 이미지를 한번에 OCR 처리"""
        try:
            # 모든 이미지를 base64로 인코딩
            image_contents = []
            for image_path in image_paths:
                with open(image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                    image_contents.append(base64_image)
            
            provider = config.get('provider', 'openai')
            api_key = config.get('api_key', '')
            base_url = config.get('base_url', 'https://api.openai.com/v1')
            model = config.get('model', 'gpt-4-vision-preview')
            temperature = config.get('temperature', 0.7)
            
            logger.info(f'🔄 Using batch OCR with {len(image_paths)} images, provider: {provider}')
            
            # 프로바이더별 LLM 클라이언트 생성
            if provider == 'openai':
                llm = ChatOpenAI(
                    model=model,
                    openai_api_key=api_key,
                    base_url=base_url,
                    temperature=temperature
                )
            elif provider == 'vllm':
                llm = ChatOpenAI(
                    model=model,
                    openai_api_key=api_key or 'dummy',
                    base_url=base_url,
                    temperature=temperature
                )
            else:
                logger.error(f"Unsupported image-text provider: {provider}")
                return [f"[이미지 파일: 지원하지 않는 프로바이더 - {provider}]" for _ in image_paths]
            
            # 배치 OCR 프롬프트
            prompt = f"""다음 {len(image_paths)}개의 이미지를 각각 정확한 텍스트로 변환해주세요. 

            **중요한 규칙:**
            1. 각 이미지의 결과를 명확히 구분해주세요
            2. 다음 형식으로 응답해주세요:

            === 이미지 1 ===
            [첫 번째 이미지의 텍스트 내용]

            === 이미지 2 ===
            [두 번째 이미지의 텍스트 내용]

            === 이미지 3 ===
            [세 번째 이미지의 텍스트 내용]

            **변환 규칙:**
            - 표가 있다면 마크다운 표 형식으로 변환
            - 원본의 레이아웃, 들여쓰기, 줄바꿈 보존
            - 모든 문자, 숫자, 기호를 정확히 인식
            - 제목, 부제목, 목록, 단락 구분을 명확히 표현
            - 특수 형식(날짜, 금액 등) 정확히 유지
            - **섹션 구분**: 각 이미지 내에서 문맥적으로 다른 내용 섹션들은 `\n\n\n` (세 개의 줄바꿈 문자)으로 명확히 구분

            **섹션 구분 예시:**
            - 제목과 본문 사이
            - 서로 다른 주제나 단락 사이  
            - 표와 다른 내용 사이
            - 차트/그래프와 설명 텍스트 사이

            **출력 형식 예시:**
            === 이미지 1 ===
            # 제목
            \n\n\n
            본문 내용이 여기에...

            \n\n\n
            ## 소제목
            다른 섹션의 내용...

            \n\n\n
            | 표 데이터 |
            |----------|
            | 내용     |

            텍스트만 출력하고, 추가 설명은 하지 마세요."""
            # 멀티 이미지 메시지 생성
            content = [{"type": "text", "text": prompt}]
            
            for i, base64_image in enumerate(image_contents):
                content.append({
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })
            
            message = HumanMessage(content=content)
            
            # 응답 생성
            response = await llm.ainvoke([message])
            response_text = response.content
            
            # 응답을 이미지별로 분할
            results = self._parse_batch_ocr_response(response_text, len(image_paths))
            
            logger.info(f"Successfully processed {len(image_paths)} images in batch")
            return results
            
        except Exception as e:
            logger.error(f"Error in batch OCR processing: {e}")
            # 실패시 개별 처리로 fallback
            logger.warning("Batch OCR failed, falling back to individual processing")
            results = []
            for image_path in image_paths:
                result = await self._convert_image_to_text(image_path)
                results.append(result)
            return results

    def _parse_batch_ocr_response(self, response_text: str, expected_count: int) -> List[str]:
        """배치 OCR 응답을 이미지별로 분할"""
        try:
            # "=== 이미지 N ===" 패턴으로 분할
            import re
            
            # 패턴 매칭으로 각 이미지 섹션 찾기
            pattern = r'=== 이미지 (\d+) ===\s*(.*?)(?=\s*=== 이미지 \d+ ===|\s*$)'
            matches = re.findall(pattern, response_text, re.DOTALL)
            
            results = []
            
            if matches and len(matches) >= expected_count:
                # 매칭된 결과 사용
                for i in range(expected_count):
                    if i < len(matches):
                        _, content = matches[i]
                        results.append(content.strip())
                    else:
                        results.append("[이미지 분할 실패]")
            else:
                # 패턴 매칭 실패시 단순 분할
                logger.warning("Pattern matching failed, using simple split")
                parts = re.split(r'=== 이미지 \d+ ===', response_text)
                
                for i in range(expected_count):
                    if i + 1 < len(parts):
                        results.append(parts[i + 1].strip())
                    else:
                        results.append("[이미지 분할 실패]")
            
            # 결과 개수 맞추기
            while len(results) < expected_count:
                results.append("[이미지 처리 실패]")
            
            return results[:expected_count]
            
        except Exception as e:
            logger.error(f"Error parsing batch OCR response: {e}")
            # 실패시 동일한 응답을 모든 이미지에 적용
            return [response_text for _ in range(expected_count)]
    
    async def _convert_image_to_text(self, image_path: str) -> str:
        """이미지를 텍스트로 변환 (실시간 설정 사용)"""
        # 🔥 실시간 설정 가져오기
        current_config = self._get_current_image_text_config()
        
        if not self._is_image_text_enabled(current_config):
            return "[이미지 파일: 이미지-텍스트 변환이 설정되지 않았습니다]"
        
        try:
            # 이미지를 base64로 인코딩
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            provider = current_config.get('provider', 'openai')
            api_key = current_config.get('api_key', '')
            base_url = current_config.get('base_url', 'https://api.openai.com/v1')
            model = current_config.get('model', 'gpt-4-vision-preview')
            temperature = current_config.get('temperature', 0.7)
            
            logger.info(f'🔄 Using real-time image-text provider: {provider}')
            logger.info(f'Model: {model}, Base URL: {base_url}')
            
            # 프로바이더별 LLM 클라이언트 생성
            if provider == 'openai':
                llm = ChatOpenAI(
                    model=model,
                    openai_api_key=api_key,
                    base_url=base_url,
                    temperature=temperature
                )
            elif provider == 'vllm':
                # vLLM의 경우 OpenAI 호환 API 사용
                llm = ChatOpenAI(
                    model=model,
                    openai_api_key=api_key or 'dummy',  # vLLM은 보통 API 키가 필요없음
                    base_url=base_url,
                    temperature=temperature
                )
            else:
                logger.error(f"Unsupported image-text provider: {provider}")
                return f"[이미지 파일: 지원하지 않는 프로바이더 - {provider}]"
            
            # OCR 프롬프트
            prompt = """이 이미지를 정확한 텍스트로 변환해주세요. 다음 규칙을 철저히 지켜주세요:

                        1. **표 구조 보존**: 표가 있다면 정확한 행과 열 구조를 유지하고, 마크다운 표 형식으로 변환해주세요
                        2. **레이아웃 유지**: 원본의 레이아웃, 들여쓰기, 줄바꿈을 최대한 보존해주세요
                        3. **정확한 텍스트**: 모든 문자, 숫자, 기호를 정확히 인식해주세요
                        4. **구조 정보**: 제목, 부제목, 목록, 단락 구분을 명확히 표현해주세요
                        5. **특수 형식**: 날짜, 금액, 주소, 전화번호 등의 형식을 정확히 유지해주세요
                        6. **슬라이드 구조**: 슬라이드 제목, 내용, 차트/그래프 설명을 구분해주세요
                        7. **섹션 구분**: 문맥적으로 다른 내용 섹션들은 `\\n\\n\\n` (세 개의 줄바꿈 문자)으로 명확히 구분해주세요

                        **섹션 구분 예시:**
                        - 제목과 본문 사이
                        - 서로 다른 주제나 단락 사이  
                        - 표와 다른 내용 사이
                        - 차트/그래프와 설명 텍스트 사이

                        만약 표가 있다면 다음과 같은 마크다운 형식으로 변환해주세요:
                        | 항목 | 내용 |
                        |------|------|
                        | 데이터1 | 값1 |
                        | 데이터2 | 값2 |

                        **출력 형식 예시:**
                        # 제목
                        \n\n\n
                        본문 내용이 여기에...

                        \n\n\n
                        ## 소제목
                        다른 섹션의 내용...

                        \n\n\n
                        | 표 데이터 |
                        |----------|
                        | 내용     |

                        텍스트만 출력하고, 추가 설명은 하지 마세요."""


            # 이미지 메시지 생성
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            )
            
            # 응답 생성
            response = await llm.ainvoke([message])
            logger.info(f"Successfully converted image to text using {provider}: {Path(image_path).name}")
            return response.content
            
        except Exception as e:
            logger.error(f"Error converting image to text {image_path}: {e}")
            return f"[이미지 파일: 텍스트 변환 중 오류 발생 - {str(e)}]"
    
    # PDF 관련 메서드들
    async def _extract_text_from_pdf(self, file_path: str) -> str:
        """PDF 파일에서 텍스트 추출 (실시간 설정에 따라 텍스트 추출 또는 OCR)"""
        try:
            # 🔥 실시간 설정 가져오기
            current_config = self._get_current_image_text_config()
            provider = current_config.get('provider', 'no_model')
            
            logger.info(f"🔄 Real-time PDF processing with provider: {provider}")
            
            # no_model인 경우에만 기본 텍스트 추출
            if provider == 'no_model':
                logger.info("Using text extraction mode (no_model)")
                
                # 1단계: pdfminer 시도
                if PDFMINER_AVAILABLE:
                    logger.info(f"Using pdfminer for {file_path}")
                    try:
                        text = extract_text(file_path)
                        cleaned_text = self.clean_text(text)
                        if len(cleaned_text.strip()) > 100:
                            logger.info(f"Text extracted via pdfminer: {len(cleaned_text)} chars")
                            return cleaned_text
                    except Exception as e:
                        logger.warning(f"pdfminer failed: {e}")
                        
                # 2단계: PyPDF2 fallback
                logger.info(f"Using PyPDF2 fallback for {file_path}")
                text = await self._extract_text_from_pdf_fallback(file_path)
                logger.info(f"Text extracted via PyPDF2: {len(text)} chars")
                return text  # no_model에서는 OCR로 넘어가지 않음
            
            else:
                # openai, vllm 등 다른 프로바이더인 경우 무조건 OCR
                logger.info(f"Using OCR mode with provider: {provider}")
                return await self._extract_text_from_pdf_via_ocr(file_path)
                
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            
            # 에러 발생시에도 실시간 설정에 따라 처리
            current_config = self._get_current_image_text_config()
            provider = current_config.get('provider', 'no_model')
            
            if provider == 'no_model':
                # no_model인 경우 기본 fallback만 시도
                try:
                    return await self._extract_text_from_pdf_fallback(file_path)
                except:
                    return "[PDF 파일: 텍스트 추출 중 오류 발생]"
            else:
                # 다른 프로바이더인 경우 OCR 시도
                return await self._extract_text_from_pdf_via_ocr(file_path)

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
    
    async def _extract_text_from_pdf_via_ocr(self, file_path: str) -> str:
        """PDF를 이미지로 변환 후 배치 OCR 메서드 사용"""
        try:
            if not PDF2IMAGE_AVAILABLE:
                logger.error("pdf2image not available for OCR processing")
                return "[PDF 파일: pdf2image 라이브러리가 필요합니다]"
            
            current_config = self._get_current_image_text_config()
            if not self._is_image_text_enabled(current_config):
                logger.warning("OCR is disabled, falling back to text extraction")
                return await self._extract_text_from_pdf_fallback(file_path)
            
            import tempfile
            import os
            
            logger.info(f"Converting PDF to images for OCR: {file_path}")
            
            # PDF를 이미지로 변환
            images = convert_from_path(file_path, dpi=300)
            
            temp_files = []
            
            try:
                # 모든 이미지를 임시 파일로 저장
                for i, image in enumerate(images):
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                        image.save(temp_file.name, 'PNG')
                        temp_files.append(temp_file.name)
                
                # 🔥 배치 처리로 OCR 수행
                batch_size = current_config.get('batch_size', 1)
                logger.info(f"Processing {len(temp_files)} pages with batch size: {batch_size}")
                
                page_texts = await self._convert_images_to_text_batch(temp_files, batch_size)
                
                # 결과 조합
                all_text = ""
                for i, page_text in enumerate(page_texts):
                    if not page_text.startswith("[이미지 파일:"):  # 오류 메시지가 아닌 경우
                        all_text += f"\n=== 페이지 {i+1} (OCR) ===\n"
                        all_text += page_text + "\n"
                    else:
                        logger.warning(f"OCR failed for page {i+1}: {page_text}")
                       
            finally:
                # 임시 파일 정리
                for temp_file in temp_files:
                    try:
                        os.unlink(temp_file)
                    except:
                        pass
            
            if all_text.strip():
                logger.info(f"Successfully extracted text via batch OCR: {len(all_text)} chars")
                return self.clean_text(all_text)
            else:
                logger.warning("OCR failed, falling back to text extraction")
                return await self._extract_text_from_pdf_fallback(file_path)
                
        except Exception as e:
            logger.error(f"OCR processing failed for {file_path}: {e}")
            logger.warning("OCR failed, falling back to text extraction")
            return await self._extract_text_from_pdf_fallback(file_path)

    # DOCX 관련 메서드들
    async def _convert_docx_to_images(self, file_path: str) -> List[str]:
        """DOCX를 이미지로 변환하여 임시 파일 리스트 반환"""
        import tempfile
        import os
        
        temp_files = []
        
        try:
            # 방법 1: docx2pdf + pdf2image 사용 (가장 권장)
            if DOCX2PDF_AVAILABLE and PDF2IMAGE_AVAILABLE:
                logger.info("Converting DOCX to PDF, then to images using docx2pdf + pdf2image")
                
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
                    # DOCX를 PDF로 변환
                    docx_to_pdf_convert(file_path, temp_pdf.name)
                    
                    # PDF를 이미지로 변환
                    images = convert_from_path(temp_pdf.name, dpi=300)
                    
                    for i, image in enumerate(images):
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_img:
                            image.save(temp_img.name, 'PNG')
                            temp_files.append(temp_img.name)
                    
                    # 임시 PDF 파일 삭제
                    os.unlink(temp_pdf.name)
                    
                return temp_files
            
            # 방법 2: LibreOffice 커맨드라인 사용
            elif PDF2IMAGE_AVAILABLE:
                logger.info("Trying LibreOffice command-line conversion")
                import subprocess
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    try:
                        # LibreOffice로 DOCX를 PDF로 변환
                        subprocess.run([
                            'libreoffice', '--headless', '--convert-to', 'pdf',
                            '--outdir', temp_dir, file_path
                        ], check=True, capture_output=True)
                        
                        # 변환된 PDF 파일 찾기
                        pdf_name = Path(file_path).stem + '.pdf'
                        pdf_path = os.path.join(temp_dir, pdf_name)
                        
                        if os.path.exists(pdf_path):
                            # PDF를 이미지로 변환
                            images = convert_from_path(pdf_path, dpi=300)
                            
                            for i, image in enumerate(images):
                                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_img:
                                    image.save(temp_img.name, 'PNG')
                                    temp_files.append(temp_img.name)
                            
                            return temp_files
                        
                    except (subprocess.CalledProcessError, FileNotFoundError) as e:
                        logger.warning(f"LibreOffice conversion failed: {e}")
            
            # 모든 방법이 실패한 경우
            logger.error("No available method to convert DOCX to images")
            return []
            
        except Exception as e:
            logger.error(f"Error converting DOCX to images: {e}")
            # 임시 파일 정리
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
            return []

    async def _extract_text_from_docx_via_ocr(self, file_path: str) -> str:
        """DOCX를 이미지로 변환 후 배치 OCR 처리"""
        try:
            current_config = self._get_current_image_text_config()
            if not self._is_image_text_enabled(current_config):
                logger.warning("OCR is disabled for DOCX, falling back to text extraction")
                return await self._extract_text_from_docx_fallback(file_path)
            
            logger.info(f"Converting DOCX to images for OCR: {file_path}")
            
            image_files = await self._convert_docx_to_images(file_path)
            
            if not image_files:
                logger.warning("Failed to convert DOCX to images, falling back to text extraction")
                return await self._extract_text_from_docx_fallback(file_path)
            
            try:
                # 🔥 배치 처리로 OCR 수행
                batch_size = current_config.get('batch_size', 1)
                logger.info(f"Processing {len(image_files)} DOCX pages with batch size: {batch_size}")
                
                page_texts = await self._convert_images_to_text_batch(image_files, batch_size)
                
                # 결과 조합
                all_text = ""
                for i, page_text in enumerate(page_texts):
                    if not page_text.startswith("[이미지 파일:"):  # 오류 메시지가 아닌 경우
                        all_text += f"\n=== 페이지 {i+1} (OCR) ===\n"
                        all_text += page_text + "\n"
                    else:
                        logger.warning(f"OCR failed for DOCX page {i+1}: {page_text}")
                        
            finally:
                # 임시 파일 정리
                for temp_file in image_files:
                    try:
                        import os
                        os.unlink(temp_file)
                    except:
                        pass
            
            if all_text.strip():
                logger.info(f"Successfully extracted DOCX text via batch OCR: {len(all_text)} chars")
                return self.clean_text(all_text)
            else:
                logger.warning("DOCX OCR failed, falling back to text extraction")
                return await self._extract_text_from_docx_fallback(file_path)
                
        except Exception as e:
            logger.error(f"DOCX OCR processing failed for {file_path}: {e}")
            logger.warning("DOCX OCR failed, falling back to text extraction")
            return await self._extract_text_from_docx_fallback(file_path)

    async def _extract_text_from_docx_fallback(self, file_path: str) -> str:
        """DOCX 파일에서 기본 텍스트 추출 (기존 방법)"""
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

    async def _extract_text_from_docx(self, file_path: str) -> str:
        """DOCX 파일에서 텍스트 추출 (실시간 설정에 따라 텍스트 추출 또는 OCR)"""
        try:
            # 🔥 실시간 설정 가져오기
            current_config = self._get_current_image_text_config()
            provider = current_config.get('provider', 'no_model')
            
            logger.info(f"🔄 Real-time DOCX processing with provider: {provider}")
            
            # no_model인 경우에만 기본 텍스트 추출
            if provider == 'no_model':
                logger.info("Using DOCX text extraction mode (no_model)")
                return await self._extract_text_from_docx_fallback(file_path)
            
            else:
                # openai, vllm 등 다른 프로바이더인 경우 OCR 시도
                logger.info(f"Using DOCX OCR mode with provider: {provider}")
                return await self._extract_text_from_docx_via_ocr(file_path)
                
        except Exception as e:
            logger.error(f"DOCX processing failed: {e}")
            
            # 에러 발생시에도 실시간 설정에 따라 처리
            current_config = self._get_current_image_text_config()
            provider = current_config.get('provider', 'no_model')
            
            if provider == 'no_model':
                # no_model인 경우 기본 fallback만 시도
                try:
                    return await self._extract_text_from_docx_fallback(file_path)
                except:
                    return "[DOCX 파일: 텍스트 추출 중 오류 발생]"
            else:
                # 다른 프로바이더인 경우 fallback으로 기본 텍스트 추출 시도
                try:
                    logger.warning("DOCX OCR failed, trying basic text extraction")
                    return await self._extract_text_from_docx_fallback(file_path)
                except:
                    return "[DOCX 파일: OCR 및 텍스트 추출 모두 실패]"

    # PPT 관련 메서드들
    async def _convert_ppt_to_images(self, file_path: str) -> List[str]:
        """PPT를 이미지로 변환하여 임시 파일 리스트 반환"""
        import tempfile
        import os
        
        temp_files = []
        
        try:
            # 방법 1: LibreOffice 커맨드라인 사용 (가장 권장)
            if PDF2IMAGE_AVAILABLE:
                logger.info("Converting PPT to PDF, then to images using LibreOffice + pdf2image")
                import subprocess
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    try:
                        # LibreOffice로 PPT를 PDF로 변환
                        subprocess.run([
                            'libreoffice', '--headless', '--convert-to', 'pdf',
                            '--outdir', temp_dir, file_path
                        ], check=True, capture_output=True)
                        
                        # 변환된 PDF 파일 찾기
                        pdf_name = Path(file_path).stem + '.pdf'
                        pdf_path = os.path.join(temp_dir, pdf_name)
                        
                        if os.path.exists(pdf_path):
                            # PDF를 이미지로 변환
                            images = convert_from_path(pdf_path, dpi=300)
                            
                            for i, image in enumerate(images):
                                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_img:
                                    image.save(temp_img.name, 'PNG')
                                    temp_files.append(temp_img.name)
                            
                            return temp_files
                        
                    except (subprocess.CalledProcessError, FileNotFoundError) as e:
                        logger.warning(f"LibreOffice PPT conversion failed: {e}")
            
            # 방법 2: python-pptx + PIL을 이용한 텍스트 렌더링 (fallback, 품질 낮음)
            if PIL_AVAILABLE and PYTHON_PPTX_AVAILABLE:
                logger.warning("Using fallback PIL text rendering for PPT (low quality)")
                return await self._render_ppt_text_to_images(file_path)
            
            # 모든 방법이 실패한 경우
            logger.error("No available method to convert PPT to images")
            return []
            
        except Exception as e:
            logger.error(f"Error converting PPT to images: {e}")
            # 임시 파일 정리
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
            return []

    async def _render_ppt_text_to_images(self, file_path: str) -> List[str]:
        """PPT 텍스트를 PIL로 이미지로 렌더링 (fallback 방법)"""
        import tempfile
        
        try:
            # PPT에서 슬라이드별 텍스트 추출
            prs = Presentation(file_path)
            temp_files = []
            
            for slide_num, slide in enumerate(prs.slides):
                # 슬라이드에서 모든 텍스트 추출
                slide_text = ""
                
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        slide_text += shape.text + "\n"
                
                if not slide_text.strip():  # 빈 슬라이드 스킵
                    continue
                
                # 이미지 생성
                img_width, img_height = 1200, 900  # 슬라이드 비율 (4:3)
                img = Image.new('RGB', (img_width, img_height), color='white')
                draw = ImageDraw.Draw(img)
                
                try:
                    # 기본 폰트 사용
                    font = ImageFont.load_default()
                except:
                    font = None
                
                y_offset = 50
                line_height = 25
                
                # 슬라이드 제목 추가
                draw.text((50, y_offset), f"=== 슬라이드 {slide_num + 1} ===", 
                            fill='black', font=font)
                y_offset += line_height * 2
                
                # 텍스트 렌더링
                lines = slide_text.split('\n')
                for line in lines:
                    if line.strip() and y_offset < img_height - 50:
                        # 긴 줄은 여러 줄로 분할
                        if len(line) > 80:
                            words = line.split()
                            current_line = ""
                            for word in words:
                                if len(current_line + word) < 80:
                                    current_line += word + " "
                                else:
                                    if current_line.strip():
                                        draw.text((50, y_offset), current_line.strip(), 
                                                fill='black', font=font)
                                        y_offset += line_height
                                    current_line = word + " "
                            if current_line.strip():
                                draw.text((50, y_offset), current_line.strip(), 
                                        fill='black', font=font)
                                y_offset += line_height
                        else:
                            draw.text((50, y_offset), line, fill='black', font=font)
                            y_offset += line_height
                
                # 임시 파일로 저장
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                    img.save(temp_file.name, 'PNG')
                    temp_files.append(temp_file.name)
            
            return temp_files
            
        except Exception as e:
            logger.error(f"Error rendering PPT text to images: {e}")
            return []

    async def _extract_text_from_ppt_via_ocr(self, file_path: str) -> str:
        """PPT를 이미지로 변환 후 배치 OCR 처리"""
        try:
            current_config = self._get_current_image_text_config()
            if not self._is_image_text_enabled(current_config):
                logger.warning("OCR is disabled for PPT, falling back to text extraction")
                return await self._extract_text_from_ppt_fallback(file_path)
            
            logger.info(f"Converting PPT to images for OCR: {file_path}")
            
            image_files = await self._convert_ppt_to_images(file_path)
            
            if not image_files:
                logger.warning("Failed to convert PPT to images, falling back to text extraction")
                return await self._extract_text_from_ppt_fallback(file_path)
            
            try:
                # 🔥 배치 처리로 OCR 수행
                batch_size = current_config.get('batch_size', 1)
                logger.info(f"Processing {len(image_files)} PPT slides with batch size: {batch_size}")
                
                slide_texts = await self._convert_images_to_text_batch(image_files, batch_size)
                
                # 결과 조합
                all_text = ""
                for i, slide_text in enumerate(slide_texts):
                    if not slide_text.startswith("[이미지 파일:"):  # 오류 메시지가 아닌 경우
                        all_text += f"\n=== 슬라이드 {i+1} (OCR) ===\n"
                        all_text += slide_text + "\n"
                    else:
                        logger.warning(f"OCR failed for PPT slide {i+1}: {slide_text}")
                        
            finally:
                # 임시 파일 정리
                for temp_file in image_files:
                    try:
                        import os
                        os.unlink(temp_file)
                    except:
                        pass
            
            if all_text.strip():
                logger.info(f"Successfully extracted PPT text via batch OCR: {len(all_text)} chars")
                return self.clean_text(all_text)
            else:
                logger.warning("PPT OCR failed, falling back to text extraction")
                return await self._extract_text_from_ppt_fallback(file_path)
                
        except Exception as e:
            logger.error(f"PPT OCR processing failed for {file_path}: {e}")
            logger.warning("PPT OCR failed, falling back to text extraction")
            return await self._extract_text_from_ppt_fallback(file_path)

    async def _extract_text_from_ppt_fallback(self, file_path: str) -> str:
        """PPT 파일에서 기본 텍스트 추출 (python-pptx 사용)"""
        if not PYTHON_PPTX_AVAILABLE:
            raise Exception("python-pptx is required for PPT file processing but is not available")
        
        try:
            prs = Presentation(file_path)
            text = ""
            
            for slide_num, slide in enumerate(prs.slides):
                logger.info(f"Processing slide {slide_num + 1}/{len(prs.slides)}")
                
                # 슬라이드 제목 추가
                text += f"\n=== 슬라이드 {slide_num + 1} ===\n"
                
                # 슬라이드의 모든 도형에서 텍스트 추출
                slide_content = ""
                tables_found = 0
                
                for shape in slide.shapes:
                    # 텍스트 도형 처리
                    if hasattr(shape, "text") and shape.text.strip():
                        slide_content += shape.text + "\n"
                    
                    # 표 처리
                    elif hasattr(shape, "table"):
                        tables_found += 1
                        slide_content += f"\n--- 표 {tables_found} ---\n"
                        
                        table = shape.table
                        for row in table.rows:
                            row_text = []
                            for cell in row.cells:
                                cell_text = cell.text.strip()
                                row_text.append(cell_text)
                            
                            if any(cell.strip() for cell in row_text):  # 빈 행이 아닌 경우만
                                slide_content += " | ".join(row_text) + "\n"
                        
                        slide_content += f"--- 표 {tables_found} 끝 ---\n\n"
                    
                    # 차트나 다른 객체의 경우 타입 정보만 추가
                    elif hasattr(shape, "chart"):
                        slide_content += "[차트 객체]\n"
                    elif hasattr(shape, "picture"):
                        slide_content += "[이미지 객체]\n"
                
                # 슬라이드 노트 추가 (있는 경우)
                if hasattr(slide, "notes_slide") and slide.notes_slide.notes_text_frame:
                    notes_text = slide.notes_slide.notes_text_frame.text.strip()
                    if notes_text:
                        slide_content += f"\n[슬라이드 노트]\n{notes_text}\n"
                
                # 빈 슬라이드가 아닌 경우만 추가
                if slide_content.strip():
                    text += slide_content + "\n"
                else:
                    text += "[빈 슬라이드]\n\n"
            
            logger.info(f"Extracted text from {len(prs.slides)} slides, found {tables_found} tables total")
            return self.clean_text(text)
            
        except Exception as e:
            logger.error(f"Error extracting text from PPT {file_path}: {e}")
            raise

    async def _extract_text_from_ppt(self, file_path: str) -> str:
        """PPT 파일에서 텍스트 추출 (실시간 설정에 따라 텍스트 추출 또는 OCR)"""
        try:
            # 🔥 실시간 설정 가져오기
            current_config = self._get_current_image_text_config()
            provider = current_config.get('provider', 'no_model')
            
            logger.info(f"🔄 Real-time PPT processing with provider: {provider}")
            
            # no_model인 경우에만 기본 텍스트 추출
            if provider == 'no_model':
                logger.info("Using PPT text extraction mode (no_model)")
                return await self._extract_text_from_ppt_fallback(file_path)
            
            else:
                # openai, vllm 등 다른 프로바이더인 경우 OCR 시도
                logger.info(f"Using PPT OCR mode with provider: {provider}")
                return await self._extract_text_from_ppt_via_ocr(file_path)
                
        except Exception as e:
            logger.error(f"PPT processing failed: {e}")
            
            # 에러 발생시에도 실시간 설정에 따라 처리
            current_config = self._get_current_image_text_config()
            provider = current_config.get('provider', 'no_model')
            
            if provider == 'no_model':
                # no_model인 경우 기본 fallback만 시도
                try:
                    return await self._extract_text_from_ppt_fallback(file_path)
                except:
                    return "[PPT 파일: 텍스트 추출 중 오류 발생]"
            else:
                # 다른 프로바이더인 경우 fallback으로 기본 텍스트 추출 시도
                try:
                    logger.warning("PPT OCR failed, trying basic text extraction")
                    return await self._extract_text_from_ppt_fallback(file_path)
                except:
                    return "[PPT 파일: OCR 및 텍스트 추출 모두 실패]"

    # 기존 헬퍼 메서드들
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
    
    async def extract_text_from_file(self, file_path: str, file_extension: str) -> str:
        """파일에서 텍스트 추출"""
        category = self.get_file_category(file_extension)
        logger.info(f"Extracting text from {file_extension} file ({category} category): {file_path}")

        if file_extension == 'pdf':
            return await self._extract_text_from_pdf(file_path)
        elif file_extension in ['docx', 'doc']:
            return await self._extract_text_from_docx(file_path)
        elif file_extension in ['pptx', 'ppt']:
            return await self._extract_text_from_ppt(file_path)
        elif file_extension in ['xlsx', 'xls']:
            return await self._extract_text_from_excel(file_path)
        elif file_extension in ['csv', 'tsv']:
            return await self._extract_text_from_csv(file_path)
        elif file_extension in self.image_types:
            return await self._convert_image_to_text(file_path)
        elif file_extension in (self.text_types + self.code_types + self.config_types + 
                                self.script_types + self.log_types + self.web_types):
            return await self._extract_text_from_text_file(file_path, file_extension)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

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
        """Excel(xlsx/xls) 파일에서 텍스트 추출 (pandas 없이)"""
        ext = os.path.splitext(file_path)[1].lower()
        text = ""

        try:
            if ext == ".xlsx":
                if not OPENPYXL_AVAILABLE:
                    raise Exception("openpyxl이 설치되어야 .xlsx 파일을 처리할 수 있습니다.")
                wb = load_workbook(file_path, data_only=True)
                for sheet_name in wb.sheetnames:
                    ws = wb[sheet_name]
                    text += f"\n=== 시트: {sheet_name} ===\n"
                    for row in ws.iter_rows(values_only=True):
                        row_text = " | ".join(str(v) for v in row if v is not None)
                        if row_text.strip():
                            text += row_text + "\n"

            elif ext == ".xls":
                if not XLRD_AVAILABLE:
                    raise Exception("xlrd가 설치되어야 .xls 파일을 처리할 수 있습니다.")
                wb = xlrd.open_workbook(file_path)
                for sheet in wb.sheets():
                    text += f"\n=== 시트: {sheet.name} ===\n"
                    for row_idx in range(sheet.nrows):
                        row_values = sheet.row_values(row_idx)
                        row_text = " | ".join(str(v) for v in row_values if v)
                        if row_text.strip():
                            text += row_text + "\n"
            else:
                raise Exception(f"지원하지 않는 Excel 형식입니다: {ext}")

            return self.clean_text(text)
        except Exception as e:
            logger.error(f"Error extracting text from Excel {file_path}: {e}")
            raise

    async def _extract_text_from_csv(self, file_path: str, encoding: str = "utf-8") -> str:
        """CSV 파일에서 텍스트 추출 (pandas 없이)"""
        text = ""
        try:
            async with aiofiles.open(file_path, mode="r", encoding=encoding) as f:
                reader = csv.reader((await f.read()).splitlines())
                for row in reader:
                    row_text = " | ".join(row)
                    if row_text.strip():
                        text += row_text + "\n"
            return self.clean_text(text)
        except Exception as e:
            logger.error(f"Error extracting text from CSV {file_path}: {e}")
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
            
            # \n\n\n이 포함되어 있으면 먼저 이것으로 분할
            if "\n\n\n" in text:
                logger.info("Found \\n\\n\\n in text, splitting by major sections first")
                major_sections = text.split("\n\n\n")
                all_chunks = []
                
                for i, section in enumerate(major_sections):
                    if not section.strip():
                        continue
                        
                    logger.info(f"Processing major section {i+1}/{len(major_sections)}")
                    
                    # 각 섹션이 chunk_size보다 작으면 그대로 사용
                    if len(section) <= chunk_size:
                        all_chunks.append(section.strip())
                    else:
                        # 큰 섹션은 추가로 청킹
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            length_function=len,
                            separators=["\n\n", "\n", " ", ""]
                        )
                        section_chunks = text_splitter.split_text(section)
                        all_chunks.extend(section_chunks)
                
                logger.info(f"Text split into {len(all_chunks)} chunks using major sections (\\n\\n\\n)")
                return all_chunks
            
            else:
                # 기존 방식으로 청킹
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

    def get_current_config_status(self) -> Dict[str, Any]:
        """현재 설정 상태 반환 (디버깅용)"""
        try:
            current_config = self._get_current_image_text_config()
            return {
                "provider": current_config.get('provider', 'unknown'),
                "ocr_enabled": self._is_image_text_enabled(current_config),
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
            current_config = self._get_current_image_text_config()
            provider = current_config.get('provider', 'no_model')
            
            logger.info(f"🔍 Test - Current provider: {provider}")
            logger.info(f"🔍 Test - Current config: {current_config}")
            logger.info(f"🔍 Test - OCR enabled: {self._is_image_text_enabled(current_config)}")
            
        except Exception as e:
            logger.error(f"Error in test method: {e}")
            raise
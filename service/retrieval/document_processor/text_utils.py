"""
텍스트 처리 유틸리티
"""

import re
import logging
from typing import List
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .constants import LANGCHAIN_CODE_LANGUAGE_MAP

logger = logging.getLogger("document-processor")

class TextUtils:
    """텍스트 처리 유틸리티 클래스"""
    
    @staticmethod
    def clean_text(text):
        """텍스트 정리"""
        if not text:
            return ""
        # 연속된 공백을 하나로 통합
        text = re.sub(r'\s+', ' ', text)
        # 연속된 줄바꿈을 두 개로 제한
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        return text.strip()
    
    @staticmethod
    def clean_code_text(text: str, file_type: str) -> str:
        """코드 텍스트 정리 (코드의 구조를 보존)"""
        if not text:
            return ""
        
        # 코드 파일의 경우 들여쓰기와 줄바꿈을 보존
        text = text.rstrip()
        
        # 탭을 4개의 스페이스로 변환 (일관성을 위해)
        text = text.replace('\t', '    ')
        
        return text
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """텍스트를 청크로 분할"""
        try:
            if not text or not text.strip():
                logger.warning("Empty text provided for chunking")
                return [""]
            
            # [색션 구분] 또는 [표 구분]이 포함되어 있으면 먼저 이것으로 분할
            if "[색션 구분]" in text or "[표 구분]" in text:
                logger.info("Found section markers in text, splitting by major sections first")
                
                # 두 구분자를 모두 고려해서 분할
                # 먼저 [색션 구분]으로 분할하고, 각 섹션에서 [표 구분]으로 추가 분할
                temp_sections = text.split("[색션 구분]")
                major_sections = []
                
                for section in temp_sections:
                    if "[표 구분]" in section:
                        table_sections = section.split("[표 구분]")
                        major_sections.extend([s.strip() for s in table_sections if s.strip()])
                    else:
                        if section.strip():
                            major_sections.append(section.strip())
                
                logger.info(f"Split into {len(major_sections)} sections using markers")
                
                # 작은 섹션들을 합치기
                merged_sections = []
                current_merged = ""
                
                for i, section in enumerate(major_sections):
                    # 현재 합쳐진 것과 새 섹션을 합쳤을 때의 길이 계산
                    if current_merged:
                        potential_merged = current_merged + "\n\n" + section
                    else:
                        potential_merged = section
                    
                    if len(potential_merged) <= chunk_size:
                        # chunk_size를 넘지 않으면 계속 합치기
                        current_merged = potential_merged
                        logger.info(f"Merging section {i+1} (total length: {len(current_merged)})")
                    else:
                        # chunk_size를 넘으면 이전까지 합친 것을 저장하고 새로 시작
                        if current_merged:
                            merged_sections.append(current_merged)
                            logger.info(f"Added merged section with length: {len(current_merged)}")
                        current_merged = section
                
                # 마지막 섹션 추가
                if current_merged:
                    merged_sections.append(current_merged)
                    logger.info(f"Added final merged section with length: {len(current_merged)}")
                
                # 합쳐진 섹션들을 최종 청킹
                all_chunks = []
                for i, section in enumerate(merged_sections):
                    logger.info(f"Processing merged section {i+1}/{len(merged_sections)} (length: {len(section)})")
                    
                    # 섹션이 chunk_size의 2배를 초과하면 추가로 청킹
                    if len(section) > chunk_size * 2:
                        logger.info(f"Merged section {i+1} is too large ({len(section)} chars), splitting further")
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            length_function=len,
                            separators=["\n\n", "\n", " ", ""]
                        )
                        section_chunks = text_splitter.split_text(section)
                        all_chunks.extend(section_chunks)
                    else:
                        # 섹션이 적당한 크기면 그대로 사용
                        all_chunks.append(section)
                
                logger.info(f"Text split into {len(all_chunks)} chunks after merging small sections")
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

    
    @staticmethod
    def chunk_code_text(text: str, file_type: str, chunk_size: int = 1500, chunk_overlap: int = 300) -> List[str]:
        """코드 텍스트를 청크로 분할 (언어별 구문 구조를 고려한 분할)"""
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
    
    @staticmethod
    def estimate_chunks_count(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> int:
        """텍스트에서 생성될 청크 수 추정"""
        if not text:
            return 0
        
        text_length = len(text)
        if text_length <= chunk_size:
            return 1
        
        # 간단한 추정 공식
        effective_chunk_size = chunk_size - chunk_overlap
        estimated_chunks = (text_length - chunk_overlap) // effective_chunk_size + 1
        return max(1, estimated_chunks)
    
    @staticmethod
    def is_similar_table_text(text1: str, text2: str, threshold: float = 0.8) -> bool:
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
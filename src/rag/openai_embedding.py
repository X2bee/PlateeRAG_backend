"""
OpenAI 임베딩 클라이언트
"""

import asyncio
from typing import List, Dict, Any
import logging
from openai import AsyncOpenAI
from .base_embedding import BaseEmbedding

logger = logging.getLogger("embeddings.openai")

class OpenAIEmbedding(BaseEmbedding):
    """OpenAI 임베딩 클라이언트"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key", "")
        self.model = config.get("model", "text-embedding-3-small")
        self.client = None
        self._dimension = None
        
        # 모델별 차원 수 매핑
        self.model_dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072
        }
        
        if self.api_key:
            try:
                self.client = AsyncOpenAI(api_key=self.api_key)
                logger.info(f"OpenAI embedding client initialized with model: {self.model}")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.client = None
        else:
            logger.warning("OpenAI API key not provided")
            self.client = None
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """여러 문서를 임베딩으로 변환"""
        if not self.client:
            raise ValueError("OpenAI client not initialized")
        
        if not texts:
            raise ValueError("Empty text list provided")
        
        # 빈 문자열이나 None 값 필터링 및 검증
        valid_texts = []
        for i, text in enumerate(texts):
            if text is None:
                logger.warning(f"None value found at index {i}, skipping")
                continue
            
            # 문자열로 변환 및 공백 제거
            text_str = str(text).strip()
            if not text_str:
                logger.warning(f"Empty text found at index {i}, using placeholder")
                text_str = "empty_text_placeholder"
            
            valid_texts.append(text_str)
        
        if not valid_texts:
            raise ValueError("No valid texts after filtering")
        
        try:
            # 배치 크기 제한 (OpenAI는 한 번에 최대 2048개)
            batch_size = 100
            all_embeddings = []
            
            for i in range(0, len(valid_texts), batch_size):
                batch_texts = valid_texts[i:i + batch_size]
                
                # 추가 검증: 각 텍스트가 문자열인지 확인
                batch_texts = [str(text) for text in batch_texts if text is not None]
                
                if not batch_texts:
                    continue
                
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=batch_texts,
                    encoding_format="float"
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
            
            logger.info(f"Generated {len(all_embeddings)} document embeddings")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate document embeddings: {e}")
            logger.error(f"Input sample: {valid_texts[:3] if valid_texts else 'empty'}")
            raise
    
    async def embed_query(self, text: str) -> List[float]:
        """단일 쿼리를 임베딩으로 변환"""
        if not self.client:
            raise ValueError("OpenAI client not initialized")
        
        # 입력 값 검증 및 처리
        if text is None:
            raise ValueError("Query text cannot be None")
        
        # 문자열로 변환 및 공백 제거
        text_str = str(text).strip()
        if not text_str:
            raise ValueError("Query text cannot be empty")
        
        try:
            # OpenAI API는 문자열을 배열로 감싸서 전달
            response = await self.client.embeddings.create(
                model=self.model,
                input=[text_str],  # 문자열을 배열로 감싸서 전달
                encoding_format="float"
            )
            
            if not response.data:
                raise ValueError("No embedding data returned from OpenAI API")
            
            embedding = response.data[0].embedding
            logger.info(f"Generated query embedding for text: {text_str[:50]}...")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            logger.error(f"Input text: '{text_str[:100]}...' (length: {len(text_str)})")
            raise
    
    def get_embedding_dimension(self) -> int:
        """임베딩 차원 수 반환"""
        if self._dimension is not None:
            return self._dimension
            
        # 모델명으로 차원 수 추정
        dimension = self.model_dimensions.get(self.model, 1536)
        self._dimension = dimension
        return dimension
    
    async def is_available(self) -> bool:
        """OpenAI 서비스 사용 가능성 확인"""
        if not self.client:
            return False
        
        try:
            # 간단한 테스트 임베딩 생성
            response = await self.client.embeddings.create(
                model=self.model,
                input=["test"],
                encoding_format="float"
            )
            
            # 실제 차원 수 업데이트
            if response.data:
                self._dimension = len(response.data[0].embedding)
            
            return True
            
        except Exception as e:
            logger.warning(f"OpenAI embedding service not available: {e}")
            return False
    
    def get_provider_info(self) -> Dict[str, Any]:
        """OpenAI 제공자 정보 반환"""
        return {
            "provider": "openai",
            "model": self.model,
            "dimension": self.get_embedding_dimension(),
            "api_key_configured": bool(self.api_key),
            "available": bool(self.client)
        } 
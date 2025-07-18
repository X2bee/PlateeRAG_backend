"""
임베딩 클라이언트의 기본 추상 클래스
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger("embeddings")

class BaseEmbedding(ABC):
    """임베딩 클라이언트의 기본 추상 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 임베딩 설정 딕셔너리
        """
        self.config = config
        self.provider_name = self.__class__.__name__.replace("Embedding", "").lower()
        
    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        여러 문서를 임베딩으로 변환
        
        Args:
            texts: 임베딩할 텍스트 리스트
            
        Returns:
            임베딩 벡터 리스트
        """
        pass
    
    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """
        단일 쿼리를 임베딩으로 변환
        
        Args:
            text: 임베딩할 텍스트
            
        Returns:
            임베딩 벡터
        """
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        임베딩 차원 수 반환
        
        Returns:
            임베딩 벡터의 차원 수
        """
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """
        임베딩 서비스가 사용 가능한지 확인
        
        Returns:
            서비스 사용 가능 여부
        """
        pass
    
    def get_provider_info(self) -> Dict[str, Any]:
        """
        임베딩 제공자 정보 반환
        
        Returns:
            제공자 정보 딕셔너리
        """
        return {
            "provider": self.provider_name,
            "model": self.config.get("model", "unknown"),
            "dimension": self.get_embedding_dimension(),
            "available": False  # 서브클래스에서 오버라이드
        }
    
    async def cleanup(self):
        """
        임베딩 클라이언트 리소스 정리
        서브클래스에서 필요시 오버라이드
        """
        logger.info("Cleaning up %s embedding client", self.provider_name) 
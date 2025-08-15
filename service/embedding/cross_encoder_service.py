"""
Cross-encoder 기반 Reranking 서비스
"""
import logging
from typing import List, Tuple
from sentence_transformers import CrossEncoder
import torch

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    """Cross-encoder 모델을 사용한 문서 재순위 서비스"""
    
    def __init__(self, model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"):
        """
        Args:
            model_name: Cross-encoder 모델 이름
        """
        self.model_name = model_name
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Cross-encoder 모델 초기화"""
        try:
            logger.info(f"Loading cross-encoder model: {self.model_name} (첫 로딩은 1-3분 소요됨)")
            
            # GPU 사용 가능하면 GPU, 아니면 CPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            # 모델 다운로드 및 로드 (시간 소요 가능)
            import time
            start_time = time.time()
            
            self.model = CrossEncoder(
                self.model_name,
                device=device,
                max_length=512  # 토큰 길이 제한
            )
            
            load_time = time.time() - start_time
            logger.info(f"Cross-encoder model loaded successfully on {device} (로딩 시간: {load_time:.2f}초)")
            
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model: {e}")
            raise
    
    def rerank(self, query: str, documents: List[str], top_k: int = None) -> List[Tuple[int, float]]:
        """
        Cross-encoder를 사용하여 문서 재순위
        
        Args:
            query: 검색 쿼리
            documents: 문서 텍스트 리스트
            top_k: 반환할 상위 문서 개수 (None이면 전체)
            
        Returns:
            List[Tuple[int, float]]: (문서_인덱스, 점수) 튜플 리스트 (점수 내림차순)
        """
        if not self.model:
            raise RuntimeError("Cross-encoder model not initialized")
        
        if not documents:
            return []
        
        try:
            # (query, document) 쌍 생성
            query_doc_pairs = [(query, doc) for doc in documents]
            
            logger.info(f"Reranking {len(documents)} documents with cross-encoder")
            
            # Cross-encoder로 점수 계산
            scores = self.model.predict(query_doc_pairs)
            
            # (인덱스, 점수) 튜플로 변환하고 점수 내림차순 정렬
            indexed_scores = [(i, float(score)) for i, score in enumerate(scores)]
            indexed_scores.sort(key=lambda x: x[1], reverse=True)
            
            # top_k 제한
            if top_k:
                indexed_scores = indexed_scores[:top_k]
            
            logger.info(f"Cross-encoder reranking completed. Top score: {indexed_scores[0][1]:.4f}")
            
            return indexed_scores
            
        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            raise
    
    def cleanup(self):
        """모델 정리"""
        if self.model:
            del self.model
            self.model = None
            
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("Cross-encoder model cleaned up")

# 전역 인스턴스
_cross_encoder_reranker = None

def get_cross_encoder_reranker(model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1") -> CrossEncoderReranker:
    """Cross-encoder 재순위 서비스 싱글톤 인스턴스 반환"""
    global _cross_encoder_reranker
    
    if _cross_encoder_reranker is None or _cross_encoder_reranker.model_name != model_name:
        if _cross_encoder_reranker:
            _cross_encoder_reranker.cleanup()
        _cross_encoder_reranker = CrossEncoderReranker(model_name)
    
    return _cross_encoder_reranker

"""
임베딩 모듈
다양한 임베딩 제공자를 지원하는 통합 임베딩 시스템
"""

from .embedding_factory import EmbeddingFactory
from .base_embedding import BaseEmbedding

__all__ = ["EmbeddingFactory", "BaseEmbedding"] 
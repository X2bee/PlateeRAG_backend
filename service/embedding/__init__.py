from .fastembed_service import FastEmbedService
from .cross_encoder_service import get_cross_encoder_reranker

_FASTEMBED_SERVICE = None

def get_fastembed_service() -> FastEmbedService:
    global _FASTEMBED_SERVICE
    if _FASTEMBED_SERVICE is None:
        _FASTEMBED_SERVICE = FastEmbedService()
    return _FASTEMBED_SERVICE

"""
Embedding module initialization file.
"""

from .embedding_factory import EmbeddingFactory
from .base_embedding import BaseEmbedding

__all__ = [
    "EmbeddingFactory", 
    "BaseEmbedding",
]
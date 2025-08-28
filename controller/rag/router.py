from fastapi import APIRouter
from controller.rag.documentController import router as document_router
from controller.rag.embeddingController import router as embedding_router
from controller.rag.retrievalController import router as retrieval_router

# RAG 라우터 통합
rag_router = APIRouter(prefix="/api", tags=["RAG"])

rag_router.include_router(document_router)
rag_router.include_router(embedding_router)
rag_router.include_router(retrieval_router)

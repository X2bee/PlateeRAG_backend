import os
import logging
import asyncio
from typing import Optional, Dict, Any
from editor.node_composer import Node
from editor.utils.helper.service_helper import AppServiceManager
from service.database.models.vectordb import VectorDB
from fastapi import Request
from controller.rag.retrievalController import list_collections
from editor.utils.helper.async_helper import sync_run_async

logger = logging.getLogger(__name__)
enhance_prompt = """You are an AI assistant that must strictly follow these guidelines when using the provided document context:

1. ANSWER ONLY BASED ON PROVIDED CONTEXT: Use only the information from the retrieved documents to answer questions. Do not add information from your general knowledge.
2. BE PRECISE AND ACCURATE: Quote specific facts, numbers, and details exactly as they appear in the documents. Include relevant quotes when appropriate.
3. ACKNOWLEDGE LIMITATIONS: If the provided documents do not contain sufficient information to answer the user's question, clearly state "I don't have enough information in the provided documents to answer this question" or "The provided documents don't contain information about [specific topic]."
4. STAY FOCUSED: Answer only what the user asked. Do not provide additional information beyond what was requested unless it's directly relevant to the question.
5. CITE SOURCES: When possible, reference which document number contains the information you're using (e.g., "According to Document 1..." or "As mentioned in Document 2...").
6. BE CONCISE: Provide clear, direct answers without unnecessary elaboration. Focus on delivering exactly what the user needs.

Remember: It's better to say "I don't know" than to provide inaccurate or fabricated information."""

embedding_model_prompt = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "

class QdrantRetrievalTool(Node):
    categoryId = "xgen"
    functionId = "document_loaders"
    nodeId = "document_loaders/Qdrant"
    nodeName = "Qdrant Search"
    description = "RAG 서비스와 검색 파라미터를 설정하여 다음 노드로 전달하는 노드"
    tags = ["document_loader", "qdrant", "vector_db", "rag", "setup"]

    inputs = []
    outputs = [
        {"id": "rag_context", "name": "RAG Context", "type": "DocsContext"},
    ]

    parameters = [
        {"id": "collection_name", "name": "Collection Name", "type": "STR", "value": "Select Collection", "required": True, "is_api": True, "api_name": "api_collection", "options": []},
        {"id": "use_model_prompt", "name": "Use Model Prompt", "type": "BOOL", "value": True, "optional": True, "description": "임베딩 벡터 변환시 모델이 요구하는 프롬프트를 사용할지를 결정합니다."},
        {"id": "top_k", "name": "Top K Results", "type": "INT", "value": 4, "required": False, "optional": True, "min": 1, "max": 10, "step": 1},
        {"id": "score_threshold", "name": "Score Threshold", "type": "FLOAT", "value": 0.2, "required": False, "optional": True, "min": 0.0, "max": 1.0, "step": 0.1},
        {"id": "rerank", "name": "Enable Rerank", "type": "BOOL", "value": False, "required": False, "optional": True},
        {"id": "rerank_top_k", "name": "Rerank Top K", "type": "INT", "value": 5, "required": False, "optional": True, "min": 1, "max": 100, "step": 1},
        {"id": "enhance_prompt", "name": "Enhance Prompt", "type": "STR", "value": enhance_prompt, "required": False, "optional": True, "expandable": True, "description": "RAG 컨텍스트를 사용하여 응답을 향상시키기 위한 프롬프트입니다."},

    ]

    def api_collection(self, request: Request) -> Dict[str, Any]:
        collections = sync_run_async(list_collections(request))
        return [{"value": collection.get("collection_name"), "label": collection.get("collection_make_name")} for collection in collections]

    def execute(self, collection_name: str, top_k: int = 4, score_threshold: float = 0.2, enhance_prompt: str = enhance_prompt, use_model_prompt: bool = True, rerank: bool = False, rerank_top_k: int = 5):
        rag_service = AppServiceManager.get_rag_service()

        try:
            if not collection_name:
                return {
                    "error": "컬렉션 이름이 제공되지 않았습니다.",
                    "rag_service": None,
                    "search_params": None,
                    "status": "error"
                }

            if not rag_service:
                return {
                    "error": "RAG 서비스를 사용할 수 없습니다.",
                    "rag_service": None,
                    "search_params": None,
                    "status": "error"
                }

            rag_context = {
                "rag_service": rag_service,
                "search_params": {
                    "collection_name": collection_name,
                    "top_k": top_k,
                    "score_threshold": score_threshold,
                    "enhance_prompt": enhance_prompt,
                    "use_model_prompt": use_model_prompt,
                    "embedding_model_prompt": embedding_model_prompt,
                    "rerank": rerank,
                    "rerank_top_k": rerank_top_k
                },
                "status": "ready"
            }
            return rag_context

        except Exception as e:
            return {
                "error": f"RAG 컨텍스트 설정 중 오류: {str(e)}",
                "rag_service": None,
                "search_params": None,
                "status": "error"
            }

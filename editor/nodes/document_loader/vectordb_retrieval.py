import os
import logging
import asyncio
from typing import Optional, Dict, Any
from editor.node_composer import Node
from langchain.agents import tool
from editor.utils.helper.service_helper import AppServiceManager
from editor.utils.tools.async_helper import sync_run_async
from service.database.models.vectordb import VectorDB
from fastapi import Request
from controller.controller_helper import extract_user_id_from_request

logger = logging.getLogger(__name__)

class QdrantRetrievalTool(Node):
    categoryId = "langchain"
    functionId = "document_loaders"
    nodeId = "document_loaders/Qdrant"
    nodeName = "Qdrant Search"
    description = "RAG 서비스와 검색 파라미터를 설정하여 다음 노드로 전달하는 노드"
    tags = ["document_loader", "qdrant", "vector_db", "rag", "setup"]

    inputs = []
    outputs = [
        {"id": "rag_context", "name": "RAG Context", "type": "DICT"},
    ]

    parameters = [
        {"id": "collection_name", "name": "Collection Name", "type": "STR", "value": "Select Collection", "required": True, "is_api": True, "api_name": "api_collection", "options": []},
        {"id": "top_k", "name": "Top K Results", "type": "INT", "value": 4, "required": False, "optional": True, "min": 1, "max": 10, "step": 1},
        {"id": "score_threshold", "name": "Score Threshold", "type": "FLOAT", "value": 0.2, "required": False, "optional": True, "min": 0.0, "max": 1.0, "step": 0.1}
    ]

    def api_collection(self, request: Request) -> Dict[str, Any]:
        user_id = extract_user_id_from_request(request)
        db_service = request.app.state.app_db
        collections = db_service.find_by_condition(
            VectorDB,
            {
            "user_id": user_id
            },
            limit=1000,
        )
        return [{"value": collection.collection_name, "label": collection.collection_make_name} for collection in collections]

    def execute(self, collection_name: str, top_k: int = 4, score_threshold: float = 0.2):
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
                    "score_threshold": score_threshold
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

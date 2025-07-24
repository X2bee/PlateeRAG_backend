import os
import logging
import asyncio
from typing import Optional, Dict, Any
from editor.node_composer import Node
from langchain.agents import tool
from editor.utils.helper.service_helper import AppServiceManager
from editor.utils.tools.async_helper import sync_run_async
from service.database.models.vectordb import VectorDB

logger = logging.getLogger(__name__)

class QdrantRetrievalTool(Node):
    categoryId = "langchain"
    functionId = "document_loaders"

    nodeId = "document_loaders/QdrantRetrievalTool"
    nodeName = "Qdrant Retrieval Tool"
    description = "VectorDB 검색 Tool을 전달"
    tags = ["document_loader", "qdrant", "vector_db", "rag", "setup"]

    inputs = []
    outputs = [
        {
            "id": "tools",
            "name": "Tools",
            "type": "TOOL"
        },
    ]

    def get_collection_options(self):
        """컬렉션 옵션을 가져오는 함수"""
        return getattr(self, '_collections_cache', [])

    parameters = [
        {
            "id": "tool_name",
            "name": "Tool Name",
            "type": "STR",
            "value": "도구의 명칭",
            "required": True,
        },
        {
            "id": "description",
            "name": "Description",
            "type": "STR",
            "value": "주어진 질문에 대해 검색을 수행하는 Tool입니다.",
            "required": True,
        },
        {
            "id": "collection_name",
            "name": "Collection Name",
            "type": "STR",
            "value": "Select Collection",
            "required": True,
            "options": get_collection_options,
        },
        {
            "id": "top_k",
            "name": "Top K Results",
            "type": "INT",
            "value": 4,
            "required": False,
            "optional": True,
            "min": 1,
            "max": 10,
            "step": 1
        },
        {
            "id": "score_threshold",
            "name": "Score Threshold",
            "type": "FLOAT",
            "value": 0.2,
            "required": False,
            "optional": True,
            "min": 0.0,
            "max": 1.0,
            "step": 0.1
        }
    ]

    def __init__(self, user_id: str = None, **kwargs):
        super().__init__(**kwargs)
        self.user_id = user_id
        self.db_service = AppServiceManager.get_db_manager()
        self.rag_service = AppServiceManager.get_rag_service()
        print(self.user_id)
        if not self.db_service:
            logger.error("RAG 서비스가 초기화되지 않았습니다. 서버가 실행 중인지 확인하세요.")
            raise RuntimeError("RAG 서비스가 초기화되지 않았습니다. 서버가 실행 중인지 확인하세요.")

        try:
            collections = self.db_service.find_by_condition(
                VectorDB,
                {
                "user_id": self.user_id
                },
                limit=1000,
            )
            self._collections_cache = [{"value": collection.collection_name, "label": collection.collection_make_name} for collection in collections]

        except Exception as e:
            logger.warning(f"초기 컬렉션 목록 로딩 실패: {e}")
            self._collections_cache = []

    def execute(self, tool_name, description, collection_name: str, top_k: int = 4, score_threshold: float = 0.2):
        def create_vectordb_tool():
            @tool(tool_name, description=description)
            def vectordb_retrieval_tool(query: str) -> str:
                try:
                    search_result = sync_run_async(self.rag_service.search_documents(
                        collection_name=collection_name,
                        query_text=query,
                        limit=top_k,
                        score_threshold=score_threshold
                    ))

                    results = search_result.get("results", [])
                    if not results:
                        return query

                    context_parts = []
                    for i, item in enumerate(results, 1):
                        if "chunk_text" in item and item["chunk_text"]:
                            score = item.get("score", 0.0)
                            chunk_text = item["chunk_text"]
                            context_parts.append(f"[문서 {i}] (관련도: {score:.3f})\n{chunk_text}")
                    if context_parts:
                        context_text = "\n".join(context_parts)
                        enhanced_prompt = f"""다음 제시되는 문서들을 참고하여 사용자 질문에 효과적으로 활용하세요:
[참고 문서]
{context_text}"""
                        return enhanced_prompt
                except Exception as e:
                    logger.error(f"RAG 검색 수행 중 오류: {e}")
                    return {"error": str(e)}

            return vectordb_retrieval_tool

        return create_vectordb_tool()

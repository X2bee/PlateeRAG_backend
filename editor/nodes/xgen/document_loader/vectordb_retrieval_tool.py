import os
import logging
import asyncio
from typing import Optional, Dict, Any
from editor.node_composer import Node
from langchain.agents import tool
from editor.utils.helper.service_helper import AppServiceManager
from editor.utils.helper.async_helper import sync_run_async
from service.database.models.vectordb import VectorDB
from fastapi import Request
from controller.rag.retrievalController import list_collections
from editor.utils.citation_prompt import citation_prompt
from controller.helper.singletonHelper import get_db_manager

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
    nodeId = "document_loaders/QdrantRetrievalTool"
    nodeName = "Qdrant Retrieval Tool"
    description = "VectorDB 검색 Tool을 전달"
    tags = ["document_loader", "qdrant", "vector_db", "rag", "setup"]

    inputs = []
    outputs = [
        {"id": "tools", "name": "Tools", "type": "TOOL"},
    ]

    parameters = [
        {"id": "tool_name", "name": "Tool Name", "type": "STR", "value": "tool_name", "required": True},
        {"id": "description", "name": "Description", "type": "STR", "value": "주어진 질문에 대해 검색을 수행하는 Tool입니다.", "required": True, "expandable": True, "description": "이 도구를 언제 사용하여야 하는지 설명합니다. AI는 해당 설명을 통해, 해당 도구를 언제 호출해야할지 결정할 수 있습니다."},
        {"id": "collection_name", "name": "Collection Name", "type": "STR", "value": "Select Collection", "required": True, "is_api": True, "api_name": "api_collection", "options": []},
        {"id": "use_model_prompt", "name": "Use Model Prompt", "type": "BOOL", "value": True, "optional": True, "description": "임베딩 벡터 변환시 모델이 요구하는 프롬프트를 사용할지를 결정합니다."},
        {"id": "top_k", "name": "Top K Results", "type": "INT", "value": 4, "required": False, "optional": True, "min": 1, "max": 10, "step": 1},
        {"id": "score_threshold", "name": "Score Threshold", "type": "FLOAT", "value": 0.2, "required": False, "optional": True, "min": 0.0, "max": 1.0, "step": 0.1},
        {"id": "rerank", "name": "Enable Rerank", "type": "BOOL", "value": False, "required": False, "optional": True},
        {"id": "rerank_top_k", "name": "Rerank Top K", "type": "INT", "value": 5, "required": False, "optional": True, "min": 1, "max": 100, "step": 1},
        {"id": "enhance_prompt", "name": "Enhance Prompt", "type": "STR", "value": enhance_prompt, "required": False, "optional": True, "expandable": True, "description": "검색된 자료를 어떻게 사용할 것인지 지시합니다."},
    ]

    def api_collection(self, request: Request) -> Dict[str, Any]:
        collections = list_collections(request)
        return [{"value": collection.get("collection_name"), "label": collection.get("collection_make_name")} for collection in collections]

    def execute(self, tool_name, description, collection_name: str, top_k: int = 4, use_model_prompt: bool = True, score_threshold: float = 0.2, enhance_prompt: str = enhance_prompt, rerank: bool = False, rerank_top_k: int = 5):
        def create_vectordb_tool():
            @tool(tool_name, description=description)
            def vectordb_retrieval_tool(query: str) -> str:
                rag_service = AppServiceManager.get_rag_service()
                try:
                    if use_model_prompt:
                        query = embedding_model_prompt + query

                    search_result = sync_run_async(rag_service.search_documents(
                        collection_name=collection_name,
                        query_text=query,
                        limit=top_k,
                        score_threshold=score_threshold,
                        rerank=rerank,
                        rerank_top_k=rerank_top_k
                    ))

                    results = search_result.get("results", [])
                    if not results:
                        return query

                    context_parts = []
                    for i, item in enumerate(results, 1):
                        if "chunk_text" in item and item["chunk_text"]:
                            item_file_name = item.get("file_name", "Unknown")
                            item_file_path = item.get("file_path", "Unknown")
                            item_page_number = item.get("page_number", 0)
                            item_line_start = item.get("line_start", 0)
                            item_line_end = item.get("line_end", 0)

                            score = item.get("score", 0.0)
                            chunk_text = item["chunk_text"]
                            context_parts.append(f"[문서 {i}](관련도: {score:.3f})\n[파일명] {item_file_name}\n[파일경로] {item_file_path}\n[페이지번호] {item_page_number}\n[문장시작줄] {item_line_start}\n[문장종료줄] {item_line_end}\n\n[내용]\n{chunk_text}")

                    if context_parts:
                        context_text = "\n".join(context_parts)
                        enhanced_prompt = f"""{enhance_prompt}
{context_text}"""
                        return enhanced_prompt
                except Exception as e:
                    logger.error(f"RAG 검색 수행 중 오류: {e}")
                    return {"error": str(e)}

            return vectordb_retrieval_tool

        return create_vectordb_tool()

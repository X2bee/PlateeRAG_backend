import os
import logging
import asyncio
from collections import Counter
from typing import Optional, Dict, Any
from editor.node_composer import Node
from editor.utils.citation_prompt import citation_prompt
from langchain.agents import tool
from editor.utils.helper.service_helper import AppServiceManager
from editor.utils.helper.async_helper import sync_run_async
from service.database.models.vectordb import VectorDB
from fastapi import Request
from controller.helper.controllerHelper import extract_user_id_from_request
from pydantic import BaseModel, Field

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

class ToolSchema(BaseModel):
    query: str = Field(..., description="Search query to get relevant information for the user's question")
    deep_search: bool = Field(False, description="Use only when exploring all information rather than appropriate search information for the user's question.")

class QdrantRetrievalTool(Node):
    categoryId = "xgen"
    functionId = "document_loaders"
    nodeId = "document_loaders/QdrantRetrievalTool_V2"
    nodeName = "Qdrant Retrieval Tool V2"
    description = "VectorDB 검색 Tool을 전달"
    tags = ["document_loader", "qdrant", "vector_db", "rag", "setup"]

    inputs = [
        {"id": "model", "name": "Model", "type": "MODEL"}
    ]
    outputs = [
        {"id": "tools", "name": "Tools", "type": "TOOL"},
    ]

    parameters = [
        {"id": "tool_name", "name": "Tool Name", "type": "STR", "value": "tool_name", "required": True},
        {"id": "description", "name": "Description", "type": "STR", "value": "주어진 질문이나 키워드에 대해 관련 문서를 검색하는 도구입니다. 구체적인 키워드나 질문을 입력하면 관련성이 높은 문서들을 찾아줍니다. 여러 번 호출하여 다양한 키워드로 검색할 수 있습니다.", "required": True, "expandable": True, "description": "이 도구를 언제 사용하여야 하는지 설명합니다. AI는 해당 설명을 통해, 해당 도구를 언제 호출해야할지 결정할 수 있습니다."},
        {"id": "collection_name", "name": "Collection Name", "type": "STR", "value": "Select Collection", "required": True, "is_api": True, "api_name": "api_collection", "options": []},
        {"id": "use_model_prompt", "name": "Use Model Prompt", "type": "BOOL", "value": True, "optional": True, "description": "임베딩 벡터 변환시 모델이 요구하는 프롬프트를 사용할지를 결정합니다."},
        {"id": "top_k", "name": "Top K Results", "type": "INT", "value": 4, "required": False, "optional": True, "min": 1, "max": 10, "step": 1},
        {"id": "score_threshold", "name": "Score Threshold", "type": "FLOAT", "value": 0.2, "required": False, "optional": True, "min": 0.0, "max": 1.0, "step": 0.1},
        {"id": "rerank", "name": "Enable Rerank", "type": "BOOL", "value": False, "required": False, "optional": True},
        {"id": "rerank_top_k", "name": "Rerank Top K", "type": "INT", "value": 5, "required": False, "optional": True, "min": 1, "max": 100, "step": 1},
        {"id": "enhance_prompt", "name": "Enhance Prompt", "type": "STR", "value": enhance_prompt, "required": False, "optional": True, "expandable": True, "description": "검색된 자료를 어떻게 사용할 것인지 지시합니다."},
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


    def execute(self, tool_name, description, collection_name: str, top_k: int = 4, use_model_prompt: bool = True, score_threshold: float = 0.2, enhance_prompt: str = enhance_prompt, rerank: bool = False, rerank_top_k: int = 5, model=None):
        def create_vectordb_tool():
            @tool(tool_name, description=description, args_schema=ToolSchema)
            def vectordb_retrieval_tool(query: str, deep_search: bool = False) -> str:
                rag_service = AppServiceManager.get_rag_service()

                if deep_search and model:
                    if use_model_prompt:
                        query = embedding_model_prompt + query

                    all_docs_in_collection = sync_run_async(rag_service.list_documents_in_collection(collection_name=collection_name))
                    documents = all_docs_in_collection.get("documents", [])

                    if not documents or len(documents) == 0:
                        return "해당 컬렉션에 문서가 없습니다."

                    search_result = sync_run_async(rag_service.search_documents(
                        collection_name=collection_name,
                        query_text=query,
                        limit=5,
                        score_threshold=0,
                    ))
                    results = search_result.get("results", [])
                    items_file_info = [(item.get("file_name", "Unknown"), item.get("file_id")) for item in results]

                    most_common_file_id = None
                    if items_file_info:
                        file_names = [file_name for file_name, _ in items_file_info]
                        file_name_counts = Counter(file_names)
                        most_common_file = file_name_counts.most_common(1)[0][0]
                        most_common_file_id = next((file_id for file_name, file_id in items_file_info if file_name == most_common_file), None)

                    if most_common_file_id:
                        document_detail = sync_run_async(rag_service.get_document_detail(
                            collection_name=collection_name,
                            document_id=most_common_file_id
                        ))

                        total_chunks = document_detail.get("total_chunks", 0)
                        if total_chunks > 4:
                            total_file_text = []
                            chunks = document_detail.get("chunks", [])
                            chunked_text_list = []
                            for i in range(0, len(chunks), 4):
                                chunk_group = chunks[i:i+4]
                                combined_text = "\n".join(chunk.get("chunk_text", "") for chunk in chunk_group if chunk.get("chunk_text"))
                                if combined_text:
                                    chunked_text_list.append(combined_text)
                            for chunked_text in chunked_text_list:
                                prompt = f"""Your role is to organize and provide the given context in a way that is easy to understand. Exclude any content that is redundant or completely unrelated to the user's request, but make sure that there is no loss of information regarding other matters that may be relevant.

User's request: {query}
Context: {chunked_text}"""
                                result = model.invoke(prompt)  # Assuming model is callable
                                response = result.content if hasattr(result, 'content') else str(result)
                                total_file_text.append(response)
                            total_file_text = "\n".join(total_file_text)

                        else:
                            chunks = document_detail.get("chunks", [])
                            total_file_text = "\n".join(chunk.get("chunk_text", "") for chunk in chunks if chunk.get("chunk_text"))

                        enhanced_prompt = f"""{enhance_prompt}:
[파일명] {most_common_file}

[내용]
{total_file_text}"""

                else:
                    if deep_search:
                        logger.warning("Deep search is enabled but no model is provided. Defaulting to basic search.")
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
                            enhanced_prompt = f"""{enhance_prompt}:
{context_text}"""
                            return enhanced_prompt
                    except Exception as e:
                        logger.error(f"RAG 검색 수행 중 오류: {e}")
                        return {"error": str(e)}

            return vectordb_retrieval_tool

        return create_vectordb_tool()

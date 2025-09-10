import os
import logging
import asyncio
from typing import Optional, Dict, Any
from editor.node_composer import Node
from langchain.agents import tool
from editor.utils.helper.service_helper import AppServiceManager
from editor.utils.helper.async_helper import sync_run_async
from fastapi import Request
from controller.rag.retrievalController import list_collections
from editor.utils.citation_prompt import citation_prompt
from .advanced_tree_search import AdvancedTreeSearchAlgorithm

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

class QdrantRetrievalToolV5(Node):
    categoryId = "xgen"
    functionId = "document_loaders"
    nodeId = "document_loaders/QdrantRetrievalTool_V5"
    nodeName = "Qdrant Retrieval Tool V5 (Advanced Tree Search)"
    description = "고도화된 트리 서치 알고리즘을 사용한 VectorDB 검색 Tool"
    tags = ["document_loader", "qdrant", "vector_db", "rag", "tree_search", "advanced"]

    inputs = []
    outputs = [
        {"id": "tools", "name": "Tools", "type": "TOOL"},
    ]

    parameters = [
        {"id": "tool_name", "name": "Tool Name", "type": "STR", "value": "advanced_search_tool", "required": True},
        {"id": "description", "name": "Description", "type": "STR", "value": "고도화된 트리 서치 알고리즘으로 최적의 문서를 검색하는 Tool입니다.", "required": True, "expandable": True},
        {"id": "collection_name", "name": "Collection Name", "type": "STR", "value": "Select Collection", "required": True, "is_api": True, "api_name": "api_collection", "options": []},
        {"id": "use_model_prompt", "name": "Use Model Prompt", "type": "BOOL", "value": True, "optional": True},
        {"id": "top_k", "name": "Top K Results", "type": "INT", "value": 5, "required": False, "optional": True, "min": 1, "max": 20, "step": 1},
        {"id": "score_threshold", "name": "Score Threshold", "type": "FLOAT", "value": 0.05, "required": False, "optional": True, "min": 0.0, "max": 1.0, "step": 0.05},
        {"id": "search_multiplier", "name": "Search Multiplier", "type": "INT", "value": 5, "required": False, "optional": True, "min": 2, "max": 10, "step": 1, "description": "검색 결과를 top_k의 몇 배로 가져올지 결정"},
        {"id": "alpha", "name": "Alpha (Original Score Weight)", "type": "FLOAT", "value": 0.6, "required": False, "optional": True, "min": 0.0, "max": 1.0, "step": 0.1},
        {"id": "beta", "name": "Beta (Tree Structure Weight)", "type": "FLOAT", "value": 0.3, "required": False, "optional": True, "min": 0.0, "max": 1.0, "step": 0.1},
        {"id": "gamma", "name": "Gamma (Diversity Weight)", "type": "FLOAT", "value": 0.1, "required": False, "optional": True, "min": 0.0, "max": 1.0, "step": 0.1},
        {"id": "diversity_threshold", "name": "Diversity Threshold", "type": "FLOAT", "value": 0.5, "required": False, "optional": True, "min": 0.0, "max": 1.0, "step": 0.1},
        {"id": "min_quality_score", "name": "Min Quality Score", "type": "FLOAT", "value": 0.15, "required": False, "optional": True, "min": 0.0, "max": 1.0, "step": 0.05, "description": "최소 품질 기준 점수"},
        {"id": "strict_citation", "name": "Strict Citation", "type": "BOOL", "value": True, "required": False, "optional": True},
        {"id": "enhance_prompt", "name": "Enhance Prompt", "type": "STR", "value": enhance_prompt, "required": False, "optional": True, "expandable": True},
    ]

    def api_collection(self, request: Request) -> Dict[str, Any]:
        collections = sync_run_async(list_collections(request))
        return [{"value": collection.get("collection_name"), "label": collection.get("collection_make_name")} for collection in collections]

    def execute(self, tool_name: str, description: str, collection_name: str, top_k: int = 5,
                use_model_prompt: bool = True, score_threshold: float = 0.05, search_multiplier: int = 5,
                alpha: float = 0.6, beta: float = 0.3, gamma: float = 0.1, diversity_threshold: float = 0.5,
                min_quality_score: float = 0.15, strict_citation: bool = True, enhance_prompt: str = enhance_prompt):

        def create_advanced_vectordb_tool():
            @tool(tool_name, description=description)
            def advanced_vectordb_retrieval_tool(query: str) -> str:
                rag_service = AppServiceManager.get_rag_service()
                try:
                    logger.info("=== Advanced Tree Search V5 시작 ===")
                    logger.info("Query: %s...", query[:100])
                    logger.info("Parameters - Alpha: %s, Beta: %s, Gamma: %s", alpha, beta, gamma)

                    if use_model_prompt:
                        query = embedding_model_prompt + query

                    # 더 많은 결과를 가져와서 알고리즘으로 최적화
                    search_limit = top_k * search_multiplier
                    search_result = sync_run_async(rag_service.search_documents(
                        collection_name=collection_name,
                        query_text=query,
                        limit=search_limit,
                        score_threshold=score_threshold,
                        rerank=False,
                        rerank_top_k=search_limit
                    ))

                    results = search_result.get("results", [])
                    if not results:
                        logger.warning("검색 결과가 없습니다.")
                        return query

                    logger.info("초기 검색 결과: %d개", len(results))

                    # Advanced Tree Search Algorithm 실행
                    algorithm = AdvancedTreeSearchAlgorithm(
                        alpha=alpha, beta=beta, gamma=gamma,
                        diversity_threshold=diversity_threshold
                    )

                    # 기본 MCTS 실행
                    mcts_results = algorithm.monte_carlo_tree_search(results, top_k * 2)  # 더 많이 가져와서 필터링

                    # 동적 Top-K 선택 (품질 기준 적용)
                    optimized_results, selection_info = algorithm.dynamic_top_k_selection(
                        mcts_results, top_k, min_quality_score=min_quality_score
                    )

                    # 선택 전략 로그
                    logger.info("선택 전략: %s - %s", selection_info.get("strategy"), selection_info.get("message"))
                    if "warning" in selection_info:
                        logger.warning("품질 경고: %s", selection_info["warning"])

                    # 품질 분석 및 적응적 조정
                    quality_analysis = algorithm.analyze_selection_quality(optimized_results)
                    logger.info("선택 품질 분석: %s", quality_analysis)

                    # 품질이 낮으면 적응적 파라미터 조정 후 재실행
                    quality_grade = quality_analysis.get("quality_grade", "F")
                    if quality_grade in ["D", "F"] and len(results) > top_k:
                        logger.info("품질 등급 %s로 인한 적응적 재조정 실행", quality_grade)

                        # 파라미터 조정
                        current_params = {"alpha": alpha, "beta": beta, "gamma": gamma}
                        adjusted_params = algorithm.adaptive_parameter_adjustment(quality_analysis, current_params)

                        # 조정된 파라미터로 알고리즘 재생성 및 재실행
                        if adjusted_params != current_params:
                            algorithm_adjusted = AdvancedTreeSearchAlgorithm(
                                alpha=adjusted_params["alpha"],
                                beta=adjusted_params["beta"],
                                gamma=adjusted_params["gamma"],
                                diversity_threshold=diversity_threshold
                            )
                            optimized_results = algorithm_adjusted.monte_carlo_tree_search(results, top_k)
                            quality_analysis = algorithm_adjusted.analyze_selection_quality(optimized_results)
                            logger.info("재조정 후 품질 분석: %s", quality_analysis)

                    context_parts = []
                    for i, item in enumerate(optimized_results, 1):
                        if "chunk_text" in item and item["chunk_text"]:
                            item_file_name = item.get("file_name", "Unknown")
                            item_file_path = item.get("file_path", "Unknown")
                            item_page_number = item.get("page_number", 0)
                            item_line_start = item.get("line_start", 0)
                            item_line_end = item.get("line_end", 0)

                            final_score = item.get("final_score", 0.0)
                            chunk_text = item["chunk_text"]
                            context_parts.append(f"[문서 {i}](관련도: {final_score:.3f})\n[파일명] {item_file_name}\n[파일경로] {item_file_path}\n[페이지번호] {item_page_number}\n[문장시작줄] {item_line_start}\n[문장종료줄] {item_line_end}\n\n[내용]\n{chunk_text}")

                    if context_parts:
                        context_text = "\n".join(context_parts)
                        enhanced_prompt = f"""{enhance_prompt}
{context_text}\n{citation_prompt if strict_citation else ""}"""
                        return enhanced_prompt

                except Exception as e:
                    logger.error("Advanced Tree Search 수행 중 오류: %s", e, exc_info=True)
                    return f"검색 중 오류가 발생했습니다: {str(e)}"

            return advanced_vectordb_retrieval_tool

        return create_advanced_vectordb_tool()

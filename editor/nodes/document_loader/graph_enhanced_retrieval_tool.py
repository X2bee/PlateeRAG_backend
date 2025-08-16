import logging
from typing import Dict, Any, List, Set
from editor.node_composer import Node
from langchain.agents import tool
from editor.utils.helper.service_helper import AppServiceManager
from editor.utils.helper.async_helper import sync_run_async
from service.database.models.vectordb import VectorDB, VectorDBChunkEdge
from fastapi import Request
from controller.controller_helper import extract_user_id_from_request

logger = logging.getLogger(__name__)

enhance_prompt = """You are an AI assistant that must strictly follow these guidelines when using the provided document context with graph-enhanced information:

1. ANSWER ONLY BASED ON PROVIDED CONTEXT: Use only the information from the retrieved documents to answer questions. Do not add information from your general knowledge.
2. UTILIZE CONCEPTUAL CONNECTIONS: Consider the conceptual relationships and entity connections provided in the enhanced context.
3. BE PRECISE AND ACCURATE: Quote specific facts, numbers, and details exactly as they appear in the documents.
4. ACKNOWLEDGE LIMITATIONS: If the provided documents do not contain sufficient information, clearly state this.
5. LEVERAGE GRAPH INSIGHTS: Use the conceptual network information to provide more comprehensive and connected answers.
6. CITE SOURCES: Reference document numbers and mention conceptual connections when relevant.

Remember: The graph-enhanced context provides richer relationships between concepts, entities, and topics."""

embedding_model_prompt = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "

class GraphEnhancedRetrievalTool(Node):
    categoryId = "xgen"
    functionId = "document_loaders"
    nodeId = "document_loaders/GraphEnhancedRetrievalTool"
    nodeName = "Graph Enhanced Retrieval Tool"
    description = "그래프 네트워크를 활용한 향상된 VectorDB 검색 Tool"
    tags = ["document_loader", "qdrant", "vector_db", "rag", "graph", "enhancement"]

    inputs = []
    outputs = [
        {"id": "tools", "name": "Tools", "type": "TOOL"},
    ]

    parameters = [
        {"id": "tool_name", "name": "Tool Name", "type": "STR", "value": "graph_enhanced_search", "required": True},
        {"id": "description", "name": "Description", "type": "STR", "value": "그래프 네트워크를 활용하여 향상된 검색을 수행하는 Tool입니다.", "required": True, "expandable": True},
        {"id": "collection_name", "name": "Collection Name", "type": "STR", "value": "Select Collection", "required": True, "is_api": True, "api_name": "api_collection", "options": []},
        {"id": "use_model_prompt", "name": "Use Model Prompt", "type": "BOOL", "value": True, "optional": True},
        {"id": "top_k", "name": "Top K Results", "type": "INT", "value": 4, "required": False, "optional": True, "min": 1, "max": 10, "step": 1},
        {"id": "score_threshold", "name": "Score Threshold", "type": "FLOAT", "value": 0.2, "required": False, "optional": True, "min": 0.0, "max": 1.0, "step": 0.1},
        {"id": "rerank", "name": "Enable Rerank", "type": "BOOL", "value": False, "required": False, "optional": True},
        {"id": "rerank_top_k", "name": "Rerank Top K", "type": "INT", "value": 5, "required": False, "optional": True, "min": 1, "max": 100, "step": 1},
        {"id": "enhance_prompt", "name": "Enhance Prompt", "type": "STR", "value": enhance_prompt, "required": False, "optional": True, "expandable": True},

        # Graph enhancement parameters
        {"id": "enable_graph_expansion", "name": "Enable Graph Expansion", "type": "BOOL", "value": True, "required": False, "optional": True, "description": "그래프 네트워크를 통한 검색 확장을 활성화합니다."},
        {"id": "expansion_depth", "name": "Expansion Depth", "type": "INT", "value": 1, "required": False, "optional": True, "min": 1, "max": 3, "step": 1, "description": "그래프 확장 깊이 (홉 수)"},
        {"id": "concept_boost_weight", "name": "Concept Boost Weight", "type": "FLOAT", "value": 0.3, "required": False, "optional": True, "min": 0.0, "max": 1.0, "step": 0.1, "description": "개념 매칭 보너스 가중치"},
        {"id": "expansion_threshold", "name": "Expansion Threshold", "type": "FLOAT", "value": 0.3, "required": False, "optional": True, "min": 0.0, "max": 1.0, "step": 0.1, "description": "그래프 확장 포함 임계값"},
        {"id": "enable_concept_rerank", "name": "Enable Concept Reranking", "type": "BOOL", "value": True, "required": False, "optional": True, "description": "개념 기반 재순위를 활성화합니다."},
    ]

    def api_collection(self, request: Request) -> Dict[str, Any]:
        user_id = extract_user_id_from_request(request)
        db_service = request.app.state.app_db
        collections = db_service.find_by_condition(
            VectorDB,
            {"user_id": user_id},
            limit=1000,
        )
        return [{"value": collection.collection_name, "label": collection.collection_make_name} for collection in collections]

    def get_chunk_concepts(self, chunk_id: str, collection_name: str, app_db, relation_types: List[str] = None) -> Dict[str, List[str]]:
        """청크와 연결된 개념들을 관계 타입별로 반환 (양방향 검색)"""
        if relation_types is None:
            relation_types = ["entity", "main_concept", "topic", "keyword"]

        concept_map = {}
        for rel_type in relation_types:
            # chunk_id가 source인 경우 (일반적인 경우)
            edges_as_source = app_db.find_by_condition(
                VectorDBChunkEdge,
                {
                    "source": chunk_id,
                    "collection_name": collection_name,
                    "relation_type": rel_type
                },
                return_list=False  # 모델 객체로 반환받기
            )

            # chunk_id가 target인 경우 (indirect 관계에서 가능)
            edges_as_target = app_db.find_by_condition(
                VectorDBChunkEdge,
                {
                    "target": chunk_id,
                    "collection_name": collection_name,
                    "relation_type": rel_type,
                    "edge_type": "indirect"  # indirect 엣지만 고려
                },
                return_list=False  # 모델 객체로 반환받기
            )

            concepts = set()

            # source에서 target으로의 관계 (모델 객체)
            for edge in edges_as_source:
                if hasattr(edge, 'target') and edge.target:
                    concepts.add(edge.target.strip())

            # target에서 source로의 관계 (역방향, 모델 객체)
            for edge in edges_as_target:
                if hasattr(edge, 'source') and edge.source:
                    concepts.add(edge.source.strip())

            # 빈 문자열 제거
            concept_map[rel_type] = [c for c in concepts if c and c.strip()]

        return concept_map

    def find_related_chunks_by_concepts(self, concepts: Dict[str, List[str]], collection_name: str, app_db, exclude_chunks: Set[str] = None) -> List[str]:
        """개념들과 연결된 다른 청크들 찾기 (양방향 검색)"""
        if exclude_chunks is None:
            exclude_chunks = set()

        related_chunks = set()

        for rel_type, concept_list in concepts.items():
            for concept in concept_list:
                if not concept:  # 빈 개념 스킵
                    continue

                # 개념이 target인 경우 (일반적)
                edges_to_concept = app_db.find_by_condition(
                    VectorDBChunkEdge,
                    {
                        "target": concept,
                        "collection_name": collection_name,
                        "relation_type": rel_type
                    },
                    return_list=False  # 모델 객체로 반환
                )

                # 개념이 source인 경우 (indirect 관계에서 가능)
                edges_from_concept = app_db.find_by_condition(
                    VectorDBChunkEdge,
                    {
                        "source": concept,
                        "collection_name": collection_name,
                        "relation_type": rel_type,
                        "edge_type": "indirect"
                    },
                    return_list=False  # 모델 객체로 반환
                )

                # target인 경우에서 source 청크들 수집 (모델 객체)
                for edge in edges_to_concept:
                    if hasattr(edge, 'source') and edge.source and edge.source not in exclude_chunks:
                        related_chunks.add(edge.source.strip())

                # source인 경우에서 target 청크들 수집 (모델 객체)
                for edge in edges_from_concept:
                    if hasattr(edge, 'target') and edge.target and edge.target not in exclude_chunks:
                        related_chunks.add(edge.target.strip())

        return list(related_chunks)

    def calculate_concept_similarity(self, query_concepts: List[str], chunk_concepts: Dict[str, List[str]]) -> float:
        """쿼리 개념과 청크 개념 간의 유사도 계산"""
        if not query_concepts:
            return 0.0

        query_set = set(concept.lower() for concept in query_concepts)
        total_matches = 0
        total_chunk_concepts = 0

        # 관계 타입별 가중치
        type_weights = {
            "entity": 1.0,
            "main_concept": 1.2,
            "topic": 0.8,
            "keyword": 0.6
        }

        for rel_type, concept_list in chunk_concepts.items():
            weight = type_weights.get(rel_type, 1.0)
            chunk_concept_set = set(concept.lower() for concept in concept_list)
            matches = len(query_set.intersection(chunk_concept_set))
            total_matches += matches * weight
            total_chunk_concepts += len(concept_list) * weight

        if total_chunk_concepts == 0:
            return 0.0

        return total_matches / max(len(query_concepts), total_chunk_concepts)

    def execute(self, *args, **kwargs):
        # 파라미터 추출
        tool_name = kwargs.get("tool_name", "graph_enhanced_search")
        description = kwargs.get("description", "그래프 네트워크를 활용하여 향상된 검색을 수행하는 Tool입니다.")
        collection_name = kwargs.get("collection_name", "")
        top_k = kwargs.get("top_k", 4)
        use_model_prompt = kwargs.get("use_model_prompt", True)
        score_threshold = kwargs.get("score_threshold", 0.2)
        enhance_prompt_param = kwargs.get("enhance_prompt", enhance_prompt)
        rerank = kwargs.get("rerank", False)
        rerank_top_k = kwargs.get("rerank_top_k", 5)
        enable_graph_expansion = kwargs.get("enable_graph_expansion", True)
        expansion_depth = kwargs.get("expansion_depth", 1)
        concept_boost_weight = kwargs.get("concept_boost_weight", 0.3)
        expansion_threshold = kwargs.get("expansion_threshold", 0.3)
        relation_types = kwargs.get("relation_types", "entity,main_concept,topic,keyword")
        enable_concept_rerank = kwargs.get("enable_concept_rerank", True)

        def create_graph_enhanced_tool():
            @tool(tool_name, description=description)
            def graph_enhanced_retrieval_tool(query: str) -> str:
                rag_service = AppServiceManager.get_rag_service()
                app_db = AppServiceManager.get_db_manager()

                try:
                    if use_model_prompt:
                        search_query = embedding_model_prompt + query
                    else:
                        search_query = query

                    # 1. 기본 벡터 검색
                    search_result = sync_run_async(rag_service.search_documents(
                        collection_name=collection_name,
                        query_text=search_query,
                        limit=top_k,
                        score_threshold=score_threshold,
                        rerank=rerank,
                        rerank_top_k=rerank_top_k
                    ))

                    results = search_result.get("results", [])
                    if not results:
                        return query

                    # 2. 그래프 확장 (옵션)
                    expanded_chunk_ids = set()
                    if enable_graph_expansion:
                        parsed_relation_types = [rt.strip() for rt in relation_types.split(",")]
                        initial_chunk_ids = set(result.get("id") for result in results)

                        current_chunks = initial_chunk_ids.copy()
                        for _ in range(expansion_depth):  # depth 변수 사용하지 않음
                            next_level_chunks = set()

                            for chunk_id in current_chunks:
                                # 청크의 개념들 가져오기
                                concepts = self.get_chunk_concepts(chunk_id, collection_name, app_db, parsed_relation_types)

                                # 관련 청크들 찾기
                                related_chunks = self.find_related_chunks_by_concepts(
                                    concepts, collection_name, app_db, initial_chunk_ids.union(expanded_chunk_ids)
                                )
                                next_level_chunks.update(related_chunks)

                            if not next_level_chunks:
                                break

                            expanded_chunk_ids.update(next_level_chunks)
                            current_chunks = next_level_chunks.copy()

                    # 3. 확장된 청크들의 실제 내용을 Qdrant에서 가져오기 및 점수 계산
                    expanded_results = []
                    if expanded_chunk_ids:
                        try:
                            logger.info("그래프 확장으로 %d개의 추가 청크 발견", len(expanded_chunk_ids))
                            parsed_relation_types = [rt.strip() for rt in relation_types.split(",")]

                            # 원본 검색 결과들의 개념 수집
                            original_concepts = set()
                            for result in results:
                                chunk_concepts = self.get_chunk_concepts(result.get("id"), collection_name, app_db, parsed_relation_types)
                                for concept_list in chunk_concepts.values():
                                    original_concepts.update(concept_list)

                            # Qdrant에서 확장된 청크들의 내용 가져오기
                            points = rag_service.vector_manager.client.retrieve(
                                collection_name=collection_name,
                                ids=list(expanded_chunk_ids)
                            )

                            for point in points:
                                if point.payload and point.payload.get("chunk_text"):
                                    # 확장된 청크의 개념들과 원본 개념들의 겹치는 정도 계산
                                    expanded_concepts = self.get_chunk_concepts(point.id, collection_name, app_db, parsed_relation_types)
                                    expanded_concept_set = set()
                                    for concept_list in expanded_concepts.values():
                                        expanded_concept_set.update(concept_list)

                                    # 개념 겹치는 비율 계산
                                    if original_concepts and expanded_concept_set:
                                        overlap_ratio = len(original_concepts.intersection(expanded_concept_set)) / len(original_concepts.union(expanded_concept_set))
                                        expansion_score = overlap_ratio
                                    else:
                                        expansion_score = 0.0

                                    # threshold 넘는 것만 포함
                                    if expansion_score >= expansion_threshold:
                                        expanded_results.append({
                                            "id": point.id,
                                            "score": expansion_score,
                                            "chunk_text": point.payload.get("chunk_text", ""),
                                            "file_name": point.payload.get("file_name", ""),
                                            "chunk_index": point.payload.get("chunk_index", 0),
                                            "is_expanded": True,
                                            "concept_overlap_ratio": overlap_ratio
                                        })
                                        logger.info("확장 청크 포함: %s, 개념겹침비율: %.3f, 점수: %.3f", point.id, overlap_ratio, expansion_score)
                                    else:
                                        logger.info("확장 청크 제외: %s, 개념겹침비율: %.3f < threshold: %.3f", point.id, overlap_ratio if 'overlap_ratio' in locals() else 0.0, expansion_threshold)

                        except (AttributeError, ValueError, KeyError) as e:
                            logger.warning("확장된 청크 내용 가져오기 실패: %s", str(e))

                    # 4. 개념 기반 재순위 (옵션)
                    all_results = results + expanded_results
                    logger.info("재순위 전 전체 결과 수: %d (기본: %d, 확장: %d)", len(all_results), len(results), len(expanded_results))

                    if enable_concept_rerank and app_db:
                        # 간단한 쿼리 개념 추출 (실제로는 더 정교한 NLP 사용 가능)
                        query_concepts = [word for word in query.replace(embedding_model_prompt, "").split() if len(word) > 2]
                        parsed_relation_types = [rt.strip() for rt in relation_types.split(",")]

                        for result in all_results:
                            chunk_id = result.get("id")
                            if chunk_id:
                                chunk_concepts = self.get_chunk_concepts(chunk_id, collection_name, app_db, parsed_relation_types)

                                # 개념 유사도 계산
                                concept_sim = self.calculate_concept_similarity(query_concepts, chunk_concepts)

                                # 기존 스코어에 개념 보너스 추가
                                original_score = result.get("score", 0.0)
                                enhanced_score = original_score + (concept_sim * concept_boost_weight)

                                result["enhanced_score"] = enhanced_score
                                result["concept_similarity"] = concept_sim
                                result["connected_concepts"] = chunk_concepts

                        # 향상된 스코어로 재정렬
                        all_results.sort(key=lambda x: x.get("enhanced_score", x.get("score", 0)), reverse=True)
                        logger.info("재순위 후 상위 5개 결과 점수:")
                        for i, result in enumerate(all_results[:5]):
                            score = result.get("enhanced_score", result.get("score", 0.0))
                            is_expanded = result.get("is_expanded", False)
                            logger.info("  %d. 점수: %.3f, 확장여부: %s, chunk_id: %s", i+1, score, is_expanded, result.get("id", ""))

                    # 5. 결과 포맷팅 (상위 top_k개만)
                    final_results = all_results[:top_k]
                    logger.info("최종 선택된 결과 수: %d (top_k: %d)", len(final_results), top_k)
                    context_parts = []
                    expansion_count = 0

                    for i, item in enumerate(final_results, 1):
                        if "chunk_text" in item and item["chunk_text"]:
                            score = item.get("enhanced_score", item.get("score", 0.0))
                            chunk_text = item["chunk_text"]
                            is_expanded = item.get("is_expanded", False)

                            if is_expanded:
                                expansion_count += 1

                            # 개념 정보 추가
                            concept_info = ""
                            if "connected_concepts" in item:
                                concepts = item["connected_concepts"]
                                concept_summary = []
                                for rel_type, concept_list in concepts.items():
                                    if concept_list:
                                        concept_summary.append(f"{rel_type}: {', '.join(concept_list[:2])}")
                                if concept_summary:
                                    concept_info = f" [연결개념: {'; '.join(concept_summary[:2])}]"

                            # 확장 표시 및 개념 겹침 정보
                            expansion_marker = ""
                            if is_expanded:
                                overlap_ratio = item.get("concept_overlap_ratio", 0.0)
                                expansion_marker = f" [그래프확장|개념겹침:{overlap_ratio:.2f}]"

                            context_parts.append(f"[문서 {i}] (관련도: {score:.3f}){concept_info}{expansion_marker}\n{chunk_text}")

                    if context_parts:
                        context_text = "\n".join(context_parts)

                        # 그래프 확장 정보 추가
                        expansion_info = ""
                        if enable_graph_expansion and expanded_chunk_ids:
                            expansion_info = f"\n\n[그래프 분석]: 네트워크를 통해 {len(expanded_chunk_ids)}개의 관련 문서를 추가 발견했으며, 상위 {expansion_count}개를 포함했습니다."

                            # 확장된 청크들의 상세 정보를 로그로 출력
                            logger.info("=== 그래프 확장으로 발견된 청크들 ===")
                            for result in expanded_results:
                                chunk_id = result.get("id", "")
                                score = result.get("enhanced_score", result.get("score", 0.0))
                                content = result.get("chunk_text", "")
                                # 내용을 100자까지만 자르기
                                content_preview = content[:100] + "..." if len(content) > 100 else content
                                logger.info("chunk_id: %s, 점수: %.3f, 내용: %s", chunk_id, score, content_preview)

                        enhanced_prompt_final = f"""{enhance_prompt_param}

[참고 문서]
{context_text}{expansion_info}"""
                        return enhanced_prompt_final

                except (AttributeError, ValueError, KeyError) as e:
                    logger.error("그래프 강화 RAG 검색 중 오류: %s", str(e))
                    return {"error": str(e)}

            return graph_enhanced_retrieval_tool

        return create_graph_enhanced_tool()

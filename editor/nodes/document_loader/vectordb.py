import os
import logging
from typing import Optional, Dict, Any
from editor.node_composer import Node

logger = logging.getLogger(__name__)

class QdrantNode(Node):
    categoryId = "langchain"
    functionId = "document_loaders"

    nodeId = "document_loaders/Qdrant"
    nodeName = "Qdrant Search"
    description = "RAG 서비스와 검색 파라미터를 설정하여 다음 노드로 전달하는 노드"
    tags = ["document_loader", "qdrant", "vector_db", "rag", "setup"]

    inputs = []
    outputs = [
        {
            "id": "rag_context",
            "name": "RAG Context",
            "type": "DICT"
        },
    ]

    def get_collection_options(self):
        """컬렉션 옵션을 가져오는 함수"""
        return getattr(self, '_collections_cache', [])

    parameters = [
        {
            "id": "collection_name",
            "name": "Collection Name",
            "type": "STR",
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
            "value": 0.5,
            "required": False,
            "optional": True,
            "min": 0.0,
            "max": 1.0,
            "step": 0.1
        }
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rag_service = self._get_rag_service()
        if not self.rag_service:
            logger.error("RAG 서비스가 초기화되지 않았습니다. 서버가 실행 중인지 확인하세요.")
            raise RuntimeError("RAG 서비스가 초기화되지 않았습니다. 서버가 실행 중인지 확인하세요.")
        self.vector_manager = self.rag_service.vector_manager

        # 컬렉션 목록을 미리 캐시해둠
        try:
            collections = self.vector_manager.list_collections().get('collections', [])
            self._collections_cache = [{"value": collection, "label": collection} for collection in collections]
            print(f"초기 컬렉션 목록 로딩 완료: {self._collections_cache}")

            print("test")
            self._collections_cache = [{"value": "test1", "label": "test1"}, {"value": "test2", "label": "test2"}, {"value": "test3", "label": "test3"}]
            print(f"테스트 컬렉션 목록 설정: {self._collections_cache}")
        except Exception as e:
            logger.warning(f"초기 컬렉션 목록 로딩 실패: {e}")
            self._collections_cache = []

    def _get_rag_service(self):
        """FastAPI 앱에서 RAG 서비스를 가져오는 함수"""
        try:
            import sys
            if 'main' in sys.modules:
                main_module = sys.modules['main']
                if hasattr(main_module, 'app') and hasattr(main_module.app, 'state'):
                    if hasattr(main_module.app.state, 'rag_service'):
                        rag_service = main_module.app.state.rag_service
                        # 캐시해서 다음번에 빠르게 접근
                        self._cached_rag_service = rag_service
                        logger.info("main 모듈에서 RAG 서비스를 찾았습니다.")
                        return rag_service

            for module_name, module in sys.modules.items():
                if hasattr(module, 'app'):
                    app = getattr(module, 'app')
                    if hasattr(app, 'state') and hasattr(app.state, 'rag_service'):
                        rag_service = app.state.rag_service
                        self._cached_rag_service = rag_service
                        logger.info(f"{module_name} 모듈에서 RAG 서비스를 찾았습니다.")
                        return rag_service

            logger.warning("RAG 서비스를 찾을 수 없습니다. 서버가 실행되지 않았을 수 있습니다.")
            return None

        except Exception as e:
            logger.error(f"RAG 서비스 접근 중 오류: {e}")
            return None

    def execute(self, collection_name: str, top_k: int = 4, score_threshold: float = 0.0) -> Dict[str, Any]:
        """
        RAG 서비스와 검색 파라미터를 준비하여 다음 노드로 전달합니다.

        Args:
            collection_name: 검색할 컬렉션 이름
            top_k: 반환할 최대 결과 수
            score_threshold: 최소 점수 임계값

        Returns:
            RAG 컨텍스트 딕셔너리 (rag_service와 파라미터 포함)
        """
        try:
            if not collection_name:
                logger.error("컬렉션 이름이 제공되지 않았습니다.")
                return {
                    "error": "컬렉션 이름이 제공되지 않았습니다.",
                    "rag_service": None,
                    "search_params": None,
                    "status": "error"
                }

            logger.info(f"RAG 컨텍스트 설정: collection='{collection_name}', top_k={top_k}, score_threshold={score_threshold}")

            if not self.rag_service:
                logger.warning("RAG 서비스를 사용할 수 없습니다.")
                return {
                    "error": "RAG 서비스를 사용할 수 없습니다.",
                    "rag_service": None,
                    "search_params": None,
                    "status": "error"
                }

            rag_context = {
                "rag_service": self.rag_service,
                "search_params": {
                    "collection_name": collection_name,
                    "top_k": top_k,
                    "score_threshold": score_threshold
                },
                "status": "ready",
                "node_info": {
                    "node_name": self.nodeName,
                    "node_id": self.nodeId,
                    "processed_at": self._get_current_timestamp()
                }
            }

            logger.info(f"RAG 컨텍스트 준비 완료: {rag_context['search_params']}")
            return rag_context

        except Exception as e:
            logger.error(f"RAG 컨텍스트 설정 중 오류 발생: {str(e)}")
            return {
                "error": f"RAG 컨텍스트 설정 중 오류: {str(e)}",
                "rag_service": None,
                "search_params": None,
                "status": "error"
            }

    def _get_current_timestamp(self) -> str:
        """현재 시간을 ISO 형식으로 반환"""
        from datetime import datetime
        return datetime.now().isoformat()

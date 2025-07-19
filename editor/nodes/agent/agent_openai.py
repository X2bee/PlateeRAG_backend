import logging
import asyncio
from typing import Dict, Any, Optional
from editor.node_composer import Node

logger = logging.getLogger(__name__)

class AgentOpenAINode(Node):
    categoryId = "langchain"
    functionId = "agents"

    nodeId = "agents/openai"
    nodeName = "Agent OpenAI"
    description = "RAG 컨텍스트를 사용하여 채팅 응답을 생성하는 Agent 노드"
    tags = ["agent", "chat", "rag", "openai"]

    inputs = [
        {
            "id": "text",
            "name": "Text",
            "type": "STR",
            "multi": False,
            "required": True
        },
        {
            "id": "rag_context",
            "name": "RAG Context",
            "type": "DICT",
            "multi": False,
            "required": False
        }
    ]
    outputs = [
        {
            "id": "result",
            "name": "Result",
            "type": "STR"
        },
    ]
    parameters = [
        {
            "id": "model",
            "name": "Model",
            "type": "STR",
            "value": "gpt-3.5-turbo",
            "required": True,
            "optional": False,
            "options": [
                {"value": "gpt-3.5-turbo", "label": "GPT-3.5 Turbo"},
                {"value": "gpt-4", "label": "GPT-4"},
                {"value": "gpt-4o", "label": "GPT-4o"}
            ]
        },
        {
            "id": "temperature",
            "name": "Temperature",
            "type": "FLOAT",
            "value": 0.7,
            "required": False,
            "optional": True,
            "min": 0.0,
            "max": 2.0,
            "step": 0.1
        },
        {
            "id": "max_tokens",
            "name": "Max Tokens",
            "type": "INTEGER",
            "value": 1000,
            "required": False,
            "optional": True,
            "min": 1,
            "max": 4000,
            "step": 1
        }
    ]

    def execute(self, text: str, rag_context: Optional[Dict[str, Any]] = None,
                model: str = "gpt-3.5-turbo", temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """
        RAG 컨텍스트를 사용하여 사용자 입력에 대한 채팅 응답을 생성합니다.

        Args:
            text: 사용자 입력
            rag_context: QdrantNode에서 전달받은 RAG 컨텍스트
            model: 사용할 언어 모델
            temperature: 생성 온도
            max_tokens: 최대 토큰 수

        Returns:
            Agent 응답
        """
        try:
            logger.info(f"Chat Agent 실행: text='{text[:50]}...', model={model}")

            # RAG 컨텍스트 확인 및 문서 검색
            enhanced_prompt = text
            if rag_context and rag_context.get("status") == "ready":
                logger.info("RAG 컨텍스트가 제공되었습니다. 문서 검색을 수행합니다.")

                rag_service = rag_context.get("rag_service")
                search_params = rag_context.get("search_params")

                if rag_service and search_params:
                    try:
                        # 문서 검색 수행
                        search_result = self._perform_rag_search(
                            rag_service,
                            text,
                            search_params
                        )

                        if search_result:
                            # 검색 결과를 프롬프트에 추가
                            enhanced_prompt = self._enhance_prompt_with_context(text, search_result)
                            logger.info(f"RAG 검색 완료: {len(search_result.get('results', []))}개 결과 추가")
                        else:
                            logger.warning("RAG 검색 결과가 없습니다.")

                    except Exception as rag_error:
                        logger.warning(f"RAG 검색 중 오류 발생: {rag_error}")
                        # RAG 실패해도 기본 채팅은 계속 진행
                else:
                    logger.warning("RAG 서비스 또는 검색 파라미터가 유효하지 않습니다.")
            else:
                logger.info("RAG 컨텍스트가 제공되지 않았거나 유효하지 않습니다. 기본 채팅으로 진행합니다.")

            # OpenAI API를 사용하여 응답 생성
            response = self._generate_chat_response(enhanced_prompt, model, temperature, max_tokens)

            logger.info(f"Chat Agent 응답 생성 완료: {len(response)}자")
            return response

        except Exception as e:
            logger.error(f"Chat Agent 실행 중 오류 발생: {str(e)}")
            return f"죄송합니다. 응답 생성 중 오류가 발생했습니다: {str(e)}"

    def _perform_rag_search(self, rag_service, query: str, search_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """RAG 서비스를 사용하여 문서 검색 수행 (동기 함수에서 비동기 호출)"""
        try:
            # 비동기 함수를 동기 환경에서 실행
            try:
                # 현재 이벤트 루프가 있는지 확인
                loop = asyncio.get_running_loop()
                # 새 태스크로 실행
                future = asyncio.create_task(
                    rag_service.search_documents(
                        collection_name=search_params["collection_name"],
                        query_text=query,
                        limit=search_params["top_k"],
                        score_threshold=search_params["score_threshold"]
                    )
                )
                result = loop.run_until_complete(future)
                return result
            except RuntimeError:
                # 이벤트 루프가 없는 경우 새로 생성
                result = asyncio.run(
                    rag_service.search_documents(
                        collection_name=search_params["collection_name"],
                        query_text=query,
                        limit=search_params["top_k"],
                        score_threshold=search_params["score_threshold"]
                    )
                )
                return result

        except Exception as e:
            logger.error(f"RAG 검색 수행 중 오류: {e}")
            return None

    def _enhance_prompt_with_context(self, text: str, search_result: Dict[str, Any]) -> str:
        """검색 결과를 사용하여 프롬프트 강화"""
        results = search_result.get("results", [])
        if not results:
            return text

        # 검색된 문서들을 컨텍스트로 구성
        context_parts = []
        for i, item in enumerate(results, 1):
            if "chunk_text" in item and item["chunk_text"]:
                score = item.get("score", 0.0)
                chunk_text = item["chunk_text"]
                context_parts.append(f"[문서 {i}] (관련도: {score:.3f})\\n{chunk_text}")

        if context_parts:
            context_text = "\\n\\n".join(context_parts)
            enhanced_prompt = f"""다음 문서들을 참고하여 질문에 답변해주세요:

[참고 문서]
{context_text}

[질문]
{text}

위 문서들의 정보를 바탕으로 정확하고 도움이 되는 답변을 제공해주세요."""
            return enhanced_prompt

        return text

    def _generate_chat_response(self, prompt: str, model: str, temperature: float, max_tokens: int) -> str:
        """OpenAI API를 사용하여 채팅 응답 생성"""
        try:
            import sys
            if 'main' in sys.modules:
                main_module = sys.modules['main']
                if hasattr(main_module, 'app') and hasattr(main_module.app, 'state'):
                    if hasattr(main_module.app.state, 'rag_service'):
                        config_composer = main_module.app.state.config_composer
                        self._cached_rag_service = config_composer
                        logger.info("main 모듈에서 RAG 서비스를 찾았습니다.")

            # 설정 초기화 (필요한 경우)
            api_key = config_composer.get_config_by_name("OPENAI_API_KEY").value

            if not api_key:
                return "OpenAI API 키가 설정되지 않았습니다."

            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # 응답 생성
            response = llm.invoke(prompt)
            return response.content

        except Exception as e:
            logger.error(f"OpenAI 응답 생성 중 오류: {e}")
            return f"응답 생성 중 오류가 발생했습니다: {str(e)}"

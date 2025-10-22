from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from typing import Dict, Any, Optional, Generator, Union
from pydantic import BaseModel
import logging
from editor.node_composer import Node
from editor.nodes.xgen.agent.functions import (
    prepare_llm_components,
    rag_context_builder,
    create_json_output_prompt,
)
from editor.utils.helper.stream_helper import (
    EnhancedAgentStreamingHandler,
    EnhancedAgentStreamingHandlerWithToolOutput,
    execute_agent_streaming,
)
from editor.utils.helper.agent_helper import (
    NonStreamingAgentHandler,
    NonStreamingAgentHandlerWithToolOutput,
    use_guarder_for_text_moderation,
    XgenJsonOutputParser,
)
from editor.utils.prefix_prompt import get_prefix_prompt
from fastapi import Request
from controller.helper.singletonHelper import get_config_composer

logger = logging.getLogger(__name__)

default_prompt = """You are a helpful AI assistant."""


class AgentXgenNode(Node):
    categoryId = "xgen"
    functionId = "agents"
    nodeId = "agents/xgen"
    nodeName = "Agent Xgen"
    description = "도구, 메모리 및 RAG 컨텍스트 등을 활용하여 채팅 응답을 생성하는 통합 Agent (스트리밍/일반 모드 지원)"
    tags = ["agent", "chat", "rag", "public_model", "stream", "xgen"]

    inputs = [
        {"id": "text", "name": "Text", "type": "STR", "multi": False, "required": True},
        {
            "id": "tools",
            "name": "Tools",
            "type": "TOOL",
            "multi": True,
            "required": False,
            "value": [],
        },
        {
            "id": "memory",
            "name": "Memory",
            "type": "OBJECT",
            "multi": False,
            "required": False,
        },
        {
            "id": "rag_context",
            "name": "RAG Context",
            "type": "DocsContext",
            "multi": False,
            "required": False,
        },
        {"id": "args_schema", "name": "ArgsSchema", "type": "OutputSchema"},
        {"id": "plan", "name": "Plan", "type": "PLAN", "required": False},
    ]
    outputs = [
        {"id": "stream", "name": "Stream", "type": "STREAM STR", "stream": True, "dependency": "streaming", "dependencyValue": True},
        {"id": "result", "name": "Result", "type": "STR", "dependency": "streaming", "dependencyValue": False},
    ]
    parameters = [
        {
            "id": "provider",
            "name": "Provider",
            "type": "STR",
            "value": "openai",
            "required": True,
            "options": [
                {"value": "openai", "label": "OpenAI"},
                {"value": "anthropic", "label": "Anthropic"},
                {"value": "google", "label": "Google"},
                {"value": "vllm", "label": "vLLM"},
            ],
        },
        {
            "id": "openai_model",
            "name": "OpenAI Model",
            "type": "STR",
            "value": "gpt-4.1",
            "required": True,
            "options": [
                {"value": "gpt-3.5-turbo", "label": "GPT-3.5 Turbo"},
                {"value": "gpt-4", "label": "GPT-4"},
                {"value": "gpt-4o", "label": "GPT-4o"},
                {"value": "o4-mini", "label": "o4 mini"},
                {"value": "gpt-4.1", "label": "GPT-4.1"},
                {"value": "gpt-4.1-mini", "label": "GPT-4.1 Mini"},
                {"value": "gpt-5", "label": "GPT-5"},
                {"value": "gpt-5-mini", "label": "GPT-5 Mini"},
                {"value": "gpt-5-nano", "label": "GPT-5 Nano"},
            ],
            "dependency": "provider",
            "dependencyValue": "openai",
        },
        {
            "id": "anthropic_model",
            "name": "Anthropic Model",
            "type": "STR",
            "value": "claude-sonnet-4-5-20250929",
            "required": True,
            "options": [
                {"value": "claude-3-5-haiku-20241022", "label": "Claude Haiku 3.5"},
                {"value": "claude-3-5-sonnet-20241022", "label": "Claude Sonnet 3.5"},
                {"value": "claude-3-7-sonnet-20250219", "label": "Claude Sonnet 3.7"},
                {"value": "claude-sonnet-4-20250514", "label": "Claude Sonnet 4"},
                {"value": "claude-opus-4-20250514", "label": "Claude Opus 4"},
                {"value": "claude-opus-4-1-20250805", "label": "Claude Opus 4.1"},
                {"value": "claude-sonnet-4-5-20250929", "label": "Claude Sonnet 4.5"},
                {"value": "claude-haiku-4-5-20251001", "label": "Claude Haiku 4.5"},
            ],
            "dependency": "provider",
            "dependencyValue": "anthropic",
        },
        {
            "id": "google_model",
            "name": "Google Model",
            "type": "STR",
            "value": "gemini-2.5-flash-lite",
            "required": True,
            "options": [
                {"value": "gemini-2.0-flash", "label": "Gemini 2.0 Flash"},
                {"value": "gemini-2.0-flash-lite", "label": "Gemini 2.0 Flash Lite"},
                {"value": "gemini-2.5-flash", "label": "Gemini 2.5 Flash"},
                {"value": "gemini-2.5-flash-lite", "label": "Gemini 2.5 Flash Lite"},
                {"value": "gemini-2.5-pro", "label": "Gemini 2.5 Pro"},
            ],
            "dependency": "provider",
            "dependencyValue": "google",
        },
        {
            "id": "vllm_model",
            "name": "vLLM Model",
            "type": "STR",
            "value": "",
            "required": True,
            "is_api": True,
            "api_name": "api_vllm_model_name",
            "dependency": "provider",
            "dependencyValue": "vllm",
        },
        {
            "id": "base_url",
            "name": "Base URL",
            "type": "STR",
            "value": "",
            "is_api": True,
            "api_name": "api_vllm_api_base_url",
            "required": True,
            "dependency": "provider",
            "dependencyValue": "vllm",
        },
        {
            "id": "temperature",
            "name": "Temperature",
            "type": "FLOAT",
            "value": 1,
            "required": False,
            "optional": True,
            "min": 0.0,
            "max": 2.0,
            "step": 0.1,
        },
        {
            "id": "max_tokens",
            "name": "Max Tokens",
            "type": "INT",
            "value": 8192,
            "required": False,
            "optional": True,
            "min": 1,
            "max": 65536,
            "step": 1,
        },
        {
            "id": "strict_citation",
            "name": "Strict Citation",
            "type": "BOOL",
            "value": True,
            "required": False,
            "optional": True,
        },
        {
            "id": "default_prompt",
            "name": "Default Prompt",
            "type": "STR",
            "value": default_prompt,
            "required": False,
            "optional": True,
            "expandable": True,
            "description": "기본 프롬프트로 AI가 따르는 System 지침을 의미합니다.",
        },
        {
            "id": "return_intermediate_steps",
            "name": "Return Intermediate Steps",
            "type": "BOOL",
            "value": False,
            "required": False,
            "optional": True,
            "description": "중간 단계를 반환할지 여부입니다.",
        },
        {
            "id": "use_guarder",
            "name": "Use Guarder Service",
            "type": "BOOL",
            "value": False,
            "required": False,
            "optional": True,
            "description": "Guarder 서비스를 사용할지 여부입니다.",
        },
        {
            "id": "streaming",
            "name": "Streaming",
            "type": "BOOL",
            "value": True,
            "required": False,
            "optional": True,
            "description": "스트리밍 모드 활성화 여부입니다.",
        },
    ]

    def api_vllm_model_name(self, request: Request) -> Dict[str, Any]:
        config_composer = get_config_composer(request)
        return config_composer.get_config_by_name("VLLM_MODEL_NAME").value

    def api_vllm_api_base_url(self, request: Request) -> Dict[str, Any]:
        config_composer = get_config_composer(request)
        return config_composer.get_config_by_name("VLLM_API_BASE_URL").value

    def _normalize_text_input(self, text: Any) -> str:
        """
        text 입력을 정규화하여 문자열로 변환합니다.
        LangChain 1.0.0의 HumanMessage는 content가 str 또는 list[str|dict]만 허용하므로
        딕셔너리나 다른 타입이 들어오면 적절히 변환합니다.

        Args:
            text: 입력 텍스트 (str, dict, list 등 다양한 타입 가능)

        Returns:
            정규화된 문자열
        """
        import json

        # 이미 문자열인 경우 그대로 반환
        if isinstance(text, str):
            return text

        # None인 경우 빈 문자열 반환
        if text is None:
            logger.warning("[NORMALIZE_TEXT] text가 None입니다. 빈 문자열로 변환합니다.")
            return ""

        # 딕셔너리인 경우
        if isinstance(text, dict):
            logger.info(f"[NORMALIZE_TEXT] 딕셔너리 입력 감지: {text}")

            # 일반적인 키들 우선순위로 확인
            priority_keys = [
                'user_text', 'text', 'content', 'message', 'user_message', 'input',
            ]

            for key in priority_keys:
                if key in text:
                    value = text[key]
                    if isinstance(value, str) and value.strip():
                        logger.info(f"[NORMALIZE_TEXT] '{key}' 키에서 텍스트 추출: {value}")
                        return value

            # 우선순위 키에서 찾지 못한 경우, 첫 번째 문자열 값 찾기
            for key, value in text.items():
                if isinstance(value, str) and value.strip():
                    logger.info(f"[NORMALIZE_TEXT] '{key}' 키에서 텍스트 추출: {value}")
                    return value

            # 문자열 값이 없으면 JSON 문자열로 변환
            try:
                json_str = json.dumps(text, ensure_ascii=False, indent=2)
                logger.info(f"[NORMALIZE_TEXT] 딕셔너리를 JSON 문자열로 변환")
                return json_str
            except (TypeError, ValueError) as e:
                logger.warning(f"[NORMALIZE_TEXT] JSON 변환 실패: {e}, str()로 변환")
                return str(text)

        # 리스트인 경우
        if isinstance(text, (list, tuple)):
            logger.info(f"[NORMALIZE_TEXT] 리스트/튜플 입력 감지: {type(text)}")

            # 문자열 리스트인 경우 결합
            if all(isinstance(item, str) for item in text):
                result = ' '.join(text)
                logger.info(f"[NORMALIZE_TEXT] 문자열 리스트를 공백으로 결합")
                return result

            # JSON 문자열로 변환
            try:
                json_str = json.dumps(text, ensure_ascii=False, indent=2)
                logger.info(f"[NORMALIZE_TEXT] 리스트를 JSON 문자열로 변환")
                return json_str
            except (TypeError, ValueError) as e:
                logger.warning(f"[NORMALIZE_TEXT] JSON 변환 실패: {e}, str()로 변환")
                return str(text)

        # 기타 타입 (int, float, bool 등)
        logger.info(f"[NORMALIZE_TEXT] 기타 타입 입력 ({type(text)}): {text}")
        try:
            return str(text)
        except Exception as e:
            logger.error(f"[NORMALIZE_TEXT] 문자열 변환 실패: {e}, 빈 문자열 반환")
            return ""

    def execute(
        self,
        text: str,
        tools: Optional[Any] = None,
        memory: Optional[Any] = None,
        rag_context: Optional[Dict[str, Any]] = None,
        args_schema: Optional[BaseModel] = None,
        plan: Optional[Dict[str, Any]] = None,
        provider: str = "openai",
        openai_model: str = "gpt-4.1",
        anthropic_model: str = "claude-sonnet-4-5-20250929",
        google_model: str = "gemini-2.5-flash-lite",
        vllm_model: str = "",
        base_url: str = "",
        temperature: float = 1,
        max_tokens: int = 8192,
        strict_citation: bool = True,
        default_prompt: str = default_prompt,
        return_intermediate_steps: bool = False,
        use_guarder: bool = False,
        streaming: bool = True,
    ) -> Union[str, Generator[str, None, None]]:
        """
        통합 Agent 실행 메서드

        Args:
            streaming: True일 경우 Generator 반환, False일 경우 str 반환
        """
        # text 입력 전처리: 딕셔너리, 리스트 등 다양한 타입을 문자열로 변환
        text = self._normalize_text_input(text)

        if provider == "openai":
            model = openai_model
            base_url = None
        elif provider == "anthropic":
            model = anthropic_model
            base_url = "https://api.anthropic.com"
        elif provider == "google":
            model = google_model
            base_url = None
        elif provider == "vllm":
            model = vllm_model
        else:
            model = openai_model
            base_url = None

        if streaming:
            return self._execute_streaming(
                text=text,
                tools=tools,
                memory=memory,
                rag_context=rag_context,
                args_schema=args_schema,
                plan=plan,
                model=model,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                strict_citation=strict_citation,
                default_prompt=default_prompt,
                return_intermediate_steps=return_intermediate_steps,
                use_guarder=use_guarder,
            )
        else:
            return self._execute_normal(
                text=text,
                tools=tools,
                memory=memory,
                rag_context=rag_context,
                args_schema=args_schema,
                plan=plan,
                model=model,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                strict_citation=strict_citation,
                default_prompt=default_prompt,
                return_intermediate_steps=return_intermediate_steps,
                use_guarder=use_guarder,
            )

    def _execute_streaming(
        self,
        text: str,
        tools: Optional[Any],
        memory: Optional[Any],
        rag_context: Optional[Dict[str, Any]],
        args_schema: Optional[BaseModel],
        plan: Optional[Dict[str, Any]],
        model: str,
        base_url: Optional[str],
        temperature: float,
        max_tokens: int,
        strict_citation: bool,
        default_prompt: str,
        return_intermediate_steps: bool,
        use_guarder: bool,
    ) -> Generator[str, None, None]:
        """스트리밍 모드 실행"""
        try:
            if use_guarder:
                is_safe, moderation_message = use_guarder_for_text_moderation(text)
                if not is_safe:
                    yield moderation_message
                    return

            default_prompt = get_prefix_prompt() + default_prompt
            llm, tools_list, chat_history = prepare_llm_components(
                text,
                tools,
                memory,
                model,
                temperature,
                max_tokens,
                streaming=True,
                plan=plan,
                base_url=base_url,
            )
            additional_rag_context = ""
            if rag_context:
                additional_rag_context = rag_context_builder(
                    text, rag_context, strict_citation
                )

            if args_schema:
                default_prompt = create_json_output_prompt(args_schema, default_prompt)

            system_prompt_text = default_prompt
            agent_summarization_middleware = SummarizationMiddleware(
                model=llm,
                max_tokens_before_summary=4000,
                messages_to_keep=10,
            )

            if tools_list:
                agent_graph = create_agent(
                    model=llm,
                    tools=tools_list,
                    system_prompt=system_prompt_text,
                    middleware=[agent_summarization_middleware],
                )
            else:
                agent_graph = create_agent(
                    model=llm,
                    system_prompt=system_prompt_text,
                    middleware=[agent_summarization_middleware],
                )

            if return_intermediate_steps:
                handler = EnhancedAgentStreamingHandlerWithToolOutput()
            else:
                handler = EnhancedAgentStreamingHandler()

            from langchain_core.messages import HumanMessage

            final_user_message = text
            if additional_rag_context:
                final_user_message = f"{text}\n\n{additional_rag_context}"

            graph_inputs = {
                "messages": chat_history + [HumanMessage(content=final_user_message)]
            }

            async_executor = lambda: agent_graph.ainvoke(
                graph_inputs, {"callbacks": [handler]}
            )

            try:
                if args_schema:
                    collected_output = []
                    for token in execute_agent_streaming(async_executor, handler):
                        collected_output.append(token)

                    full_output = "".join(collected_output)
                    try:
                        parser = XgenJsonOutputParser()
                        parsed_output = parser.parse(full_output)
                        logger.info("[AGENT_STREAM_EXECUTE] JSON 파싱 성공")
                        logger.info(
                            f"[AGENT_STREAM_EXECUTE] 파싱된 출력: {parsed_output}"
                        )
                        yield full_output
                    except Exception as parse_error:
                        logger.error(
                            f"[AGENT_STREAM_EXECUTE] JSON 파싱 실패: {parse_error}"
                        )
                        logger.error(
                            f"[AGENT_STREAM_EXECUTE] 수집된 출력: {full_output}"
                        )
                        yield full_output
                else:
                    for token in execute_agent_streaming(async_executor, handler):
                        yield token

            except Exception as e:
                logger.error(f"Agent streaming error: {str(e)}", exc_info=True)
                yield f"\nStreaming Error: {str(e)}\n"

        except Exception as e:
            logger.error(
                f"[AGENT_STREAM_EXECUTE] 스트리밍 Agent 실행 중 오류 발생: {str(e)}",
                exc_info=True,
            )
            yield f"죄송합니다. 응답 생성 중 오류가 발생했습니다: {str(e)}"

    def _execute_normal(
        self,
        text: str,
        tools: Optional[Any],
        memory: Optional[Any],
        rag_context: Optional[Dict[str, Any]],
        args_schema: Optional[BaseModel],
        plan: Optional[Dict[str, Any]],
        model: str,
        base_url: Optional[str],
        temperature: float,
        max_tokens: int,
        strict_citation: bool,
        default_prompt: str,
        return_intermediate_steps: bool,
        use_guarder: bool,
    ) -> str:
        """일반 모드 실행"""
        try:
            if use_guarder:
                is_safe, moderation_message = use_guarder_for_text_moderation(text)
                if not is_safe:
                    return moderation_message

            default_prompt = get_prefix_prompt() + default_prompt
            llm, tools_list, chat_history = prepare_llm_components(
                text,
                tools,
                memory,
                model,
                temperature,
                max_tokens,
                streaming=False,
                plan=plan,
                base_url=base_url,
            )

            additional_rag_context = ""
            if rag_context:
                additional_rag_context = rag_context_builder(
                    text, rag_context, strict_citation
                )

            if args_schema:
                default_prompt = create_json_output_prompt(args_schema, default_prompt)

            system_prompt_text = default_prompt
            agent_summarization_middleware = SummarizationMiddleware(
                model=llm,
                max_tokens_before_summary=4000,
                messages_to_keep=10,
            )

            if tools_list:
                agent_graph = create_agent(
                    model=llm,
                    tools=tools_list,
                    system_prompt=system_prompt_text,
                    middleware=[agent_summarization_middleware],
                )
            else:
                agent_graph = create_agent(
                    model=llm,
                    system_prompt=system_prompt_text,
                    middleware=[agent_summarization_middleware],
                )

            if return_intermediate_steps:
                handler = NonStreamingAgentHandlerWithToolOutput()
            else:
                handler = NonStreamingAgentHandler()

            from langchain_core.messages import HumanMessage

            final_user_message = text
            if additional_rag_context:
                final_user_message = f"{text}\n\n{additional_rag_context}"

            graph_inputs = {
                "messages": chat_history + [HumanMessage(content=final_user_message)]
            }

            response = agent_graph.invoke(graph_inputs, {"callbacks": [handler]})
            output = (
                response.get("messages", [])[-1].content
                if response.get("messages")
                else ""
            )

            formatted_output = handler.get_formatted_output(output)

            if args_schema:
                try:
                    parser = XgenJsonOutputParser()
                    parsed_output = parser.parse(output)
                    logger.info("[AGENT_EXECUTE] JSON 파싱 성공")
                    return parsed_output
                except Exception as parse_error:
                    logger.error(f"[AGENT_EXECUTE] JSON 파싱 실패: {parse_error}")
                    logger.error(f"[AGENT_EXECUTE] 원본 출력: {output}")
                    return formatted_output

            return formatted_output

        except Exception as e:
            logger.error(
                f"[AGENT_EXECUTE] Agent 실행 중 오류 발생: {str(e)}",
                exc_info=True,
            )
            return f"죄송합니다. 응답 생성 중 오류가 발생했습니다: {str(e)}"

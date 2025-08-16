import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel
from editor.node_composer import Node
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from editor.utils.helper.service_helper import AppServiceManager
from editor.utils.helper.async_helper import sync_run_async
from editor.utils.prefix_prompt import prefix_prompt
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from fastapi import Request

logger = logging.getLogger(__name__)

default_prompt = """You are a helpful AI assistant."""

class AgentVLLMNode(Node):
    categoryId = "xgen"
    functionId = "agents"
    nodeId = "agents/vllm"
    nodeName = "Agent VLLM"
    description = "RAG 컨텍스트를 사용하여 채팅 응답을 생성하는 Agent 노드"
    tags = ["agent", "chat", "rag", "vllm"]

    inputs = [
        {"id": "text", "name": "Text", "type": "STR", "multi": False, "required": True},
        {"id": "tools", "name": "Tools", "type": "TOOL", "multi": True, "required": False, "value": []},
        {"id": "memory", "name": "Memory", "type": "OBJECT", "multi": False, "required": False},
        {"id": "rag_context", "name": "RAG Context", "type": "DocsContext", "multi": False, "required": False},
        {"id": "args_schema", "name": "ArgsSchema", "type": "OutputSchema"},
    ]
    outputs = [
        {"id": "result", "name": "Result", "type": "STR"},
    ]
    parameters = [
        {"id": "model", "name": "Model", "type": "STR", "value": "", "is_api": True, "api_name": "api_vllm_model_name", "required": False, "optional": True},
        {"id": "temperature", "name": "Temperature", "type": "FLOAT", "value": 0.0, "required": False, "optional": True, "min": 0.0, "max": 2.0, "step": 0.1},
        {"id": "max_tokens", "name": "Max Tokens", "type": "INT", "value": 8192, "required": False, "optional": True, "min": 1, "max": 65536, "step": 1},
        {"id": "n_messages", "name": "Max Memory", "type": "INT", "value": 3, "min": 1, "max": 10, "step": 1, "optional": True},
        {"id": "base_url", "name": "Base URL", "type": "STR", "value": "", "is_api": True, "api_name": "api_vllm_api_base_url", "required": False, "optional": True},
        {"id": "default_prompt", "name": "Default Prompt", "type": "STR", "value": default_prompt, "required": False, "optional": True, "expandable": True, "description": "기본 프롬프트로 AI가 따르는 System 지침을 의미합니다."},
    ]

    def __init__(self, user_id: str = None, **kwargs):
        super().__init__(**kwargs)
        self.user_id = user_id
        self.config_composer = AppServiceManager.get_config_composer()
        self.llm_provider = self.config_composer.get_config_by_name("DEFAULT_LLM_PROVIDER").value
        self.vllm_api_base_url = self.config_composer.get_config_by_name("VLLM_API_BASE_URL").value
        self.vllm_model_name = self.config_composer.get_config_by_name("VLLM_MODEL_NAME").value
        self.vllm_temperature_default = self.config_composer.get_config_by_name("VLLM_TEMPERATURE_DEFAULT").value
        self.vllm_max_tokens_default = self.config_composer.get_config_by_name("VLLM_MAX_TOKENS_DEFAULT").value
        self.vllm_top_p = self.config_composer.get_config_by_name("VLLM_TOP_P").value
        self.vllm_top_k = self.config_composer.get_config_by_name("VLLM_TOP_K").value
        self.vllm_frequency_penalty = self.config_composer.get_config_by_name("VLLM_FREQUENCY_PENALTY").value
        self.vllm_repetition_penalty = self.config_composer.get_config_by_name("VLLM_REPETITION_PENALTY").value
        self.vllm_best_of = self.config_composer.get_config_by_name("VLLM_BEST_OF").value

    def api_vllm_model_name(self, request: Request) -> Dict[str, Any]:
        config_composer = request.app.state.config_composer
        return config_composer.get_config_by_name("VLLM_MODEL_NAME").value

    def api_vllm_api_base_url(self, request: Request) -> Dict[str, Any]:
        config_composer = request.app.state.config_composer
        return config_composer.get_config_by_name("VLLM_API_BASE_URL").value

    def execute(
        self,
        text: str,
        tools: Optional[Any] = None,
        memory: Optional[Any] = None,
        rag_context: Optional[Dict[str, Any]] = None,
        args_schema: Optional[BaseModel] = None,
        model: str = "",
        temperature: float = None,
        max_tokens: int = 8192,
        n_messages: int = 3,
        base_url: str = "",
        default_prompt: str = default_prompt,
    ) -> str:
        """
        RAG 컨텍스트를 사용하여 사용자 입력에 대한 채팅 응답을 생성합니다.

        Args:
            text: 사용자 입력
            tools: 사용할 도구 목록
            model: 사용할 언어 모델
            temperature: 생성 온도
            max_tokens: 최대 토큰 수

        Returns:
            Agent 응답
        """
        try:
            if self.llm_provider != "vllm":
                logger.warning(f"현재 설정된 LLM 제공자가 vLLM가 아닙니다: {self.llm_provider}. vLLM를 사용하려면 설정을 확인하세요.")

            if not model or model.strip() == "":
                model = self.vllm_model_name if model.strip() == "" else model
            if not base_url or base_url.strip() == "":
                base_url = self.vllm_api_base_url if base_url.strip() == "" else base_url
            if temperature is None or temperature < 0.0:
                temperature = self.vllm_temperature_default if temperature is None else temperature
            if max_tokens is None or max_tokens <= 0:
                max_tokens = self.vllm_max_tokens_default if max_tokens is None else max_tokens

            logger.info(f"Chat Agent 실행: text='{text[:50]}...', model={model}")

            # tools 처리 로직
            if tools is None:
                tools = None
            elif isinstance(tools, list):
                if len(tools) == 0:
                    tools = None
            else:
                # 단일 StructuredTool인 경우 리스트로 감싸기
                tools = [tools]

            additional_rag_context = None
            if rag_context:
                if rag_context['search_params']['use_model_prompt']:
                    query = rag_context['search_params']['embedding_model_prompt'] + text
                else:
                    query = text
                search_result = sync_run_async(rag_context['rag_service'].search_documents(
                    collection_name=rag_context['search_params']['collection_name'],
                    query_text=query,
                    limit=rag_context['search_params']['top_k'],
                    score_threshold=rag_context['search_params']['score_threshold']
                ))
                results = search_result.get("results", [])
                if results:
                    context_parts = []
                    for i, item in enumerate(results, 1):
                        if "chunk_text" in item and item["chunk_text"]:
                            score = item.get("score", 0.0)
                            chunk_text = item["chunk_text"]
                            context_parts.append(f"[문서 {i}] (관련도: {score:.3f})\n{chunk_text}")
                    if context_parts:
                        context_text = "\n".join(context_parts)
                        additional_rag_context = f"""{rag_context['search_params']['enhance_prompt']}
[Context]
{context_text}"""
            prompt = prefix_prompt+default_prompt

            # OpenAI API를 사용하여 응답 생성
            response = self._generate_chat_response(text, prompt, model, tools, memory, temperature, max_tokens, n_messages, base_url, additional_rag_context)

            logger.info(f"Chat Agent 응답 생성 완료: {len(response)}자")
            return response

        except Exception as e:
            logger.error(f"Chat Agent 실행 중 오류 발생: {str(e)}")
            return f"죄송합니다. 응답 생성 중 오류가 발생했습니다: {str(e)}"


    def _generate_chat_response(self, text: str, prompt: str, model: str, tools: Optional[Any], memory: Optional[Any], temperature: float, max_tokens: int, n_messages: int, base_url: str, additional_rag_context: Optional[str]) -> str:
        """OpenAI API를 사용하여 채팅 응답 생성"""
        try:
            config_composer = AppServiceManager.get_config_composer()
            if not config_composer:
                return "Config Composer가 설정되지 않았습니다."

            # OpenAI API 키 설정
            llm_provider = config_composer.get_config_by_name("DEFAULT_LLM_PROVIDER").value
            if llm_provider == "openai":
                api_key = config_composer.get_config_by_name("OPENAI_API_KEY").value
                if not api_key:
                    return "OpenAI API 키가 설정되지 않았습니다."

            elif llm_provider == "vllm":
                print("vLLM API를 사용합니다.")
                api_key = None # 현재 vLLM API 키는 별도로 설정하지 않음

                # TODO: vLLM API 키 설정 로직 추가
                # api_key = config_composer.get_config_by_name("VLLM_API_KEY").value
                # if not api_key:
                #     return "vLLM API 키가 설정되지 않았습니다."

            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                base_url=base_url
            )

            chat_history = []
            if memory:
                try:
                    memory_vars = memory.load_memory_variables({})
                    chat_history = memory_vars.get("chat_history", [])
                except Exception as e:
                    chat_history = []
            else:
                logger.info(f"[CHAT_RESPONSE] 메모리가 없어 빈 채팅 히스토리 사용")

            inputs = {
                "chat_history": chat_history,
                "input": text,
                "additional_rag_context": additional_rag_context if additional_rag_context else ""
            }

            if tools is not None:
                if additional_rag_context and additional_rag_context.strip():
                    final_prompt = ChatPromptTemplate.from_messages([
                        ("system", prompt),
                        MessagesPlaceholder(variable_name="chat_history", n_messages=n_messages),
                        ("user", "{input}"),
                        ("user", "{additional_rag_context}"),
                        MessagesPlaceholder(variable_name="agent_scratchpad", n_messages=2)
                    ])
                else:
                    final_prompt = ChatPromptTemplate.from_messages([
                        ("system", prompt),
                        MessagesPlaceholder(variable_name="chat_history", n_messages=n_messages),
                        ("user", "{input}"),
                        MessagesPlaceholder(variable_name="agent_scratchpad", n_messages=2)
                    ])

                agent = create_tool_calling_agent(llm, tools, final_prompt)
                agent_executor = AgentExecutor(
                    agent=agent,
                    tools=tools,
                    verbose=True,
                    handle_parsing_errors=True,
                )
                response = agent_executor.invoke(inputs)
                output = response["output"]

                return output

            else:
                if additional_rag_context and additional_rag_context.strip():
                    final_prompt = ChatPromptTemplate.from_messages([
                        ("system", prompt),
                        MessagesPlaceholder(variable_name="chat_history", n_messages=n_messages),
                        ("user", "{input}"),
                        ("user", "{additional_rag_context}"),
                    ])
                else:
                    final_prompt = ChatPromptTemplate.from_messages([
                        ("system", prompt),
                        MessagesPlaceholder(variable_name="chat_history", n_messages=n_messages),
                        ("user", "{input}")
                    ])
                chain = final_prompt | llm | StrOutputParser()
                response = chain.invoke(inputs)
                return response

        except Exception as e:
            logger.error(f"OpenAI 응답 생성 중 오류: {e}")
            return f"응답 생성 중 오류가 발생했습니다: {str(e)}"

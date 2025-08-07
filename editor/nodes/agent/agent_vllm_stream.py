from typing import Dict, Any
import logging
from typing import Any, Optional, Generator
from editor.node_composer import Node
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from editor.utils.helper.service_helper import AppServiceManager
from editor.utils.tools.async_helper import sync_run_async
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from fastapi import Request

logger = logging.getLogger(__name__)

default_prompt = """You are a helpful AI assistant."""
enhance_prompt = """Use the context from the documents to enhance your responses."""

class AgentVLLMStreamNode(Node):
    categoryId = "langchain"
    functionId = "agents"
    nodeId = "agents/vllm_stream"
    nodeName = "Agent VLLM Stream"
    description = "RAG 컨텍스트를 사용하여 채팅 응답을 스트리밍으로 생성하는 Agent 노드"
    tags = ["agent", "chat", "rag", "vllm", "stream"]

    inputs = [
        {"id": "text", "name": "Text", "type": "STR", "multi": False, "required": True},
        {"id": "tools", "name": "Tools", "type": "TOOL", "multi": True, "required": False, "value": []},
        {"id": "memory", "name": "Memory", "type": "OBJECT", "multi": False, "required": False},
        {"id": "rag_context", "name": "RAG Context", "type": "DICT", "multi": False, "required": False}
    ]
    outputs = [
        {"id": "stream", "name": "Stream", "type": "STREAM STR", "stream": True}
    ]
    parameters = [
        {"id": "model", "name": "Model", "type": "STR", "value": "", "is_api": True, "api_name": "api_vllm_model_name", "required": False, "optional": True},
        {"id": "temperature", "name": "Temperature", "type": "FLOAT", "value": 0.0, "required": False, "optional": True, "min": 0.0, "max": 2.0, "step": 0.1},
        {"id": "max_tokens", "name": "Max Tokens", "type": "INT", "value": 8192, "required": False, "optional": True, "min": 1, "max": 65536, "step": 1},
        {"id": "n_messages", "name": "Max Memory", "type": "INT", "value": 3, "min": 1, "max": 10, "step": 1, "optional": True},
        {"id": "base_url", "name": "Base URL", "type": "STR", "value": "", "is_api": True, "api_name": "api_vllm_api_base_url", "required": False, "optional": True},
        {"id": "default_prompt", "name": "Default Prompt", "type": "STR", "value": default_prompt, "required": False, "optional": True, "expandable": True, "description": "기본 프롬프트로 AI가 따르는 System 지침을 의미합니다."},
        {"id": "enhance_prompt", "name": "Enhance Prompt", "type": "STR", "value": enhance_prompt, "required": False, "optional": True, "expandable": True, "description": "RAG 컨텍스트를 사용하여 응답을 향상시키기 위한 프롬프트입니다."},
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
        model: str = "x2bee/Polar-14B",
        temperature: float = 0.7,
        max_tokens: int = 8192,
        n_messages: int = 3,
        base_url: str = "",
        default_prompt: str = default_prompt,
        enhance_prompt: str = enhance_prompt
    ) -> Generator[str, None, None]:

        try:
            llm, tools_list, chat_history = self._prepare_llm_and_inputs(tools, memory, model, temperature, max_tokens, base_url)

            if rag_context:
                search_result = sync_run_async(rag_context.rag_service.search_documents(
                    collection_name=rag_context.search_params.collection_name,
                    query_text=text,
                    limit=rag_context.search_params.top_k,
                    score_threshold=rag_context.search_params.score_threshold
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
                        text = f"""{text}
{enhance_prompt}
[참고 문서]
{context_text}"""
            inputs = {"input": text, "chat_history": chat_history}

            if tools_list:
                final_prompt = ChatPromptTemplate.from_messages([
                    ("system", default_prompt),
                    MessagesPlaceholder(variable_name="chat_history", n_messages=n_messages),
                    ("user", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad", n_messages=2)
                ])
                agent = create_tool_calling_agent(llm, tools_list, final_prompt)
                agent_executor = AgentExecutor(agent=agent, tools=tools_list, verbose=True, handle_parsing_errors=True)

                # stream() 메소드를 사용하여 스트리밍 응답 생성
                for chunk in agent_executor.stream(inputs):
                    if "output" in chunk:
                        yield chunk["output"]
            else:
                # 도구가 없을 경우 간단한 체인으로 스트리밍
                final_prompt = ChatPromptTemplate.from_messages([
                    ("system", default_prompt),
                    MessagesPlaceholder(variable_name="chat_history", n_messages=n_messages),
                    ("user", "{input}")
                ])
                chain = final_prompt | llm
                for chunk in chain.stream(inputs):
                    yield chunk.content

        except Exception as e:
            logger.error(f"[AGENT_STREAM_EXECUTE] 스트리밍 Agent 실행 중 오류 발생: {str(e)}", exc_info=True)
            yield f"죄송합니다. 응답 생성 중 오류가 발생했습니다: {str(e)}"

    def _prepare_llm_and_inputs(self, tools, memory, model, temperature, max_tokens, base_url):
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens, base_url=base_url, streaming=True)

        tools_list = []
        if tools:
            tools_list = tools if isinstance(tools, list) else [tools]

        chat_history = []
        if memory:
            try:
                memory_vars = memory.load_memory_variables({})
                chat_history = memory_vars.get("chat_history", [])
            except Exception:
                chat_history = []

        return llm, tools_list, chat_history

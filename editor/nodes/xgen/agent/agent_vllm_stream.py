import logging
from pydantic import BaseModel
from typing import Dict, Any, Optional, Generator
from editor.node_composer import Node
from editor.utils.helper.stream_helper import EnhancedAgentStreamingHandler, execute_agent_streaming
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from editor.utils.helper.service_helper import AppServiceManager
from editor.utils.helper.async_helper import sync_run_async
from editor.utils.prefix_prompt import prefix_prompt
from editor.utils.citation_prompt import citation_prompt
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from fastapi import Request
from langchain_core.output_parsers import JsonOutputParser

logger = logging.getLogger(__name__)

default_prompt = """You are a helpful AI assistant."""

class AgentVLLMStreamNode(Node):
    categoryId = "xgen"
    functionId = "agents"
    nodeId = "agents/vllm_stream"
    nodeName = "Agent VLLM Stream"
    description = "RAG 컨텍스트를 사용하여 채팅 응답을 스트리밍으로 생성하는 Agent 노드"
    tags = ["agent", "chat", "rag", "vllm", "stream"]

    inputs = [
        {"id": "text", "name": "Text", "type": "STR", "multi": False, "required": True},
        {"id": "tools", "name": "Tools", "type": "TOOL", "multi": True, "required": False, "value": []},
        {"id": "memory", "name": "Memory", "type": "OBJECT", "multi": False, "required": False},
        {"id": "rag_context", "name": "RAG Context", "type": "DocsContext", "multi": False, "required": False},
        {"id": "args_schema", "name": "ArgsSchema", "type": "OutputSchema"},
    ]
    outputs = [
        {"id": "stream", "name": "Stream", "type": "STREAM STR", "stream": True}
    ]
    parameters = [
        {"id": "model", "name": "Model", "type": "STR", "value": "", "is_api": True, "api_name": "api_vllm_model_name", "required": True},
        {"id": "temperature", "name": "Temperature", "type": "FLOAT", "value": 0.0, "required": False, "optional": True, "min": 0.0, "max": 2.0, "step": 0.1},
        {"id": "max_tokens", "name": "Max Tokens", "type": "INT", "value": 8192, "required": False, "optional": True, "min": 1, "max": 65536, "step": 1},
        {"id": "n_messages", "name": "Max Memory", "type": "INT", "value": 3, "min": 1, "max": 10, "step": 1, "optional": True},
        {"id": "base_url", "name": "Base URL", "type": "STR", "value": "", "is_api": True, "api_name": "api_vllm_api_base_url", "required": True},
        {"id": "strict_citation", "name": "Strict Citation", "type": "BOOL", "value": True, "required": False, "optional": True},
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
        model: str = "x2bee/Polar-14B",
        temperature: float = 0.7,
        max_tokens: int = 8192,
        n_messages: int = 3,
        base_url: str = "",
        strict_citation: bool = True,
        default_prompt: str = default_prompt,
    ) -> Generator[str, None, None]:

        try:
            default_prompt = prefix_prompt + default_prompt
            llm, tools_list, chat_history = self._prepare_llm_and_inputs(tools, memory, model, temperature, max_tokens, base_url)

            additional_rag_context = ""
            if rag_context:
                # rag_context['search_params']에서 옵션을 받아서 처리
                search_params = rag_context.get('search_params', {})
                rerank_flag = search_params.get('rerank', False)
                rerank_top_k = search_params.get('rerank_top_k', search_params.get('top_k', 20))

                # use_model_prompt 옵션이 있으면 embedding_model_prompt를 query에 추가
                if search_params.get('use_model_prompt'):
                    query = search_params.get('embedding_model_prompt', '') + text
                else:
                    query = text

                search_result = sync_run_async(rag_context['rag_service'].search_documents(
                    collection_name=search_params.get('collection_name'),
                    query_text=query,
                    limit=search_params.get('top_k', 5),
                    score_threshold=search_params.get('score_threshold', 0.7),
                    rerank=rerank_flag,
                    rerank_top_k=rerank_top_k
                ))
                results = search_result.get("results", [])
                if results:
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
                        additional_rag_context = f"""{rag_context['search_params']['enhance_prompt']}
{context_text}"""
            inputs = {"input": text, "chat_history": chat_history, "additional_rag_context": additional_rag_context if rag_context else ""}

            if args_schema:
                parser = JsonOutputParser(pydantic_object=args_schema)
                format_instructions = parser.get_format_instructions()
                escaped_instructions = format_instructions.replace("{", "{{").replace("}", "}}")
                default_prompt = f"{default_prompt}\n\n{escaped_instructions}"

            if tools_list:
                if additional_rag_context and additional_rag_context.strip():
                    final_prompt = ChatPromptTemplate.from_messages([
                        ("system", default_prompt),
                        MessagesPlaceholder(variable_name="chat_history", n_messages=n_messages),
                        ("user", "{input}"),
                        ("user", "{additional_rag_context}"),
                        MessagesPlaceholder(variable_name="agent_scratchpad")
                    ])
                else:
                    final_prompt = ChatPromptTemplate.from_messages([
                        ("system", default_prompt),
                        MessagesPlaceholder(variable_name="chat_history", n_messages=n_messages),
                        ("user", "{input}"),
                        MessagesPlaceholder(variable_name="agent_scratchpad")
                    ])

                agent = create_tool_calling_agent(llm, tools_list, final_prompt)
                # Agent가 더 많은 반복(iteration)을 할 수 있도록 max_iterations 증가
                agent_executor = AgentExecutor(
                    agent=agent,
                    tools=tools_list,
                    verbose=True,
                    handle_parsing_errors=True,
                    max_iterations=10,  # 최대 10번까지 tool 호출 가능
                    max_execution_time=300,  # 최대 5분까지 실행
                    early_stopping_method="generate"  # 충분한 정보를 얻으면 조기 종료
                )
                handler = EnhancedAgentStreamingHandler()

                # Helper 함수를 사용하여 Agent 실행을 스트리밍으로 처리
                async_executor = lambda: agent_executor.ainvoke(inputs, {"callbacks": [handler]})

                try:
                    for token in execute_agent_streaming(async_executor, handler):
                        yield token
                except Exception as e:
                    yield f"\nStreaming Error: {str(e)}\n"
            else:
                if additional_rag_context and additional_rag_context.strip():
                    if strict_citation:
                        default_prompt = default_prompt + citation_prompt
                    final_prompt = ChatPromptTemplate.from_messages([
                        ("system", default_prompt),
                        MessagesPlaceholder(variable_name="chat_history", n_messages=n_messages),
                        ("user", "{input}"),
                        ("user", "{additional_rag_context}"),
                    ])
                else:
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

from typing import Dict, Any, Optional, Generator
from pydantic import BaseModel
import logging
from editor.node_composer import Node
from editor.utils.helper.stream_helper import EnhancedAgentStreamingHandler, execute_agent_streaming
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from editor.utils.helper.service_helper import AppServiceManager
from editor.utils.helper.async_helper import sync_run_async
from editor.utils.prefix_prompt import prefix_prompt
from editor.utils.citation_prompt import citation_prompt
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain_core.output_parsers import JsonOutputParser

logger = logging.getLogger(__name__)

default_prompt = """You are a helpful AI assistant."""

class AgentOpenAIStreamNode(Node):
    categoryId = "xgen"
    functionId = "agents"
    nodeId = "agents/openai_stream"
    nodeName = "Agent OpenAI Stream"
    description = "RAG 컨텍스트를 사용하여 채팅 응답을 스트리밍으로 생성하는 Agent 노드"
    tags = ["agent", "chat", "rag", "openai", "stream"]

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
        {
            "id": "model", "name": "Model", "type": "STR", "value": "gpt-5", "required": True,
            "options": [
                {"value": "gpt-oss-20b", "label": "GPT-OSS-20B"},
                {"value": "gpt-oss-120b", "label": "GPT-OSS-120B"},
                {"value": "gpt-3.5-turbo", "label": "GPT-3.5 Turbo"},
                {"value": "gpt-4", "label": "GPT-4"},
                {"value": "gpt-4o", "label": "GPT-4o"},
                {"value": "gpt-5", "label": "GPT-5"},
                {"value": "gpt-5-mini", "label": "GPT-5 Mini"},
            ]
        },
        {"id": "temperature", "name": "Temperature", "type": "FLOAT", "value": 0.7, "min": 0.0, "max": 2.0, "step": 0.1},
        {"id": "max_tokens", "name": "Max Tokens", "type": "INT", "value": 8192, "min": 1, "max": 65536, "step": 1},
        {"id": "n_messages", "name": "Max Memory", "type": "INT", "value": 3, "min": 1, "max": 10, "step": 1, "optional": True},
        {"id": "base_url", "name": "Base URL", "type": "STR", "value": "https://api.openai.com/v1", "optional": True},
        {"id": "strict_citation", "name": "Strict Citation", "type": "BOOL", "value": True, "required": False, "optional": True},
        {"id": "default_prompt", "name": "Default Prompt", "type": "STR", "value": default_prompt, "required": False, "optional": True, "expandable": True, "description": "기본 프롬프트로 AI가 따르는 System 지침을 의미합니다."},
    ]

    def execute(
        self,
        text: str,
        tools: Optional[Any] = None,
        memory: Optional[Any] = None,
        rag_context: Optional[Dict[str, Any]] = None,
        args_schema: Optional[BaseModel] = None,
        model: str = "gpt-5",
        temperature: float = 0.7,
        max_tokens: int = 8192,
        n_messages: int = 3,
        base_url: str = "https://api.openai.com/v1",
        strict_citation: bool = True,
        default_prompt: str = default_prompt,
    ) -> Generator[str, None, None]:

        try:
            default_prompt= prefix_prompt+default_prompt
            llm, tools_list, chat_history = self._prepare_llm_and_inputs(tools, memory, model, temperature, max_tokens, base_url)

            additional_rag_context = ""
            if rag_context:
                # rag_context.search_params에서 옵션을 지원 (rerank 등)
                search_params = rag_context.get('search_params', {})
                rerank_flag = search_params.get('rerank', False)
                rerank_top_k = search_params.get('rerank_top_k', search_params.get('top_k', 20))

                # use_model_prompt 옵션 처리
                if search_params.get('use_model_prompt', False):
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
                        context_parts.append(f"{citation_prompt}")
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
                if strict_citation:
                    default_prompt = default_prompt + citation_prompt
                if additional_rag_context and additional_rag_context.strip():
                    final_prompt = ChatPromptTemplate.from_messages([
                        ("system", default_prompt),
                        MessagesPlaceholder(variable_name="chat_history", n_messages=n_messages),
                        ("user", "{input}"),
                        ("user", "{additional_rag_context}"),
                        MessagesPlaceholder(variable_name="agent_scratchpad", n_messages=2)
                    ])
                else:
                    final_prompt = ChatPromptTemplate.from_messages([
                        ("system", default_prompt),
                        MessagesPlaceholder(variable_name="chat_history", n_messages=n_messages),
                        ("user", "{input}"),
                        MessagesPlaceholder(variable_name="agent_scratchpad", n_messages=2)
                    ])

                agent = create_tool_calling_agent(llm, tools_list, final_prompt)
                agent_executor = AgentExecutor(agent=agent, tools=tools_list, verbose=True, handle_parsing_errors=True)
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
        # (기존 _generate_chat_response와 유사한 로직으로 LLM 및 입력 준비)
        from langchain_openai import ChatOpenAI

        config_composer = AppServiceManager.get_config_composer()
        if not config_composer:
            raise ValueError("Config Composer가 설정되지 않았습니다.")

        llm_provider = config_composer.get_config_by_name("DEFAULT_LLM_PROVIDER").value

        if llm_provider == "openai":
            api_key = config_composer.get_config_by_name("OPENAI_API_KEY").value
            print(api_key)
            if not api_key:
                logger.error(f"[CHAT_RESPONSE] OpenAI API 키가 설정되지 않았습니다")
                return "OpenAI API 키가 설정되지 않았습니다."

        elif llm_provider == "vllm":
            api_key = None # 현재 vLLM API 키는 별도로 설정하지 않음
            logger.info(f"[CHAT_RESPONSE] vLLM API 키는 None으로 설정")

            # TODO: vLLM API 키 설정 로직 추가
            # api_key = config_composer.get_config_by_name("VLLM_API_KEY").value
            # if not api_key:
            #     return "vLLM API 키가 설정되지 않았습니다."
        else:
            logger.error(f"[CHAT_RESPONSE] 지원하지 않는 LLM Provider: {llm_provider}")
            return f"지원하지 않는 LLM Provider: {llm_provider}"

        llm = ChatOpenAI(api_key=api_key, model=model, temperature=temperature, max_tokens=max_tokens, base_url=base_url, streaming=True)

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

import logging
from pydantic import BaseModel
from typing import Dict, Any, Optional
from editor.node_composer import Node
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from editor.utils.helper.service_helper import AppServiceManager
from editor.utils.helper.async_helper import sync_run_async
from editor.utils.prefix_prompt import prefix_prompt
from editor.utils.citation_prompt import citation_prompt
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor

logger = logging.getLogger(__name__)

default_prompt = """You are a helpful AI assistant."""

class AgentOpenAINode(Node):
    categoryId = "xgen"
    functionId = "agents"
    nodeId = "agents/openai"
    nodeName = "Agent OpenAI"
    description = "RAG 컨텍스트를 사용하여 채팅 응답을 생성하는 Agent 노드"
    tags = ["agent", "chat", "rag", "openai"]

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
        {
            "id": "model", "name": "Model", "type": "STR", "value": "gpt-4o", "required": True, "optional": False,
            "options": [
                {"value": "gpt-3.5-turbo", "label": "GPT-3.5 Turbo"},
                {"value": "gpt-4", "label": "GPT-4"},
                {"value": "gpt-4o", "label": "GPT-4o"}
            ]
        },
        {"id": "temperature", "name": "Temperature", "type": "FLOAT", "value": 0.7, "required": False, "optional": True, "min": 0.0, "max": 2.0, "step": 0.1},
        {"id": "max_tokens", "name": "Max Tokens", "type": "INT", "value": 8192, "required": False, "optional": True, "min": 1, "max": 65536, "step": 1},
        {"id": "n_messages", "name": "Max Memory", "type": "INT", "value": 3, "min": 1, "max": 10, "step": 1, "optional": True},
        {"id": "base_url", "name": "Base URL", "type": "STR", "value": "https://api.openai.com/v1", "required": False, "optional": True},
        {"id": "default_prompt", "name": "Default Prompt", "type": "STR", "value": default_prompt, "required": False, "optional": True, "expandable": True, "description": "기본 프롬프트로 AI가 따르는 System 지침을 의미합니다."},
    ]

    def execute(
        self,
        text: str,
        tools: Optional[Any] = None,
        memory: Optional[Any] = None,
        rag_context: Optional[Dict[str, Any]] = None,
        args_schema: Optional[BaseModel] = None,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 8192,
        n_messages: int = 3,
        base_url: str = "https://api.openai.com/v1",
        default_prompt: str = default_prompt,
    ) -> str:
        try:
            default_prompt  = prefix_prompt+default_prompt
            if tools is None:
                logger.info(f"[AGENT_EXECUTE] Tools가 None으로 설정됨")
                tools = None
            elif isinstance(tools, list):
                if len(tools) == 0:
                    tools = None
                else:
                    for i, tool in enumerate(tools):
                        logger.info(f"    [{i}] {type(tool)} - {getattr(tool, 'name', 'unknown')}")
            else:
                logger.info(f"[AGENT_EXECUTE] 단일 tool을 리스트로 변환: {type(tools)}")
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
                        additional_rag_context = f"""{rag_context['search_params']['enhance_prompt']}{citation_prompt}

[Context]
{context_text}"""

            response = self._generate_chat_response(text, default_prompt, model, tools, memory, temperature, max_tokens, n_messages, base_url, additional_rag_context)
            logger.info(f"[AGENT_EXECUTE] Chat Agent 응답 생성 완료: {len(response)}자")
            logger.debug(f"[AGENT_EXECUTE] 생성된 응답: {response[:200]}{'...' if len(response) > 200 else ''}")
            return response

        except Exception as e:
            logger.error(f"[AGENT_EXECUTE] Chat Agent 실행 중 오류 발생: {str(e)}")
            logger.error(f"[AGENT_EXECUTE] 오류 타입: {type(e)}")
            logger.exception(f"[AGENT_EXECUTE] 상세 스택 트레이스:")
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
            logger.error(f"[CHAT_RESPONSE] OpenAI 응답 생성 중 오류: {e}")
            logger.error(f"[CHAT_RESPONSE] 오류 타입: {type(e)}")
            logger.exception(f"[CHAT_RESPONSE] 상세 스택 트레이스:")
            return f"응답 생성 중 오류가 발생했습니다: {str(e)}"

import logging
from typing import Any, Optional, Generator
from editor.node_composer import Node
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from editor.utils.helper.service_helper import AppServiceManager
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor

logger = logging.getLogger(__name__)

default_prompt = """You are a helpful AI assistant."""

class AgentOpenAIStreamNode(Node):
    categoryId = "langchain"
    functionId = "agents"
    nodeId = "agents/openai_stream"
    nodeName = "Agent OpenAI Stream"
    description = "RAG 컨텍스트를 사용하여 채팅 응답을 스트리밍으로 생성하는 Agent 노드"
    tags = ["agent", "chat", "rag", "openai", "stream"]

    inputs = [
        {"id": "text", "name": "Text", "type": "STR", "multi": False, "required": True},
        {"id": "tools", "name": "Tools", "type": "TOOL", "multi": True, "required": False, "value": []},
        {"id": "memory", "name": "Memory", "type": "OBJECT", "multi": False, "required": False}
    ]
    outputs = [
        {"id": "stream", "name": "Stream", "type": "STREAM STR", "stream": True}
    ]
    parameters = [
        {
            "id": "model", "name": "Model", "type": "STR", "value": "gpt-4o", "required": True,
            "options": [
                {"value": "gpt-3.5-turbo", "label": "GPT-3.5 Turbo"},
                {"value": "gpt-4", "label": "GPT-4"},
                {"value": "gpt-4o", "label": "GPT-4o"}
            ]
        },
        {"id": "temperature", "name": "Temperature", "type": "FLOAT", "value": 0.7, "min": 0.0, "max": 2.0, "step": 0.1},
        {"id": "max_tokens", "name": "Max Tokens", "type": "INT", "value": 4096, "min": 1, "max": 8192, "step": 1},
        {"id": "n_messages", "name": "Max Memory", "type": "INT", "value": 3, "min": 1, "max": 10, "step": 1, "optional": True},
        {"id": "base_url", "name": "Base URL", "type": "STRING", "value": "https://api.openai.com/v1", "optional": True},
    ]

    def execute(self, text: str, tools: Optional[Any] = None, memory: Optional[Any] = None,
                model: str = "gpt-4o", temperature: float = 0.7, max_tokens: int = 4096, n_messages: int = 3, base_url: str = "https://api.openai.com/v1") -> Generator[str, None, None]:

        try:
            llm, tools_list, chat_history = self._prepare_llm_and_inputs(tools, memory, model, temperature, max_tokens, base_url)

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

import logging
import asyncio
from typing import Dict, Any, Optional
from editor.node_composer import Node
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from editor.utils.helper.service_helper import AppServiceManager
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor

logger = logging.getLogger(__name__)

default_prompt = """당신은 사용자의 요청에 대해 도움을 제공하는 AI 어시스턴트입니다."""

class AgentOpenAINodeV2(Node):
    categoryId = "langchain"
    functionId = "agents"
    nodeId = "agents/openai_v2"
    nodeName = "Agent OpenAI V2"
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
            "id": "tools",
            "name": "Tools",
            "type": "TOOL",
            "multi": True,
            "required": False,
            "value": []
        },
        {
            "id": "memory",
            "name": "Memory",
            "type": "OBJECT",
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
            "value": "gpt-4o",
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
        },
        {
            "id": "base_url",
            "name": "Base URL",
            "type": "STRING",
            "value": "https://api.openai.com/v1",
            "required": False,
            "optional": True
        },
    ]

    def execute(self, text: str, tools: Optional[Any] = None, memory: Optional[Any] = None,
                model: str = "gpt-4o", temperature: float = 0.7, max_tokens: int = 1000, base_url: str = "https://api.openai.com/v1") -> str:
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


            # OpenAI API를 사용하여 응답 생성
            response = self._generate_chat_response(text, default_prompt, model, tools, memory, temperature, max_tokens, base_url)
            logger.info(f"[AGENT_EXECUTE] Chat Agent 응답 생성 완료: {len(response)}자")
            logger.debug(f"[AGENT_EXECUTE] 생성된 응답: {response[:200]}{'...' if len(response) > 200 else ''}")
            return response

        except Exception as e:
            logger.error(f"[AGENT_EXECUTE] Chat Agent 실행 중 오류 발생: {str(e)}")
            logger.error(f"[AGENT_EXECUTE] 오류 타입: {type(e)}")
            logger.exception(f"[AGENT_EXECUTE] 상세 스택 트레이스:")
            return f"죄송합니다. 응답 생성 중 오류가 발생했습니다: {str(e)}"


    def _generate_chat_response(self, text: str, prompt: str, model: str, tools: Optional[Any], memory: Optional[Any], temperature: float, max_tokens: int, base_url: str) -> str:
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
                "input": text
            }

            if tools is not None:
                final_prompt = ChatPromptTemplate.from_messages([
                    ("system", prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("user", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad")
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
                final_prompt = ChatPromptTemplate.from_messages([
                    ("system", prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
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

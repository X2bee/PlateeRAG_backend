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

class AgentVLLMNodeV2(Node):
    categoryId = "langchain"
    functionId = "agents"
    nodeId = "agents/vllm_v2"
    nodeName = "Agent VLLM V2"
    description = "RAG 컨텍스트를 사용하여 채팅 응답을 생성하는 Agent 노드"
    tags = ["agent", "chat", "rag", "vllm"]

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
            "value": "",
            "required": False,
            "optional": True,
        },
        {
            "id": "temperature",
            "name": "Temperature",
            "type": "FLOAT",
            "value": None,
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
            "value": None,
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
            "value": "",
            "required": False,
            "optional": True
        },
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

    def execute(self, text: str, tools, memory: Optional[Any] = None,
                model: str = "", temperature: float = None, max_tokens: int = None, base_url: str = "") -> str:
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

            prompt = default_prompt

            # OpenAI API를 사용하여 응답 생성
            response = self._generate_chat_response(text, prompt, model, tools, memory, temperature, max_tokens, base_url)

            logger.info(f"Chat Agent 응답 생성 완료: {len(response)}자")
            return response

        except Exception as e:
            logger.error(f"Chat Agent 실행 중 오류 발생: {str(e)}")
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

            final_prompt = ChatPromptTemplate.from_messages([
                ("system", prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])

            chat_history = []
            if memory:
                chat_history = memory.load_memory_variables({})["chat_history"]

            inputs = {
                "chat_history": chat_history,
                "input": text
            }

            agent = create_tool_calling_agent(llm, tools, final_prompt)
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                handle_parsing_errors=True,
            )

            response = agent_executor.invoke(inputs)
            return response["output"]

        except Exception as e:
            logger.error(f"OpenAI 응답 생성 중 오류: {e}")
            return f"응답 생성 중 오류가 발생했습니다: {str(e)}"

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

    def execute(self, text: str, tools, memory: Optional[Any] = None,
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
            logger.info(f"[AGENT_EXECUTE] Chat Agent 실행 시작")
            logger.info(f"[AGENT_EXECUTE] 입력 파라미터:")
            logger.info(f"  - text: '{text[:100]}{'...' if len(text) > 100 else ''}'")
            logger.info(f"  - model: {model}")
            logger.info(f"  - temperature: {temperature}")
            logger.info(f"  - max_tokens: {max_tokens}")
            logger.info(f"  - base_url: {base_url}")
            logger.info(f"  - tools 타입: {type(tools)}")
            logger.info(f"  - memory 타입: {type(memory)}")

            # tools 처리 로직
            original_tools = tools
            if tools is None:
                logger.info(f"[AGENT_EXECUTE] Tools가 None으로 설정됨")
                tools = None
            elif isinstance(tools, list):
                logger.info(f"[AGENT_EXECUTE] Tools는 리스트 타입, 길이: {len(tools)}")
                if len(tools) == 0:
                    logger.info(f"[AGENT_EXECUTE] 빈 tools 리스트, None으로 설정")
                    tools = None
                else:
                    logger.info(f"[AGENT_EXECUTE] Tools 리스트 내용:")
                    for i, tool in enumerate(tools):
                        logger.info(f"    [{i}] {type(tool)} - {getattr(tool, 'name', 'unknown')}")
            else:
                # 단일 StructuredTool인 경우 리스트로 감싸기
                logger.info(f"[AGENT_EXECUTE] 단일 tool을 리스트로 변환: {type(tools)}")
                tools = [tools]
                logger.info(f"[AGENT_EXECUTE] 변환된 tools: {[getattr(tool, 'name', 'unknown') for tool in tools]}")

            logger.info(f"[AGENT_EXECUTE] 프롬프트 설정: {default_prompt[:50]}...")

            # OpenAI API를 사용하여 응답 생성
            logger.info(f"[AGENT_EXECUTE] _generate_chat_response 호출 시작")
            response = self._generate_chat_response(text, default_prompt, model, tools, memory, temperature, max_tokens, base_url)
            logger.info(f"[AGENT_EXECUTE] _generate_chat_response 호출 완료")

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
            logger.info(f"[CHAT_RESPONSE] 채팅 응답 생성 시작")
            logger.info(f"[CHAT_RESPONSE] 파라미터 확인:")
            logger.info(f"  - text 길이: {len(text)}")
            logger.info(f"  - prompt 길이: {len(prompt)}")
            logger.info(f"  - model: {model}")
            logger.info(f"  - tools: {tools is not None} ({len(tools) if tools else 0} 개)")
            logger.info(f"  - memory: {memory is not None}")
            logger.info(f"  - temperature: {temperature}")
            logger.info(f"  - max_tokens: {max_tokens}")
            logger.info(f"  - base_url: {base_url}")

            logger.info(f"[CHAT_RESPONSE] Config Composer 가져오기 시도")
            config_composer = AppServiceManager.get_config_composer()
            if not config_composer:
                logger.error(f"[CHAT_RESPONSE] Config Composer가 None입니다")
                return "Config Composer가 설정되지 않았습니다."
            logger.info(f"[CHAT_RESPONSE] Config Composer 획득 성공")

            # OpenAI API 키 설정
            logger.info(f"[CHAT_RESPONSE] LLM Provider 설정 확인")
            llm_provider = config_composer.get_config_by_name("DEFAULT_LLM_PROVIDER").value
            logger.info(f"[CHAT_RESPONSE] LLM Provider: {llm_provider}")

            if llm_provider == "openai":
                logger.info(f"[CHAT_RESPONSE] OpenAI API 키 가져오기")
                api_key = config_composer.get_config_by_name("OPENAI_API_KEY").value
                if not api_key:
                    logger.error(f"[CHAT_RESPONSE] OpenAI API 키가 설정되지 않았습니다")
                    return "OpenAI API 키가 설정되지 않았습니다."
                logger.info(f"[CHAT_RESPONSE] OpenAI API 키 확인 완료 (길이: {len(api_key)})")

            elif llm_provider == "vllm":
                logger.info(f"[CHAT_RESPONSE] vLLM API를 사용합니다")
                api_key = None # 현재 vLLM API 키는 별도로 설정하지 않음
                logger.info(f"[CHAT_RESPONSE] vLLM API 키는 None으로 설정")

                # TODO: vLLM API 키 설정 로직 추가
                # api_key = config_composer.get_config_by_name("VLLM_API_KEY").value
                # if not api_key:
                #     return "vLLM API 키가 설정되지 않았습니다."
            else:
                logger.error(f"[CHAT_RESPONSE] 지원하지 않는 LLM Provider: {llm_provider}")
                return f"지원하지 않는 LLM Provider: {llm_provider}"

            logger.info(f"[CHAT_RESPONSE] ChatOpenAI 임포트")
            from langchain_openai import ChatOpenAI

            logger.info(f"[CHAT_RESPONSE] ChatOpenAI 인스턴스 생성")
            llm = ChatOpenAI(
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                base_url=base_url
            )
            logger.info(f"[CHAT_RESPONSE] ChatOpenAI 인스턴스 생성 완료")

            logger.info(f"[CHAT_RESPONSE] 프롬프트 템플릿 생성")
            final_prompt = ChatPromptTemplate.from_messages([
                ("system", prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])
            logger.info(f"[CHAT_RESPONSE] 프롬프트 템플릿 생성 완료")

            logger.info(f"[CHAT_RESPONSE] 채팅 히스토리 처리")
            chat_history = []
            if memory:
                logger.info(f"[CHAT_RESPONSE] 메모리에서 채팅 히스토리 로드")
                try:
                    memory_vars = memory.load_memory_variables({})
                    chat_history = memory_vars.get("chat_history", [])
                    logger.info(f"[CHAT_RESPONSE] 채팅 히스토리 로드 완료: {len(chat_history)}개 메시지")
                except Exception as e:
                    logger.warning(f"[CHAT_RESPONSE] 메모리에서 채팅 히스토리 로드 실패: {e}")
                    chat_history = []
            else:
                logger.info(f"[CHAT_RESPONSE] 메모리가 없어 빈 채팅 히스토리 사용")

            logger.info(f"[CHAT_RESPONSE] 입력 데이터 준비")
            inputs = {
                "chat_history": chat_history,
                "input": text
            }
            logger.debug(f"[CHAT_RESPONSE] 입력 데이터: {inputs}")

            logger.info(f"[CHAT_RESPONSE] Tool calling agent 생성")
            if tools:
                logger.info(f"[CHAT_RESPONSE] {len(tools)}개의 도구와 함께 agent 생성")
                for i, tool in enumerate(tools):
                    logger.info(f"[CHAT_RESPONSE] Tool [{i}]: {getattr(tool, 'name', 'unknown')} - {getattr(tool, 'description', 'no description')}")
            else:
                logger.info(f"[CHAT_RESPONSE] 도구 없이 agent 생성")

            agent = create_tool_calling_agent(llm, tools, final_prompt)
            logger.info(f"[CHAT_RESPONSE] Tool calling agent 생성 완료")

            logger.info(f"[CHAT_RESPONSE] AgentExecutor 생성")
            if tools == None:
                tools = []
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                handle_parsing_errors=True,
            )
            logger.info(f"[CHAT_RESPONSE] AgentExecutor 생성 완료")

            logger.info(f"[CHAT_RESPONSE] Agent 실행 시작")
            response = agent_executor.invoke(inputs)
            logger.info(f"[CHAT_RESPONSE] Agent 실행 완료")

            logger.info(f"[CHAT_RESPONSE] 응답 추출")
            output = response["output"]
            logger.info(f"[CHAT_RESPONSE] 최종 응답 길이: {len(output)}")
            logger.debug(f"[CHAT_RESPONSE] 최종 응답: {output[:200]}{'...' if len(output) > 200 else ''}")

            return output

        except Exception as e:
            logger.error(f"[CHAT_RESPONSE] OpenAI 응답 생성 중 오류: {e}")
            logger.error(f"[CHAT_RESPONSE] 오류 타입: {type(e)}")
            logger.exception(f"[CHAT_RESPONSE] 상세 스택 트레이스:")
            return f"응답 생성 중 오류가 발생했습니다: {str(e)}"

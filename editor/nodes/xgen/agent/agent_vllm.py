import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel
from editor.node_composer import Node
from langchain.schema.output_parser import StrOutputParser
from langchain_core.exceptions import OutputParserException
from editor.utils.helper.agent_helper import (
    NonStreamingAgentHandler,
    NonStreamingAgentHandlerWithToolOutput,
    use_guarder_for_text_moderation,
    XgenJsonOutputParser
)
from editor.utils.prefix_prompt import get_prefix_prompt
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from fastapi import Request
from controller.helper.singletonHelper import get_config_composer
from editor.nodes.xgen.agent.functions import prepare_llm_components, rag_context_builder, create_json_output_prompt, create_tool_context_prompt, create_context_prompt
logger = logging.getLogger(__name__)

default_prompt = """You are a helpful AI assistant."""

class AgentVLLMNode(Node):
    categoryId = "xgen"
    functionId = "agents"
    nodeId = "agents/vllm"
    nodeName = "Agent VLLM"
    description = "RAG 컨텍스트를 사용하여 채팅 응답을 생성하는 Agent 노드"
    tags = ["agent", "chat", "rag", "vllm"]
    disable = True

    inputs = [
        {"id": "text", "name": "Text", "type": "STR", "multi": False, "required": True},
        {"id": "tools", "name": "Tools", "type": "TOOL", "multi": True, "required": False, "value": []},
        {"id": "memory", "name": "Memory", "type": "OBJECT", "multi": False, "required": False},
        {"id": "rag_context", "name": "RAG Context", "type": "DocsContext", "multi": False, "required": False},
        {"id": "args_schema", "name": "ArgsSchema", "type": "OutputSchema"},
        {"id": "plan", "name": "Plan", "type": "PLAN", "required": False},
    ]
    outputs = [
        {"id": "result", "name": "Result", "type": "STR"},
    ]
    parameters = [
        {"id": "model", "name": "Model", "type": "STR", "value": "", "is_api": True, "api_name": "api_vllm_model_name", "required": True},
        {"id": "temperature", "name": "Temperature", "type": "FLOAT", "value": 0.0, "required": False, "optional": True, "min": 0.0, "max": 2.0, "step": 0.1},
        {"id": "max_tokens", "name": "Max Tokens", "type": "INT", "value": 8192, "required": False, "optional": True, "min": 1, "max": 65536, "step": 1},
        {"id": "base_url", "name": "Base URL", "type": "STR", "value": "", "is_api": True, "api_name": "api_vllm_api_base_url", "required": True},
        {"id": "strict_citation", "name": "Strict Citation", "type": "BOOL", "value": True, "required": False, "optional": True},
        {"id": "return_intermediate_steps", "name": "Return Intermediate Steps", "type": "BOOL", "value": False, "required": False, "optional": True, "description": "Tool 사용시 해당 과정을 출력할지 여부를 결정합니다."},
        {"id": "default_prompt", "name": "Default Prompt", "type": "STR", "value": default_prompt, "required": False, "optional": True, "expandable": True, "description": "기본 프롬프트로 AI가 따르는 System 지침을 의미합니다."},
        {"id": "use_guarder", "name": "Use Guarder Service", "type": "BOOL", "value": False, "required": False, "optional": True, "description": "Guarder 서비스를 사용할지 여부입니다."},
    ]

    def api_vllm_model_name(self, request: Request) -> Dict[str, Any]:
        config_composer = get_config_composer(request)
        return config_composer.get_config_by_name("VLLM_MODEL_NAME").value

    def api_vllm_api_base_url(self, request: Request) -> Dict[str, Any]:
        config_composer = get_config_composer(request)
        return config_composer.get_config_by_name("VLLM_API_BASE_URL").value

    def execute(
        self,
        text: str,
        tools: Optional[Any] = None,
        memory: Optional[Any] = None,
        rag_context: Optional[Dict[str, Any]] = None,
        args_schema: Optional[BaseModel] = None,
        plan: Optional[Dict[str, Any]] = None,
        model: str = "",
        temperature: float = None,
        max_tokens: int = 8192,
        base_url: str = "",
        strict_citation: bool = True,
        return_intermediate_steps: bool = False,
        default_prompt: str = default_prompt,
        use_guarder: bool = False,
    ) -> str:
        try:
            if use_guarder:
                is_safe, moderation_message = use_guarder_for_text_moderation(text)
                if not is_safe:
                    return moderation_message

            default_prompt = get_prefix_prompt() + default_prompt
            llm, tools_list, chat_history = prepare_llm_components(text, tools, memory, model, temperature, max_tokens, base_url, streaming=True, plan=plan)

            additional_rag_context = ""
            if rag_context:
                additional_rag_context = rag_context_builder(text, rag_context, strict_citation)
            inputs = {"input": text, "chat_history": chat_history, "additional_rag_context": additional_rag_context}

            if args_schema:
                default_prompt = create_json_output_prompt(args_schema, default_prompt)
            if tools_list and len(tools_list) > 0:
                final_prompt = create_tool_context_prompt(additional_rag_context, default_prompt, plan=plan)
                agent = create_tool_calling_agent(llm, tools_list, final_prompt)
                agent_executor = AgentExecutor(
                    agent=agent,
                    tools=tools_list,
                    verbose=True,
                    handle_parsing_errors=True,
                    max_iterations=10,
                )
                if return_intermediate_steps:
                    handler = NonStreamingAgentHandlerWithToolOutput()
                else:
                    handler = NonStreamingAgentHandler()
                response = agent_executor.invoke(inputs, {"callbacks": [handler]})
                output = response["output"]
                return handler.get_formatted_output(output)

            else:
                final_prompt = create_context_prompt(additional_rag_context, default_prompt, strict_citation, plan=plan)

                # XgenJsonOutputParser 또는 StrOutputParser 사용
                if args_schema:
                    parser = XgenJsonOutputParser()
                else:
                    parser = StrOutputParser()

                chain = final_prompt | llm | parser
                response = chain.invoke(inputs)
                return response

        except OutputParserException as e:
            # XgenJsonOutputParser에서도 실패한 경우
            logger.error(f"[AGENT_EXECUTE] OutputParser 예외 발생 (모든 파싱 시도 실패): {str(e)}")

            # 구조화된 에러 응답 반환
            if args_schema:
                llm_output = getattr(e, 'llm_output', str(e))
                return {
                    "error": "JSON 파싱 실패",
                    "raw_output": llm_output,
                    "parse_error": str(e),
                    "expected_schema": args_schema.__name__ if hasattr(args_schema, '__name__') else str(args_schema),
                    "suggestion": "LLM이 올바른 JSON 형식으로 응답하지 않았습니다."
                }
            else:
                # 문자열 출력이 예상되는 경우 원본 출력 반환
                llm_output = getattr(e, 'llm_output', None)
                if llm_output:
                    return llm_output
                raise
        except Exception as e:
            logger.error(f"[AGENT_EXECUTE] Chat Agent 실행 중 오류 발생: {str(e)}")
            logger.error(f"[AGENT_EXECUTE] 오류 타입: {type(e)}")
            logger.exception(f"[AGENT_EXECUTE] 상세 스택 트레이스:")

            # 구조화된 에러 응답 반환 (라우팅 가능하도록)
            if args_schema:
                return {
                    "error": "Agent 실행 실패",
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "user_message": f"죄송합니다. 응답 생성 중 오류가 발생했습니다: {str(e)}"
                }
            else:
                return f"죄송합니다. 응답 생성 중 오류가 발생했습니다: {str(e)}"

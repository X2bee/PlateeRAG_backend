import logging
from pydantic import BaseModel
from typing import Dict, Any, Optional, Generator
from editor.node_composer import Node
from editor.nodes.xgen.agent.functions import prepare_llm_components, rag_context_builder, create_json_output_prompt, create_tool_context_prompt, create_context_prompt
from editor.utils.helper.stream_helper import EnhancedAgentStreamingHandler, EnhancedAgentStreamingHandlerWithToolOutput, execute_agent_streaming
from editor.utils.helper.agent_helper import use_guarder_for_text_moderation
from editor.utils.prefix_prompt import get_prefix_prompt
from langchain.agents import create_agent
from fastapi import Request
from controller.helper.singletonHelper import get_config_composer

logger = logging.getLogger(__name__)

default_prompt = """You are a helpful AI assistant."""

class AgentVLLMStreamNode(Node):
    categoryId = "xgen"
    functionId = "agents"
    nodeId = "agents/vllm_stream"
    nodeName = "Agent VLLM Stream"
    description = "RAG 컨텍스트를 사용하여 채팅 응답을 스트리밍으로 생성하는 Agent 노드"
    tags = ["agent", "chat", "rag", "vllm", "stream"]
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
        {"id": "stream", "name": "Stream", "type": "STREAM STR", "stream": True}
    ]
    parameters = [
        {"id": "model", "name": "Model", "type": "STR", "value": "", "is_api": True, "api_name": "api_vllm_model_name", "required": True},
        {"id": "temperature", "name": "Temperature", "type": "FLOAT", "value": 0.0, "required": False, "optional": True, "min": 0.0, "max": 2.0, "step": 0.1},
        {"id": "max_tokens", "name": "Max Tokens", "type": "INT", "value": 8192, "required": False, "optional": True, "min": 1, "max": 65536, "step": 1},
        {"id": "base_url", "name": "Base URL", "type": "STR", "value": "", "is_api": True, "api_name": "api_vllm_api_base_url", "required": True},
        {"id": "strict_citation", "name": "Strict Citation", "type": "BOOL", "value": True, "required": False, "optional": True},
        {"id": "default_prompt", "name": "Default Prompt", "type": "STR", "value": default_prompt, "required": False, "optional": True, "expandable": True, "description": "기본 프롬프트로 AI가 따르는 System 지침을 의미합니다."},
        {"id": "return_intermediate_steps", "name": "Return Intermediate Steps", "type": "BOOL", "value": False, "required": False, "optional": True, "description": "중간 단계를 반환할지 여부입니다."},
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
        model: str = "x2bee/Polar-14B",
        temperature: float = 0.7,
        max_tokens: int = 8192,
        base_url: str = "",
        strict_citation: bool = True,
        default_prompt: str = default_prompt,
        return_intermediate_steps: bool = False,
        use_guarder: bool = False,
    ) -> Generator[str, None, None]:

        try:
            if use_guarder:
                is_safe, moderation_message = use_guarder_for_text_moderation(text)
                if not is_safe:
                    yield moderation_message
                    return

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

                # LangChain 1.0.0의 create_agent는 system_prompt로 문자열만 받습니다
                # ChatPromptTemplate에서 system message 추출
                system_prompt_text = default_prompt
                if hasattr(final_prompt, 'messages') and len(final_prompt.messages) > 0:
                    first_msg = final_prompt.messages[0]
                    if hasattr(first_msg, 'prompt') and hasattr(first_msg.prompt, 'template'):
                        system_prompt_text = first_msg.prompt.template

                # create_agent는 이제 CompiledStateGraph를 반환합니다
                agent_graph = create_agent(
                    model=llm,
                    tools=tools_list,
                    system_prompt=system_prompt_text
                )

                if return_intermediate_steps:
                    handler = EnhancedAgentStreamingHandlerWithToolOutput()
                else:
                    handler = EnhancedAgentStreamingHandler()

                # LangGraph의 새로운 입력 형식: messages 리스트 사용
                from langchain_core.messages import HumanMessage

                graph_inputs = {"messages": chat_history + [HumanMessage(content=text)]}
                if additional_rag_context:
                    graph_inputs["additional_rag_context"] = additional_rag_context

                # LangGraph 기반 agent 실행을 위한 async executor
                async_executor = lambda: agent_graph.ainvoke(
                    graph_inputs,
                    {"callbacks": [handler]}
                )

                try:
                    for token in execute_agent_streaming(async_executor, handler):
                        yield token
                except Exception as e:
                    logger.error(f"Agent streaming error: {str(e)}", exc_info=True)
                    yield f"\nStreaming Error: {str(e)}\n"
            else:
                final_prompt = create_context_prompt(additional_rag_context, default_prompt, strict_citation, plan=plan)
                chain = final_prompt | llm
                for chunk in chain.stream(inputs):
                    yield chunk.content

        except Exception as e:
            logger.error(f"[AGENT_STREAM_EXECUTE] 스트리밍 Agent 실행 중 오류 발생: {str(e)}", exc_info=True)
            yield f"죄송합니다. 응답 생성 중 오류가 발생했습니다: {str(e)}"

import logging
from pydantic import BaseModel
from typing import Dict, Any, Optional
from editor.node_composer import Node
from editor.nodes.xgen.agent.functions import (
    prepare_llm_components,
    rag_context_builder,
    create_json_output_prompt,
)
from editor.utils.helper.agent_helper import (
    NonStreamingAgentHandler,
    NonStreamingAgentHandlerWithToolOutput,
    use_guarder_for_text_moderation,
    XgenJsonOutputParser,
)
from editor.utils.prefix_prompt import get_prefix_prompt
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

logger = logging.getLogger(__name__)

default_prompt = """You are a helpful AI assistant."""

class AgentOpenAINode(Node):
    categoryId = "xgen"
    functionId = "agents"
    nodeId = "agents/openai"
    nodeName = "Agent OpenAI"
    description = "RAG 컨텍스트를 사용하여 채팅 응답을 생성하는 Agent 노드"
    tags = ["agent", "chat", "rag", "openai"]
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
        {
            "id": "model", "name": "Model", "type": "STR", "value": "gpt-5", "required": True,
            "options": [
                {"value": "gpt-3.5-turbo", "label": "GPT-3.5 Turbo"},
                {"value": "gpt-4", "label": "GPT-4"},
                {"value": "gpt-4o", "label": "GPT-4o"},
                {"value": "o4-mini", "label": "o4 mini"},
                {"value": "gpt-4.1", "label": "GPT-4.1"},
                {"value": "gpt-4.1-mini", "label": "GPT-4.1 Mini"},
                {"value": "gpt-5", "label": "GPT-5"},
                {"value": "gpt-5-mini", "label": "GPT-5 Mini"},
                {"value": "gpt-5-nano", "label": "GPT-5 Nano"}
            ]
        },
        {"id": "temperature", "name": "Temperature", "type": "FLOAT", "value": 1, "required": False, "optional": True, "min": 0.0, "max": 2.0, "step": 0.1},
        {"id": "max_tokens", "name": "Max Tokens", "type": "INT", "value": 8192, "required": False, "optional": True, "min": 1, "max": 65536, "step": 1},
        {"id": "base_url", "name": "Base URL", "type": "STR", "value": "https://api.openai.com/v1", "required": False, "optional": True},
        {"id": "strict_citation", "name": "Strict Citation", "type": "BOOL", "value": True, "required": False, "optional": True},
        {"id": "return_intermediate_steps", "name": "Return Intermediate Steps", "type": "BOOL", "value": False, "required": False, "optional": True, "description": "Tool 사용시 해당 과정을 출력할지 여부를 결정합니다."},
        {"id": "default_prompt", "name": "Default Prompt", "type": "STR", "value": default_prompt, "required": False, "optional": True, "expandable": True, "description": "기본 프롬프트로 AI가 따르는 System 지침을 의미합니다."},
        {"id": "use_guarder", "name": "Use Guarder Service", "type": "BOOL", "value": False, "required": False, "optional": True, "description": "Guarder 서비스를 사용할지 여부입니다."},
    ]

    def execute(
        self,
        text: str,
        tools: Optional[Any] = None,
        memory: Optional[Any] = None,
        rag_context: Optional[Dict[str, Any]] = None,
        args_schema: Optional[BaseModel] = None,
        plan: Optional[Dict[str, Any]] = None,
        model: str = "gpt-5",
        temperature: float = 1,
        max_tokens: int = 8192,
        base_url: str = "https://api.openai.com/v1",
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
            llm, tools_list, chat_history = prepare_llm_components(
                text,
                tools,
                memory,
                model,
                temperature,
                max_tokens,
                base_url,
                streaming=False,
                plan=plan,
            )

            additional_rag_context = ""
            if rag_context:
                additional_rag_context = rag_context_builder(
                    text, rag_context, strict_citation
                )

            if args_schema:
                default_prompt = create_json_output_prompt(args_schema, default_prompt)

            system_prompt_text = default_prompt

            agent_summarization_model = f"openai:{model}"

            agent_summarization_middleware = SummarizationMiddleware(
                model=agent_summarization_model,
                max_tokens_before_summary=4000,
                messages_to_keep=10,
            )

            if tools_list:
                agent_graph = create_agent(
                    model=llm,
                    tools=tools_list,
                    system_prompt=system_prompt_text,
                    middleware=[agent_summarization_middleware],
                )
            else:
                agent_graph = create_agent(
                    model=llm,
                    system_prompt=system_prompt_text,
                    middleware=[agent_summarization_middleware],
                )

            if return_intermediate_steps:
                handler = NonStreamingAgentHandlerWithToolOutput()
            else:
                handler = NonStreamingAgentHandler()

            # LangGraph의 새로운 입력 형식: messages 리스트 사용
            from langchain_core.messages import HumanMessage

            # additional_rag_context를 HumanMessage content에 포함
            final_user_message = text
            if additional_rag_context:
                final_user_message = f"{text}\n\n{additional_rag_context}"

            graph_inputs = {"messages": chat_history + [HumanMessage(content=final_user_message)]}

            # LangGraph 기반 agent 실행
            response = agent_graph.invoke(graph_inputs, {"callbacks": [handler]})
            output = (
                response.get("messages", [])[-1].content
                if response.get("messages")
                else ""
            )

            formatted_output = handler.get_formatted_output(output)

            # args_schema가 있으면 XgenJsonOutputParser로 파싱하여 정합성 검증
            if args_schema:
                try:
                    parser = XgenJsonOutputParser()
                    parsed_output = parser.parse(output)
                    logger.info("[AGENT_EXECUTE] JSON 파싱 성공")
                    return parsed_output
                except Exception as parse_error:
                    logger.error(f"[AGENT_EXECUTE] JSON 파싱 실패: {parse_error}")
                    logger.error(f"[AGENT_EXECUTE] 원본 출력: {output}")
                    # 파싱 실패 시 원본 formatted_output 반환
                    return formatted_output

            return formatted_output

        except Exception as e:
            logger.error(
                f"[AGENT_EXECUTE] Agent 실행 중 오류 발생: {str(e)}",
                exc_info=True,
            )
            return f"죄송합니다. 응답 생성 중 오류가 발생했습니다: {str(e)}"

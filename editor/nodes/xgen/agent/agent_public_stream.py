from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from typing import Dict, Any, Optional, Generator
from pydantic import BaseModel
import logging
from editor.node_composer import Node
from editor.nodes.xgen.agent.functions import (
    prepare_llm_components,
    rag_context_builder,
    create_json_output_prompt,
    create_tool_context_prompt,
    create_context_prompt,
)
from editor.utils.helper.stream_helper import (
    EnhancedAgentStreamingHandler,
    EnhancedAgentStreamingHandlerWithToolOutput,
    execute_agent_streaming,
)
from editor.utils.prefix_prompt import get_prefix_prompt
from editor.utils.helper.agent_helper import use_guarder_for_text_moderation

logger = logging.getLogger(__name__)

default_prompt = """You are a helpful AI assistant."""


class AgentPublicStreamNode(Node):
    categoryId = "xgen"
    functionId = "agents"
    nodeId = "agents/public_stream"
    nodeName = "Agent Public Stream"
    description = "도구, 메모리 및 RAG 컨텍스트 등을 활용하여 채팅 응답을 스트리밍으로 생성하는 Agent"
    tags = ["agent", "chat", "rag", "public_model", "stream"]

    inputs = [
        {"id": "text", "name": "Text", "type": "STR", "multi": False, "required": True},
        {
            "id": "tools",
            "name": "Tools",
            "type": "TOOL",
            "multi": True,
            "required": False,
            "value": [],
        },
        {
            "id": "memory",
            "name": "Memory",
            "type": "OBJECT",
            "multi": False,
            "required": False,
        },
        {
            "id": "rag_context",
            "name": "RAG Context",
            "type": "DocsContext",
            "multi": False,
            "required": False,
        },
        {"id": "args_schema", "name": "ArgsSchema", "type": "OutputSchema"},
        {"id": "plan", "name": "Plan", "type": "PLAN", "required": False},
    ]
    outputs = [{"id": "stream", "name": "Stream", "type": "STREAM STR", "stream": True}]
    parameters = [
        {
            "id": "model",
            "name": "Model",
            "type": "STR",
            "value": "gpt-4.1",
            "required": True,
            "options": [
                {"value": "gpt-3.5-turbo", "label": "GPT-3.5 Turbo"},
                {"value": "gpt-4", "label": "GPT-4"},
                {"value": "gpt-4o", "label": "GPT-4o"},
                {"value": "o4-mini", "label": "o4 mini"},
                {"value": "gpt-4.1", "label": "GPT-4.1"},
                {"value": "gpt-4.1-mini", "label": "GPT-4.1 Mini"},
                {"value": "gpt-5", "label": "GPT-5"},
                {"value": "gpt-5-mini", "label": "GPT-5 Mini"},
                {"value": "gpt-5-nano", "label": "GPT-5 Nano"},
                {"value": "claude-3-5-haiku-20241022", "label": "Claude Haiku 3.5"},
                {"value": "claude-3-5-sonnet-20241022", "label": "Claude Sonnet 3.5"},
                {"value": "claude-3-7-sonnet-20250219", "label": "Claude Sonnet 3.7"},
                {"value": "claude-sonnet-4-20250514", "label": "Claude Sonnet 4"},
                {"value": "claude-opus-4-20250514", "label": "Claude Opus 4"},
                {"value": "claude-opus-4-1-20250805", "label": "Claude Opus 4.1"},
                {"value": "claude-sonnet-4-5-20250929", "label": "Claude Sonnet 4.5"},
                {"value": "claude-haiku-4-5-20251001", "label": "Claude Haiku 4.5"},
                {"value": "gemini-2.0-flash", "label": "Gemini 2.0 Flash"},
                {"value": "gemini-2.0-flash-lite", "label": "Gemini 2.0 Flash Lite"},
                {"value": "gemini-2.5-flash", "label": "Gemini 2.5 Flash"},
                {"value": "gemini-2.5-flash-lite", "label": "Gemini 2.5 Flash Lite"},
                {"value": "gemini-2.5-pro", "label": "Gemini 2.5 Pro"},
            ],
        },
        {
            "id": "temperature",
            "name": "Temperature",
            "type": "FLOAT",
            "value": 1,
            "min": 0.0,
            "max": 2.0,
            "step": 0.1,
        },
        {
            "id": "max_tokens",
            "name": "Max Tokens",
            "type": "INT",
            "value": 8192,
            "min": 1,
            "max": 65536,
            "step": 1,
        },
        {
            "id": "strict_citation",
            "name": "Strict Citation",
            "type": "BOOL",
            "value": True,
            "required": False,
            "optional": True,
        },
        {
            "id": "default_prompt",
            "name": "Default Prompt",
            "type": "STR",
            "value": default_prompt,
            "required": False,
            "optional": True,
            "expandable": True,
            "description": "기본 프롬프트로 AI가 따르는 System 지침을 의미합니다.",
        },
        {
            "id": "return_intermediate_steps",
            "name": "Return Intermediate Steps",
            "type": "BOOL",
            "value": False,
            "required": False,
            "optional": True,
            "description": "중간 단계를 반환할지 여부입니다.",
        },
        {
            "id": "use_guarder",
            "name": "Use Guarder Service",
            "type": "BOOL",
            "value": False,
            "required": False,
            "optional": True,
            "description": "Guarder 서비스를 사용할지 여부입니다.",
        },
    ]

    def execute(
        self,
        text: str,
        tools: Optional[Any] = None,
        memory: Optional[Any] = None,
        rag_context: Optional[Dict[str, Any]] = None,
        args_schema: Optional[BaseModel] = None,
        plan: Optional[Dict[str, Any]] = None,
        model: str = "gpt-4.1",
        temperature: float = 1,
        max_tokens: int = 8192,
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
            llm, tools_list, chat_history = prepare_llm_components(
                text,
                tools,
                memory,
                model,
                temperature,
                max_tokens,
                streaming=True,
                plan=plan,
            )
            additional_rag_context = ""
            if rag_context:
                additional_rag_context = rag_context_builder(
                    text, rag_context, strict_citation
                )

            if args_schema:
                default_prompt = create_json_output_prompt(args_schema, default_prompt)


            final_prompt = create_tool_context_prompt(
                additional_rag_context, default_prompt, plan=plan
            )

            system_prompt_text = default_prompt
            if hasattr(final_prompt, "messages") and len(final_prompt.messages) > 0:
                first_msg = final_prompt.messages[0]
                if hasattr(first_msg, "prompt") and hasattr(
                    first_msg.prompt, "template"
                ):
                    system_prompt_text = first_msg.prompt.template

            is_anthropic_model = model.startswith('claude-')
            is_google_model = model.startswith('gemini-')

            if is_anthropic_model:
                agent_summarization_model = f"anthropic:{model}"
            elif is_google_model:
                agent_summarization_model = f"google_genai:{model}"
            else:
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
                graph_inputs, {"callbacks": [handler]}
            )

            try:
                for token in execute_agent_streaming(async_executor, handler):
                    yield token
            except Exception as e:
                logger.error(f"Agent streaming error: {str(e)}", exc_info=True)
                yield f"\nStreaming Error: {str(e)}\n"

        except Exception as e:
            logger.error(
                f"[AGENT_STREAM_EXECUTE] 스트리밍 Agent 실행 중 오류 발생: {str(e)}",
                exc_info=True,
            )
            yield f"죄송합니다. 응답 생성 중 오류가 발생했습니다: {str(e)}"

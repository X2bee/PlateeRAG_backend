import logging
from pydantic import BaseModel
from typing import Dict, Any, Optional
from editor.node_composer import Node
from langchain.schema.output_parser import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from editor.nodes.xgen.agent.functions import prepare_llm_components, rag_context_builder, create_json_output_prompt, create_tool_context_prompt, create_context_prompt
from editor.utils.helper.agent_helper import NonStreamingAgentHandler, NonStreamingAgentHandlerWithToolOutput
from editor.utils.prefix_prompt import prefix_prompt
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
        {"id": "temperature", "name": "Temperature", "type": "FLOAT", "value": 0.7, "required": False, "optional": True, "min": 0.0, "max": 2.0, "step": 0.1},
        {"id": "max_tokens", "name": "Max Tokens", "type": "INT", "value": 8192, "required": False, "optional": True, "min": 1, "max": 65536, "step": 1},
        {"id": "n_messages", "name": "Max Memory", "type": "INT", "value": 3, "min": 1, "max": 10, "step": 1, "optional": True},
        {"id": "base_url", "name": "Base URL", "type": "STR", "value": "https://api.openai.com/v1", "required": False, "optional": True},
        {"id": "strict_citation", "name": "Strict Citation", "type": "BOOL", "value": True, "required": False, "optional": True},
        {"id": "return_intermediate_steps", "name": "Return Intermediate Steps", "type": "BOOL", "value": False, "required": False, "optional": True, "description": "Tool 사용시 해당 과정을 출력할지 여부를 결정합니다."},
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
        return_intermediate_steps: bool = False,
        default_prompt: str = default_prompt,
    ) -> str:
        try:
            default_prompt  = prefix_prompt+default_prompt
            llm, tools_list, chat_history = prepare_llm_components(text, tools, memory, model, temperature, max_tokens, base_url, n_messages, streaming=False)

            additional_rag_context = None
            if rag_context:
                additional_rag_context = rag_context_builder(text, rag_context, strict_citation)
            inputs = {"input": text, "chat_history": chat_history, "additional_rag_context": additional_rag_context}

            if args_schema:
                default_prompt = create_json_output_prompt(args_schema, default_prompt)

            if tools_list and len(tools_list) > 0:
                final_prompt = create_tool_context_prompt(additional_rag_context, default_prompt, n_messages)
                agent = create_tool_calling_agent(llm, tools_list, final_prompt)
                agent_executor = AgentExecutor(
                    agent=agent,
                    tools=tools,
                    verbose=True,
                    handle_parsing_errors=True,
                    max_iterations=3,
                )
                if return_intermediate_steps:
                    handler = NonStreamingAgentHandlerWithToolOutput()
                else:
                    handler = NonStreamingAgentHandler()
                response = agent_executor.invoke(inputs, {"callbacks": [handler]})
                output = response["output"]
                return handler.get_formatted_output(output)

            else:
                final_prompt = create_context_prompt(additional_rag_context, default_prompt, n_messages, strict_citation)
                if args_schema:
                    parser = JsonOutputParser()
                else:
                    parser = StrOutputParser()
                chain = final_prompt | llm | parser
                response = chain.invoke(inputs)
                return response

        except Exception as e:
            logger.error(f"[AGENT_EXECUTE] Chat Agent 실행 중 오류 발생: {str(e)}")
            logger.error(f"[AGENT_EXECUTE] 오류 타입: {type(e)}")
            logger.exception(f"[AGENT_EXECUTE] 상세 스택 트레이스:")
            return f"죄송합니다. 응답 생성 중 오류가 발생했습니다: {str(e)}"

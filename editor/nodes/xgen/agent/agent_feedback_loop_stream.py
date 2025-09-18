import logging
import queue
import threading
from typing import Any, Dict, Generator, Optional, Tuple

from pydantic import BaseModel

from editor.node_composer import Node
from editor.nodes.xgen.agent.functions import (
    create_context_prompt,
    create_json_output_prompt,
    create_tool_context_prompt,
    prepare_llm_components,
    rag_context_builder,
)
from editor.nodes.xgen.tool.print_format import PrintAnyNode
from editor.utils.feedback.create_feedback_graph import create_feedback_graph
from editor.utils.feedback.create_todos import create_todos
from editor.utils.helper.feedback_stream_helper import FeedbackStreamEmitter
from editor.utils.feedback.todo_executor import todo_executor
from editor.utils.prefix_prompt import prefix_prompt

logger = logging.getLogger(__name__)

default_prompt = """You are a helpful AI assistant with feedback loop capabilities. \
You will execute tasks, evaluate results against user requirements, and iterate until satisfactory results are achieved."""

class AgentFeedbackLoopStreamNode(Node):
    categoryId = "xgen"
    functionId = "agents"
    nodeId = "agents/feedback_loop_stream"
    nodeName = "Agent Stream Feedback Loop"
    description = "LangGraph 기반 피드백 루프를 스트리밍 형태로 제공하는 Agent 노드"
    tags = ["agent", "feedback", "loop", "langgraph", "iterative", "evaluation", "stream"]

    inputs = [
        {"id": "text", "name": "Text", "type": "STR", "multi": False, "required": True},
        {"id": "tools", "name": "Tools", "type": "TOOL", "multi": True, "required": False, "value": []},
        {"id": "memory", "name": "Memory", "type": "OBJECT", "multi": False, "required": False},
        {"id": "rag_context", "name": "RAG Context", "type": "DocsContext", "multi": False, "required": False},
        {"id": "args_schema", "name": "ArgsSchema", "type": "OutputSchema", "required": False},
        {"id": "feedback_criteria", "name": "Feedback Criteria", "type": "FeedbackCrit", "multi": False, "required": False, "value": ""},
    ]

    outputs = [
        {"id": "stream", "name": "Stream", "type": "STREAM STR", "stream": True},
    ]

    parameters = [
        {"id": "model", "name": "Model", "type": "STR", "value": "gpt-4.1-mini", "required": True},
        {"id": "temperature", "name": "Temperature", "type": "FLOAT", "value": 0.7, "required": False, "optional": True, "min": 0.0, "max": 2.0, "step": 0.1},
        {"id": "max_tokens", "name": "Max Tokens", "type": "INT", "value": 8192, "required": False, "optional": True, "min": 1, "max": 65536, "step": 1},
        {"id": "n_messages", "name": "Max Memory", "type": "INT", "value": 3, "min": 1, "max": 10, "step": 1, "optional": True},
        {"id": "base_url", "name": "Base URL", "type": "STR", "value": "https://api.openai.com/v1", "required": False, "optional": True},
        {"id": "strict_citation", "name": "Strict Citation", "type": "BOOL", "value": True, "required": False, "optional": True},
        {"id": "return_intermediate_steps", "name": "Return Intermediate Steps", "type": "BOOL", "value": True, "required": False, "optional": True, "description": "중간 단계를 반환할지 여부입니다."},
        {"id": "default_prompt", "name": "Default Prompt", "type": "STR", "value": default_prompt, "required": False, "optional": True, "expandable": True, "description": "기본 프롬프트로 AI가 따르는 System 지침을 의미합니다."},
        {"id": "max_iterations", "name": "Max Iterations", "type": "INT", "value": 5, "min": 1, "max": 20, "step": 1, "optional": True, "description": "최대 반복 횟수"},
        {"id": "feedback_threshold", "name": "Feedback Threshold", "type": "INT", "value": 8, "min": 1, "max": 10, "step": 1, "optional": True, "description": "만족스러운 결과로 간주할 점수 임계값 (1-10)"},
        {"id": "enable_auto_feedback", "name": "Enable Auto Feedback", "type": "BOOL", "value": True, "required": False, "optional": True, "description": "자동 피드백 평가 활성화"},
        {"id": "format_style", "name": "Format Style", "type": "STR", "value": "detailed", "required": False, "optional": True, "options": [
            {"value": "summary", "label": "요약만 표시"},
            {"value": "detailed", "label": "상세 정보 표시"},
            {"value": "compact", "label": "압축된 형태"},
            {"value": "markdown", "label": "마크다운 형식"}
        ]},
        {"id": "show_scores", "name": "Show Scores", "type": "BOOL", "value": True, "required": False, "optional": True, "description": "점수 정보를 표시할지 여부"},
        {"id": "show_timestamps", "name": "Show Timestamps", "type": "BOOL", "value": False, "required": False, "optional": True, "description": "타임스탬프를 표시할지 여부"},
        {"id": "max_iteration_display", "name": "Max Iterations Display", "type": "INT", "value": 5, "min": 1, "max": 20, "step": 1, "optional": True, "description": "표시할 최대 반복 횟수"},
        {"id": "show_todo_details", "name": "Show TODO Details", "type": "BOOL", "value": True, "required": False, "optional": True, "description": "TODO 실행 과정을 상세히 표시할지 여부"},
    ]

    def __init__(self):
        super().__init__()

    def execute(
        self,
        text: str,
        tools: Optional[Any] = None,
        memory: Optional[Any] = None,
        rag_context: Optional[Dict[str, Any]] = None,
        args_schema: Optional[BaseModel] = None,
        feedback_criteria: str = "",
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 8192,
        n_messages: int = 3,
        base_url: str = "https://api.openai.com/v1",
        strict_citation: bool = True,
        return_intermediate_steps: bool = True,
        default_prompt: str = default_prompt,
        max_iterations: int = 5,
        feedback_threshold: int = 8,
        enable_auto_feedback: bool = True,
        format_style: str = "detailed",
        show_scores: bool = True,
        show_timestamps: bool = False,
        max_iteration_display: int = 5,
        show_todo_details: bool = True,
        **kwargs,
    ) -> Generator[str, None, None]:
        stream_queue: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        emitter = FeedbackStreamEmitter(stream_queue)
        printer = PrintAnyNode()

        def chunk_output(text: str) -> Generator[str, None, None]:
            if not text:
                return
            chunk_size = 512
            for i in range(0, len(text), chunk_size):
                yield text[i : i + chunk_size]

        def worker() -> None:
            try:
                enhanced_prompt = prefix_prompt + default_prompt
                llm, tools_list, chat_history = prepare_llm_components(
                    text,
                    tools,
                    memory,
                    model,
                    temperature,
                    max_tokens,
                    base_url,
                    n_messages,
                    streaming=True,
                )

                additional_rag_context = ""
                if rag_context:
                    additional_rag_context = rag_context_builder(text, rag_context, strict_citation)

                if args_schema:
                    enhanced_prompt = create_json_output_prompt(args_schema, enhanced_prompt)

                prompt_template_with_tool = create_tool_context_prompt(
                    additional_rag_context,
                    enhanced_prompt,
                    n_messages,
                )
                prompt_template_without_tool = create_context_prompt(
                    additional_rag_context,
                    enhanced_prompt,
                    n_messages,
                    strict_citation,
                )

                emitter.emit_status("<FEEDBACK_STATUS>TODO 리스트 생성 중...</FEEDBACK_STATUS>\n")
                todos = create_todos(llm, text)
                emitter.emit_todo_list(todos)

                if not feedback_criteria:
                    feedback_criteria_value = """
사용자 요청을 얼마나 잘 충족했는지 평가해주세요:
- 정확성: 요청한 내용과 일치하는가?
- 완성도: 필요한 모든 정보가 포함되었는가?
- 품질: 결과의 품질이 높은가?
- 유용성: 사용자에게 도움이 되는가?
"""
                else:
                    feedback_criteria_value = feedback_criteria

                workflow = create_feedback_graph(
                    llm,
                    tools_list,
                    chat_history,
                    prompt_template_with_tool,
                    prompt_template_without_tool,
                    additional_rag_context,
                    feedback_criteria_value,
                    return_intermediate_steps,
                    feedback_threshold,
                    enable_auto_feedback,
                    stream_emitter=emitter,
                )

                all_results, todo_execution_log = todo_executor(
                    todos,
                    text,
                    max_iterations,
                    workflow,
                    return_intermediate_steps,
                    stream_emitter=emitter,
                )

                all_scores = [result["score"] for result in all_results if "score" in result]
                completed_todos = [todo for todo in todo_execution_log if todo.get("requirements_met", False)]

                if todo_execution_log:
                    final_summary = todo_execution_log[-1].get("result", "No TODOs executed.")
                else:
                    final_summary = "No TODOs executed."

                result_dict: Dict[str, Any] = {
                    "result": final_summary,
                    "todos_generated": len(todos),
                    "todos_completed": len(completed_todos),
                    "total_iterations": len(all_results),
                    "final_score": all_scores[-1] if all_scores else 0,
                    "average_score": sum(all_scores) / len(all_scores) if all_scores else 0,
                    "completion_rate": len(completed_todos) / len(todos) if todos else 0,
                }

                if return_intermediate_steps:
                    result_dict.update(
                        {
                            "todos_list": todos,
                            "todo_execution_log": todo_execution_log,
                            "iteration_log": all_results,
                            "feedback_scores": all_scores,
                        }
                    )

                emitter.emit_final_dict(result_dict)

            except Exception as exc:  # pragma: no cover - runtime safety
                logger.error(
                    f"[FEEDBACK_LOOP_STREAM] Feedback Loop Agent 스트리밍 실행 중 오류 발생: {str(exc)}",
                    exc_info=True,
                )
                emitter.emit_error(
                    f"죄송합니다. TODO 기반 피드백 루프 스트리밍 중 오류가 발생했습니다: {str(exc)}"
                )
            finally:
                emitter.close()

        worker_thread = threading.Thread(target=worker, daemon=True)
        worker_thread.start()

        loop_open = False

        def split_feedback_output(text: str) -> Tuple[str, str]:
            if "<FEEDBACK_LOOP>" in text and "</FEEDBACK_LOOP>" in text:
                _, after_open = text.split("<FEEDBACK_LOOP>", 1)
                inside, after_close = after_open.split("</FEEDBACK_LOOP>", 1)
                return inside, after_close
            return text, ""

        while True:
            msg_type, payload = stream_queue.get()
            if msg_type == "token":
                if not loop_open:
                    yield "<FEEDBACK_LOOP>"
                    loop_open = True
                yield payload
            elif msg_type == "final":
                formatted_output = printer.execute(
                    payload,
                    format_style=format_style,
                    show_scores=show_scores,
                    show_timestamps=show_timestamps,
                    max_iteration_display=max_iteration_display,
                    show_todo_details=show_todo_details,
                )
                loop_body, remainder = split_feedback_output(formatted_output)

                if loop_body:
                    if not loop_open:
                        yield "<FEEDBACK_LOOP>"
                        loop_open = True
                    for chunk in chunk_output(loop_body):
                        yield chunk

                if loop_open:
                    yield "</FEEDBACK_LOOP>"
                    loop_open = False

                if remainder:
                    yield remainder
            elif msg_type == "error":
                if payload:
                    if not loop_open:
                        yield "<FEEDBACK_LOOP>"
                        loop_open = True
                    yield f"<FEEDBACK_STATUS>{payload}</FEEDBACK_STATUS>\n"
                if loop_open:
                    yield "</FEEDBACK_LOOP>"
                    loop_open = False
            elif msg_type == "done":
                if loop_open:
                    yield "</FEEDBACK_LOOP>"
                    loop_open = False
                break

        if worker_thread.is_alive():
            worker_thread.join(timeout=2.0)

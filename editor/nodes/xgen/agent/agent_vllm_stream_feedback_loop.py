import logging
import queue
import threading
from typing import Any, Dict, Generator, Optional, Tuple
from urllib.request import Request

from controller.helper.singletonHelper import get_config_composer
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
from editor.utils.helper.agent_helper import use_guarder_for_text_moderation
from editor.utils.feedback.todo_executor import todo_executor, build_final_summary
from editor.utils.prefix_prompt import prefix_prompt
from editor.type_model.feedback_state import FeedbackState

logger = logging.getLogger(__name__)

default_prompt = """You are a helpful AI assistant with feedback loop capabilities. \
You will execute tasks, evaluate results against user requirements, and iterate until satisfactory results are achieved."""

class AgentVLLMFeedbackLoopStreamNode(Node):
    categoryId = "xgen"
    functionId = "agents"
    nodeId = "agents/vllm_feedback_loop_stream"
    nodeName = "Agent VLLM Stream Feedback Loop"
    description = "LangGraph 기반 피드백 루프를 스트리밍 형태로 제공하는 Agent 노드"
    tags = ["agent", "feedback", "loop", "langgraph", "iterative", "evaluation", "stream"]

    inputs = [
        {"id": "text", "name": "Text", "type": "STR", "multi": False, "required": True},
        {"id": "tools", "name": "Tools", "type": "TOOL", "multi": True, "required": False, "value": []},
        {"id": "memory", "name": "Memory", "type": "OBJECT", "multi": False, "required": False},
        {"id": "rag_context", "name": "RAG Context", "type": "DocsContext", "multi": False, "required": False},
        {"id": "args_schema", "name": "ArgsSchema", "type": "OutputSchema", "required": False},
        {"id": "plan", "name": "Plan", "type": "PLAN", "required": False},
        {"id": "feedback_criteria", "name": "Feedback Criteria", "type": "FeedbackCrit", "multi": False, "required": False, "value": ""},
    ]

    outputs = [
        {"id": "stream", "name": "Stream", "type": "STREAM STR", "stream": True},
    ]

    parameters = [
        {"id": "model", "name": "Model", "type": "STR", "value": "", "is_api": True, "api_name": "api_vllm_model_name", "required": True},
        {"id": "temperature", "name": "Temperature", "type": "FLOAT", "value": 0.7, "required": False, "optional": True, "min": 0.0, "max": 2.0, "step": 0.1},
        {"id": "max_tokens", "name": "Max Tokens", "type": "INT", "value": 8192, "required": False, "optional": True, "min": 1, "max": 65536, "step": 1},
        {"id": "base_url", "name": "Base URL", "type": "STR", "value": "", "is_api": True, "api_name": "api_vllm_api_base_url", "required": True},
        {"id": "strict_citation", "name": "Strict Citation", "type": "BOOL", "value": True, "required": False, "optional": True},
        {"id": "return_intermediate_steps", "name": "Return Intermediate Steps", "type": "BOOL", "value": True, "required": False, "optional": True, "description": "중간 단계를 반환할지 여부입니다."},
        {"id": "default_prompt", "name": "Default Prompt", "type": "STR", "value": default_prompt, "required": False, "optional": True, "expandable": True, "description": "기본 프롬프트로 AI가 따르는 System 지침을 의미합니다."},
        {"id": "max_iterations", "name": "Max Iterations", "type": "INT", "value": 5, "min": 1, "max": 20, "step": 1, "optional": True, "description": "최대 반복 횟수"},
        {"id": "feedback_threshold", "name": "Feedback Threshold", "type": "INT", "value": 8, "min": 1, "max": 10, "step": 1, "optional": True, "description": "만족스러운 결과로 간주할 점수 임계값 (1-10)"},
        {"id": "enable_auto_feedback", "name": "Enable Auto Feedback", "type": "BOOL", "value": True, "required": False, "optional": True, "description": "자동 피드백 평가 활성화"},
        {"id": "enable_formatted_output", "name": "Enable Formatted Output", "type": "BOOL", "value": True, "required": False, "optional": True, "description": "형식화된 출력 활성화"},
        {"id": "format_style", "name": "Format Style", "type": "STR", "value": "detailed", "required": False, "optional": True, "options": [
            {"value": "summary", "label": "요약만 표시"},
            {"value": "detailed", "label": "상세 정보 표시"},
            {"value": "compact", "label": "압축된 형태"},
            {"value": "markdown", "label": "마크다운 형식"}
        ], "dependency": "enable_formatted_output"},
        {"id": "show_scores", "name": "Show Scores", "type": "BOOL", "value": True, "required": False, "optional": True, "description": "점수 정보를 표시할지 여부", "dependency": "enable_formatted_output"},
        {"id": "show_timestamps", "name": "Show Timestamps", "type": "BOOL", "value": False, "required": False, "optional": True, "description": "타임스탬프를 표시할지 여부", "dependency": "enable_formatted_output"},
        {"id": "max_iteration_display", "name": "Max Iterations Display", "type": "INT", "value": 5, "min": 1, "max": 20, "step": 1, "optional": True, "description": "표시할 최대 반복 횟수", "dependency": "enable_formatted_output"},
        {"id": "show_todo_details", "name": "Show TODO Details", "type": "BOOL", "value": True, "required": False, "optional": True, "description": "TODO 실행 과정을 상세히 표시할지 여부", "dependency": "enable_formatted_output"},
        {"id": "use_guarder", "name": "Use Guarder Service", "type": "BOOL", "value": False, "required": False, "optional": True, "description": "Guarder 서비스를 사용할지 여부입니다."},
    ]

    def __init__(self):
        super().__init__()

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
        feedback_criteria: str = "",
        model: str = "x2bee/Polar-14B",
        temperature: float = 0.7,
        max_tokens: int = 8192,
        base_url: str = "",
        strict_citation: bool = True,
        return_intermediate_steps: bool = True,
        default_prompt: str = default_prompt,
        max_iterations: int = 5,
        feedback_threshold: int = 8,
        enable_auto_feedback: bool = True,
        enable_formatted_output: bool = False,
        format_style: str = "detailed",
        show_scores: bool = True,
        show_timestamps: bool = False,
        max_iteration_display: int = 5,
        show_todo_details: bool = True,
        use_guarder: bool = False,
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
                if use_guarder:
                    is_safe, moderation_message = use_guarder_for_text_moderation(text)
                    if not is_safe:
                        emitter.emit_error(moderation_message)
                        return

                enhanced_prompt = prefix_prompt + default_prompt
                llm, tools_list, chat_history = prepare_llm_components(
                    text,
                    tools,
                    memory,
                    model,
                    temperature,
                    max_tokens,
                    base_url,
                    streaming=True,
                    plan=plan,
                )

                additional_rag_context = ""
                if rag_context:
                    additional_rag_context = rag_context_builder(text, rag_context, strict_citation)

                if args_schema:
                    enhanced_prompt = create_json_output_prompt(args_schema, enhanced_prompt)

                prompt_template_with_tool = create_tool_context_prompt(
                    additional_rag_context,
                    enhanced_prompt,
                    plan=plan
                )
                prompt_template_without_tool = create_context_prompt(
                    additional_rag_context,
                    enhanced_prompt,
                    strict_citation,
                    plan=plan
                )

                emitter.emit_status("<FEEDBACK_STATUS>실행 전략 평가 중...</FEEDBACK_STATUS>\n")
                todo_plan = create_todos(llm, text, tools_list)
                todos = todo_plan.get("todos", [])
                execution_mode = todo_plan.get("mode", "todo")
                todo_strategy_reason = todo_plan.get("reason", "")
                direct_tool_usage = todo_plan.get("tool_usage", "simple")

                if execution_mode == "todo":
                    emitter.emit_todo_list(todos)
                else:
                    emitter.emit_status("<FEEDBACK_STATUS>단일 실행 모드로 전환합니다.</FEEDBACK_STATUS>\n")

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

                if execution_mode == "direct":
                    direct_requires_tools = direct_tool_usage == "complex" and bool(tools_list)
                    direct_state = FeedbackState(
                        messages=[{"role": "user", "content": text}],
                        user_input=text,
                        tool_results=[],
                        feedback_score=0,
                        iteration_count=1,
                        final_result=None,
                        requirements_met=False,
                        max_iterations=1,
                        todo_requires_tools=direct_requires_tools,
                        current_todo_id=None,
                        current_todo_title="Direct Execution",
                        current_todo_index=1,
                        total_todos=0,
                        execution_mode="direct",
                        skip_feedback_eval=True,
                        remediation_notes=[],
                        seen_results=[],
                        last_result_signature=None,
                        last_result_duplicate=False,
                        stagnation_count=0,
                    )

                    import time

                    thread_id = f"direct_{hash(text)}_{int(time.time())}"
                    final_state = workflow.invoke(
                        direct_state,
                        config={"configurable": {"thread_id": thread_id}},
                    )

                    iteration_log = []
                    feedback_scores = []
                    for result in final_state.get("tool_results", []):
                        iteration_log.append(
                            {
                                "iteration": result.get("iteration"),
                                "result": result.get("result"),
                                "score": result.get("feedback_score", 0),
                                "evaluation": result.get("evaluation", {}),
                                "timestamp": result.get("timestamp"),
                            }
                        )
                        feedback_scores.append(result.get("feedback_score", 0))

                    final_summary = final_state.get("final_result") or ""
                    raw_todos = todo_plan.get("raw_todos", [])

                    result_dict = {
                        "result": final_summary,
                        "todos_generated": len(raw_todos),
                        "todos_completed": 1 if final_state.get("requirements_met") else 0,
                        "total_iterations": len(iteration_log),
                        "final_score": feedback_scores[-1] if feedback_scores else 0,
                        "average_score": sum(feedback_scores) / len(feedback_scores) if feedback_scores else 0,
                        "completion_rate": 1 if final_state.get("requirements_met") else 0,
                        "workflow_mode": "direct",
                        "workflow_reason": todo_strategy_reason,
                    }

                    if return_intermediate_steps:
                        result_dict.update(
                            {
                                "todos_list": raw_todos,
                                "todo_execution_log": [
                                    {
                                        "mode": "direct",
                                        "request": text,
                                        "result": final_summary,
                                        "iterations": len(iteration_log),
                                        "final_score": feedback_scores[-1] if feedback_scores else 0,
                                        "average_score": result_dict["average_score"],
                                        "requirements_met": final_state.get("requirements_met", False),
                                        "reason": todo_strategy_reason,
                                    }
                                ],
                                "iteration_log": iteration_log,
                                "feedback_scores": feedback_scores,
                                "todo_planner_raw": todo_plan.get("llm_raw", ""),
                            }
                        )

                    emitter.emit_final_dict(result_dict)
                    return

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

                final_summary = build_final_summary(todo_execution_log)

                result_dict: Dict[str, Any] = {
                    "result": final_summary,
                    "todos_generated": len(todos),
                    "todos_completed": len(completed_todos),
                    "total_iterations": len(all_results),
                    "final_score": all_scores[-1] if all_scores else 0,
                    "average_score": sum(all_scores) / len(all_scores) if all_scores else 0,
                    "completion_rate": len(completed_todos) / len(todos) if todos else 0,
                    "workflow_mode": "todo",
                    "workflow_reason": todo_strategy_reason,
                }

                if return_intermediate_steps:
                    result_dict.update(
                        {
                            "todos_list": todos,
                            "todo_execution_log": todo_execution_log,
                            "iteration_log": all_results,
                            "feedback_scores": all_scores,
                            "todo_planner_raw": todo_plan.get("llm_raw", ""),
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
            if "<FEEDBACK_RESULT>" in text and "</FEEDBACK_RESULT>" in text:
                _, after_open = text.split("<FEEDBACK_RESULT>", 1)
                inside, after_close = after_open.split("</FEEDBACK_RESULT>", 1)
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
                if loop_open:
                    yield "</FEEDBACK_LOOP>"
                    loop_open = False

                formatted_output = printer.execute(
                    payload,
                    enable_formatted_output=enable_formatted_output,
                    format_style=format_style,
                    show_scores=show_scores,
                    show_timestamps=show_timestamps,
                    max_iteration_display=max_iteration_display,
                    show_todo_details=show_todo_details,
                )
                loop_body, remainder = split_feedback_output(formatted_output)

                if loop_body:
                    for chunk in chunk_output(loop_body):
                        yield chunk
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

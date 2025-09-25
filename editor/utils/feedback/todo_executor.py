import logging
import textwrap
from typing import List

from editor.type_model.feedback_state import FeedbackState

logger = logging.getLogger(__name__)

def _trim_text(value: str, limit: int = 600) -> str:
    """Limit long context strings to keep prompts efficient."""
    if not value:
        return ""
    stripped = value.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[:limit].rstrip() + "..."


def _format_previous_results(previous_results: List[dict]) -> str:
    if not previous_results:
        return ""

    formatted_entries = []
    for idx, prev in enumerate(previous_results, start=1):
        title = prev.get('title') or f"TODO {idx}"
        result_text = _trim_text(prev.get('result') or "결과 없음")
        formatted_entries.append(f"{idx}. {title}: {result_text}")

    if not formatted_entries:
        return ""

    return "이전 TODO 결과 요약:\n" + "\n".join(formatted_entries)


def todo_executor(todos, text, max_iterations, workflow, return_intermediate_steps=False, stream_emitter=None):
    # 2단계: 각 TODO를 순차적으로 실행
    all_results = []
    todo_execution_log = []
    previous_todo_results = []  # 이전 TODO 결과를 누적하기 위한 리스트

    for i, todo in enumerate(todos):
        todo_requires_tools = todo.get('tool_required', 'complex') == 'complex'
        logger.info(f"Executing TODO {i+1}/{len(todos)}: {todo.get('title', 'Untitled')} [{'with tools' if todo_requires_tools else 'without tools'}]")

        if stream_emitter:
            stream_emitter.emit_todo_start(i + 1, len(todos), todo)

        # TODO를 위한 명확한 지침과 컨텍스트 구성
        title = todo.get('title', '').strip() or "Untitled"
        description = todo.get('description', '').strip()
        previous_context = _format_previous_results(previous_todo_results)

        directive_body = textwrap.dedent(
            f"""
            당신은 복합 요청을 단계별 TODO로 해결하는 보조자입니다.
            현재 진행 중인 단계는 전체 {len(todos)}개 중 {i + 1}번째입니다.

            현재 TODO 목표:
            - 제목: {title}
            - 설명: {description or '설명 없음'}

            필수 지침:
            1. 오직 현재 TODO 목표만 달성하세요. 다른 TODO의 요구사항을 미리 수행하지 마세요.
            2. 이전 단계 결과는 참고만 하고 중복 작업을 피하세요.
            3. 결과는 다음 단계에서 활용하기 쉽도록 명확하고 검증 가능한 형태로 정리하세요.
            4. 필요하다면 중간 계산 과정을 설명하되, 최종 답변은 현재 TODO에 필요한 핵심 결과만 포함하세요.
            5. 쿼리나 계산을 수행했다면 실행 결과(반환된 데이터 요약 또는 "결과 없음"과 같은 명시적 설명)를 반드시 포함하세요.
            """
        ).strip()

        context_section = ""
        if previous_context:
            context_section = "\n\n" + previous_context

        todo_text = (
            directive_body
            + "\n\n원본 사용자 요청:\n"
            + _trim_text(text)
            + context_section
            + "\n\n출력 형식: 현재 TODO 목표를 충족하는 데 필요한 결과만 명확히 제시하고, 실행 결과 데이터 또는 '결과 없음'과 같은 확인 문구를 함께 제공하세요."
        )

        initial_state = FeedbackState(
            messages=[{"role": "user", "content": todo_text}],
            user_input=todo_text,
            tool_results=[],
            feedback_score=0,
            iteration_count=1,
            final_result=None,
            requirements_met=False,
            max_iterations=max_iterations,  # 개별 TODO의 sub-task iterations
            todo_requires_tools=todo_requires_tools,  # 도구 필요성 전달
            current_todo_id=todo.get('id', i + 1),
            current_todo_title=todo.get('title', 'Untitled'),
            current_todo_description=todo.get('description', ''),
            current_todo_index=i + 1,
            total_todos=len(todos),
            execution_mode="todo",
            skip_feedback_eval=False,
            remediation_notes=[],
            seen_results=[],
            last_result_signature=None,
            last_result_duplicate=False,
            stagnation_count=0,
            result_frequencies={},
            duplicate_run_length=0,
            original_user_request=text,
            previous_results_context=previous_context,
            todo_directive=todo_text,
        )

        # TODO 실행
        import time
        thread_id = f"todo_{todo.get('id', i)}_{hash(todo_text)}_{int(time.time())}"
        todo_final_state = workflow.invoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}}
        )

        # TODO 결과 저장
        todo_results = []
        todo_scores = []

        for result in todo_final_state["tool_results"]:
            todo_results.append({
                "iteration": result["iteration"],
                "result": result.get("clean_result", result.get("result", "")),
                "score": result.get("feedback_score", 0),
                "evaluation": result.get("evaluation", {}),
                "timestamp": result["timestamp"]
            })
            todo_scores.append(result.get("feedback_score", 0))

        todo_log = {
            "todo_id": todo.get("id", i+1),
            "todo_title": todo.get("title", "Untitled"),
            "todo_description": todo.get("description", ""),
            "todo_priority": todo.get("priority", "medium"),
            "result": todo_final_state.get("final_result_clean") or todo_final_state.get("final_result") or "No result generated",
            "iterations": len(todo_results),
            "final_score": todo_scores[-1] if todo_scores else 0,
            "average_score": sum(todo_scores) / len(todo_scores) if todo_scores else 0,
            "requirements_met": todo_final_state.get("requirements_met", False),
            "status": "completed" if todo_final_state.get("requirements_met", False) else "incomplete",
        }

        if todo_results:
            best_iteration_entry = max(todo_results, key=lambda r: r.get("score", 0))
            todo_log.update(
                {
                    "best_iteration": best_iteration_entry.get("iteration"),
                    "best_score": best_iteration_entry.get("score", 0),
                }
            )

        if return_intermediate_steps:
            todo_log.update({
                "iteration_log": todo_results,
                "feedback_scores": todo_scores
            })

        todo_execution_log.append(todo_log)
        all_results.extend(todo_results)

        # 현재 TODO 결과를 다음 TODO를 위해 저장
        previous_todo_results.append({
            'title': todo.get('title', 'Untitled'),
            'description': todo.get('description', ''),
            'result': todo_final_state["final_result"] or "No result generated",
            'score': todo_scores[-1] if todo_scores else 0,
            'requirements_met': todo_final_state.get("requirements_met", False)
        })

        if stream_emitter:
            stream_emitter.emit_todo_summary(todo_log)

    return all_results, todo_execution_log


def build_final_summary(todo_execution_log):
    """최종 결과를 간결하게 반환"""
    if not todo_execution_log:
        return "No TODOs executed."

    # 가장 최근 TODO의 결과가 있으면 사용
    for entry in reversed(todo_execution_log):
        result_text = (entry.get("result") or "").strip()
        if result_text:
            return result_text

    return "No result generated."

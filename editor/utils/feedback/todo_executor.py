import logging
from editor.type_model.feedback_state import FeedbackState

logger = logging.getLogger(__name__)

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

        # 이전 TODO 결과들을 컨텍스트로 구성
        previous_context = ""
        if previous_todo_results:
            previous_context = "\n\n이전 TODO 결과들:\n"
            for idx, prev_result in enumerate(previous_todo_results):
                previous_context += f"- TODO {idx+1} ({prev_result['title']}): {prev_result['result']}\n"

        # TODO 실행을 위한 상태 설정 (이전 결과 포함)
        todo_text = f"TODO: {todo.get('title', '')}\n설명: {todo.get('description', '')}\n\n원본 요청 컨텍스트: {text}{previous_context}"

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
            current_todo_index=i + 1,
            total_todos=len(todos),
            execution_mode="todo",
            skip_feedback_eval=False,
            remediation_notes=[],
            seen_results=[],
            last_result_signature=None,
            last_result_duplicate=False,
            stagnation_count=0,
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

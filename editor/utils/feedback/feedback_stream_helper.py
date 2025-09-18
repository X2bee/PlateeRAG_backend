from typing import Any, Dict, List, Optional

class FeedbackStreamEmitter:
    """단일 피드백 루프 실행에서 스트리밍 출력을 관리하는 간단한 큐 기반 헬퍼"""

    def __init__(self, queue):
        self._queue = queue

    # 기본 큐 인터페이스 -------------------------------------------------
    def _put(self, msg_type: str, payload: Any) -> None:
        self._queue.put((msg_type, payload))

    def emit_token(self, text: str) -> None:
        if text:
            self._put("token", text)

    def emit_status(self, text: str) -> None:
        if text:
            self.emit_token(text)

    def emit_error(self, message: str) -> None:
        if message:
            self._put("error", message)

    def emit_final_dict(self, payload: Dict[str, Any]) -> None:
        self._put("final", payload)

    def close(self) -> None:
        self._put("done", None)

    # 스트리밍 전용 헬퍼 -------------------------------------------------
    def emit_todo_list(self, todos: List[Dict[str, Any]]) -> None:
        if not todos:
            return
        lines = []
        for todo in todos:
            title = todo.get("title", "Untitled")
            priority = todo.get("priority", "medium")
            requires_tool = todo.get("tool_required", "complex")
            lines.append(f"- {title} (priority: {priority}, tool: {requires_tool})")
        summary = "\n".join(lines)
        self.emit_status(f"<FEEDBACK_STATUS>TODOs 준비 완료:\n{summary}</FEEDBACK_STATUS>\n")

    def emit_todo_start(self, index: int, total: int, todo: Dict[str, Any]) -> None:
        title = todo.get("title", "Untitled")
        description = todo.get("description", "")
        self.emit_status(
            f"<FEEDBACK_STATUS>TODO {index}/{total} 시작: {title}\n{description}</FEEDBACK_STATUS>\n"
        )

    def emit_iteration_start(
        self,
        todo_index: Optional[int],
        total_todos: Optional[int],
        iteration: int,
        todo_title: str,
    ) -> None:
        parts = []
        if todo_index and total_todos:
            parts.append(f"TODO {todo_index}/{total_todos}")
        if iteration:
            parts.append(f"iteration {iteration}")
        header = " ".join(parts) if parts else f"iteration {iteration}"
        title_segment = f" ({todo_title})" if todo_title else ""
        self.emit_status(f"<FEEDBACK_STATUS>{header}{title_segment} 진행 중...</FEEDBACK_STATUS>\n")

    def emit_llm_chunk(self, todo_index: Optional[int], iteration: int, chunk: str) -> None:
        self.emit_token(chunk)

    def emit_iteration_error(self, todo_index: Optional[int], iteration: Optional[int], error: Exception) -> None:
        prefix = f"TODO {todo_index} iteration {iteration}" if todo_index else f"iteration {iteration}"
        self.emit_status(f"<FEEDBACK_STATUS>{prefix} 오류: {error}</FEEDBACK_STATUS>\n")

    def emit_iteration_complete(self, todo_index: Optional[int], iteration: int, result: str) -> None:
        snippet = (result or "").strip().split("\n", 1)[0]
        if len(snippet) > 160:
            snippet = snippet[:160] + "..."
        label = f"TODO {todo_index} iteration {iteration}" if todo_index else f"iteration {iteration}"
        self.emit_status(f"<FEEDBACK_STATUS>{label} 완료: {snippet}</FEEDBACK_STATUS>\n")

    def emit_feedback_score(self, todo_index: Optional[int], iteration: Optional[int], evaluation: Dict[str, Any]) -> None:
        score = evaluation.get("score", 0)
        reasoning = evaluation.get("reasoning", "")
        label = f"TODO {todo_index} iteration {iteration}" if todo_index else f"iteration {iteration}"
        self.emit_status(
            f"<FEEDBACK_STATUS>{label} 평가 점수: {score}/10 - {reasoning}</FEEDBACK_STATUS>\n"
        )

    def emit_todo_finalization(
        self,
        todo_index: Optional[int],
        todo_title: str,
        final_result: str,
        requirements_met: bool,
    ) -> None:
        status_icon = "✅" if requirements_met else "⚠️"
        snippet = (final_result or "").strip().split("\n", 1)[0]
        if len(snippet) > 180:
            snippet = snippet[:180] + "..."
        label = f"TODO {todo_index}" if todo_index else "TODO"
        title_segment = f" {todo_title}" if todo_title else ""
        self.emit_status(
            f"<FEEDBACK_STATUS>{status_icon} {label}{title_segment} 최종 결과: {snippet}</FEEDBACK_STATUS>\n"
        )

    def emit_todo_summary(self, todo_log: Dict[str, Any]) -> None:
        if not todo_log:
            return
        title = todo_log.get("todo_title", "Untitled")
        score = todo_log.get("final_score", 0)
        iterations = todo_log.get("iterations", 0)
        met = todo_log.get("requirements_met", False)
        status_icon = "✅" if met else "⏳"
        self.emit_status(
            f"<FEEDBACK_STATUS>{status_icon} '{title}' 완료 - 반복 {iterations}회, 최종 점수 {score}/10</FEEDBACK_STATUS>\n"
        )

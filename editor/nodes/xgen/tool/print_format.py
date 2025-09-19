import logging
from datetime import datetime
from typing import Any, Dict, List
from editor.node_composer import Node

logger = logging.getLogger(__name__)

class PrintAnyNode(Node):
    categoryId = "xgen"
    functionId = "endnode"
    nodeId = "tools/print_format"
    nodeName = "Print Format"
    description = "임의의 타입의 데이터를 입력받아 그대로 반환하는 출력 노드입니다. 워크플로우의 최종 결과를 확인하는데 사용됩니다."
    tags = ["output", "print", "display", "debug", "end_node", "utility", "any_type"]

    inputs = [
        {"id": "input_print", "name": "Print", "type": "FeedbackDICT", "multi": False, "required": True},
    ]

    parameters = [
        {"id": "enable_formatted_output", "name": "Enable Formatted Output", "type": "BOOL", "value": False, "required": False, "optional": True, "description": "형식화된 출력 활성화"},
        {
            "id": "format_style", "name": "Format Style", "type": "STR", "value": "detailed", "required": False,
            "options": [
                {"value": "summary", "label": "요약만 표시"},
                {"value": "detailed", "label": "상세 정보 표시"},
                {"value": "compact", "label": "압축된 형태"},
                {"value": "markdown", "label": "마크다운 형식"}
            ],
            "optional": True
        },
        {"id": "show_scores", "name": "Show Scores", "type": "BOOL", "value": True, "required": False, "optional": True, "description": "점수 정보를 표시할지 여부"},
        {"id": "show_timestamps", "name": "Show Timestamps", "type": "BOOL", "value": False, "required": False, "optional": True, "description": "타임스탬프를 표시할지 여부"},
        {"id": "max_iteration_display", "name": "Max Iterations Display", "type": "INT", "value": 5, "min": 1, "max": 20, "step": 1, "optional": True, "description": "표시할 최대 반복 횟수"},
        {"id": "show_todo_details", "name": "Show TODO Details", "type": "BOOL", "value": True, "required": False, "optional": True, "description": "TODO 실행 과정을 상세히 표시할지 여부"},
        # {"id": "truncate_results", "name": "Truncate Results", "type": "INT", "value": 150, "min": 50, "max": 500, "step": 25, "optional": True, "description": "각 반복 결과를 자를 최대 문자 수"},
    ]

    def execute(
        self,
        input_print: Dict[str, Any],
        enable_formatted_output: bool = False,
        format_style: str = "detailed",
        show_scores: bool = True,
        show_timestamps: bool = False,
        max_iteration_display: int = 5,
        show_todo_details: bool = True,
        truncate_results: int = 150,
        **kwargs
    ) -> str:
        try:
            # 피드백 결과에서 데이터 추출
            result = input_print.get("result", "결과 없음")
            iteration_log = input_print.get("iteration_log", [])
            feedback_scores = input_print.get("feedback_scores", [])
            total_iterations = input_print.get("total_iterations", len(iteration_log))
            final_score = input_print.get("final_score", 0)
            average_score = input_print.get("average_score", 0)
            has_error = input_print.get("error", False)
            
            if format_style == "summary":
                formatted_output = self._format_summary(result, iteration_log, feedback_scores, total_iterations, final_score, average_score, has_error, show_todo_details, input_print)
            elif format_style == "compact":
                formatted_output = self._format_compact(result, iteration_log, feedback_scores, show_scores, total_iterations, final_score, has_error, show_todo_details, input_print)
            elif format_style == "markdown":
                formatted_output = self._format_markdown(result, iteration_log, feedback_scores, show_scores, show_timestamps, max_iteration_display, truncate_results, total_iterations, final_score, average_score, has_error, show_todo_details, input_print)
            else:  # detailed
                formatted_output = self._format_detailed(result, iteration_log, feedback_scores, show_scores, show_timestamps, max_iteration_display, truncate_results, total_iterations, final_score, average_score, has_error, show_todo_details, input_print)
            
            if not enable_formatted_output:
                return str(result)
            
            return f"<FEEDBACK_RESULT><FEEDBACK_REPORT>{formatted_output}</FEEDBACK_REPORT></FEEDBACK_RESULT>{self._format_todo_details(input_print)}{str(result)}"
            
        except Exception as e:
            logger.error(f"[FEEDBACK_FORMATTER] 포매팅 중 오류 발생: {str(e)}")
            return f"포매팅 오류: {str(e)}\n\n원본 결과: {str(input_print)}"

    def _format_summary(self, result: str, iteration_log: List[Dict], feedback_scores: List[int],
                       total_iterations: int, final_score: int, average_score: float, has_error: bool,
                       show_todo_details: bool, input_print: Dict[str, Any]) -> str:
        """요약 형태로 포매팅"""
        error_indicator = "⚠️ 오류 발생 " if has_error else ""

        feedback_output = f"""{error_indicator}=== 피드백 루프 실행 요약 ===
📊 실행 통계:
- 총 반복 횟수: {total_iterations}회
- 최종 점수: {final_score}/10
- 평균 점수: {average_score:.1f}/10"""

        return f"""{feedback_output}"""

    def _format_compact(self, result: str, iteration_log: List[Dict], feedback_scores: List[int],
                       show_scores: bool, total_iterations: int, final_score: int, has_error: bool,
                       show_todo_details: bool, input_print: Dict[str, Any]) -> str:
        """압축된 형태로 포매팅"""
        score_info = f" (점수: {' → '.join(map(str, feedback_scores))})" if show_scores and feedback_scores else ""
        error_indicator = "⚠️ " if has_error else "🔄 "

        feedback_output = f"{error_indicator}피드백 루프 완료: {total_iterations}회 반복{score_info}"

        # TODO 세부 정보 추가
        todo_output = ""
        if show_todo_details:
            todo_output = self._format_todo_details(input_print)

        return f"""{feedback_output}
{todo_output}
{str(result)}"""

    def _format_markdown(self, result: str, iteration_log: List[Dict], feedback_scores: List[int],
                        show_scores: bool, show_timestamps: bool, max_iterations: int, truncate_len: int,
                        total_iterations: int, final_score: int, average_score: float, has_error: bool,
                        show_todo_details: bool, input_print: Dict[str, Any]) -> str:
        """마크다운 형태로 포매팅"""
        error_indicator = "⚠️ " if has_error else "🔄 "
        markdown = f"# {error_indicator}피드백 루프 실행 결과\n\n"
        
        markdown += "## 📊 실행 통계\n\n"
        markdown += f"- **총 반복 횟수**: {total_iterations}회\n"
        if show_scores and feedback_scores:
            markdown += f"- **최종 점수**: {final_score}/10\n"
            markdown += f"- **평균 점수**: {average_score:.1f}/10\n"
            markdown += f"- **점수 변화**: {' → '.join(map(str, feedback_scores))}\n"
        markdown += "\n"
        
        # 반복 과정
        if len(iteration_log) > 0:
            markdown += "## 🔄 반복 과정\n\n"
            display_iterations = min(max_iterations, len(iteration_log))
            
            for i, log_entry in enumerate(iteration_log[:display_iterations]):
                iteration_num = log_entry.get('iteration', i + 1)
                iteration_result = str(log_entry.get('result', '결과 없음'))
                score = log_entry.get('score', 0)
                
                # 결과 자르기
                # if len(iteration_result) > truncate_len:
                #     iteration_result = iteration_result[:truncate_len] + "..."
                
                markdown += f"### 반복 {iteration_num}"
                if show_scores:
                    markdown += f" (점수: {score}/10)"
                markdown += "\n\n"
                
                if show_timestamps and 'timestamp' in log_entry:
                    timestamp = datetime.fromtimestamp(log_entry['timestamp']).strftime('%H:%M:%S')
                    markdown += f"**시간**: {timestamp}\n\n"
                
                markdown += f"```\n{iteration_result}\n```\n\n"

        return markdown

    def _format_detailed(self, result: str, iteration_log: List[Dict], feedback_scores: List[int],
                        show_scores: bool, show_timestamps: bool, max_iterations: int, truncate_len: int,
                        total_iterations: int, final_score: int, average_score: float, has_error: bool,
                        show_todo_details: bool, input_print: Dict[str, Any]) -> str:
        """상세한 형태로 포매팅"""
        error_indicator = "⚠️ 오류 발생 - " if has_error else ""
        output = ""
        output += f"{error_indicator}🔄 피드백 루프 실행 결과\n"
        output += "=" * 60 + "\n\n"
        
        # 실행 통계
        output += f"📊 실행 통계:\n"
        output += f"   - 총 반복 횟수: {total_iterations}회\n"
        
        if show_scores and feedback_scores:
            min_score = min(feedback_scores)
            max_score = max(feedback_scores)
            
            output += f"   - 최종 점수: {final_score}/10\n"
            output += f"   - 평균 점수: {average_score:.1f}/10\n"
            output += f"   - 최고 점수: {max_score}/10\n"
            output += f"   - 최저 점수: {min_score}/10\n"
            output += f"   - 점수 변화: {' → '.join(map(str, feedback_scores))}\n"
        
        output += "\n" + "-" * 60 + "\n\n"
        
        # 반복 과정 상세 표시
        if len(iteration_log) > 0:
            output += "🔄 반복 과정:\n\n"
            display_iterations = min(max_iterations, len(iteration_log))
            
            for i, log_entry in enumerate(iteration_log[:display_iterations]):
                iteration_num = log_entry.get('iteration', i + 1)
                iteration_result = str(log_entry.get('result', '결과 없음'))
                score = log_entry.get('score', 0)
                
                output += f"[반복 {iteration_num}]"
                if show_scores:
                    score_emoji = self._get_score_emoji(score)
                    output += f" {score_emoji} {score}/10"
                
                if show_timestamps and 'timestamp' in log_entry:
                    timestamp = datetime.fromtimestamp(log_entry['timestamp']).strftime('%H:%M:%S')
                    output += f" ({timestamp})"
                
                output += "\n"
                
                # 결과 표시
                # if len(iteration_result) > truncate_len:
                #     truncated_result = iteration_result[:truncate_len] + "..."
                #     output += f"결과: {truncated_result}\n"
                # else:
                #     output += f"결과: {iteration_result}\n"
                
                output += f"결과: {iteration_result}\n"
                # 평가 정보
                if 'evaluation' in log_entry:
                    eval_info = log_entry['evaluation']
                    if isinstance(eval_info, dict):
                        if 'reasoning' in eval_info:
                            output += f"평가: {str(eval_info['reasoning'])}\n"
                        if 'improvements' in eval_info and eval_info['improvements']:
                            improvements = eval_info['improvements']
                            if isinstance(improvements, list):
                                output += f"개선사항: {', '.join(str(item) for item in improvements)}\n"
                            else:
                                output += f"개선사항: {str(improvements)}\n"
                
                output += "\n"
            
            # if len(iteration_log) > max_iterations:
            #     output += f"... 및 {len(iteration_log) - max_iterations}개의 추가 반복\n\n"
        
        # 최종 결과
        output += "=" * 60 + "\n\n"

        return output

    def _format_todo_details(self, input_print: Dict[str, Any]) -> str:
        """TODO 실행 과정을 블록 형식으로 포매팅"""
        todos_list = input_print.get("todos_list", [])
        todo_execution_log = input_print.get("todo_execution_log", [])
        todos_generated = input_print.get("todos_generated", 0)
        todos_completed = input_print.get("todos_completed", 0)
        completion_rate = input_print.get("completion_rate", 0)

        if not todos_list and not todo_execution_log:
            return ""

        output = "<TODO_DETAILS>\n"
        output += "📋 TODO 실행 상세 정보\n"
        output += "=" * 40 + "\n\n"

        # TODO 통계
        output += f"📊 TODO 통계:\n"
        output += f"   - 생성된 TODO: {todos_generated}개\n"
        output += f"   - 완료된 TODO: {todos_completed}개\n"
        output += f"   - 완료율: {completion_rate:.1%}\n\n"

        # TODO 목록
        if todos_list:
            output += "📝 생성된 TODO 목록:\n"
            for i, todo in enumerate(todos_list, 1):
                todo_title = todo.get("title", f"TODO {i}")
                todo_description = todo.get("description", "")
                output += f"   {i}. {todo_title}\n"
                if todo_description:
                    output += f"      {todo_description}\n"
            output += "\n"

        # TODO 실행 로그
        if todo_execution_log:
            output += "🔄 TODO 실행 과정:\n\n"
            for i, log_entry in enumerate(todo_execution_log, 1):
                todo_title = log_entry.get("todo_title", f"TODO {i}")
                result = log_entry.get("result", "결과 없음")
                status = log_entry.get("status", "unknown")

                status_emoji = "✅" if status == "completed" else "❌" if status == "failed" else "⏳"

                output += f"[TODO {i}] {status_emoji} {todo_title}\n"
                output += f"결과: {str(result)}\n\n"

        output += "</TODO_DETAILS>"

        return output

    def _get_score_emoji(self, score: int) -> str:
        """점수에 따른 이모지 반환"""
        if score >= 9:
            return "🟢"  # 매우 좋음
        elif score >= 7:
            return "🟡"  # 좋음
        elif score >= 5:
            return "🟠"  # 보통
        else:
            return "🔴"  # 나쁨

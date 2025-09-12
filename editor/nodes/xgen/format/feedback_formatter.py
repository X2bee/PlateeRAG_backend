import logging
from typing import Dict, Any, Optional, List
from editor.node_composer import Node
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class FeedbackLoopFormatterNode(Node):
    categoryId = "xgen" 
    functionId = "format"
    nodeId = "format/feedback_loop"
    nodeName = "Feedback Loop Formatter"
    description = "피드백 루프 노드의 출력을 가독성 좋게 포매팅하여 하나의 문자열로 반환하는 노드"
    tags = ["format", "feedback", "display", "output"]

    inputs = [
        {"id": "result", "name": "Final Result", "type": "STR", "multi": False, "required": True},
        {"id": "iteration_log", "name": "Iteration Log", "type": "LIST", "multi": False, "required": True},
        {"id": "feedback_scores", "name": "Feedback Scores", "type": "LIST", "multi": False, "required": True},
    ]
    outputs = [
        {"id": "formatted_output", "name": "Formatted Output", "type": "STR"},
    ]
    parameters = [
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
        {"id": "truncate_results", "name": "Truncate Results", "type": "INT", "value": 150, "min": 50, "max": 500, "step": 25, "optional": True, "description": "각 반복 결과를 자를 최대 문자 수"},
    ]

    def execute(
        self,
        result: str,
        iteration_log: List[Dict],
        feedback_scores: List[int],
        format_style: str = "detailed",
        show_scores: bool = True,
        show_timestamps: bool = False,
        max_iteration_display: int = 5,
        truncate_results: int = 150,
        **kwargs
    ) -> str:
        
        try:
            if format_style == "summary":
                formatted_output = self._format_summary(result, iteration_log, feedback_scores)
            elif format_style == "compact":
                formatted_output = self._format_compact(result, iteration_log, feedback_scores, show_scores)
            elif format_style == "markdown":
                formatted_output = self._format_markdown(result, iteration_log, feedback_scores, show_scores, show_timestamps, max_iteration_display, truncate_results)
            else:  # detailed
                formatted_output = self._format_detailed(result, iteration_log, feedback_scores, show_scores, show_timestamps, max_iteration_display, truncate_results)
            
            return formatted_output
            
        except Exception as e:
            logger.error(f"[FEEDBACK_FORMATTER] 포매팅 중 오류 발생: {str(e)}")
            return f"포매팅 오류: {str(e)}\n\n원본 결과: {result}"

    def _format_summary(self, result: str, iteration_log: List[Dict], feedback_scores: List[int]) -> str:
        """요약 형태로 포매팅"""
        total_iterations = len(iteration_log)
        final_score = feedback_scores[-1] if feedback_scores else 0
        avg_score = sum(feedback_scores) / len(feedback_scores) if feedback_scores else 0
        
        return f"""=== 피드백 루프 실행 요약 ===

📊 실행 통계:
- 총 반복 횟수: {total_iterations}회
- 최종 점수: {final_score}/10
- 평균 점수: {avg_score:.1f}/10

✅ 최종 결과:
{result}
"""

    def _format_compact(self, result: str, iteration_log: List[Dict], feedback_scores: List[int], show_scores: bool) -> str:
        """압축된 형태로 포매팅"""
        total_iterations = len(iteration_log)
        score_info = f" (점수: {' → '.join(map(str, feedback_scores))})" if show_scores and feedback_scores else ""
        
        return f"""🔄 피드백 루프 완료: {total_iterations}회 반복{score_info}

📝 결과: {result}"""

    def _format_markdown(self, result: str, iteration_log: List[Dict], feedback_scores: List[int], 
                        show_scores: bool, show_timestamps: bool, max_iterations: int, truncate_len: int) -> str:
        """마크다운 형태로 포매팅"""
        markdown = "# 🔄 피드백 루프 실행 결과\n\n"
        
        # 통계 정보
        total_iterations = len(iteration_log)
        final_score = feedback_scores[-1] if feedback_scores else 0
        avg_score = sum(feedback_scores) / len(feedback_scores) if feedback_scores else 0
        
        markdown += "## 📊 실행 통계\n\n"
        markdown += f"- **총 반복 횟수**: {total_iterations}회\n"
        if show_scores and feedback_scores:
            markdown += f"- **최종 점수**: {final_score}/10\n"
            markdown += f"- **평균 점수**: {avg_score:.1f}/10\n"
            markdown += f"- **점수 변화**: {' → '.join(map(str, feedback_scores))}\n"
        markdown += "\n"
        
        # 반복 과정
        if len(iteration_log) > 0:
            markdown += "## 🔄 반복 과정\n\n"
            display_iterations = min(max_iterations, len(iteration_log))
            
            for i, log_entry in enumerate(iteration_log[:display_iterations]):
                iteration_num = log_entry.get('iteration', i + 1)
                iteration_result = log_entry.get('result', '결과 없음')
                score = log_entry.get('score', 0)
                
                # 결과 자르기
                if len(iteration_result) > truncate_len:
                    iteration_result = iteration_result[:truncate_len] + "..."
                
                markdown += f"### 반복 {iteration_num}"
                if show_scores:
                    markdown += f" (점수: {score}/10)"
                markdown += "\n\n"
                
                if show_timestamps and 'timestamp' in log_entry:
                    timestamp = datetime.fromtimestamp(log_entry['timestamp']).strftime('%H:%M:%S')
                    markdown += f"**시간**: {timestamp}\n\n"
                
                markdown += f"```\n{iteration_result}\n```\n\n"
            
            if len(iteration_log) > max_iterations:
                markdown += f"*... 및 {len(iteration_log) - max_iterations}개의 추가 반복*\n\n"
        
        # 최종 결과
        markdown += "## ✅ 최종 결과\n\n"
        markdown += f"```\n{result}\n```\n"
        
        return markdown

    def _format_detailed(self, result: str, iteration_log: List[Dict], feedback_scores: List[int],
                        show_scores: bool, show_timestamps: bool, max_iterations: int, truncate_len: int) -> str:
        """상세한 형태로 포매팅"""
        output = "=" * 60 + "\n"
        output += "🔄 피드백 루프 실행 결과\n"
        output += "=" * 60 + "\n\n"
        
        # 실행 통계
        total_iterations = len(iteration_log)
        output += f"📊 실행 통계:\n"
        output += f"   - 총 반복 횟수: {total_iterations}회\n"
        
        if show_scores and feedback_scores:
            final_score = feedback_scores[-1]
            avg_score = sum(feedback_scores) / len(feedback_scores)
            min_score = min(feedback_scores)
            max_score = max(feedback_scores)
            
            output += f"   - 최종 점수: {final_score}/10\n"
            output += f"   - 평균 점수: {avg_score:.1f}/10\n"
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
                iteration_result = log_entry.get('result', '결과 없음')
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
                if len(iteration_result) > truncate_len:
                    truncated_result = iteration_result[:truncate_len] + "..."
                    output += f"결과: {truncated_result}\n"
                else:
                    output += f"결과: {iteration_result}\n"
                
                # 평가 정보
                if 'evaluation' in log_entry:
                    eval_info = log_entry['evaluation']
                    if isinstance(eval_info, dict):
                        if 'reasoning' in eval_info:
                            output += f"평가: {eval_info['reasoning']}\n"
                        if 'improvements' in eval_info and eval_info['improvements']:
                            improvements = eval_info['improvements']
                            if isinstance(improvements, list):
                                output += f"개선사항: {', '.join(improvements)}\n"
                            else:
                                output += f"개선사항: {improvements}\n"
                
                output += "\n"
            
            if len(iteration_log) > max_iterations:
                output += f"... 및 {len(iteration_log) - max_iterations}개의 추가 반복\n\n"
        
        # 최종 결과
        output += "=" * 60 + "\n"
        output += "✅ 최종 결과:\n"
        output += "=" * 60 + "\n\n"
        output += result
        
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
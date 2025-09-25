import logging
import re
import json
from langchain.callbacks.base import BaseCallbackHandler
from editor.utils.helper.service_helper import AppServiceManager
from editor.utils.helper.async_helper import sync_run_async

logger = logging.getLogger(__name__)

def use_guarder_for_text_moderation(text: str) -> tuple[bool, str]:
    try:
        guarder = AppServiceManager.get_guarder_service()
        if guarder:
            moderation_result = sync_run_async(guarder.moderate_text(text))
            if not moderation_result.get("safe", True):
                categories = moderation_result.get("categories", [])
                categories_str = ", ".join(categories) if categories else "알 수 없는 이유"
                return False, f"이러한 요청은 허용되지 않습니다. 원인: {categories_str}"
            return True, ""
        else:
            return True, ""
    except Exception as guarder_error:
        logger.warning(f"Guarder 서비스를 사용할 수 없습니다: {guarder_error}")
        return True, ""

def _parse_document_citations(text: str) -> str:
    """문서 인용 정보를 파싱하여 JSON 형태로 변환"""
    document_pattern = r'\[문서 (\d+)\]\(관련도: ([\d.]+)\)\n\[파일명\] ([^\n]+)\n\[파일경로\] ([^\n]+)\n\[페이지번호\] ([^\n]+)\n\[문장시작줄\] ([^\n]+)\n\[문장종료줄\] ([^\n]+)'

    matches = re.findall(document_pattern, text)

    if matches:
        citations = []
        for match in matches:
            doc_num, score, file_name, file_path, page_num, line_start, line_end = match

            citation = {
                "document_number": int(doc_num.strip()) if doc_num.strip().isdigit() else doc_num.strip(),
                "relevance_score": float(score.strip()) if score.strip().replace('.', '', 1).isdigit() else score.strip(),
                "file_name": file_name.strip(),
                "file_path": file_path.strip(),
                "page_number": int(page_num.strip()) if page_num.strip().isdigit() else page_num.strip(),
                "line_start": int(line_start.strip()) if line_start.strip().isdigit() else line_start.strip(),
                "line_end": int(line_end.strip()) if line_end.strip().isdigit() else line_end.strip()
            }

            cite_json = json.dumps(citation, ensure_ascii=False)
            escaped_cite_json = cite_json.replace('\"', '"')
            escaped_cite_json = escaped_cite_json.replace('\\"', '"')
            citations.append(f"[Tool_Cite. {escaped_cite_json}]")

        return "\n".join(citations)

    return ""

class NonStreamingAgentHandler(BaseCallbackHandler):
    """스트리밍이 아닌 Agent용 기본 callback handler"""

    def __init__(self):
        self.tool_logs = []
        self.tool_outputs = []

    def on_agent_action(self, action, **kwargs) -> None:
        """Agent가 도구를 호출할 때 호출"""
        # 기본 핸들러에서는 아무 작업도 하지 않음
        return

    def on_tool_end(self, output, **kwargs) -> None:
        """도구 실행이 완료될 때 호출"""
        # 기본 핸들러에서는 아무 작업도 하지 않음
        return

    def on_tool_error(self, error, **kwargs) -> None:
        """도구 실행 오류 시 호출"""
        # 기본 핸들러에서는 아무 작업도 하지 않음
        return

    def get_formatted_output(self, original_output: str) -> str:
        """원본 출력을 그대로 반환"""
        return original_output

class NonStreamingAgentHandlerWithToolOutput(BaseCallbackHandler):
    """스트리밍이 아닌 Agent용 도구 출력 포함 callback handler"""

    def __init__(self):
        self.tool_logs = []
        self.tool_outputs = []

    def on_agent_action(self, action, **kwargs) -> None:
        """Agent가 도구를 호출할 때 호출"""
        tool_name = action.tool
        tool_input = str(action.tool_input)
        self.tool_logs.append(f"<TOOLUSELOG>{tool_name}\n{tool_input}</TOOLUSELOG>")

    def on_tool_end(self, output, **kwargs) -> None:
        """도구 실행이 완료될 때 호출"""
        if isinstance(output, (dict, list)):
            try:
                tool_output = json.dumps(output, ensure_ascii=False, indent=2)
            except Exception:
                tool_output = str(output)
        else:
            tool_output = str(output)
        self.tool_outputs.append(output)

        parsed_output = _parse_document_citations(tool_output)
        display_output = parsed_output.strip() if parsed_output.strip() else tool_output.strip()

        if len(display_output) > 1200:
            display_output = display_output[:1200].rstrip() + "..."

        self.tool_logs.append(f"<TOOLOUTPUTLOG>{display_output}</TOOLOUTPUTLOG>")

    def on_tool_error(self, error, **kwargs) -> None:
        """도구 실행 오류 시 호출"""
        self.tool_logs.append(f"❌ 도구 실행 오류: {str(error)}")

    def get_formatted_output(self, original_output: str) -> str:
        """원본 출력에 도구 사용 정보를 추가하여 반환"""
        if not self.tool_logs:
            return original_output

        tool_section = "\n".join(self.tool_logs)
        if original_output:
            return f"{original_output}\n\n{tool_section}"
        return tool_section

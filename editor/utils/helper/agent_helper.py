import logging
import re
import json
from typing import Any, Optional, Union, List, Dict
from json import JSONDecodeError
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.outputs import Generation
from editor.utils.helper.service_helper import AppServiceManager
from editor.utils.helper.async_helper import sync_run_async

logger = logging.getLogger(__name__)


class XgenJsonOutputParser(JsonOutputParser):
    """
    LangChain의 JsonOutputParser를 래핑하여 강건한 JSON 파싱 기능을 제공하는 커스텀 파서.

    주요 기능:
    1. 표준 JSON 파싱 시도
    2. 마크다운 코드 블록 제거 후 파싱
    3. 임베디드 JSON 추출 (에러 메시지 내부 등)
    4. OutputParserException 발생 시 자동 fallback
    5. 상세한 로깅
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _extract_and_parse_json_robust(self, text: str) -> Union[Dict, List, str]:
        """
        텍스트에서 JSON을 추출하고 파싱하는 강건한 메서드.

        Args:
            text: JSON이 포함된 텍스트

        Returns:
            파싱된 JSON 객체 또는 원본 텍스트
        """
        if not isinstance(text, str):
            return text

        logger.info("[XGEN_JSON_PARSER] Robust JSON 파싱 시도 중...")

        # 1. 마크다운 코드 블록 제거 후 시도 (먼저 시도)
        # ```json ... ``` 또는 ``` ... ``` 형태 처리
        code_block_patterns = [
            r'```json\s*\n(.*?)\n\s*```',  # ```json\n...\n```
            r'```json\s+(.*?)\s*```',      # ```json ... ```
            r'```\s*\n(.*?)\n\s*```',      # ```\n...\n``` (가장 흔한 케이스)
            r'```\s+(.*?)\s+```',          # ``` ... ``` (공백 있음)
            r'```(.*?)```',                # ```...``` (공백 없음, 최후 수단)
        ]

        for pattern in code_block_patterns:
            match = re.search(pattern, text.strip(), re.DOTALL | re.IGNORECASE)
            if match:
                json_content = match.group(1).strip()
                # 빈 문자열이면 건너뛰기
                if not json_content:
                    continue
                try:
                    parsed = json.loads(json_content)
                    logger.info("[XGEN_JSON_PARSER] 코드 블록 제거 후 JSON 파싱 성공")
                    return parsed
                except (JSONDecodeError, ValueError) as e:
                    logger.debug(f"[XGEN_JSON_PARSER] 코드 블록 파싱 실패 (패턴: {pattern}): {e}")
                    continue        # 2. 전체 텍스트를 JSON으로 파싱 시도
        try:
            cleaned_text = text.strip()
            parsed = json.loads(cleaned_text)
            logger.info("[XGEN_JSON_PARSER] 전체 텍스트 JSON 파싱 성공")
            return parsed
        except (JSONDecodeError, ValueError) as e:
            logger.debug(f"[XGEN_JSON_PARSER] 직접 파싱 실패: {e}")

        # 3. 텍스트 내부에서 JSON 객체 추출 시도 (중괄호로 감싸진 부분)
        json_patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # 중첩된 객체 포함
            r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]',  # 중첩된 배열 포함
        ]

        for pattern in json_patterns:
            matches = list(re.finditer(pattern, text, re.DOTALL))
            # 가장 긴 매칭부터 시도 (더 완전한 JSON일 가능성이 높음)
            matches.sort(key=lambda m: len(m.group(0)), reverse=True)

            for match in matches:
                json_str = match.group(0)
                try:
                    parsed = json.loads(json_str)
                    logger.info(f"[XGEN_JSON_PARSER] 임베디드 JSON 추출 성공 (길이: {len(json_str)})")
                    return parsed
                except (JSONDecodeError, ValueError):
                    continue

        # 4. 모든 시도 실패 - 원본 텍스트 반환
        logger.warning("[XGEN_JSON_PARSER] 모든 JSON 파싱 시도 실패, 원본 텍스트 반환")
        return text

    def parse_result(self, result: List[Generation], *, partial: bool = False) -> Any:
        """
        LLM 결과를 파싱하며, 실패 시 robust 파싱 시도.

        Args:
            result: LLM 생성 결과 리스트
            partial: 부분 파싱 여부

        Returns:
            파싱된 JSON 객체

        Raises:
            OutputParserException: 모든 파싱 시도가 실패한 경우
        """
        text = result[0].text
        text = text.strip()

        # LangChain의 parse_json_markdown를 건너뛰고 직접 파싱 시도
        try:
            # 직접 JSON 파싱 시도
            parsed = json.loads(text)
            logger.info("[XGEN_JSON_PARSER] 직접 JSON 파싱 성공")
            return parsed
        except (JSONDecodeError, ValueError) as direct_error:
            logger.debug(f"[XGEN_JSON_PARSER] 직접 파싱 실패: {direct_error}")

        # 직접 파싱 실패 시 부모 클래스 시도
        try:
            return super().parse_result(result, partial=partial)
        except OutputParserException as e:
            logger.warning(f"[XGEN_JSON_PARSER] 표준 파싱 실패: {str(e)}")

            # partial 모드에서는 None 반환
            if partial:
                return None

            # robust 파싱 시도
            logger.info("[XGEN_JSON_PARSER] Robust 파싱 시도 중...")
            parsed_result = self._extract_and_parse_json_robust(text)

            # 파싱 성공 시 반환
            if isinstance(parsed_result, (dict, list)):
                logger.info("[XGEN_JSON_PARSER] Robust 파싱 성공!")
                return parsed_result

            # 완전히 실패한 경우 원본 예외 재발생 (llm_output 포함)
            logger.error("[XGEN_JSON_PARSER] 모든 파싱 시도 실패")
            msg = f"Invalid json output: {text}"
            raise OutputParserException(msg, llm_output=text) from e
        except Exception as e:
            # 예상치 못한 오류
            logger.error(f"[XGEN_JSON_PARSER] 예상치 못한 오류: {str(e)}", exc_info=True)
            raise

    def parse(self, text: str) -> Any:
        """
        텍스트를 직접 파싱.

        Args:
            text: 파싱할 텍스트

        Returns:
            파싱된 JSON 객체
        """
        return self.parse_result([Generation(text=text)])


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

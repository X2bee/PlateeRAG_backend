import asyncio
import logging
import queue
import threading
from typing import Any, Generator, Callable, Awaitable
from langchain_core.callbacks import AsyncCallbackHandler
import re
import json

logger = logging.getLogger(__name__)

def _parse_document_citations(text: str) -> str:
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

        # 인용 정보만 반환 (줄바꿈으로 구분)
        return "\n".join(citations)

    # 문서 인용 패턴이 없으면 빈 문자열 반환
    return ""

class EnhancedAgentStreamingHandler(AsyncCallbackHandler):
    def __init__(self):
        self.token_queue = queue.Queue()
        self.is_done = False
        self.current_step = 0
        self.tool_outputs = []
        self.streamed_tokens = []

    def put_token(self, token):
        if not self.is_done and token:
            self.token_queue.put(('token', str(token)))

    def put_status(self, status):
        if not self.is_done:
            self.token_queue.put(('status', status))

    def finish(self):
        self.is_done = True
        self.token_queue.put(('done', None))

    def put_error(self, error):
        self.is_done = True
        self.token_queue.put(('error', error))

    async def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        """LLM 호출 시작 시"""
        self.current_step += 1
        if self.current_step > 1:
            pass
            # self.put_status(f"\n💭 단계 {self.current_step}: 답변 생성 중...\n")

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """LLM이 새 토큰을 생성할 때 호출 (OpenAI/Claude 모두 지원)"""
        if not token:
            return

        # Claude 모델의 경우 토큰이 리스트 형태로 올 수 있음
        if isinstance(token, list):
            for item in token:
                if isinstance(item, dict) and item.get('type') == 'text' and 'text' in item:
                    text_content = item['text']
                    if text_content:
                        self.streamed_tokens.append(text_content)
                        self.put_token(text_content)
            return

        # Claude 모델의 경우 토큰이 딕셔너리 형태로 올 수 있음
        if isinstance(token, dict):
            # text 타입의 청크만 추출
            if token.get('type') == 'text' and 'text' in token:
                text_content = token['text']
                if text_content:
                    self.streamed_tokens.append(text_content)
                    self.put_token(text_content)
            # tool_use나 다른 타입은 무시
            return

        # OpenAI 등 문자열로 오는 경우
        if isinstance(token, str):
            self.streamed_tokens.append(token)
            self.put_token(token)

    async def on_agent_action(self, action, **kwargs) -> None:
        """Agent가 도구를 호출할 때 호출"""
        pass

    async def on_tool_start(self, serialized, input_str, **kwargs) -> None:
        """도구 실행 시작 시"""
        pass

    async def on_tool_end(self, output, **kwargs) -> None:
        """도구 실행이 완료될 때 호출"""
        pass

    async def on_tool_error(self, error, **kwargs) -> None:
        self.put_status(f"❌ 도구 실행 오류: {str(error)}\n")

    async def on_agent_finish(self, finish, **kwargs) -> None:
        pass

    # LangGraph 지원을 위한 추가 메서드
    async def on_chain_start(self, serialized, inputs, **kwargs) -> None:
        """Chain 시작 시 호출 (LangGraph 노드 실행 시작)"""
        pass

    async def on_chain_end(self, outputs, **kwargs) -> None:
        """Chain 종료 시 호출 (LangGraph 노드 실행 완료)"""
        pass

    async def on_chain_error(self, error, **kwargs) -> None:
        """Chain 오류 시 호출"""
        self.put_error(error)

class EnhancedAgentStreamingHandlerWithToolOutput(AsyncCallbackHandler):
    def __init__(self):
        self.token_queue = queue.Queue()
        self.is_done = False
        self.current_step = 0
        self.tool_outputs = []
        self.streamed_tokens = []
        self.tool_logs = []

    def put_token(self, token):
        if not self.is_done and token:
            self.token_queue.put(('token', str(token)))

    def put_status(self, status):
        if not self.is_done:
            self.token_queue.put(('status', status))

    def finish(self):
        self.is_done = True
        self.token_queue.put(('done', None))

    def put_error(self, error):
        self.is_done = True
        self.token_queue.put(('error', error))

    async def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        """LLM 호출 시작 시"""
        self.current_step += 1
        if self.current_step > 1:
            pass
            # self.put_status(f"\n💭 단계 {self.current_step}: 답변 생성 중...\n")

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """LLM이 새 토큰을 생성할 때 호출 (OpenAI/Claude 모두 지원)"""
        if not token:
            return

        # Claude 모델의 경우 토큰이 리스트 형태로 올 수 있음
        if isinstance(token, list):
            for item in token:
                if isinstance(item, dict) and item.get('type') == 'text' and 'text' in item:
                    text_content = item['text']
                    if text_content:
                        self.streamed_tokens.append(text_content)
                        self.put_token(text_content)
            return

        # Claude 모델의 경우 토큰이 딕셔너리 형태로 올 수 있음
        if isinstance(token, dict):
            # text 타입의 청크만 추출
            if token.get('type') == 'text' and 'text' in token:
                text_content = token['text']
                if text_content:
                    self.streamed_tokens.append(text_content)
                    self.put_token(text_content)
            # tool_use나 다른 타입은 무시
            return

        # OpenAI 등 문자열로 오는 경우
        if isinstance(token, str):
            self.streamed_tokens.append(token)
            self.put_token(token)

    async def on_agent_action(self, action, **kwargs) -> None:
        """Agent가 도구를 호출할 때 호출"""
        # pass
        tool_name = action.tool
        tool_input = str(action.tool_input)
        log_entry = f"<TOOLUSELOG>{tool_name}\n{tool_input}</TOOLUSELOG>"
        self.tool_logs.append(log_entry)
        self.put_status(f"{log_entry}\n")

    async def on_tool_start(self, serialized, input_str, **kwargs) -> None:
        """도구 실행 시작 시"""
        pass

    async def on_tool_end(self, output, **kwargs) -> None:
        """도구 실행이 완료될 때 호출"""

        self.tool_outputs.append(output)

        if isinstance(output, (dict, list)):
            try:
                import json

                tool_output = json.dumps(output, ensure_ascii=False, indent=2)
            except Exception:
                tool_output = str(output)
        else:
            tool_output = str(output)

        parsed_output = _parse_document_citations(tool_output)
        display_output = parsed_output.strip() if parsed_output.strip() else tool_output.strip()

        if len(display_output) > 1200:
            display_output = display_output[:1200].rstrip() + "..."

        log_entry = f"<TOOLOUTPUTLOG>{display_output}</TOOLOUTPUTLOG>"
        self.tool_logs.append(log_entry)

        self.put_status(f"{log_entry}\n")

    async def on_tool_error(self, error, **kwargs) -> None:
        self.put_status(f"❌ 도구 실행 오류: {str(error)}\n")

    async def on_agent_finish(self, finish, **kwargs) -> None:
        pass

    # LangGraph 지원을 위한 추가 메서드
    async def on_chain_start(self, serialized, inputs, **kwargs) -> None:
        """Chain 시작 시 호출 (LangGraph 노드 실행 시작)"""
        pass

    async def on_chain_end(self, outputs, **kwargs) -> None:
        """Chain 종료 시 호출 (LangGraph 노드 실행 완료)"""
        pass

    async def on_chain_error(self, error, **kwargs) -> None:
        """Chain 오류 시 호출"""
        self.put_error(error)

def execute_agent_streaming(
    async_executor_func: Callable[[], Awaitable[Any]],
    handler: EnhancedAgentStreamingHandler
) -> Generator[str, None, None]:
    """
    Agent 실행을 스트리밍으로 처리하는 범용 함수
    LangGraph 1.0.0+ CompiledStateGraph 지원

    Args:
        async_executor_func: 실행할 비동기 함수 (예: lambda: agent_graph.ainvoke(inputs, {"callbacks": [handler]}))
        handler: 스트리밍 핸들러

    Yields:
        str: 스트리밍된 토큰들
    """

    exception_container = [None]
    execution_finished = [False]

    def run_agent():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def execute_agent():
                try:
                    # 전달받은 비동기 함수 실행
                    result = await async_executor_func()
                    await asyncio.sleep(0.05)
                    handler.finish()
                    execution_finished[0] = True
                    return result
                except Exception as e:
                    logger.error(f"Agent 실행 중 오류: {str(e)}", exc_info=True)

                    # 사용자 친화적인 에러 메시지 생성
                    error_detail = str(e)

                    # OpenAI API 에러 처리
                    if "404" in error_detail and "does not exist" in error_detail:
                        if "claude" in error_detail.lower():
                            error_message = "❌ Claude 모델을 OpenAI API로 호출하려고 했습니다. Anthropic API 키가 올바르게 설정되었는지 확인하세요."
                        else:
                            error_message = f"❌ 모델을 찾을 수 없습니다: {error_detail}"
                    elif "401" in error_detail or "authentication" in error_detail.lower():
                        error_message = "❌ API 인증 실패: API 키를 확인하세요."
                    elif "429" in error_detail or "rate limit" in error_detail.lower():
                        error_message = "❌ API 요청 한도 초과: 잠시 후 다시 시도하세요."
                    else:
                        error_message = f"❌ 오류 발생: {error_detail}"

                    # 에러 메시지를 handler를 통해 전달
                    handler.put_status(f"\n{error_message}\n")

                    # LangGraph 관련 오류에 대한 추가 정보 제공
                    if "validation error" in str(e).lower():
                        if "function.arguments" in str(e):
                            logger.error("OpenAI Tool Calling validation error detected. This may be caused by:")
                            logger.error("1. Tool with empty or None args_schema")
                            logger.error("2. Tool function returning invalid argument format")
                            logger.error("3. Pydantic schema validation failure")
                        elif "messages" in str(e).lower():
                            logger.error("LangGraph state validation error. Check that:")
                            logger.error("1. Input format matches AgentState schema (must contain 'messages' key)")
                            logger.error("2. Messages are proper langchain_core.messages objects")

                    handler.put_error(e)
                    execution_finished[0] = True
                    raise e

            loop.run_until_complete(execute_agent())
        except Exception as e:
            exception_container[0] = e
            handler.put_error(e)
            execution_finished[0] = True
        finally:
            loop.close()

    thread = threading.Thread(target=run_agent, daemon=True)
    thread.start()

    try:
        while True:
            try:
                msg_type, value = handler.token_queue.get(timeout=2.0)
                if msg_type == 'token':
                    yield value
                elif msg_type == 'status':
                    yield value
                elif msg_type == 'done':
                    break
                elif msg_type == 'error':
                    if exception_container[0]:
                        raise exception_container[0]
                    elif value:
                        raise value
                    else:
                        raise RuntimeError("Unknown error occurred")

            except queue.Empty:
                # 큐가 비어있을 때 스레드 상태 확인
                if execution_finished[0] or not thread.is_alive():
                    # 실행이 완료되었지만 큐에 'done' 신호가 없는 경우
                    if execution_finished[0] and handler.token_queue.empty():
                        break
                    # 스레드가 비정상 종료된 경우
                    elif not thread.is_alive() and exception_container[0]:
                        raise exception_container[0]
                    elif not thread.is_alive():
                        break
                continue

    except Exception as e:
        logger.error(f"스트리밍 중 오류: {str(e)}", exc_info=True)
        # 에러 메시지를 사용자에게 yield
        yield f"\n❌ 오류 발생: {str(e)}\n"
    finally:
        # 스레드 정리
        if thread.is_alive():
            thread.join(timeout=10.0)

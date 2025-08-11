import asyncio
import logging
import queue
import threading
from typing import Any, Generator, Callable, Awaitable
from langchain.callbacks.base import AsyncCallbackHandler

logger = logging.getLogger(__name__)

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
        """LLM이 새 토큰을 생성할 때 호출"""
        if token:
            self.streamed_tokens.append(token)
            self.put_token(token)

    async def on_agent_action(self, action, **kwargs) -> None:
        """Agent가 도구를 호출할 때 호출"""
        pass
        # tool_name = action.tool
        # tool_input = str(action.tool_input)[:100] + "..." if len(str(action.tool_input)) > 100 else str(action.tool_input)
        # self.put_status(f"\n🔧 도구 '{tool_name}' 실행 중...\n입력: {tool_input}\n")

    async def on_tool_start(self, serialized, input_str, **kwargs) -> None:
        """도구 실행 시작 시"""
        pass

    async def on_tool_end(self, output, **kwargs) -> None:
        """도구 실행이 완료될 때 호출"""
        pass
        # tool_output = str(output)[:200] + "..." if len(str(output)) > 200 else str(output)
        # self.tool_outputs.append(output)
        # self.put_status(f"✅ 도구 실행 완료\n결과: {tool_output}\n")

    async def on_tool_error(self, error, **kwargs) -> None:
        self.put_status(f"❌ 도구 실행 오류: {str(error)}\n")

    async def on_agent_finish(self, finish, **kwargs) -> None:
        pass


def execute_agent_streaming(
    async_executor_func: Callable[[], Awaitable[Any]],
    handler: EnhancedAgentStreamingHandler
) -> Generator[str, None, None]:
    """
    Agent 실행을 스트리밍으로 처리하는 범용 함수

    Args:
        async_executor_func: 실행할 비동기 함수 (예: lambda: agent_executor.ainvoke(inputs, {"callbacks": [handler]}))
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
                    raise value

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
        yield f"\n❌ 오류 발생: {str(e)}\n"
    finally:
        # 스레드 정리
        if thread.is_alive():
            thread.join(timeout=10.0)

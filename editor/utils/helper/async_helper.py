"""
Async Helper 유틸리티

이 모듈은 비동기 함수를 동기적으로 호출할 수 있게 해주는 헬퍼 함수들을 제공합니다.
주로 Langchain Tool과 같이 동기 함수만 지원하는 환경에서 비동기 함수를 사용할 때 활용됩니다.
"""

import asyncio
import concurrent.futures
import logging
from typing import Any, Callable, Coroutine
from functools import wraps

logger = logging.getLogger(__name__)


def run_async_in_sync(async_func: Callable[..., Coroutine[Any, Any, Any]]) -> Callable[..., Any]:
    """
    비동기 함수를 동기적으로 실행할 수 있게 래핑하는 데코레이터

    Args:
        async_func: 래핑할 비동기 함수

    Returns:
        동기적으로 실행되는 래퍼 함수

    Example:
        @run_async_in_sync
        async def my_async_function(arg1, arg2):
            return await some_async_operation(arg1, arg2)

        # 이제 동기적으로 호출 가능
        result = my_async_function(value1, value2)
    """
    @wraps(async_func)
    def sync_wrapper(*args, **kwargs):
        return sync_run_async(async_func(*args, **kwargs))

    return sync_wrapper


def sync_run_async(coro: Coroutine[Any, Any, Any]) -> Any:
    """
    코루틴을 동기적으로 실행하는 함수

    이미 실행 중인 이벤트 루프가 있어도 안전하게 작동합니다.
    새로운 스레드에서 asyncio.run()을 실행하여 이벤트 루프 충돌을 방지합니다.

    Args:
        coro: 실행할 코루틴 객체

    Returns:
        코루틴의 실행 결과

    Raises:
        Exception: 코루틴 실행 중 발생한 예외

    Example:
        async def my_async_func():
            return "Hello, World!"

        result = sync_run_async(my_async_func())
        print(result)  # "Hello, World!"
    """
    def run_in_thread():
        """새로운 스레드에서 asyncio.run을 실행"""
        try:
            return asyncio.run(coro)
        except Exception as e:
            logger.error(f"비동기 함수 실행 중 오류 발생: {e}")
            raise

    try:
        # 새로운 스레드에서 asyncio.run 실행
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            return future.result()
    except Exception as e:
        logger.error(f"sync_run_async 실행 중 오류: {e}")
        raise


class AsyncToSyncWrapper:
    """
    비동기 함수를 동기 함수로 래핑하는 클래스

    특정 비동기 함수를 여러 번 호출해야 할 때 유용합니다.
    """

    def __init__(self, async_func: Callable[..., Coroutine[Any, Any, Any]]):
        """
        Args:
            async_func: 래핑할 비동기 함수
        """
        self.async_func = async_func
        self.__name__ = getattr(async_func, '__name__', 'wrapped_async_func')
        self.__doc__ = getattr(async_func, '__doc__', None)

    def __call__(self, *args, **kwargs) -> Any:
        """
        비동기 함수를 동기적으로 호출

        Returns:
            비동기 함수의 실행 결과
        """
        coro = self.async_func(*args, **kwargs)
        return sync_run_async(coro)


# 편의를 위한 별칭
async_to_sync = AsyncToSyncWrapper

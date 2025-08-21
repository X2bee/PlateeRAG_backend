import asyncio
import concurrent.futures
import inspect
import logging
import json
import threading
import time
import queue
from collections import deque
from typing import Dict, Any, Generator, List, Optional, AsyncGenerator
from editor.node_composer import NODE_CLASS_REGISTRY
from service.monitoring.performance_logger import PerformanceLogger
from service.database.models.executor import ExecutionIO

logger = logging.getLogger('Async-Workflow-Executor')

class AsyncWorkflowExecutor:
    """
    비동기 워크플로우 실행기
    백그라운드에서 워크플로우를 실행하여 다른 API 호출을 블로킹하지 않습니다.
    """

    def __init__(self,
                 workflow_data: Dict[str, Any],
                 db_manager=None,
                 interaction_id: Optional[str] = None,
                 user_id: Optional[int] = None,
                 executor_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None,
                 expected_output: Optional[str] = None,
                 test_mode: Optional[bool] = False):
        self.workflow_id: str = workflow_data['workflow_id']
        self.workflow_name: str = workflow_data['workflow_name']
        self.interaction_id: str = interaction_id or 'default'
        self.user_id: int = user_id or 0
        self.db_manager = db_manager
        self.nodes: Dict[str, Dict[str, Any]] = {node['id']: node for node in workflow_data['nodes']}
        self.edges: List[Dict[str, Any]] = workflow_data['edges']
        self.graph: Dict[str, List[str]] = {node_id: [] for node_id in self.nodes}
        self.in_degree: Dict[str, int] = {node_id: 0 for node_id in self.nodes}

        # 스레드 풀 executor 설정 (공유 가능)
        self.executor_pool = executor_pool
        self._should_use_thread_pool = True

        # 실행 상태 추적
        self._execution_status = "pending"  # pending, running, completed, error
        self._error_message = None
        self._start_time = None
        self._end_time = None

        # 스트리밍을 위한 큐 (스레드 간 통신용)
        self._streaming_queue = queue.Queue()
        self._is_streaming = False

        if expected_output and expected_output.strip() != "":
            self.expected_output = expected_output
        else:
            self.expected_output = ""
        self.test_mode = test_mode

    def _build_graph(self) -> None:
        """워크플로우 데이터로부터 그래프와 진입 차수를 계산합니다."""
        for edge in self.edges:
            source_id: str = edge['source']['nodeId']
            target_id: str = edge['target']['nodeId']
            if source_id in self.graph and target_id in self.graph:
                self.graph[source_id].append(target_id)
                self.in_degree[target_id] += 1

    def _topological_sort(self) -> List[str]:
        """위상 정렬을 사용하여 노드의 실행 순서를 결정합니다."""
        task_queue: deque[str] = deque([node_id for node_id, degree in self.in_degree.items() if degree == 0])
        sorted_nodes: List[str] = []

        while task_queue:
            node_id: str = task_queue.popleft()
            sorted_nodes.append(node_id)

            for neighbor_id in self.graph.get(node_id, []):
                self.in_degree[neighbor_id] -= 1
                if self.in_degree[neighbor_id] == 0:
                    task_queue.append(neighbor_id)

        if len(sorted_nodes) != len(self.nodes):
            raise ValueError("그래프에 순환(cycle)이 존재하여 워크플로우를 실행할 수 없습니다.")

        return sorted_nodes

    def _execute_workflow_sync(self) -> Generator[Any, None, None]:
        """
        동기 워크플로우 실행 (기존 로직과 동일)
        이 메서드는 스레드 풀에서 실행됩니다.
        """
        self._execution_status = "running"
        self._start_time = time.time()

        try:
            self._build_graph()
            execution_order: List[str] = self._topological_sort()

            node_outputs: Dict[str, Dict[str, Any]] = {}
            start_node_data: Optional[Dict[str, Any]] = None

            logger.info("--- 워크플로우 실행 시작 ---")
            logger.info(f"실행 순서: {' -> '.join(execution_order)}")

            streaming_output_started = False

            for node_id in execution_order:
                node_info: Dict[str, Any] = self.nodes[node_id]
                node_spec_id: str = node_info['data']['id']

                NodeClass = NODE_CLASS_REGISTRY.get(node_spec_id)
                if not NodeClass:
                    raise ValueError(f"ID가 '{node_spec_id}'인 노드 클래스를 찾을 수 없습니다.")

                kwargs: Dict[str, Any] = {}
                connected_edges: List[Dict[str, Any]] = [edge for edge in self.edges if edge['target']['nodeId'] == node_id]

                # 포트별로 들어오는 값들을 수집
                port_values: Dict[str, List[Any]] = {}
                for edge in connected_edges:
                    source_node_id: str = edge['source']['nodeId']
                    source_port_id: str = edge['source']['portId']
                    target_port_id: str = edge['target']['portId']

                    if source_node_id in node_outputs and source_port_id in node_outputs[source_node_id]:
                        value = node_outputs[source_node_id][source_port_id]
                        if target_port_id not in port_values:
                            port_values[target_port_id] = []
                        port_values[target_port_id].append(value)

                # 포트별로 값 할당 (단일 값이면 그대로, 여러 값이면 리스트로)
                for port_id, values in port_values.items():
                    if len(values) == 1:
                        kwargs[port_id] = values[0]  # 단일 값은 그대로
                    else:
                        kwargs[port_id] = values  # 여러 값은 리스트로

                if 'parameters' in node_info['data'] and node_info['data']['parameters']:
                    for param in node_info['data']['parameters']:
                        param_key: Optional[str] = param.get('id')
                        param_value: Any = param.get('value')
                        if param_key and param_key not in kwargs:
                            kwargs[param_key] = param_value

                logger.info(f"\n[실행] {node_info['data']['nodeName']} ({node_id})")
                logger.info(f" -> 입력: {kwargs}")

                # 노드 인스턴스 생성 및 실행
                instance = NodeClass()
                node_name_for_logging: str = node_info['data']['nodeName']

                result: Any = None
                try:
                    # PerformanceLogger 컨텍스트 시작
                    with PerformanceLogger(
                        workflow_name=self.workflow_name,
                        workflow_id=self.workflow_id,
                        node_id=node_id,
                        user_id=self.user_id,
                        node_name=node_name_for_logging,
                        db_manager=self.db_manager
                    ) as perf_logger:

                        result = instance.execute(**kwargs)
                        is_generator = inspect.isgenerator(result)

                        # 노드 실행 완료 후 로그 기록
                        log_output = "<streaming_output>" if is_generator else result
                        perf_logger.log(input_data=kwargs, output_data=log_output)
                        logger.info(" -> 완료. 결과: %s", str(log_output)[:100] + "..." if len(str(log_output)) > 100 else str(log_output))

                except Exception as e:
                    logger.error("Error executing node %s: %s", node_id, str(e), exc_info=True)
                    raise

                # functionId에 따른 특별 처리
                function_id: str = node_info['data']['functionId']

                if function_id == 'startnode':
                    # startnode의 경우 입력 데이터 기록
                    start_node_data = {
                        'node_id': node_id,
                        'node_name': node_info['data']['nodeName'],
                        'inputs': kwargs,
                        'result': result
                    }
                    logger.info(f" -> startnode 데이터 수집: {start_node_data}")

                if function_id == 'endnode':
                    is_generator = inspect.isgenerator(result)

                    end_node_result_for_db = "<streaming_output>" if is_generator else result
                    end_node_data = {'node_id': node_id, 'node_name': node_info['data']['nodeName'], 'inputs': kwargs, 'result': end_node_result_for_db}
                    input_data_for_db = start_node_data if start_node_data else {}
                    self._save_execution_io(input_data_for_db, end_node_data)

                    if is_generator:
                        logger.info(f" -> endnode 스트리밍 출력 시작.")
                        streaming_output_started = True
                        yield from result
                        logger.info("\n--- 워크플로우 스트리밍 실행 완료 ---")
                        self._execution_status = "completed"
                        self._end_time = time.time()
                        return
                    else:
                        logger.info(f" -> endnode 완료. 최종 출력: {result}")
                        node_outputs[node_id] = {'result': result}

                # 일반 노드 처리
                if function_id != 'endnode':
                    if not node_info['data']['outputs']:
                        logger.info(f" -> 출력 없음. (결과: {result})")
                        node_outputs[node_id] = {}
                        continue

                    if not node_info['data']['outputs'][0].get('id'):
                        raise ValueError(f"노드 '{node_info['data']['nodeName']}'의 첫 번째 출력 포트에 ID가 정의되어 있지 않습니다.")

                    output_port_id: str = node_info['data']['outputs'][0]['id']
                    node_outputs[node_id] = {output_port_id: result}
                    logger.info(f" -> 출력: {node_outputs[node_id]}")

            if not streaming_output_started:
                logger.info("\n--- 워크플로우 실행 완료 ---")
                final_output = None
                for node_id, output_data in node_outputs.items():
                    if self.nodes[node_id]['data']['functionId'] == 'endnode':
                        final_output = output_data.get('result')
                        break

                if final_output is not None:
                    logger.info("최종 출력: %s", final_output)
                    yield final_output
                else:
                    logger.info("최종 출력이 정의되지 않았습니다. 모든 노드의 중간 결과물을 반환합니다.")
                    yield node_outputs

            self._execution_status = "completed"
            self._end_time = time.time()

        except Exception as e:
            self._execution_status = "error"
            self._error_message = str(e)
            self._end_time = time.time()
            logger.error(f"워크플로우 실행 중 오류 발생: {e}", exc_info=True)
            raise

    def _execute_workflow_sync_streaming(self) -> Generator[Any, None, None]:
        """
        스트리밍용 동기 워크플로우 실행
        결과를 큐에 넣어 실시간 전송이 가능하도록 합니다.
        """
        self._is_streaming = True

        try:
            for result in self._execute_workflow_sync():
                # 결과를 큐에 넣어 비동기 제너레이터에서 사용할 수 있도록 함
                self._streaming_queue.put(('data', result))
                yield result
        except Exception as e:
            self._streaming_queue.put(('error', str(e)))
            raise
        finally:
            # 스트림 종료 신호
            self._streaming_queue.put(('end', None))

    async def execute_workflow_async_streaming(self) -> AsyncGenerator[Any, None]:
        """
        스트리밍을 위한 비동기 워크플로우 실행
        실시간으로 결과를 전송할 수 있습니다.
        """
        if self.executor_pool and self._should_use_thread_pool:
            # 스레드 풀에서 스트리밍 실행 시작
            loop = asyncio.get_event_loop()

            def run_streaming_workflow():
                """스트리밍 워크플로우를 실행"""
                try:
                    # 제너레이터를 소비하여 큐에 결과 전송
                    for _ in self._execute_workflow_sync_streaming():
                        pass  # 결과는 큐를 통해 전달됨
                except Exception as e:
                    logger.error(f"스트리밍 워크플로우 실행 중 오류: {e}", exc_info=True)
                    self._streaming_queue.put(('error', str(e)))

            # 백그라운드에서 워크플로우 실행 시작
            future = loop.run_in_executor(self.executor_pool, run_streaming_workflow)

            try:
                # 큐에서 결과를 실시간으로 읽어 yield
                while True:
                    try:
                        # 짧은 타임아웃으로 큐 확인
                        item_type, item_data = self._streaming_queue.get(timeout=0.1)

                        if item_type == 'data':
                            yield item_data
                        elif item_type == 'error':
                            raise Exception(item_data)
                        elif item_type == 'end':
                            break

                    except queue.Empty:
                        # 큐가 비어있으면 잠시 대기
                        await asyncio.sleep(0.01)
                        continue

                # 백그라운드 작업 완료 대기
                await future

            except Exception as e:
                logger.error(f"스트리밍 실행 중 오류: {e}", exc_info=True)
                raise
        else:
            # 직접 실행 (테스트용)
            for result in self._execute_workflow_sync():
                yield result
    async def execute_workflow_async(self) -> AsyncGenerator[Any, None]:
        """
        비동기적으로 워크플로우를 실행합니다.
        스레드 풀을 사용하여 CPU 집약적 작업을 백그라운드에서 처리합니다.
        """
        if self.executor_pool and self._should_use_thread_pool:
            # 스레드 풀에서 실행 - 결과를 리스트로 수집
            loop = asyncio.get_event_loop()

            def run_sync_workflow():
                """동기 워크플로우를 실행하고 결과를 리스트로 반환"""
                try:
                    results = []
                    for result in self._execute_workflow_sync():
                        results.append(result)
                    return results
                except Exception as e:
                    logger.error(f"스레드 풀에서 워크플로우 실행 중 오류: {e}", exc_info=True)
                    raise

            future = loop.run_in_executor(self.executor_pool, run_sync_workflow)

            try:
                results = await future
                for result in results:
                    yield result
            except Exception as e:
                logger.error(f"스레드 풀에서 워크플로우 실행 중 오류 발생: {e}", exc_info=True)
                raise
        else:
            # 직접 실행 (테스트용)
            for result in self._execute_workflow_sync():
                yield result

    def get_execution_status(self) -> Dict[str, Any]:
        """실행 상태 정보를 반환합니다."""
        execution_time = None
        if self._start_time:
            end_time = self._end_time if self._end_time else time.time()
            execution_time = end_time - self._start_time

        return {
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "interaction_id": self.interaction_id,
            "user_id": self.user_id,
            "status": self._execution_status,
            "error_message": self._error_message,
            "start_time": self._start_time,
            "end_time": self._end_time,
            "execution_time": execution_time
        }

    def _save_execution_io(self, input_data: Dict[str, Any], output_data: Dict[str, Any]) -> None:
        """워크플로우의 입출력 데이터를 ExecutionIO 모델을 통해 DB에 저장합니다."""
        if not self.db_manager:
            logger.warning("DB manager가 없어 ExecutionIO 데이터를 저장할 수 없습니다.")
            return

        def safe_json_dumps(data: Any) -> str:
            """순환 참조 및 미지원 타입을 처리하여 안전하게 JSON으로 직렬화하는 함수"""
            def default_encoder(o: Any) -> Any:
                if isinstance(o, (bytes, bytearray)):
                    return o.decode('utf-8', errors='ignore')
                if inspect.isgenerator(o):
                    return "<generator_output>"
                try:
                    return str(o)
                except Exception:
                    return f"<unserializable: {type(o).__name__}>"
            try:
                return json.dumps(data, ensure_ascii=False, default=default_encoder)
            except TypeError:
                return json.dumps({"error": "unserializable data"}, ensure_ascii=False)

        try:
            # JSON 형태로 변환하여 저장
            input_json = safe_json_dumps(input_data)
            output_json = safe_json_dumps(output_data)

            insert_data = ExecutionIO(
                user_id=self.user_id,
                interaction_id=self.interaction_id,
                workflow_id=self.workflow_id,
                workflow_name=self.workflow_name,
                input_data=input_json,
                output_data=output_json,
                expected_output=self.expected_output,
                test_mode=self.test_mode
            )
            self.db_manager.insert(insert_data)

            logger.info("ExecutionIO 데이터가 성공적으로 저장되었습니다. workflow_id: %s", self.workflow_id)

        except (ValueError, TypeError) as e:
            logger.error("ExecutionIO 데이터 JSON 변환 중 오류 발생: %s", str(e), exc_info=True)
        except (AttributeError, KeyError) as e:
            logger.error("ExecutionIO 데이터 저장 중 DB 오류 발생: %s", str(e), exc_info=True)


class WorkflowExecutionManager:
    """
    워크플로우 실행을 관리하는 싱글톤 클래스
    스레드 풀과 실행 상태를 중앙에서 관리합니다.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.executor_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=4,  # 동시 실행 가능한 워크플로우 수
            thread_name_prefix="workflow-executor"
        )
        self.active_executions: Dict[str, AsyncWorkflowExecutor] = {}
        self._execution_lock = threading.Lock()
        self._initialized = True

        logger.info("WorkflowExecutionManager 초기화 완료. 최대 동시 실행: 4개")

    def create_executor(self,
                       workflow_data: Dict[str, Any],
                       db_manager=None,
                       interaction_id: Optional[str] = None,
                       user_id: Optional[int] = None,
                       expected_output: Optional[str] = None,
                       test_mode: Optional[bool] = False) -> AsyncWorkflowExecutor:
        """새로운 비동기 워크플로우 실행기를 생성합니다."""
        executor = AsyncWorkflowExecutor(
            workflow_data=workflow_data,
            db_manager=db_manager,
            interaction_id=interaction_id,
            user_id=user_id,
            executor_pool=self.executor_pool,
            expected_output=expected_output,
            test_mode=test_mode
        )

        # 실행 ID 생성 (interaction_id + workflow_id 조합)
        execution_id = f"{interaction_id}_{workflow_data['workflow_id']}_{user_id}"

        with self._execution_lock:
            self.active_executions[execution_id] = executor

        logger.info(f"새로운 워크플로우 실행기 생성: {execution_id}")
        return executor

    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """특정 실행의 상태를 조회합니다."""
        with self._execution_lock:
            executor = self.active_executions.get(execution_id)
            if executor:
                return executor.get_execution_status()
        return None

    def get_all_execution_status(self) -> Dict[str, Dict[str, Any]]:
        """모든 활성 실행의 상태를 조회합니다."""
        with self._execution_lock:
            return {
                execution_id: executor.get_execution_status()
                for execution_id, executor in self.active_executions.items()
            }

    def cleanup_completed_executions(self):
        """완료된 실행들을 정리합니다."""
        with self._execution_lock:
            completed_ids = []
            for execution_id, executor in self.active_executions.items():
                status = executor.get_execution_status()
                if status['status'] in ['completed', 'error']:
                    completed_ids.append(execution_id)

            for execution_id in completed_ids:
                del self.active_executions[execution_id]
                logger.info(f"완료된 실행 정리: {execution_id}")

    def shutdown(self):
        """실행 매니저를 종료합니다."""
        logger.info("WorkflowExecutionManager 종료 중...")
        self.executor_pool.shutdown(wait=True)
        with self._execution_lock:
            self.active_executions.clear()
        logger.info("WorkflowExecutionManager 종료 완료")

# 글로벌 실행 매니저 인스턴스
execution_manager = WorkflowExecutionManager()

import asyncio
import concurrent.futures
import inspect
import logging
import json
import threading
import time
import queue
import os
from collections import deque
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, Any, Generator, List, Optional, AsyncGenerator
from editor.node_composer import NODE_CLASS_REGISTRY
from service.monitoring.performance_logger import PerformanceLogger
from service.database.models.executor import ExecutionIO

# WorkflowExecutor를 위한 helper 함수들

def extract_json_from_code_block(text: str) -> str:
    """
    마크다운 코드 블록에서 JSON 내용을 추출합니다.
    ```json...``` 또는 ```...``` 형태의 코드 블록을 처리합니다.
    agent_helper.py의 강건한 로직을 적용합니다.

    Args:
        text: 처리할 텍스트

    Returns:
        추출된 JSON 내용 또는 원본 텍스트
    """
    import re

    clean_data = text.strip()

    # 여러 패턴을 순차적으로 시도 (agent_helper.py와 동일)
    code_block_patterns = [
        r'```json\s*\n(.*?)\n```',  # ```json\n...\n```
        r'```json\s+(.*?)```',      # ```json ... ```
        r'```\s*\n(.*?)\n```',      # ```\n...\n```
        r'```\s*(.*?)```',          # ``` ... ```
    ]

    for pattern in code_block_patterns:
        match = re.search(pattern, clean_data, re.DOTALL | re.IGNORECASE)
        if match:
            json_content = match.group(1).strip()
            # 추출된 내용이 JSON처럼 보이는지 확인
            if json_content.startswith(('{', '[')):
                return json_content

    return clean_data

def extract_embedded_json(text: str, logger=None) -> tuple[Any, bool]:
    """
    텍스트 내부에 임베디드된 JSON 객체를 찾아 추출합니다.
    에러 메시지나 일반 텍스트 안에 포함된 JSON을 처리합니다.

    Args:
        text: JSON이 포함된 텍스트
        logger: 로깅을 위한 logger 객체 (선택사항)

    Returns:
        (추출된 JSON 데이터, 성공 여부) 튜플
    """
    import re
    import json

    # 중괄호로 시작하는 JSON 객체 패턴 찾기
    # 가장 긴 매칭을 찾기 위해 여러 패턴 시도
    patterns = [
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # 중첩된 객체 포함
        r'\{[^}]+\}',  # 간단한 객체
    ]

    for pattern in patterns:
        matches = re.finditer(pattern, text, re.DOTALL)
        for match in matches:
            json_str = match.group(0)
            try:
                parsed = json.loads(json_str)
                if logger:
                    logger.info(f"임베디드 JSON 추출 성공: {parsed}")
                return parsed, True
            except (json.JSONDecodeError, ValueError):
                continue

    # 대괄호로 시작하는 JSON 배열도 시도
    array_pattern = r'\[[^\]]+\]'
    matches = re.finditer(array_pattern, text, re.DOTALL)
    for match in matches:
        json_str = match.group(0)
        try:
            parsed = json.loads(json_str)
            if logger:
                logger.info(f"임베디드 JSON 배열 추출 성공: {parsed}")
            return parsed, True
        except (json.JSONDecodeError, ValueError):
            continue

    return text, False

def parse_json_safely(text: str, logger=None) -> tuple[Any, bool]:
    """
    텍스트를 JSON으로 안전하게 파싱합니다.
    코드 블록으로 감싸진 JSON도 자동으로 처리합니다.
    agent_helper.py의 XgenJsonOutputParser와 동일한 강건한 로직을 적용합니다.

    Args:
        text: 파싱할 텍스트
        logger: 로깅을 위한 logger 객체 (선택사항)

    Returns:
        (파싱된 데이터, 성공 여부) 튜플
    """
    import json
    import re

    if not isinstance(text, str):
        return text, False

    # 1. 마크다운 코드 블록 제거 후 시도 (먼저 시도)
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
                parsed_data = json.loads(json_content)
                if logger:
                    logger.info(f"코드 블록 제거 후 JSON 파싱 성공: {parsed_data}")
                return parsed_data, True
            except (json.JSONDecodeError, ValueError):
                continue

    # 2. 전체 텍스트를 직접 JSON으로 파싱 시도
    try:
        parsed_data = json.loads(text.strip())
        if logger:
            logger.info(f"직접 JSON 파싱 성공: {parsed_data}")
        return parsed_data, True
    except (json.JSONDecodeError, ValueError) as e:
        if logger:
            logger.debug(f"직접 파싱 실패: {e}")

    # 3. 임베디드 JSON 추출 시도
    if logger:
        logger.info(f"임베디드 JSON 추출 시도 중...")

    embedded_result, is_embedded = extract_embedded_json(text, logger)
    if is_embedded:
        return embedded_result, True

    if logger:
        logger.info(f"모든 JSON 파싱 시도 실패, 원본 텍스트 반환")

    return text, False

def collect_generator_data(generator, logger=None) -> tuple[str, int]:
    """
    Generator 객체에서 모든 데이터를 수집하여 문자열로 결합합니다.

    Args:
        generator: 수집할 Generator 객체
        logger: 로깅을 위한 logger 객체 (선택사항)

    Returns:
        (수집된 데이터, 청크 수) 튜플

    Raises:
        StopIteration, GeneratorExit, RuntimeError, ValueError: Generator 처리 중 오류
    """
    chunks = []

    for chunk in generator:
        if chunk is not None:
            # 타입 안전성을 위해 명시적으로 문자열로 변환
            try:
                if isinstance(chunk, (str, bytes)):
                    # bytes는 디코딩
                    chunk_str = chunk.decode('utf-8') if isinstance(chunk, bytes) else chunk
                elif isinstance(chunk, (int, float, bool)):
                    chunk_str = str(chunk)
                elif isinstance(chunk, (dict, list)):
                    # JSON 직렬화 가능한 객체는 JSON 문자열로
                    try:
                        chunk_str = json.dumps(chunk, ensure_ascii=False)
                    except (TypeError, ValueError):
                        chunk_str = str(chunk)
                else:
                    chunk_str = str(chunk)

                chunks.append(chunk_str)
            except Exception as e:
                if logger:
                    logger.warning(f"청크 변환 중 오류 ({type(chunk)}): {e}, 빈 문자열로 대체")
                chunks.append("")

    collected_data = ''.join(chunks) if chunks else ""

    if logger:
        logger.info(f"Generator 수집 완료: 총 청크 수: {len(chunks)}, 결과 길이: {len(collected_data)}")

    return collected_data, len(chunks)

def process_generator_input(key: str, generator, logger=None) -> Any:
    """
    Generator 입력을 처리하여 적절한 데이터 형태로 변환합니다.
    JSON 파싱도 자동으로 시도합니다.

    Args:
        key: 입력 키 이름 (로깅용)
        generator: 처리할 Generator 객체
        logger: 로깅을 위한 logger 객체 (선택사항)

    Returns:
        처리된 데이터 (Dict, List 또는 문자열)
    """
    if logger:
        logger.info(f"Generator 입력 감지: {key} ({type(generator)})")

    try:
        # Generator에서 데이터 수집
        collected_data, chunk_count = collect_generator_data(generator, logger)

        # 타입 안전성 검증
        if not isinstance(collected_data, str):
            if logger:
                logger.warning(f"수집된 데이터가 문자열이 아님 ({type(collected_data)}), 문자열로 변환")
            collected_data = str(collected_data) if collected_data is not None else ""

        # 빈 문자열 처리
        if not collected_data or collected_data.strip() == "":
            if logger:
                logger.info(f"Generator에서 빈 데이터 수집: {key}")
            return ""

        # JSON 파싱 시도
        parsed_data, is_json = parse_json_safely(collected_data, logger)

        if is_json:
            return parsed_data
        else:
            if logger:
                preview = collected_data[:100] if len(collected_data) > 100 else collected_data
                logger.info(f"문자열로 처리: {key} = {preview}...")
            return collected_data

    except (StopIteration, GeneratorExit, RuntimeError, ValueError) as e:
        error_msg = f"Generator 수집 오류: {str(e)}"
        if logger:
            logger.error(f"Generator 수집 중 오류: {key}, {e}")
        return error_msg
    except Exception as e:
        # 예상치 못한 오류에 대한 안전망
        error_msg = f"예상치 못한 Generator 처리 오류: {str(e)}"
        if logger:
            logger.error(f"예상치 못한 오류 발생: {key}, {e}", exc_info=True)
        return error_msg

def clean_router_input_text(text: str) -> str:
    """
    RouterNode 입력 텍스트에서 불필요한 태그들을 제거합니다.

    Args:
        text: 정리할 텍스트

    Returns:
        정리된 텍스트
    """
    import re

    if not isinstance(text, str):
        return text

    # 각종 태그들 제거
    text = re.sub(r"<FEEDBACK_(LOOP|RESULT|STATUS)>.*?</FEEDBACK_(LOOP|RESULT|STATUS)>", '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<TODO_DETAILS>.*?</TODO_DETAILS>", '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<TOOLUSELOG>.*?</TOOLUSELOG>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<TOOLOUTPUTLOG>.*?</TOOLOUTPUTLOG>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # 여러 개의 연속된 공백/줄바꿈 정리
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    return text

def get_route_key_from_data(data: Any, routing_criteria: str, logger_instance=None) -> str:
    """
    RouterNode의 데이터와 라우팅 기준을 바탕으로 라우팅 키를 결정합니다.
    Boolean과 String 값에 대해 강건한 처리를 제공합니다.
    문자열 데이터인 경우 JSON 파싱을 시도합니다.
    에러 메시지 안에 포함된 JSON도 추출합니다.
    """
    try:
        if not routing_criteria or not isinstance(routing_criteria, str) or routing_criteria.strip() == "":
            if logger_instance:
                logger_instance.warning(" -> 라우팅 기준이 없거나 유효하지 않음, 'default' 반환")
            return "default"

        # 데이터가 None인 경우
        if data is None:
            if logger_instance:
                logger_instance.warning(" -> 라우팅 데이터가 None, 'default' 반환")
            return "default"

        # 데이터가 문자열인 경우 JSON 파싱 시도
        if isinstance(data, str):
            if logger_instance:
                logger_instance.info(" -> 라우팅 데이터가 문자열입니다. JSON 파싱 시도 중...")

            # 빈 문자열 체크
            if data.strip() == "":
                if logger_instance:
                    logger_instance.warning(" -> 빈 문자열 데이터, 'default' 반환")
                return "default"

            # JSON 파싱 시도 (코드 블록 및 임베디드 JSON 포함)
            parsed_data, is_json = parse_json_safely(data, logger_instance)
            if is_json and isinstance(parsed_data, dict):
                if logger_instance:
                    logger_instance.info(" -> JSON 파싱 성공, Dict로 변환됨")
                data = parsed_data
            else:
                if logger_instance:
                    logger_instance.warning(" -> JSON 파싱 실패 또는 Dict가 아님")
                    # 에러 메시지인 경우 특별 처리
                    if "오류" in data or "error" in data.lower() or "exception" in data.lower():
                        logger_instance.warning(" -> 에러 메시지로 판단됨, 'error' 라우팅 키 반환")
                        return "error"
                    logger_instance.warning(" -> 'default' 반환")
                return "default"

        if not isinstance(data, dict):
            if logger_instance:
                logger_instance.warning(f" -> 데이터가 Dict가 아님 ({type(data)}), 'default' 반환")
            return "default"

        routing_key = routing_criteria.strip()
        if routing_key not in data:
            if logger_instance:
                logger_instance.warning(f" -> 라우팅 키 '{routing_key}'가 데이터에 없음, 'default' 반환")
            return "default"

        route_value = data[routing_key]

        # Boolean 값에 대한 강건한 처리 (실제 Python bool 타입만)
        if isinstance(route_value, bool):
            return "true" if route_value else "false"

        # None 값 처리
        if route_value is None:
            return "null"

        # 숫자 타입 처리 (int, float)
        if isinstance(route_value, (int, float)):
            return str(route_value)

        # 문자열로 안전하게 변환
        try:
            str_value = str(route_value).strip()
        except Exception as e:
            if logger_instance:
                logger_instance.error(f" -> 라우팅 값 문자열 변환 실패: {e}, 'default' 반환")
            return "default"

        # 빈 문자열 처리
        if str_value == "":
            return "default"

        # Boolean-like 문자열들을 정규화 (대소문자 구분 없이)
        # 단, 순수 숫자 문자열은 제외
        str_lower = str_value.lower()

        # 순수 숫자 문자열인지 확인
        try:
            float(str_value)
            # 숫자 문자열이면 소문자로만 변환하여 반환
            return str_lower
        except ValueError:
            # 숫자가 아닌 문자열에 대해서만 Boolean-like 변환 적용
            if str_lower in ["true", "yes", "on", "enabled"]:
                return "true"
            elif str_lower in ["false", "no", "off", "disabled"]:
                return "false"

            # 일반 문자열은 소문자로 정규화하여 반환
            return str_lower

    except Exception as e:
        # 예상치 못한 오류에 대한 최종 안전망
        if logger_instance:
            logger_instance.error(f" -> 라우팅 키 결정 중 예상치 못한 오류: {e}, 'default' 반환", exc_info=True)
        return "default"

logger = logging.getLogger('Async-Workflow-Executor')

# 환경변수에서 타임존 가져오기 (기본값: 서울 시간)
TIMEZONE = ZoneInfo(os.getenv('TIMEZONE', 'Asia/Seoul'))

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

    def _get_current_time(self) -> datetime:
        """현재 시간을 타임존이 적용된 datetime으로 반환"""
        return datetime.now(TIMEZONE)

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

    def _is_port_match(self, port_id: str, route_key: str) -> bool:
        """
        포트 ID와 라우팅 키가 매칭되는지 강건하게 확인합니다.
        Boolean, 문자열, 숫자 등 다양한 타입에 대해 유연한 매칭을 제공합니다.
        """
        if not port_id or not route_key:
            return False

        # 정확한 매칭 (대소문자 구분 없음)
        if port_id.lower().strip() == route_key.lower().strip():
            return True

        # Boolean 값들에 대한 추가 매칭
        port_lower = port_id.lower().strip()
        route_lower = route_key.lower().strip()

        # true 계열 매칭
        true_variants = ["true", "1", "yes", "on", "enabled", "t", "y"]
        if port_lower in true_variants and route_lower in true_variants:
            return True

        # false 계열 매칭
        false_variants = ["false", "0", "no", "off", "disabled", "f", "n"]
        if port_lower in false_variants and route_lower in false_variants:
            return True

        # 숫자 매칭 (문자열로 된 숫자와 실제 숫자)
        try:
            port_num = float(port_id)
            route_num = float(route_key)
            return port_num == route_num
        except (ValueError, TypeError):
            pass

        return False

    def _exclude_node_and_descendants(self, node_id: str, excluded_nodes: set) -> None:
        """
        RouterNode에서 선택되지 않은 경로의 노드와 그 후속 노드들을 제외합니다.
        DFS를 사용하여 재귀적으로 모든 후속 노드들을 찾습니다.
        """
        if node_id in excluded_nodes:
            return  # 이미 제외된 노드면 스킵

        excluded_nodes.add(node_id)

        # 이 노드에서 나가는 모든 edge를 찾아 후속 노드들도 제외
        outgoing_edges = [edge for edge in self.edges if edge['source']['nodeId'] == node_id]
        for edge in outgoing_edges:
            target_node_id = edge['target']['nodeId']
            self._exclude_node_and_descendants(target_node_id, excluded_nodes)

    def _execute_workflow_sync(self) -> Generator[Any, None, None]:
        """
        동기 워크플로우 실행 (기존 로직과 동일)
        이 메서드는 스레드 풀에서 실행됩니다.
        """
        self._execution_status = "running"
        self._start_time = self._get_current_time()

        try:
            self._build_graph()
            execution_order: List[str] = self._topological_sort()

            node_outputs: Dict[str, Dict[str, Any]] = {}
            start_node_data: Optional[Dict[str, Any]] = None
            # 라우팅으로 인해 제외된 노드들을 추적
            excluded_nodes: set = set()

            logger.info("--- 워크플로우 실행 시작 ---")
            logger.info(f"실행 순서: {' -> '.join(execution_order)}")

            streaming_output_started = False

            for node_id in execution_order:
                # 라우팅으로 인해 제외된 노드는 건너뛰기
                if node_id in excluded_nodes:
                    logger.info(" -> 노드 '%s'는 라우팅으로 인해 제외되어 건너뜁니다.", node_id)
                    continue

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

                # RouterNode를 위한 특별한 전처리: Generator 입력 수집
                function_id: str = node_info['data']['functionId']
                if function_id == 'router':
                    logger.info(" -> RouterNode 감지: Generator 입력 확인 중...")

                    # 모든 입력 값에서 generator가 있는지 확인하고 수집
                    processed_kwargs = {}
                    for key, value in kwargs.items():
                        try:
                            if hasattr(value, '__iter__') and hasattr(value, '__next__'):
                                # 헬퍼 함수를 사용하여 Generator 처리
                                processed_value = process_generator_input(key, value, logger)

                                # 타입 안전성 재확인
                                if processed_value is None:
                                    logger.warning(f" -> Generator 처리 결과가 None: {key}, 빈 문자열로 대체")
                                    processed_value = ""

                                # RouterNode 전용 텍스트 정리 (마지막 전처리)
                                if isinstance(processed_value, str):
                                    processed_value = clean_router_input_text(processed_value)
                                    logger.info(f" -> RouterNode 텍스트 정리 완료: {key}")

                                processed_kwargs[key] = processed_value
                            else:
                                # 일반 값 처리
                                if value is None:
                                    logger.warning(f" -> 입력 값이 None: {key}, 빈 문자열로 대체")
                                    processed_kwargs[key] = ""
                                elif isinstance(value, str):
                                    # 먼저 텍스트 정리
                                    cleaned_value = clean_router_input_text(value)

                                    # JSON 형태로 보이는 경우에만 파싱 시도 (중괄호나 대괄호로 시작)
                                    if cleaned_value.strip().startswith(('{', '[')):
                                        logger.info(f" -> RouterNode JSON 형태 감지: {key}, 파싱 시도")
                                        parsed_value, is_json = parse_json_safely(cleaned_value, logger)
                                        if is_json:
                                            logger.info(f" -> RouterNode JSON 파싱 성공: {key}")
                                            processed_kwargs[key] = parsed_value
                                        else:
                                            processed_kwargs[key] = cleaned_value
                                    else:
                                        # 일반 문자열은 정리만 하고 그대로 전달
                                        logger.info(f" -> RouterNode 일반 문자열: {key}")
                                        processed_kwargs[key] = cleaned_value
                                elif isinstance(value, (dict, list)):
                                    # Dict나 List는 그대로 전달
                                    processed_kwargs[key] = value
                                elif isinstance(value, (int, float, bool)):
                                    # 기본 타입은 그대로 전달
                                    processed_kwargs[key] = value
                                else:
                                    # 기타 타입은 안전하게 문자열로 변환
                                    try:
                                        processed_kwargs[key] = str(value)
                                        logger.info(f" -> 알 수 없는 타입을 문자열로 변환: {key} ({type(value)})")
                                    except Exception as conv_err:
                                        logger.error(f" -> 값 변환 실패: {key}, {conv_err}, 빈 문자열로 대체")
                                        processed_kwargs[key] = ""

                        except Exception as e:
                            logger.error(f" -> RouterNode 전처리 중 오류 발생: {key}, {e}, 빈 문자열로 대체", exc_info=True)
                            processed_kwargs[key] = ""

                    kwargs = processed_kwargs
                    logger.info(" -> RouterNode 전처리 완료")

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

                # functionId에 따른 특별 처리 (이미 위에서 선언됨)

                if function_id == 'startnode':
                    start_node_data = {
                        'node_id': node_id,
                        'node_name': node_info['data']['nodeName'],
                        'inputs': kwargs,
                        'result': result
                    }
                    logger.info(f" -> startnode 데이터 수집: {start_node_data}")

                if function_id == 'endnode':
                    is_generator = inspect.isgenerator(result)

                    if is_generator:
                        logger.info(f" -> endnode 스트리밍 출력 시작.")
                        streaming_output_started = True
                        collected_output = []
                        for chunk in result:
                            collected_output.append(str(chunk))
                            yield chunk

                        final_streaming_result = ''.join(collected_output)
                        end_node_data = {'node_id': node_id, 'node_name': node_info['data']['nodeName'], 'inputs': kwargs, 'result': final_streaming_result}
                        input_data_for_db = start_node_data if start_node_data else {}
                        self._save_execution_io(input_data_for_db, end_node_data)

                        logger.info("\n--- 워크플로우 스트리밍 실행 완료 ---")
                        self._execution_status = "completed"
                        self._end_time = self._get_current_time()
                        return
                    else:
                        end_node_result_for_db = result
                        end_node_data = {'node_id': node_id, 'node_name': node_info['data']['nodeName'], 'inputs': kwargs, 'result': end_node_result_for_db}
                        input_data_for_db = start_node_data if start_node_data else {}
                        self._save_execution_io(input_data_for_db, end_node_data)
                        logger.info(f" -> endnode 완료. 최종 출력: {result}")
                        node_outputs[node_id] = {'result': result}

                if function_id != 'endnode':
                    if not node_info['data']['outputs']:
                        logger.info(" -> 출력 없음. (결과: %s)", result)
                        node_outputs[node_id] = {}
                        continue

                    # RouterNode의 특별한 처리
                    if function_id == 'router':
                        logger.info(" -> RouterNode 결과 처리 중...")

                        # routing_criteria 파라미터 찾기
                        routing_criteria = ""
                        for param in node_info['data'].get('parameters', []):
                            if param.get('id') == 'routing_criteria':
                                routing_criteria = param.get('value', '')
                                break

                        # 라우팅 키 결정 (logger 전달)
                        routed_port_id = get_route_key_from_data(result, routing_criteria, logger)
                        output_ports = node_info['data']['outputs']

                        logger.info(" -> 라우팅 기준: '%s', 결정된 포트: '%s'", routing_criteria, routed_port_id)

                        # 선택된 포트와 선택되지 않은 포트들 분리
                        selected_port = None
                        unselected_ports = []

                        for port in output_ports:
                            port_id = port.get('id', '').strip()

                            # 강건한 포트 ID 매칭
                            if self._is_port_match(port_id, routed_port_id):
                                selected_port = port
                            else:
                                unselected_ports.append(port)

                        # 선택되지 않은 포트들로 연결된 노드들을 excluded_nodes에 추가
                        for port in unselected_ports:
                            port_id = port.get('id')
                            # 이 포트와 연결된 모든 다음 노드들을 찾아서 제외
                            connected_edges = [edge for edge in self.edges
                                             if edge['source']['nodeId'] == node_id and edge['source']['portId'] == port_id]

                            for edge in connected_edges:
                                target_node_id = edge['target']['nodeId']
                                # 해당 노드와 그 후속 노드들을 모두 제외
                                self._exclude_node_and_descendants(target_node_id, excluded_nodes)
                                logger.info(" -> 라우팅으로 인해 노드 '%s' 및 후속 노드들 제외", target_node_id)

                        # 선택된 포트로만 데이터 전달
                        if selected_port:
                            actual_port_id = selected_port.get('id')
                            node_outputs[node_id] = {actual_port_id: result}
                            logger.info(" -> 라우팅 성공: 포트 '%s'로 데이터 전달", actual_port_id)
                        else:
                            # 매칭되는 포트가 없으면 default 포트 또는 첫 번째 포트 사용
                            default_port = None
                            for port in output_ports:
                                if port.get('id') == 'default':
                                    default_port = port
                                    break

                            if default_port:
                                node_outputs[node_id] = {'default': result}
                                logger.info(" -> 기본 포트 'default'로 데이터 전달")
                            elif output_ports:
                                first_port_id = output_ports[0].get('id')
                                if first_port_id:
                                    node_outputs[node_id] = {first_port_id: result}
                                    logger.info(" -> 첫 번째 포트 '%s'로 데이터 전달", first_port_id)
                                else:
                                    raise ValueError(f"RouterNode '{node_info['data']['nodeName']}'의 출력 포트에 ID가 정의되어 있지 않습니다.")
                            else:
                                logger.warning(" -> RouterNode에 출력 포트가 없습니다.")
                                node_outputs[node_id] = {}

                        logger.info(" -> 라우팅 완료: %s", node_outputs[node_id])
                    else:
                        # 일반 노드의 기존 처리 로직
                        if not node_info['data']['outputs'][0].get('id'):
                            raise ValueError(f"노드 '{node_info['data']['nodeName']}'의 첫 번째 출력 포트에 ID가 정의되어 있지 않습니다.")

                        output_port_id: str = node_info['data']['outputs'][0]['id']
                        node_outputs[node_id] = {output_port_id: result}
                        logger.info(" -> 출력: %s", node_outputs[node_id])

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
            self._end_time = self._get_current_time()

        except Exception as e:
            self._execution_status = "error"
            self._error_message = str(e)
            self._end_time = self._get_current_time()
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
            end_time = self._end_time if self._end_time else self._get_current_time()
            execution_time = (end_time - self._start_time).total_seconds()

        return {
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "interaction_id": self.interaction_id,
            "user_id": self.user_id,
            "status": self._execution_status,
            "error_message": self._error_message,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "end_time": self._end_time.isoformat() if self._end_time else None,
            "execution_time": execution_time,
            "current_timezone": str(TIMEZONE)
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

            logger.info("ExecutionIO 데이터가 성공적으로 저장되었습니다. workflow_id: %s, timezone: %s",
                       self.workflow_id, str(TIMEZONE))

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

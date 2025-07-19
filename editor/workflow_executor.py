from collections import deque
import logging
import json
from typing import Dict, Any, List, Optional
from editor.node_composer import NODE_CLASS_REGISTRY, run_discovery
from service.monitoring.performance_logger import PerformanceLogger

logger = logging.getLogger('Workflow-Executor')

class WorkflowExecutor:
    def __init__(self, workflow_data: Dict[str, Any], db_manager=None, interaction_id: Optional[str] = None):
        self.workflow_id: str = workflow_data['workflow_id']
        self.workflow_name: str = workflow_data['workflow_name']
        self.interaction_id: str = interaction_id or 'default'
        self.db_manager = db_manager
        self.nodes: Dict[str, Dict[str, Any]] = {node['id']: node for node in workflow_data['nodes']}
        self.edges: List[Dict[str, Any]] = workflow_data['edges']
        self.graph: Dict[str, List[str]] = {node_id: [] for node_id in self.nodes}
        self.in_degree: Dict[str, int] = {node_id: 0 for node_id in self.nodes}

        if not NODE_CLASS_REGISTRY:
            print("Node class registry is empty. Running discovery...")
            run_discovery()

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
        queue: deque[str] = deque([node_id for node_id, degree in self.in_degree.items() if degree == 0])
        sorted_nodes: List[str] = []

        while queue:
            node_id: str = queue.popleft()
            sorted_nodes.append(node_id)

            for neighbor_id in self.graph.get(node_id, []):
                self.in_degree[neighbor_id] -= 1
                if self.in_degree[neighbor_id] == 0:
                    queue.append(neighbor_id)

        if len(sorted_nodes) != len(self.nodes):
            raise ValueError("그래프에 순환(cycle)이 존재하여 워크플로우를 실행할 수 없습니다.")

        return sorted_nodes

    def execute_workflow(self) -> Dict[str, Any]:
        """워크플로우를 실행하고 최종 결과물을 반환합니다."""
        self._build_graph()
        execution_order: List[str] = self._topological_sort()

        node_outputs: Dict[str, Dict[str, Any]] = {}
        start_node_data: Optional[Dict[str, Any]] = None
        end_node_data: Optional[Dict[str, Any]] = None

        print("--- 워크플로우 실행 시작 ---")
        print(f"실행 순서: {' -> '.join(execution_order)}")

        execution_final_outputs: Optional[Any] = None

        for node_id in execution_order:
            node_info: Dict[str, Any] = self.nodes[node_id]
            node_spec_id: str = node_info['data']['id']

            NodeClass = NODE_CLASS_REGISTRY.get(node_spec_id)
            if not NodeClass:
                raise ValueError(f"ID가 '{node_spec_id}'인 노드 클래스를 찾을 수 없습니다.")

            kwargs: Dict[str, Any] = {}
            connected_edges: List[Dict[str, Any]] = [edge for edge in self.edges if edge['target']['nodeId'] == node_id]
            for edge in connected_edges:
                source_node_id: str = edge['source']['nodeId']
                source_port_id: str = edge['source']['portId']
                target_port_id: str = edge['target']['portId']

                if source_node_id in node_outputs and source_port_id in node_outputs[source_node_id]:
                    kwargs[target_port_id] = node_outputs[source_node_id][source_port_id]

            if 'parameters' in node_info['data'] and node_info['data']['parameters']:
                for param in node_info['data']['parameters']:
                    param_key: Optional[str] = param.get('id')
                    param_value: Any = param.get('value')
                    if param_key and param_key not in kwargs:
                        kwargs[param_key] = param_value

            print(f"\n[실행] {node_info['data']['nodeName']} ({node_id})")
            print(f" -> 입력: {kwargs}")

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
                    node_name=node_name_for_logging,
                    db_manager=self.db_manager
                ) as perf_logger:

                    result = instance.execute(**kwargs)

                    # 노드 실행 완료 후 로그 기록
                    perf_logger.log(input_data=kwargs, output_data=result)
                    logger.info(" -> 완료. 결과: %s", str(result)[:100] + "..." if len(str(result)) > 100 else str(result))

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
                print(f" -> startnode 데이터 수집: {start_node_data}")

            elif function_id == 'endnode':
                # endnode의 경우 출력 데이터 기록 및 ExecutionIO 저장
                end_node_data = {
                    'node_id': node_id,
                    'node_name': node_info['data']['nodeName'],
                    'inputs': kwargs,
                    'result': result
                }

                # ExecutionIO에 저장할 데이터 준비
                input_data_for_db = start_node_data if start_node_data else {}
                output_data_for_db = end_node_data

                # ExecutionIO 저장
                self._save_execution_io(input_data_for_db, output_data_for_db)

                execution_final_outputs = result
                print(f" -> endnode 완료 및 ExecutionIO 저장. 최종 출력: {execution_final_outputs}")

            # 일반 노드 처리
            if function_id != 'endnode':
                if not node_info['data']['outputs']:
                    print(f" -> 출력 없음. (결과: {result})")
                    node_outputs[node_id] = {}
                    continue

                if not node_info['data']['outputs'][0].get('id'):
                    raise ValueError(f"노드 '{node_info['data']['nodeName']}'의 첫 번째 출력 포트에 ID가 정의되어 있지 않습니다.")

                output_port_id: str = node_info['data']['outputs'][0]['id']
                node_outputs[node_id] = {output_port_id: result}

                print(f" -> 출력: {node_outputs[node_id]}")

        if execution_final_outputs is not None:
            print("\n--- 워크플로우 실행 완료 ---")
            print("최종 출력:", execution_final_outputs)
            return execution_final_outputs

        else:
            print("\n--- 워크플로우 실행 완료 ---")
            print("최종 출력이 정의되지 않았습니다. 모든 노드의 중간 결과물을 반환합니다.")
            print("노드 출력 데이터:")
            return node_outputs

    def _save_execution_io(self, input_data: Dict[str, Any], output_data: Dict[str, Any]) -> None:
        """워크플로우의 입출력 데이터를 ExecutionIO 모델을 통해 DB에 저장합니다."""
        if not self.db_manager:
            logger.warning("DB manager가 없어 ExecutionIO 데이터를 저장할 수 없습니다.")
            return

        try:
            # JSON 형태로 변환하여 저장
            input_json = json.dumps(input_data, ensure_ascii=False)
            output_json = json.dumps(output_data, ensure_ascii=False)

            # DB 타입에 따른 쿼리 준비
            db_type = self.db_manager.config_db_manager.db_type
            if db_type == "postgresql":
                query = """
                    INSERT INTO execution_io (interaction_id, workflow_id, workflow_name, input_data, output_data, created_at)
                    VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                """
            else:  # SQLite
                query = """
                    INSERT INTO execution_io (interaction_id, workflow_id, workflow_name, input_data, output_data, created_at)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """

            self.db_manager.config_db_manager.execute_query(
                query,
                (self.interaction_id, self.workflow_id, self.workflow_name, input_json, output_json)
            )

            logger.info("ExecutionIO 데이터가 성공적으로 저장되었습니다. workflow_id: %s", self.workflow_id)

        except (ValueError, TypeError) as e:
            logger.error("ExecutionIO 데이터 JSON 변환 중 오류 발생: %s", str(e), exc_info=True)
        except (AttributeError, KeyError) as e:
            logger.error("ExecutionIO 데이터 저장 중 DB 오류 발생: %s", str(e), exc_info=True)

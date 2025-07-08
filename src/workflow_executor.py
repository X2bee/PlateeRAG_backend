from collections import deque
from typing import Dict, Any, List
from src.node_composer import NODE_CLASS_REGISTRY, run_discovery
from src.model.node import Port, Parameter, NodeSpec

class WorkflowExecutor:
    def __init__(self, workflow_data: Dict[str, Any]):
        self.nodes = {node['id']: node for node in workflow_data['nodes']}
        self.edges = workflow_data['edges']
        self.graph: Dict[str, List[str]] = {node_id: [] for node_id in self.nodes}
        self.in_degree: Dict[str, int] = {node_id: 0 for node_id in self.nodes}
        
        if not NODE_CLASS_REGISTRY:
            print("Node class registry is empty. Running discovery...")
            run_discovery()

    def _build_graph(self):
        """워크플로우 데이터로부터 그래프와 진입 차수를 계산합니다."""
        for edge in self.edges:
            source_id = edge['source']['nodeId']
            target_id = edge['target']['nodeId']
            if source_id in self.graph and target_id in self.graph:
                self.graph[source_id].append(target_id)
                self.in_degree[target_id] += 1

    def _topological_sort(self) -> List[str]:
        """위상 정렬을 사용하여 노드의 실행 순서를 결정합니다."""
        queue = deque([node_id for node_id, degree in self.in_degree.items() if degree == 0])
        sorted_nodes = []
        
        while queue:
            node_id = queue.popleft()
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
        execution_order = self._topological_sort()
        
        # 각 노드의 출력 결과를 저장하는 딕셔너리
        node_outputs: Dict[str, Dict[str, Any]] = {}

        print("--- 워크플로우 실행 시작 ---")
        print(f"실행 순서: {' -> '.join(execution_order)}")
        
        execution_final_outputs = None

        for node_id in execution_order:
            node_info = self.nodes[node_id]
            node_spec_id = node_info['data']['id']
            
            NodeClass = NODE_CLASS_REGISTRY.get(node_spec_id)
            if not NodeClass:
                raise ValueError(f"ID가 '{node_spec_id}'인 노드 클래스를 찾을 수 없습니다.")

            kwargs = {}
            connected_edges = [edge for edge in self.edges if edge['target']['nodeId'] == node_id]
            for edge in connected_edges:
                source_node_id = edge['source']['nodeId']
                source_port_id = edge['source']['portId']
                target_port_id = edge['target']['portId']
                
                if source_node_id in node_outputs and source_port_id in node_outputs[source_node_id]:
                    kwargs[target_port_id] = node_outputs[source_node_id][source_port_id]

            if 'parameters' in node_info['data'] and node_info['data']['parameters']:
                for param in node_info['data']['parameters']:
                    param_key = param.get('id')
                    param_value = param.get('value')
                    if param_key and param_key not in kwargs:
                        kwargs[param_key] = param_value
            
            print(f"\n[실행] {node_info['data']['nodeName']} ({node_id})")
            print(f" -> 입력: {kwargs}")
            
            # 노드 인스턴스 생성 및 실행
            instance = NodeClass()
            result = instance.execute(**kwargs)
            
            if node_info['data']['functionId'] == 'endnode':
                execution_final_outputs = result
                print(f" -> 최종 출력: {execution_final_outputs}")
            
            else:
                if not node_info['data']['outputs']:
                    print(f" -> 출력 없음. (결과: {result})")
                    node_outputs[node_id] = {}
                    continue
                
                if not node_info['data']['outputs'][0].get('id'):
                    raise ValueError(f"노드 '{node_info['data']['nodeName']}'의 첫 번째 출력 포트에 ID가 정의되어 있지 않습니다.")
                
                output_port_id = node_info['data']['outputs'][0]['id']
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
    

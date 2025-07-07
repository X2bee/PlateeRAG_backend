from collections import deque
from typing import Dict, Any, List
from src.node_composer import NODE_CLASS_REGISTRY, run_discovery

class WorkflowExecutor:
    def __init__(self, workflow_data: Dict[str, Any]):
        self.nodes = {node['id']: node for node in workflow_data['nodes']}
        self.edges = workflow_data['edges']
        self.graph: Dict[str, List[str]] = {node_id: [] for node_id in self.nodes}
        self.in_degree: Dict[str, int] = {node_id: 0 for node_id in self.nodes}
        
        # 노드 디스커버리를 실행하여 NODE_CLASS_REGISTRY를 채웁니다.
        # 실제 애플리케이션에서는 시작 시 한 번만 호출하는 것이 좋습니다.
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

        for node_id in execution_order:
            node_info = self.nodes[node_id]
            node_spec_id = node_info['data']['id']
            
            NodeClass = NODE_CLASS_REGISTRY.get(node_spec_id)
            if not NodeClass:
                raise ValueError(f"ID가 '{node_spec_id}'인 노드 클래스를 찾을 수 없습니다.")

            # 노드 실행에 필요한 입력값 준비
            kwargs = {}
            # 엣지로부터 입력값을 가져옴
            connected_edges = [edge for edge in self.edges if edge['target']['nodeId'] == node_id]
            for edge in connected_edges:
                source_node_id = edge['source']['nodeId']
                source_port_id = edge['source']['portId']
                target_port_id = edge['target']['portId']
                
                # 소스 노드의 출력값에서 필요한 값을 찾아 입력으로 연결
                if source_node_id in node_outputs and source_port_id in node_outputs[source_node_id]:
                    kwargs[target_port_id] = node_outputs[source_node_id][source_port_id]

            # TODO: 노드의 'parameters' 값을 kwargs에 추가하는 로직 (필요 시)

            print(f"\n[실행] {node_info['data']['nodeName']} ({node_id})")
            print(f" -> 입력: {kwargs}")
            
            # 노드 인스턴스 생성 및 실행
            instance = NodeClass()
            result = instance.execute(**kwargs)
            
            # 실행 결과를 node_outputs에 저장
            # 현재는 출력이 하나라고 가정, 다중 출력 시 수정 필요
            output_port_id = node_info['data']['outputs'][0]['id']
            node_outputs[node_id] = {output_port_id: result}
            
            print(f" -> 출력: {node_outputs[node_id]}")

        print("\n--- 워크플로우 실행 종료 ---")
        return node_outputs
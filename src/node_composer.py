import inspect
import json
import pkgutil
import importlib
from pathlib import Path
NODE_REGISTRY = []

TYPE_MAP = {
    int: "INT",
    float: "FLOAT",
    str: "STR",
}

def node(id: str, name: str, category: str = "Default"):
    def decorator(func):
        sig = inspect.signature(func)
        
        inputs = []
        parameters = []
        
        for param in sig.parameters.values():
            param_type = TYPE_MAP.get(param.annotation, "ANY")
            # 기본값이 없는 파라미터는 '입력(Input)' 포트로 간주
            if param.default is inspect.Parameter.empty:
                inputs.append({"id": param.name, "name": param.name.capitalize(), "type": param_type})
            # 기본값이 있는 파라미터는 '파라미터(Parameter)'로 간주
            else:
                parameters.append({
                    "id": f"p_{param.name}", 
                    "name": param.name.capitalize(), 
                    "value": param.default,
                    "type": param_type
                })

        # 반환값 타입 힌트를 '출력(Output)' 포트로 간주
        return_type = TYPE_MAP.get(sig.return_annotation, "ANY")
        outputs = [{"id": "output", "name": "Output", "type": return_type}]

        node_spec = {
            "id": id,
            "nodeName": name,
            "category": category,
            "inputs": inputs,
            "outputs": outputs,
            "parameters": parameters,
        }
        
        NODE_REGISTRY.append(node_spec)
        return func
    return decorator


def run_discovery():
    """'nodes' 디렉토리의 모든 모듈을 임포트하여 노드를 등록합니다."""
    nodes_dir = Path(__file__).parent / "nodes"
    # nodes 디렉토리의 모든 파이썬 파일을 순회하며 동적으로 임포트
    for (_, module_name, _) in pkgutil.iter_modules([str(nodes_dir)]):
        importlib.import_module(f"nodes.{module_name}")


def generate_json_spec(output_path="node_spec.json"):
    """등록된 노드 정보를 JSON 파일로 저장합니다."""
    
    # 카테고리별로 노드를 그룹화
    grouped_nodes = {}
    for node in NODE_REGISTRY:
        category = node.pop("category", "Default")
        if category not in grouped_nodes:
            grouped_nodes[category] = []
        grouped_nodes[category].append(node)
        
    # 프론트엔드 `NODE_DATA` 형식에 맞게 최종 데이터 구조화
    # 이 부분은 프론트엔드 형식에 맞게 자유롭게 커스터마이징 가능
    final_spec = [
        {
            "id": "dynamic-nodes",
            "name": "My Nodes",
            "icon": "LuWrench", # 예시 아이콘
            "categories": [
                {"id": cat_name.lower(), "name": cat_name, "nodes": nodes}
                for cat_name, nodes in grouped_nodes.items()
            ]
        }
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_spec, f, indent=4, ensure_ascii=False)
    print(f"✅ Node specification generated at {output_path}")
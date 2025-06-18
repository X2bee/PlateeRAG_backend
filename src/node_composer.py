# backend/node_discovery.py

import json
import pkgutil
import importlib
from pathlib import Path
from typing import Literal, List, Dict, Any

from node_config import CATEGORIES_LABEL_MAP, FUNCTION_LABEL_MAP

CATEGORIES_ID = Literal[*CATEGORIES_LABEL_MAP.keys()]
CATEGORIES_LABEL = Literal[*CATEGORIES_LABEL_MAP.values()]
FUNCTION_ID = Literal[*FUNCTION_LABEL_MAP.keys()]
FUNCTION_LABEL = Literal[*FUNCTION_LABEL_MAP.values()]

NODE_REGISTRY = []

class BaseNode:
    categoryId: CATEGORIES_ID
    functionId: FUNCTION_ID
    
    nodeId: str = "unimplemented"
    nodeName: str = "Unimplemented Node"
    
    inputs: List[Dict[str, Any]] = []
    outputs: List[Dict[str, Any]] = []
    parameters: List[Dict[str, Any]] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__name__ == 'BaseNode':
            return
            
        spec = {
            # 나중에 그룹화를 위해 모든 정보를 일단 저장
            "categoryId": cls.categoryId,
            "categoryName": cls.categoryName,
            "functionId": cls.functionId,
            "functionName": cls.functionName,
            # 프론트엔드 노드 레벨에서 사용하는 최종 데이터
            "id": cls.nodeId,
            "nodeName": cls.nodeName,
            "inputs": cls.inputs,
            "outputs": cls.outputs,
            "parameters": cls.parameters,
        }
        NODE_REGISTRY.append(spec)

    def execute(self, *args, **kwargs):
        raise NotImplementedError("모든 노드는 execute 메서드를 구현해야 합니다.")

def run_discovery():
    nodes_root_dir = Path(__file__).parent / "nodes"
    for module_info in pkgutil.walk_packages(path=[str(nodes_root_dir)], prefix='nodes.'):
        try:
            importlib.import_module(module_info.name)
        except Exception as e:
            print(f"Error importing module {module_info.name}: {e}")

def generate_json_spec(output_path="export_nodes.json"):
    """등록된 노드 정보를 새로운 프론트엔드 형식에 맞춰 JSON 파일로 저장합니다."""
    
    categories = {}
    for node_spec in NODE_REGISTRY:
        cat_id = node_spec["categoryId"]
        if cat_id not in categories:
            categories[cat_id] = {
                "categoryId": cat_id,
                "categoryName": node_spec["categoryName"],
                "icon": "LuWrench",
                "functions": {}
            }
        
        # 2. Category 내부에서 Function별로 그룹화
        func_id = node_spec["functionId"]
        if func_id not in categories[cat_id]["functions"]:
            categories[cat_id]["functions"][func_id] = {
                "functionId": func_id,
                "functionName": node_spec["functionName"],
                "nodes": []
            }
        
        # 노드 정보 추가
        node_info = {k: v for k, v in node_spec.items() if k not in ['categoryId', 'categoryName', 'functionId', 'functionName']}
        categories[cat_id]["functions"][func_id]["nodes"].append(node_info)

    # 최종 포맷으로 변환
    final_spec = []
    for cat in categories.values():
        cat["functions"] = list(cat["functions"].values())
        final_spec.append(cat)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_spec, f, indent=4, ensure_ascii=False)
    print(f"✅ Node specification generated with {len(NODE_REGISTRY)} nodes.")
import os
import json
import pkgutil
import importlib
from pathlib import Path
from typing import List, Dict, Any, Type
from abc import ABC, abstractmethod
from src.model.node import NodeSpec, Port, Parameter

from src.node_config import CATEGORIES_LABEL_MAP, FUNCTION_LABEL_MAP

NODE_REGISTRY = []
NODE_CLASS_REGISTRY: Dict[str, Type['Node']] = {}

class Node(ABC):
    categoryId: str = "Default"
    functionId: str = "Default"
    nodeId: str = "Default"
    nodeName: str = "Default"
    inputs: List[Port] = []
    outputs: List[Port] = []
    parameters: List[Parameter] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__name__ == 'Node':
            return
        
        is_valid = True
        
        if not hasattr(cls, 'categoryId') or cls.categoryId not in CATEGORIES_LABEL_MAP:
            is_valid = False
            allowed = list(CATEGORIES_LABEL_MAP.keys())
            print(
                f"[Node Registration Failed] Node '{cls.__name__}': 'categoryId' is invalid.\n"
                f"-> Assigned value: '{getattr(cls, 'categoryId', 'not defined')}' (Allowed values: {allowed})\n"
            )

        # 2. functionId 유효성 검사
        if not hasattr(cls, 'functionId') or cls.functionId not in FUNCTION_LABEL_MAP:
            is_valid = False
            allowed = list(FUNCTION_LABEL_MAP.keys())
            print(
                f"[Node Registration Failed] Node '{cls.__name__}': 'functionId' is invalid.\n"
                f"-> Assigned value: '{getattr(cls, 'functionId', 'not defined')}' (Allowed values: {allowed})\n"
            )
            
        if not is_valid:
            return
        
        category_name = CATEGORIES_LABEL_MAP.get(cls.categoryId, "Unknown Category")
        function_name = FUNCTION_LABEL_MAP.get(cls.functionId, "Unknown Function")

        spec: NodeSpec = {
            "categoryId": cls.categoryId,
            "categoryName": category_name,
            "functionId": cls.functionId,
            "functionName": function_name,
            "id": cls.nodeId,
            "nodeName": cls.nodeName,
            "inputs": cls.inputs,
            "outputs": cls.outputs,
            "parameters": cls.parameters
        }
        NODE_REGISTRY.append(spec)
        NODE_CLASS_REGISTRY[cls.nodeId] = cls

        print(f" -> 노드 '{cls.nodeName}' 등록 완료.")

    @abstractmethod
    def execute(self, *args, **kwargs):
        """desc"""
        raise NotImplementedError("모든 노드는 execute 메서드를 구현해야 합니다.")

def run_discovery() -> None:
    """Desc"""
    nodes_root_dir = Path(__file__).parent / "nodes"
    for module_info in pkgutil.walk_packages(path=[str(nodes_root_dir)], prefix='src.nodes.'):
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
        
        func_id = node_spec["functionId"]
        if func_id not in categories[cat_id]["functions"]:
            categories[cat_id]["functions"][func_id] = {
                "functionId": func_id,
                "functionName": node_spec["functionName"],
                "nodes": []
            }
        
        node_info = {k: v for k, v in node_spec.items() if k not in ['categoryId', 'categoryName', 'functionName']}
        categories[cat_id]["functions"][func_id]["nodes"].append(node_info)

    final_spec = []
    for cat in categories.values():
        cat["functions"] = list(cat["functions"].values())
        final_spec.append(cat)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_spec, f, indent=4, ensure_ascii=False)
    print(f"\n✅ Node specification generated with {len(NODE_REGISTRY)} nodes.")
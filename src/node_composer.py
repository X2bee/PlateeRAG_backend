import os
import json
import pkgutil
import importlib
from pathlib import Path
from typing import List, Dict, Any

from node_config import CATEGORIES_LABEL_MAP, FUNCTION_LABEL_MAP

NODE_REGISTRY = []

class BaseNode:
    categoryId: str = "Default"
    functionId: str = "Default"
    nodeId: str = "Default"
    nodeName: str = "Default"
    inputs: List[Dict[str, Any]] = []
    outputs: List[Dict[str, Any]] = []
    parameters: List[Dict[str, Any]] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__name__ == 'BaseNode':
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

        spec = {
            "categoryId": cls.categoryId,
            "categoryName": category_name,
            "functionId": cls.functionId,
            "functionName": function_name,
            "id": cls.nodeId,
            "nodeName": cls.nodeName,
            "inputs": cls.inputs,
            "outputs": cls.outputs,
            "parameters": cls.parameters,
        }
        NODE_REGISTRY.append(spec)
        print(f" -> 노드 '{cls.nodeName}' 등록 완료.")

    def execute(self, *args, **kwargs):
        """desc"""
        raise NotImplementedError("모든 노드는 execute 메서드를 구현해야 합니다.")

def run_discovery() -> None:
    """Desc"""
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
        
        func_id = node_spec["functionId"]
        if func_id not in categories[cat_id]["functions"]:
            categories[cat_id]["functions"][func_id] = {
                "functionId": func_id,
                "functionName": node_spec["functionName"],
                "nodes": []
            }
        
        node_info = {k: v for k, v in node_spec.items() if k not in ['categoryId', 'categoryName', 'functionId', 'functionName']}
        categories[cat_id]["functions"][func_id]["nodes"].append(node_info)

    final_spec = []
    for cat in categories.values():
        cat["functions"] = list(cat["functions"].values())
        final_spec.append(cat)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_spec, f, indent=4, ensure_ascii=False)
    print(f"\n✅ Node specification generated with {len(NODE_REGISTRY)} nodes.")


if __name__ == "__main__":
    print("\n\n--- 🚀 실행 테스트 시작 🚀 ---")
    run_discovery()
    print(f"\n✅ Step 4: 노드 등록 결과 확인")
    print(f" -> 총 {len(NODE_REGISTRY)}개의 노드가 레지스트리에 등록되었습니다.")
    print(" -> 등록된 노드 상세 정보:")
    print(json.dumps(NODE_REGISTRY, indent=2, ensure_ascii=False))

    output_filename = "test_export_nodes.json"
    print(f"\n✅ Step 5: '{output_filename}' 파일 생성 시작")
    generate_json_spec(output_path=output_filename) # JSON 생성 함수 호출

    print(f"\n✅ Step 6: 생성된 '{output_filename}' 파일 내용 확인")
    try:
        with open(output_filename, "r", encoding="utf-8") as f:
            generated_json_content = f.read()
            print("--- 파일 내용 시작 ---")
            print(generated_json_content)
            print("--- 파일 내용 종료 ---")
        
        # 테스트 후 생성된 파일 삭제
        # os.remove(output_filename)
    except FileNotFoundError:
        print(f" -> 🔴 에러: JSON 파일 '{output_filename}'이 생성되지 않았습니다.")
    
    print("\n--- ✨ 실행 테스트 성공적으로 종료 ✨ ---")
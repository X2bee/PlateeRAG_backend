import os
import sys
import json
import pkgutil
import importlib
from pathlib import Path
from typing import List, Dict, Any, Type
from abc import ABC, abstractmethod
from editor.model.node import NodeSpec, Port, Parameter, CATEGORIES_LABEL_MAP, FUNCTION_LABEL_MAP, ICON_LABEL_MAP, validate_parameters

NODE_REGISTRY = []
NODE_CLASS_REGISTRY: Dict[str, Type['Node']] = {}
CURRENT_USER_ID: str = None  # 현재 discovery 과정에서 사용할 user_id 저장

def get_node_registry() -> List[NodeSpec]:
    """전역 NODE_REGISTRY에 접근하는 함수"""
    return NODE_REGISTRY

def get_node_class_registry() -> Dict[str, Type['Node']]:
    """전역 NODE_CLASS_REGISTRY에 접근하는 함수"""
    return NODE_CLASS_REGISTRY

def get_node_by_id(node_id: str) -> Type['Node']:
    return NODE_CLASS_REGISTRY.get(node_id)

def clear_registries():
    global NODE_REGISTRY, NODE_CLASS_REGISTRY, CURRENT_USER_ID
    NODE_REGISTRY = []
    NODE_CLASS_REGISTRY = {}
    CURRENT_USER_ID = None

class Node(ABC):
    categoryId: str = "Default"
    functionId: str = "Default"
    nodeId: str = "Default"
    nodeName: str = "Default"
    description: str = "Default description"
    tags: List[str] = []
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

        # 3. 파라미터 유효성 검사
        if hasattr(cls, 'parameters') and cls.parameters:
            params_valid, param_errors = validate_parameters(cls.parameters)
            if not params_valid:
                is_valid = False
                print(f"[Node Registration Failed] Node '{cls.__name__}': Parameter validation failed.")
                for error in param_errors:
                    print(f"  -> {error}")

        if not is_valid:
            return

        # 파라미터의 함수형 options를 처리
        processed_parameters = []
        for param in cls.parameters:
            processed_param = param.copy()

            # options가 함수인 경우 처리
            if 'options' in processed_param and callable(processed_param['options']):
                try:
                    # 실제 인스턴스를 생성해서 함수 호출
                    try:
                        # user_id가 있고 __init__이 user_id를 받을 수 있는지 확인
                        import inspect
                        init_signature = inspect.signature(cls.__init__)
                        init_params = list(init_signature.parameters.keys())

                        if CURRENT_USER_ID and 'user_id' in init_params:
                            temp_instance = cls(user_id=CURRENT_USER_ID)
                            print(f"  -> Created instance with user_id: {CURRENT_USER_ID}")
                        else:
                            temp_instance = cls()

                        options_result = processed_param['options'](temp_instance)
                        processed_param['options'] = options_result
                        print(f"  -> Parameter '{param['id']}' options resolved: {len(options_result) if options_result else 0} items")
                    except Exception as init_error:
                        # __init__ 실패 시 빈 배열로 대체
                        print(f"  -> Warning: Could not initialize instance for '{param['id']}' options: {init_error}")
                        processed_param['options'] = []
                except Exception as e:
                    print(f"  -> Warning: Failed to resolve options for parameter '{param['id']}': {e}")
                    processed_param['options'] = []

            processed_parameters.append(processed_param)

        category_name = CATEGORIES_LABEL_MAP.get(cls.categoryId, "Unknown Category")
        function_name = FUNCTION_LABEL_MAP.get(cls.functionId, "Unknown Function")

        spec: NodeSpec = {
            "categoryId": cls.categoryId,
            "categoryName": category_name,
            "functionId": cls.functionId,
            "functionName": function_name,
            "id": cls.nodeId,
            "nodeName": cls.nodeName,
            "description": cls.description,
            "tags": cls.tags,
            "inputs": cls.inputs,
            "outputs": cls.outputs,
            "parameters": processed_parameters
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
    for module_info in pkgutil.walk_packages(path=[str(nodes_root_dir)], prefix='editor.nodes.'):
        try:
            importlib.import_module(module_info.name)
        except Exception as e:
            print(f"Error importing module {module_info.name}: {e}")

def run_force_discovery(user_id: str = None) -> None:
    """모든 노드 모듈을 강제로 다시 로드하여 __init_subclass__ 메서드가 다시 실행되도록 합니다."""
    # 기존 레지스트리 초기화
    clear_registries()

    # user_id가 제공된 경우 전역 변수에 설정
    global CURRENT_USER_ID
    if user_id:
        CURRENT_USER_ID = user_id
        print(f"Setting user_id for node discovery: {user_id}")

    nodes_root_dir = Path(__file__).parent / "nodes"
    for module_info in pkgutil.walk_packages(path=[str(nodes_root_dir)], prefix='editor.nodes.'):
        try:
            # 모듈이 이미 로드된 경우 다시 로드
            if module_info.name in sys.modules:
                module = sys.modules[module_info.name]

                # 모듈에 Node 서브클래스가 있는지 확인
                has_node_classes = False
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and
                        issubclass(attr, Node) and
                        attr != Node and
                        hasattr(attr, '__init_subclass__')):
                        has_node_classes = True
                        break

                if has_node_classes:
                    print(f"Force reloading module: {module_info.name}")
                    importlib.reload(module)
                else:
                    # Node 클래스가 없는 모듈은 일반 import
                    importlib.import_module(module_info.name)
            else:
                # 새로운 모듈은 일반 import
                importlib.import_module(module_info.name)

        except Exception as e:
            print(f"Error force importing module {module_info.name}: {e}")

def generate_json_spec(output_path="export_nodes.json"):
    """등록된 노드 정보를 새로운 프론트엔드 형식에 맞춰 JSON 파일로 저장합니다."""
    # 출력 디렉토리가 존재하지 않으면 생성
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created directory: {output_dir}")

    categories = {}
    for node_spec in NODE_REGISTRY:
        cat_id = node_spec["categoryId"]
        if cat_id not in categories:

            icon_name = ICON_LABEL_MAP.get(cat_id, "Unknown Function")
            categories[cat_id] = {
                "categoryId": cat_id,
                "categoryName": node_spec["categoryName"],
                "icon": icon_name,
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

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_spec, f, indent=4, ensure_ascii=False)
        print(f"\n✅ Node specification generated with {len(NODE_REGISTRY)} nodes at: {output_path}")
    except Exception as e:
        print(f"\n❌ Error generating node specification: {e}")
        raise

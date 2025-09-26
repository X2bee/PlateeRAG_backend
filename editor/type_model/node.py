from typing import List, Any, TypedDict, Literal, Type, Optional, Union, Callable, NotRequired

class Port(TypedDict):
    """노드의 입력 또는 출력 포트 구조를 정의합니다."""
    id: str
    name: str
    type: str
    required: NotRequired[Optional[bool]]
    multi: NotRequired[Optional[bool]]
    stream: NotRequired[Optional[bool]]

class Parameter(TypedDict):
    """노드가 사용할 파라미터의 구조를 정의합니다."""
    id: str
    name: str
    type: Literal["STR", "INT", "FLOAT", "BOOL"]
    value: Any
    required: NotRequired[Optional[bool]]
    optional: NotRequired[Optional[bool]]  # True이면 advanced 모드에서만 표시
    step: NotRequired[Optional[float]]
    min: NotRequired[Optional[float]]
    max: NotRequired[Optional[float]]
    options: NotRequired[Optional[Union[List[dict[str, Any]], Callable]]]
    is_api: NotRequired[bool]  # API 호출 여부
    api_name: NotRequired[Optional[str]] # API 호출 시 사용할 이름
    handle_id: NotRequired[Optional[bool]] # True이면 id를 키로 사용
    expandable: NotRequired[Optional[bool]]  # True이면 확장 가능한 파라미터
    description: NotRequired[Optional[str]]  # 파라미터 설명
    is_added: NotRequired[Optional[bool]]  # True이면 추가된
    dependency: NotRequired[Optional[str]]  # 다른 파라미터에 의존하는 경우 그 파라미터의 id


class NodeSpec(TypedDict):
    """NODE_REGISTRY에 저장될 노드의 전체 명세 구조를 정의합니다."""
    id: str
    nodeName: str
    description: str
    tags: List[str]
    categoryId: str
    categoryName: str
    functionId: str
    functionName: str
    inputs: List[Port]
    outputs: List[Port]
    parameters: List[Parameter]


CATEGORIES_LABEL_MAP = {
    'langchain': 'LangChain',
    'xgen': 'XGen',
    'utilities': 'Utilities',
    'math': 'Math'
}

ICON_LABEL_MAP = {
    'langchain': 'SiLangchain',
    'xgen': 'GiRollingEnergy',
    'utilities': 'LuWrench',
    'math': 'LuWrench'
}


FUNCTION_LABEL_MAP = {
    'agents': 'Agent',
    'cache': 'Cache',
    'chain': 'Chain',
    'chat_models': 'Chat Model',
    'document_loaders': 'Document Loader',
    'embeddings': 'Embedding',
    'graph': 'Graph',
    'memory': 'Memory',
    'moderation': 'Moderation',
    'output_parsers': 'Output Parser',
    'feedback':'Feedback',
    'tools': 'Tool',
    'api_loader': 'API Loader',
    'mcp': 'MCP',
    'arithmetic':'Arithmetic',
    'router': 'Router',
    'endnode': 'End Node',
    'startnode': 'Start Node',
}

def validate_parameter(param: Parameter) -> tuple[bool, str]:
    """
    파라미터의 유효성을 검사합니다.

    Args:
        param: 검사할 파라미터

    Returns:
        tuple[bool, str]: (유효한지 여부, 에러 메시지)
    """
    # required=True 이면서 optional=True 인 경우는 불가능
    if param.get("required", False) and param.get("optional", False):
        return False, f"Parameter '{param.get('id', 'unknown')}': 'required=True' and 'optional=True' cannot be both set. A required parameter cannot be optional."

    return True, ""

def validate_parameters(parameters: List[Parameter]) -> tuple[bool, List[str]]:
    """
    파라미터 리스트의 유효성을 검사합니다.

    Args:
        parameters: 검사할 파라미터 리스트

    Returns:
        tuple[bool, List[str]]: (모두 유효한지 여부, 에러 메시지 리스트)
    """
    errors = []

    for param in parameters:
        is_valid, error_msg = validate_parameter(param)
        if not is_valid:
            errors.append(error_msg)

    return len(errors) == 0, errors

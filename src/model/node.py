from typing import List, Any, TypedDict, Literal, Type, Optional

class Port(TypedDict):
    """노드의 입력 또는 출력 포트 구조를 정의합니다."""
    id: str
    name: str
    type: str
    required: bool = False
    multi: Optional[bool] = False

class Parameter(TypedDict):
    """노드가 사용할 파라미터의 구조를 정의합니다."""
    id: str
    name: str
    type: Literal["STRING", "INTEGER", "FLOAT", "BOOLEAN"]
    value: Any
    step: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    options: Optional[List[dict[str, Any]]] = None


class NodeSpec(TypedDict):
    """NODE_REGISTRY에 저장될 노드의 전체 명세 구조를 정의합니다."""
    id: str
    nodeName: str
    categoryId: str
    categoryName: str
    functionId: str
    functionName: str
    inputs: List[Port]
    outputs: List[Port]
    parameters: List[Parameter]


CATEGORIES_LABEL_MAP = {
    'langchain': 'LangChain',
    'polar': 'POLAR',
    'utilities': 'Utilities',
    'math': 'Math'
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
    'tools': 'Tool',
    'arithmetic':'Arithmetic',
    'endnode': 'End Node',
}
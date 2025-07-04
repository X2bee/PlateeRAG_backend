from pydantic import BaseModel
from typing import List, Dict, Any

class BaseNode(BaseModel):
    categoryId: str = "Default"
    functionId: str = "Default"
    nodeId: str = "Default"
    nodeName: str = "Default"
    inputs: List[Dict[str, Any]] = []
    outputs: List[Dict[str, Any]] = []
    parameters: List[Dict[str, Any]] = []

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
    'arithmetic':'Arithmetic'
}
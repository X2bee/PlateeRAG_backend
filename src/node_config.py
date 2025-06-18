from typing import Literal

CATEGORIES_LABEL_MAP = {
    'langchain': 'LangChain',
    'polar': 'POLAR',
    'utilities': 'Utilities'
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
    'tools': 'Tool'
}

CATEGORIES_ID = Literal[tuple(CATEGORIES_LABEL_MAP.keys())]
CATEGORIES_LABEL = Literal[tuple(CATEGORIES_LABEL_MAP.values())]
FUNCTION_ID = Literal[tuple(FUNCTION_LABEL_MAP.keys())]
FUNCTION_LABEL = Literal[tuple(FUNCTION_LABEL_MAP.values())]

if __name__ == "__main__":
    print(FUNCTION_ID)

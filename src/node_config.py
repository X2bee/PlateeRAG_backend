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

CATEGORIES = Literal[*(CATEGORIES_LABEL_MAP.keys())]
FUNCTIONS = Literal[*FUNCTION_LABEL_MAP.keys()]

if __name__ == "__main__":
    print(CATEGORIES)

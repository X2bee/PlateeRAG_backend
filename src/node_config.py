from typing import Literal
from src.model.node import (
    CATEGORIES_LABEL_MAP,
    FUNCTION_LABEL_MAP,
    ICON_LABEL_MAP,
)

CATEGORIES = Literal[*(CATEGORIES_LABEL_MAP.keys())]
FUNCTIONS = Literal[*FUNCTION_LABEL_MAP.keys()]
ICONS = Literal[*ICON_LABEL_MAP.keys()]

if __name__ == "__main__":
    print(CATEGORIES)

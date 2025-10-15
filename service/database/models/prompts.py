from typing import Dict, Optional
from service.database.models.base_model import BaseModel

class Prompts(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.user_id: Optional[int] = kwargs.get('user_id')
        self.prompt_uid: str = kwargs.get('prompt_uid', '')
        self.prompt_title: str = kwargs.get('prompt_title', '')
        self.prompt_content: str = kwargs.get('prompt_content', '')
        self.public_available: bool = kwargs.get('public_available', False)
        self.is_template: bool = kwargs.get('is_template', False)
        self.language: str = kwargs.get('language', 'ko')
        self.metadata: Optional[Dict] = kwargs.get('metadata', {})
        self.rating_count: int = kwargs.get('rating_count', 0)
        self.rating_sum: int = kwargs.get('rating_sum', 0)

    def get_table_name(self) -> str:
        return "prompts"

    def get_schema(self) -> Dict[str, str]:
        return {
            'user_id': 'INTEGER REFERENCES users(id) ON DELETE SET NULL',
            'prompt_uid': 'VARCHAR(100) NOT NULL',
            'prompt_title': 'VARCHAR(200) NOT NULL',
            'prompt_content': 'TEXT NOT NULL',
            'public_available': 'BOOLEAN DEFAULT FALSE',
            'is_template': 'BOOLEAN DEFAULT FALSE',
            'language': 'VARCHAR(30) DEFAULT \'ko\'',
            'metadata': 'TEXT',
            'rating_count': 'INTEGER DEFAULT 0',
            'rating_sum': 'INTEGER DEFAULT 0'
        }

class PromptStoreRating(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.user_id: int = kwargs.get('user_id')
        self.prompt_store_id: int = kwargs.get('prompt_store_id')
        self.prompt_uid: str = kwargs.get('prompt_uid', '')
        self.prompt_title: str = kwargs.get('prompt_title', '')
        self.rating: int = kwargs.get('rating', 1)  # 1 to 5

    def get_table_name(self) -> str:
        return "prompt_store_rating"

    def get_schema(self) -> Dict[str, str]:
        return {
            'user_id': 'INTEGER REFERENCES users(id) ON DELETE SET NULL',
            'prompt_store_id': 'INTEGER REFERENCES prompts(id) ON DELETE CASCADE',
            'prompt_uid': 'VARCHAR(100) NOT NULL',
            'prompt_title': 'VARCHAR(200) NOT NULL',
            'rating': 'INTEGER CHECK (rating >= 1 AND rating <= 5)'
        }

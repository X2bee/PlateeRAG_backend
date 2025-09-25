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
            'metadata': 'TEXT'
        }

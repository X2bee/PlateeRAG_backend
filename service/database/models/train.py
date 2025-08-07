
from typing import Dict, Optional
from service.database.models.base_model import BaseModel

class TrainMeta(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.user_id: Optional[int] = kwargs.get('user_id')
        self.model_info_name: str = kwargs.get('model_info_name', '')
        self.model_info_type: str = kwargs.get('model_info_type', '')
        self.train_data: str = kwargs.get('train_data', '')
        self.test_data: str = kwargs.get('test_data', '')
        self.mlflow_url: Optional[str] = kwargs.get('mlflow_url', None)
        self.mlflow_run_id: Optional[str] = kwargs.get('mlflow_run_id', None)
        self.status: str = kwargs.get('status', '')

    def get_table_name(self) -> str:
        return "train_meta"

    def get_schema(self) -> Dict[str, str]:
        return {
            'user_id': 'INTEGER REFERENCES users(id) ON DELETE SET NULL',
            'model_info_name': 'VARCHAR(100) NOT NULL',
            'model_info_type': 'VARCHAR(100) NOT NULL',
            'train_data': 'TEXT NOT NULL',
            'test_data': 'TEXT NOT NULL',
            'mlflow_url': 'VARCHAR(255)',
            'mlflow_run_id': 'VARCHAR(100)',
            'status': 'VARCHAR(50)'
        }

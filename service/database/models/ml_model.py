"""Machine learning model registry data model."""
from typing import Dict, Optional, List, Any
import json
from service.database.models.base_model import BaseModel

class MLModel(BaseModel):
    """Metadata record for an uploaded machine learning model."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name: str = kwargs.get("model_name", "")
        self.model_version: Optional[str] = kwargs.get("model_version")
        self.framework: str = kwargs.get("framework", "sklearn")
        self.file_path: str = kwargs.get("file_path", "")
        self.file_size: Optional[int] = kwargs.get("file_size")
        self.file_checksum: Optional[str] = kwargs.get("file_checksum")
        self.description: Optional[str] = kwargs.get("description")
        self.input_schema: Optional[List[str]] = self._parse_schema(kwargs.get("input_schema"))
        self.output_schema: Optional[List[str]] = self._parse_schema(kwargs.get("output_schema"))
        self.metadata: Dict[str, Any] = self._parse_metadata(kwargs.get("metadata"))
        self.status: str = kwargs.get("status", "active")
        self.uploaded_by: Optional[int] = kwargs.get("uploaded_by")

    def get_table_name(self) -> str:
        return "ml_models"

    def get_schema(self) -> Dict[str, str]:
        return {
            "model_name": "VARCHAR(200) NOT NULL",
            "model_version": "VARCHAR(100)",
            "framework": "VARCHAR(100) NOT NULL",
            "file_path": "TEXT NOT NULL",
            "file_size": "INTEGER",
            "file_checksum": "VARCHAR(128)",
            "description": "TEXT",
            "input_schema": "TEXT",
            "output_schema": "TEXT",
            "metadata": "TEXT",
            "status": "VARCHAR(50) DEFAULT 'active'",
            "uploaded_by": "INTEGER REFERENCES users(id) ON DELETE SET NULL"
        }

    @staticmethod
    def _parse_schema(value: Optional[Any]) -> Optional[List[str]]:
        if value is None or value == "":
            return None
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
            return [item.strip() for item in value.split(",") if item.strip()]
        return None

    @staticmethod
    def _parse_metadata(value: Optional[Any]) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value:
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                return {}
        return {}

    def get_input_schema(self) -> Optional[List[str]]:
        return self.input_schema

    def get_output_schema(self) -> Optional[List[str]]:
        return self.output_schema

    def get_metadata(self) -> Dict[str, Any]:
        return self.metadata

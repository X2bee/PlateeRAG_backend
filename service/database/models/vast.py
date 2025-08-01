"""
VastAI 인스턴스 관리 모델
"""
from typing import Dict, Any, Optional
from datetime import datetime
import json

from service.database.models.base_model import BaseModel

class VastInstance(BaseModel):
    """VastAI 인스턴스 정보를 저장하는 모델"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.instance_id: Optional[str] = kwargs.get('instance_id')
        self.offer_id: Optional[str] = kwargs.get('offer_id')
        self.image_name: Optional[str] = kwargs.get('image_name')
        self.status: Optional[str] = kwargs.get('status', 'creating')
        self.public_ip: Optional[str] = kwargs.get('public_ip')
        self.ssh_port: Optional[int] = kwargs.get('ssh_port')
        self.port_mappings: Optional[str] = kwargs.get('port_mappings')  # JSON string
        self.start_command: Optional[str] = kwargs.get('start_command')
        self.gpu_info: Optional[str] = kwargs.get('gpu_info')  # JSON string
        self.auto_destroy: Optional[bool] = kwargs.get('auto_destroy', False)
        self.template_name: Optional[str] = kwargs.get('template_name')
        self.model_name: Optional[str] = kwargs.get('model_name')
        self.max_model_length: Optional[int] = kwargs.get('max_model_length')
        self.dph_total: Optional[float] = kwargs.get('dph_total', 0.0)
        self.cpu_name: Optional[str] = kwargs.get('cpu_name')
        self.cpu_cores: Optional[int] = kwargs.get('cpu_cores')
        self.ram: Optional[int] = kwargs.get('ram')
        self.cuda_max_good: Optional[float] = kwargs.get('cuda_max_good')
        self.destroyed_at: Optional[datetime] = kwargs.get('destroyed_at')

    def get_table_name(self) -> str:
        return "vast_instances"

    def get_schema(self) -> Dict[str, str]:
        return {
            "instance_id": "VARCHAR(255) UNIQUE",
            "offer_id": "VARCHAR(255)",
            "image_name": "VARCHAR(500)",
            "status": "VARCHAR(50) DEFAULT 'creating'",
            "public_ip": "VARCHAR(45)",
            "ssh_port": "INTEGER",
            "port_mappings": "TEXT",
            "start_command": "TEXT",
            "gpu_info": "TEXT",
            "auto_destroy": "BOOLEAN DEFAULT FALSE",
            "template_name": "VARCHAR(100)",
            "model_name": "VARCHAR(100)",
            "max_model_length": "INTEGER",
            "dph_total": "DECIMAL(10,4) DEFAULT 0.0",
            "cpu_name": "VARCHAR(100)",
            "cpu_cores": "INTEGER",
            "ram": "INTEGER",
            "cuda_max_good": "DECIMAL(10,4) DEFAULT 0.0",
            "destroyed_at": "TIMESTAMP"
        }

    def get_port_mappings_dict(self) -> Dict[str, Any]:
        """포트 매핑을 딕셔너리로 반환"""
        if self.port_mappings:
            try:
                return json.loads(self.port_mappings)
            except json.JSONDecodeError:
                return {}
        return {}

    def set_port_mappings(self, mappings: Dict[str, Any]):
        """포트 매핑을 JSON 문자열로 저장"""
        self.port_mappings = json.dumps(mappings)

    def get_gpu_info_dict(self) -> Dict[str, Any]:
        """GPU 정보를 딕셔너리로 반환"""
        if self.gpu_info:
            try:
                return json.loads(self.gpu_info)
            except json.JSONDecodeError:
                return {}
        return {}

    def set_gpu_info(self, info: Dict[str, Any]):
        """GPU 정보를 JSON 문자열로 저장"""
        self.gpu_info = json.dumps(info)

class VastExecutionLog(BaseModel):
    """VastAI 실행 로그를 저장하는 모델"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.instance_id: Optional[str] = kwargs.get('instance_id')
        self.operation: Optional[str] = kwargs.get('operation')  # create, destroy, execute, etc.
        self.command: Optional[str] = kwargs.get('command')
        self.result: Optional[str] = kwargs.get('result')
        self.error_message: Optional[str] = kwargs.get('error_message')
        self.execution_time: Optional[float] = kwargs.get('execution_time')
        self.success: Optional[bool] = kwargs.get('success', True)
        self.metadata: Optional[str] = kwargs.get('metadata')  # JSON string for additional data

    def get_table_name(self) -> str:
        return "vast_execution_logs"

    def get_schema(self) -> Dict[str, str]:
        return {
            "instance_id": "VARCHAR(255)",
            "operation": "VARCHAR(100)",
            "command": "TEXT",
            "result": "TEXT",
            "error_message": "TEXT",
            "execution_time": "DECIMAL(10,3)",
            "success": "BOOLEAN DEFAULT TRUE",
            "metadata": "TEXT"
        }

    def get_metadata_dict(self) -> Dict[str, Any]:
        """메타데이터를 딕셔너리로 반환"""
        if self.metadata:
            try:
                return json.loads(self.metadata)
            except json.JSONDecodeError:
                return {}
        return {}

    def set_metadata(self, metadata: Dict[str, Any]):
        """메타데이터를 JSON 문자열로 저장"""
        self.metadata = json.dumps(metadata)

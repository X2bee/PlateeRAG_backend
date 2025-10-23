"""
Tool 관련 요청 모델
"""
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class ToolData(BaseModel):
    """툴 데이터 모델"""
    function_name: str
    function_id: str
    description: str = ""
    api_header: Optional[Dict[str, Any]] = {}
    api_body: Optional[Dict[str, Any]] = {}
    static_body: Optional[Dict[str, Any]] = {}
    body_type: Optional[str] = "application/json"
    api_url: str
    api_method: str = "GET"
    api_timeout: int = 30
    response_filter: bool = False
    response_filter_path: Optional[str] = ""
    response_filter_field: Optional[str] = ""
    status: Optional[str] = ""
    metadata: Optional[Dict[str, Any]] = {}

class SaveToolRequest(BaseModel):
    """툴 저장 요청 모델"""
    function_name: str
    content: ToolData
    user_id: Optional[int | str] = None

class UploadToolStoreRequest(BaseModel):
    """툴 스토어 업로드 요청 모델"""
    function_upload_id: str = Field(description="업로드할 툴의 고유 ID")
    description: Optional[str] = Field(default="", description="툴 설명")
    tags: Optional[list] = Field(default_factory=list, description="툴 태그")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="추가 메타데이터")

class ApiTestRequest(BaseModel):
    """API 테스트 요청 모델"""
    api_url: str = Field(description="테스트할 API URL")
    api_method: str = Field(default="GET", description="HTTP 메서드")
    api_headers: Optional[Dict[str, Any]] = Field(default_factory=dict, description="요청 헤더")
    api_body: Optional[Dict[str, Any]] = Field(default_factory=dict, description="요청 바디")
    api_timeout: int = Field(default=30, description="타임아웃 (초)")

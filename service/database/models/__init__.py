"""
모델 모듈 초기화
"""
from service.database.models.base_model import BaseModel
from service.database.models.user import User, UserSession
# from service.database.models.chat import ChatSession, ChatMessage
from service.database.models.performance import WorkflowExecution, NodeExecution, SystemMetrics, NodePerformance
from service.database.models.persistent_config_model import PersistentConfigModel
from service.database.models.executor import ExecutionIO, ExecutionMeta
from service.database.models.workflow import WorkflowMeta
from service.database.models.vectordb import VectorDB
from service.database.models.vast import VastInstance, VastExecutionLog

# 사용 가능한 모델들
__all__ = [
    'BaseModel',
    'User', 'UserSession',
    'WorkflowExecution', 'NodeExecution', 'SystemMetrics', 'NodePerformance',
    'PersistentConfigModel',
    'ExecutionIO', 'ExecutionMeta', 'WorkflowMeta',
    'VectorDB',
    'VastInstance', 'VastExecutionLog'
]

# 애플리케이션에서 사용할 모델 목록
APPLICATION_MODELS = [
    PersistentConfigModel,  # 가장 먼저 생성되어야 함
    User,
    UserSession,
    WorkflowExecution,
    NodeExecution,
    SystemMetrics,
    NodePerformance,
    ExecutionIO, ExecutionMeta,
    WorkflowMeta,
    VectorDB,
    VastInstance,
    VastExecutionLog
]

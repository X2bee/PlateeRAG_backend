"""
모델 모듈 초기화
"""
from service.database.models.base_model import BaseModel
from service.database.models.user import User, UserSession
# from service.database.models.chat import ChatSession, ChatMessage
from service.database.models.performance import WorkflowExecution, NodePerformance
from service.database.models.persistent_config import PersistentConfigModel
from service.database.models.executor import ExecutionIO, ExecutionMeta
from service.database.models.workflow import WorkflowMeta
from service.database.models.deploy import DeployMeta
from service.database.models.vectordb import VectorDB, VectorDBChunkMeta, VectorDBChunkEdge
from service.database.models.vast import VastInstance, VastExecutionLog
from service.database.models.train import TrainMeta
from service.database.models.group import GroupMeta

# 사용 가능한 모델들
__all__ = [
    'BaseModel',
    'User', 'UserSession',
    'WorkflowExecution', 'NodePerformance',
    'PersistentConfigModel',
    'ExecutionIO', 'ExecutionMeta', 'WorkflowMeta', 'DeployMeta',
    'VectorDB', 'VectorDBChunkMeta', 'VectorDBChunkEdge',
    'VastInstance', 'VastExecutionLog',
    'TrainMeta',
    'GroupMeta'
]

# 애플리케이션에서 사용할 모델 목록
APPLICATION_MODELS = [
    PersistentConfigModel,  # 가장 먼저 생성되어야 함
    User,
    UserSession,
    WorkflowExecution,
    NodePerformance,
    ExecutionIO, ExecutionMeta,
    WorkflowMeta,
    DeployMeta,
    VectorDB,
    VectorDBChunkMeta,
    VectorDBChunkEdge,
    VastInstance,
    VastExecutionLog,
    TrainMeta,
    GroupMeta
]

"""
모델 모듈 초기화
"""
from service.database.models.base_model import BaseModel
from service.database.models.user import User, UserSession
# from service.database.models.chat import ChatSession, ChatMessage
from service.database.models.performance import WorkflowExecution, NodePerformance
from service.database.models.persistent_config import PersistentConfigModel
from service.database.models.executor import ExecutionIO, ExecutionMeta
from service.database.models.workflow import WorkflowMeta, WorkflowVersion, WorkflowStoreMeta, WorkflowStoreRating
from service.database.models.deploy import DeployMeta
from service.database.models.vectordb import VectorDB, VectorDBFolders, VectorDBChunkMeta, VectorDBChunkEdge
from service.database.models.vast import VastInstance, VastExecutionLog
from service.database.models.train import TrainMeta
from service.database.models.ml_model import MLModel
from service.database.models.group import GroupMeta
from service.database.models.backend import BackendLogs
from service.database.models.prompts import Prompts, PromptStoreRating
from service.database.models.tools import Tools, ToolStoreMeta, ToolStoreRating
from service.database.models.db_sync_config import DBSyncConfig  # ⭐ 추가
from service.database.models.data_scraper import (
    ScraperConfig,
    ScraperRun,
    ScrapedItem,
    ScrapedAsset,
    ScraperStats,
)

# 사용 가능한 모델들
__all__ = [
    'BaseModel',
    'User', 'UserSession',
    'WorkflowExecution', 'NodePerformance',
    'PersistentConfigModel',
    'ExecutionIO', 'ExecutionMeta', 'WorkflowMeta', 'WorkflowVersion', 'WorkflowStoreMeta', 'WorkflowStoreRating', 'DeployMeta',
    'VectorDB', 'VectorDBFolders', 'VectorDBChunkMeta', 'VectorDBChunkEdge',
    'VastInstance', 'VastExecutionLog',
    'TrainMeta',
    'GroupMeta',
    'BackendLogs',
    'MLModel',
    'Prompts', 'PromptStoreRating',
    'Tools', 'ToolStoreMeta', 'ToolStoreRating',
    'DBSyncConfig',  # ⭐ 추가
    'ScraperConfig', 'ScraperRun', 'ScrapedItem', 'ScrapedAsset', 'ScraperStats',
]

# 애플리케이션에서 사용할 모델 목록
APPLICATION_MODELS = [
    PersistentConfigModel,
    User, UserSession,
    WorkflowExecution,
    NodePerformance,
    ExecutionIO, ExecutionMeta,
    WorkflowMeta, WorkflowVersion, WorkflowStoreMeta, WorkflowStoreRating, DeployMeta,
    VectorDB, VectorDBFolders, VectorDBChunkMeta, VectorDBChunkEdge,
    VastInstance, VastExecutionLog,
    TrainMeta,
    GroupMeta,
    BackendLogs,
    MLModel,
    Prompts, PromptStoreRating,
    Tools, ToolStoreMeta, ToolStoreRating,
    DBSyncConfig,  # ⭐ 추가
    ScraperConfig, ScraperRun, ScrapedItem, ScrapedAsset, ScraperStats,
]

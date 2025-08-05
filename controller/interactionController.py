from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import json
import logging
from datetime import datetime
from editor.workflow_executor import WorkflowExecutor
from service.database.execution_meta_service import get_or_create_execution_meta, update_execution_meta_count
from controller.controller_helper import extract_user_id_from_request

from service.database.models.executor import ExecutionMeta

logger = logging.getLogger("interaction-controller")
router = APIRouter(prefix="/api/interaction", tags=["interaction"])

class WorkflowRequest(BaseModel):
    workflow_name: str
    workflow_id: str
    input_data: str = ""
    interaction_id: str = "default"

class WorkflowData(BaseModel):
    workflow_name: str
    workflow_id: str
    view: Dict[str, Any]
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    interaction_id: str = "default"

class SaveWorkflowRequest(BaseModel):
    workflow_id: str
    content: WorkflowData

@router.get("/list")
async def list_interaction(request: Request, interaction_id: str = None, workflow_id: str = None, limit: int = 100):
    """
    ExecutionMeta 정보들을 리스트 형태로 반환합니다.

    Args:
        interaction_id: 특정 상호작용 ID로 필터링 (선택적)
        workflow_id: 특정 워크플로우 ID로 필터링 (선택적)
        limit: 반환할 최대 레코드 수 (기본값: 100)

    Returns:
        ExecutionMeta 데이터 리스트
    """
    try:
        user_id = extract_user_id_from_request(request)

        app_db = request.app.state.app_db
        if not app_db:
            raise HTTPException(status_code=500, detail="Database connection not available")

        # SQL 쿼리 작성 - 조건에 따라 동적으로 구성
        where_conditions = {}
        query_params = []
        where_conditions["user_id"] = user_id

        if interaction_id:
            where_conditions["interaction_id"] = interaction_id

        # 로직 제거
        # if workflow_id:
        #     where_conditions["workflow_id"] = workflow_id

        result = app_db.find_by_condition(
            ExecutionMeta,
            conditions=where_conditions,
            orderby="updated_at",
            orderby_asc=False,
            limit=1000000,
            return_list=True
        )

        print(result)

        # # SQLite인 경우 파라미터 플레이스홀더 변경
        # if db_manager.config_db_manager.db_type == "sqlite":
        #     query = query.replace("%s", "?")

        # # 쿼리 실행
        # result = db_manager.config_db_manager.execute_query(query, tuple(query_params))

        if not result:
            return JSONResponse(content={
                "execution_meta_list": [],
                "total_count": 0,
                "filters": {
                    "interaction_id": interaction_id,
                    "workflow_id": workflow_id,
                    "limit": limit
                },
                "message": "No execution meta data found"
            })

        execution_meta_list = []
        for row in result:
            meta_data = {
                "id": row.get('id'),
                "interaction_id": row['interaction_id'],
                "workflow_id": row['workflow_id'],
                "workflow_name": row['workflow_name'],
                "interaction_count": row['interaction_count'],
                "metadata": json.loads(row['metadata']) if row.get('metadata') else {},
                "created_at": row['created_at'].isoformat() if isinstance(row['created_at'], datetime) else row['created_at'],
                "updated_at": row['updated_at'].isoformat() if isinstance(row['updated_at'], datetime) else row['updated_at']
            }
            execution_meta_list.append(meta_data)

        response_data = {
            "execution_meta_list": execution_meta_list,
            "total_count": len(execution_meta_list),
            "filters": {
                "interaction_id": interaction_id,
                "workflow_id": workflow_id,
                "limit": limit
            },
            "message": f"Found {len(execution_meta_list)} execution meta records"
        }

        logger.info(f"Retrieved {len(execution_meta_list)} execution meta records with filters: interaction_id={interaction_id}, workflow_id={workflow_id}")
        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"Error listing execution meta: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list execution meta: {str(e)}")

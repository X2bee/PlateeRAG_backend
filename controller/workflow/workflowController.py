import asyncio
from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import hashlib
import os
import copy
import json
import logging
import re
from datetime import datetime
from editor.async_workflow_executor import execution_manager
from service.database.execution_meta_service import get_or_create_execution_meta, update_execution_meta_count
from controller.helper.controllerHelper import extract_user_id_from_request
from controller.workflow.helper import _workflow_parameter_helper, _default_workflow_parameter_helper
from controller.workflow.model import WorkflowRequest, WorkflowData, SaveWorkflowRequest, ConversationRequest
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from service.database.models.user import User
from service.database.models.executor import ExecutionMeta, ExecutionIO
from service.database.models.workflow import WorkflowMeta
from service.database.models.performance import NodePerformance
from service.database.models.deploy import DeployMeta
from controller.helper.singletonHelper import get_config_composer, get_vector_manager, get_rag_service, get_document_processor, get_db_manager

import uuid
import time

logger = logging.getLogger("workflow-controller")
router = APIRouter(prefix="", tags=["workflow"])

def workflow_user_id_extractor(app_db, login_user_id, requested_user_id, workflow_id):
    if login_user_id is not None:
        login_user_id = str(login_user_id).strip()

    if requested_user_id is not None:
        requested_user_id = str(requested_user_id).strip()
    else:
        requested_user_id = None

    if (login_user_id == requested_user_id) or requested_user_id == None or len(requested_user_id) == 0:
        return login_user_id
    else:
        user = app_db.find_by_id(User, login_user_id)
        if not user:
            logger.error(f"Login user not found in database: {login_user_id}")
            return login_user_id

        groups = user.groups
        requested_workflow_meta = app_db.find_by_condition(WorkflowMeta, {'user_id': requested_user_id, 'workflow_name': workflow_id}, limit=1)

        if requested_workflow_meta:
            requested_workflow_meta = requested_workflow_meta[0]
            if requested_workflow_meta.is_shared:
                if requested_workflow_meta.share_group in groups:
                    logger.info(f"✓ Access granted! Login user belongs to share group '{requested_workflow_meta.share_group}'. Using requested_user_id: {requested_user_id}")
                    return requested_user_id
                else:
                    return login_user_id
            else:
                logger.warning(f"✗ Access denied! Workflow is not shared (is_shared=False). Using login_user_id: {login_user_id}")
                return login_user_id
        else:
            logger.warning(f"✗ No workflow metadata found for user_id: {requested_user_id}, workflow_name: {workflow_id}. Using login_user_id: {login_user_id}")
            return login_user_id

def extract_collection_name(collection_full_name: str) -> str:
    """
    컬렉션 이름에서 UUID 부분을 제거하고 실제 이름만 추출합니다.

    예: '장하렴연구_3a6a552d-d277-490d-9f3c-cead80d651f7' -> '장하렴연구'

    Args:
        collection_full_name: UUID가 포함된 전체 컬렉션 이름

    Returns:
        UUID 부분이 제거된 깨끗한 컬렉션 이름
    """
    # UUID 패턴: 8-4-4-4-12 형태의 16진수 문자열
    uuid_pattern = r'_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'

    # UUID 부분을 제거하고 앞의 이름만 반환
    clean_name = re.sub(uuid_pattern, '', collection_full_name, flags=re.IGNORECASE)

    return clean_name


@router.get("/list")
async def list_workflows(request: Request):
    """
    downloads 폴더에 있는 모든 workflow 파일들의 이름을 반환합니다.
    """
    try:
        user_id = extract_user_id_from_request(request)

        downloads_path = os.path.join(os.getcwd(), "downloads")
        download_path_id = os.path.join(downloads_path, user_id)

        # downloads 폴더가 존재하지 않으면 생성
        if not os.path.exists(download_path_id):
            os.makedirs(download_path_id)
            return JSONResponse(content={"workflows": []})

        # .json 확장자를 가진 파일들만 필터링
        workflow_files = []
        for file in os.listdir(download_path_id):
            if file.endswith('.json'):
                workflow_files.append(file)

        logger.info(f"Found {len(workflow_files)} workflow files")
        return JSONResponse(content={"workflows": workflow_files})

    except Exception as e:
        logger.error(f"Error listing workflows: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list workflows: {str(e)}")

@router.post("/save")
async def save_workflow(request: Request, workflow_request: SaveWorkflowRequest):
    """
    Frontend에서 받은 workflow 정보를 파일로 저장합니다.
    파일명: {workflow_name}.json
    """
    try:
        user_id = extract_user_id_from_request(request)

        downloads_path = os.path.join(os.getcwd(), "downloads")
        download_path_id = os.path.join(downloads_path, user_id)
        workflow_data = workflow_request.content.dict()
        if not os.path.exists(download_path_id):
            os.makedirs(download_path_id)

        if not workflow_request.workflow_name.endswith('.json'):
            filename = f"{workflow_request.workflow_name}.json"
        else:
            filename = workflow_request.workflow_name
        file_path = os.path.join(download_path_id, filename)

        app_db = get_db_manager(request)

        # nodes 수 계산
        nodes = workflow_data.get('nodes', [])
        node_count = len(nodes) if isinstance(nodes, list) else 0
        has_startnode = any(
            node.get('data', {}).get('functionId') == 'startnode' for node in nodes
        )
        has_endnode = any(
            node.get('data', {}).get('functionId') == 'endnode' for node in nodes
        )

        # edges 수 계산
        edges = workflow_data.get('edges', [])
        edge_count = len(edges) if isinstance(edges, list) else 0

        workflow_meta = WorkflowMeta(
            user_id=user_id,
            workflow_id=workflow_request.content.workflow_id,
            workflow_name=workflow_request.workflow_name,
            node_count=node_count,
            edge_count=edge_count,
            has_startnode=has_startnode,
            has_endnode=has_endnode,
            is_completed=(has_startnode and has_endnode),
        )

        existing_data = app_db.find_by_condition(
            WorkflowMeta,
            {
                "user_id": user_id,
                "workflow_name": workflow_request.workflow_name,
            },
            limit=1
        )
        if existing_data:
            existing_data_id = existing_data[0].id
            workflow_meta.id = existing_data_id
            insert_result = app_db.update(workflow_meta)
        else:
            insert_result = app_db.insert(workflow_meta)


        if insert_result and insert_result.get("result") == "success":
            # Deploy metadata 생성
            deploy_meta = DeployMeta(
                user_id=user_id,
                workflow_id=workflow_request.content.workflow_id,
                workflow_name=workflow_request.workflow_name,
                is_deployed=False,
                deploy_key=""
            )

            # 기존 Deploy metadata 확인 및 생성/업데이트
            existing_deploy_data = app_db.find_by_condition(
                DeployMeta,
                {
                    "user_id": user_id,
                    "workflow_name": workflow_request.workflow_name,
                },
                limit=1
            )
            if existing_deploy_data:
                existing_deploy_id = existing_deploy_data[0].id
                deploy_meta.id = existing_deploy_id
                app_db.update(deploy_meta)
            else:
                app_db.insert(deploy_meta)

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(workflow_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Workflow metadata and deploy metadata saved successfully: {workflow_request.workflow_name}")
        else:
            logger.error(f"Failed to save workflow metadata: {insert_result.get('error', 'Unknown error')}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save workflow metadata: {insert_result.get('error', 'Unknown error')}"
            )

        logger.info(f"Workflow saved successfully: {filename}")
        return JSONResponse(content={
            "success": True,
            "message": f"Workflow '{workflow_request.workflow_name}' saved successfully",
            "filename": filename
        })

    except Exception as e:
        logger.error(f"Error saving workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save workflow: {str(e)}")

@router.get("/load/{workflow_name}")
async def load_workflow(request: Request, workflow_name: str, user_id):
    """
    특정 workflow를 로드합니다.
    """
    try:
        login_user_id = extract_user_id_from_request(request)
        downloads_path = os.path.join(os.getcwd(), "downloads")
        app_db = get_db_manager(request)
        using_id = workflow_user_id_extractor(app_db, login_user_id, user_id, workflow_name)
        download_path_id = os.path.join(downloads_path, using_id)

        filename = f"{workflow_name}.json"
        file_path = os.path.join(download_path_id, filename)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Workflow '{workflow_name}' not found")

        with open(file_path, 'r', encoding='utf-8') as f:
            workflow_data = json.load(f)

        logger.info(f"Workflow loaded successfully: {filename}")
        return JSONResponse(content=workflow_data)

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_name}' not found")
    except Exception as e:
        logger.error(f"Error loading workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load workflow: {str(e)}")

@router.get("/duplicate/{workflow_name}")
async def duplicate_workflow(request: Request, workflow_name: str, user_id):
    """
    특정 workflow를 복제합니다.
    """
    try:
        login_user_id = extract_user_id_from_request(request)
        downloads_path = os.path.join(os.getcwd(), "downloads")
        app_db = get_db_manager(request)
        using_id = workflow_user_id_extractor(app_db, login_user_id, user_id, workflow_name)

        origin_path_id = os.path.join(downloads_path, using_id)
        target_path_id = os.path.join(downloads_path, login_user_id)

        filename = f"{workflow_name}.json"
        origin_path = os.path.join(origin_path_id, filename)

        if not os.path.exists(origin_path):
            logger.info(f"Reading workflow data from: {origin_path}")
            raise HTTPException(status_code=404, detail=f"Workflow '{workflow_name}' not found")

        copy_workflow_name = f"{workflow_name}_copy"
        copy_file_name = f"{copy_workflow_name}.json"
        target_path = os.path.join(target_path_id, filename)
        
        if os.path.exists(target_path):
            logger.warning(f"Workflow already exists for user '{login_user_id}': {filename}. Change target file name.")
            counter = 1
            while os.path.exists(target_path):
                copy_workflow_name = f"{workflow_name}_copy_{counter}"
                copy_file_name = f"{copy_workflow_name}.json"
                target_path = os.path.join(target_path_id, copy_file_name)
                counter += 1

        with open(origin_path, 'r', encoding='utf-8') as f:
            logger.info(f"Reading workflow data from: {origin_path}")
            workflow_data = json.load(f)

        nodes = workflow_data.get('nodes', [])
        node_count = len(nodes) if isinstance(nodes, list) else 0
        has_startnode = any(node.get('data', {}).get('functionId') == 'startnode' for node in nodes)
        has_endnode = any(node.get('data', {}).get('functionId') == 'endnode' for node in nodes)

        edges = workflow_data.get('edges', [])
        edge_count = len(edges) if isinstance(edges, list) else 0

        workflow_meta = WorkflowMeta(
            user_id=login_user_id,
            workflow_id=workflow_data.get('workflow_id'),
            workflow_name=copy_workflow_name,
            node_count=node_count,
            edge_count=edge_count,
            has_startnode=has_startnode,
            has_endnode=has_endnode,
            is_completed=(has_startnode and has_endnode),
        )

        insert_result = app_db.insert(workflow_meta)
        with open(target_path, 'w', encoding='utf-8') as wf:
            json.dump(workflow_data, wf, ensure_ascii=False, indent=2)

        logger.info(f"Workflow duplicated successfully: {filename}")
        return {"success": True, "message": f"Workflow '{workflow_name}' duplicated successfully", "filename": filename}

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_name}' not found")
    except Exception as e:
        logger.error(f"Error loading workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load workflow: {str(e)}")

@router.post("/update/{workflow_name}")
async def update_workflow(request: Request, workflow_name: str, update_dict: dict):
    user_id = extract_user_id_from_request(request)
    app_db = get_db_manager(request)

    try:
        existing_data = app_db.find_by_condition(
            WorkflowMeta,
            {
                "user_id": user_id,
                "workflow_name": workflow_name
            },
        )

        if not existing_data:
            raise HTTPException(status_code=404, detail="Workflow not found")
        existing_data = existing_data[0]

        existing_data.is_shared = update_dict.get("is_shared", existing_data.is_shared)
        existing_data.share_group = update_dict.get("share_group", existing_data.share_group)

        app_db.update(existing_data)

        return {
            "message": "Workflow updated successfully",
            "workflow_name": existing_data.workflow_name
        }

    except Exception as e:
        logger.error(f"Failed to update workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update workflow: {str(e)}")

@router.delete("/delete/{workflow_name}")
async def delete_workflow(request: Request, workflow_name: str):
    """
    특정 workflow를 삭제합니다.
    """
    try:
        user_id = extract_user_id_from_request(request)
        app_db = get_db_manager(request)

        existing_data = app_db.find_by_condition(
            WorkflowMeta,
            {
                "user_id": user_id,
                "workflow_name": workflow_name,
            },
            limit=1
        )

        app_db.delete(WorkflowMeta, existing_data[0].id if existing_data else None)

        downloads_path = os.path.join(os.getcwd(), "downloads")
        download_path_id = os.path.join(downloads_path, user_id)
        filename = f"{workflow_name}.json"
        file_path = os.path.join(download_path_id, filename)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Workflow '{workflow_name}' not found")

        os.remove(file_path)

        logger.info(f"Workflow deleted successfully: {filename}")
        return JSONResponse(content={
            "success": True,
            "message": f"Workflow '{workflow_name}' deleted successfully"
        })

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_name}' not found")
    except Exception as e:
        logger.error(f"Error deleting workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete workflow: {str(e)}")

@router.get("/list/detail")
async def list_workflows_detail(request: Request):
    """
    downloads 폴더에 있는 모든 workflow 파일들의 상세 정보를 반환합니다.
    각 워크플로우에 대해 파일명, workflow_id, 노드 수, 마지막 수정일자를 포함합니다.
    """
    try:
        user_id = extract_user_id_from_request(request)
        app_db = get_db_manager(request)
        user = app_db.find_by_id(User, user_id)
        groups = user.groups
        user_name = user.username if user else "Unknown User"

        existing_data = app_db.find_by_condition(
            WorkflowMeta,
            {
                "user_id": user_id,
            },
            limit=10000,
            orderby="updated_at",
        )
        if groups and groups != None and groups != [] and len(groups) > 0:
            for group_name in groups:
                shared_data = app_db.find_by_condition(
                    WorkflowMeta,
                    {
                        "share_group": group_name,
                        "is_shared": True,
                    },
                    limit=10000,
                    orderby="updated_at",
                )
                existing_data.extend(shared_data)

        seen_ids = set()
        unique_data = []
        for item in existing_data:
            if item.id not in seen_ids:
                seen_ids.add(item.id)
                unique_data.append(item)

        response_data = []
        for data in unique_data:
            data_dict = data.to_dict()
            data_dict['user_name'] = user_name
            response_data.append(data_dict)

        logger.info(f"Found {len(existing_data)} workflow files with detailed information")

        return JSONResponse(content={"workflows": response_data})

    except Exception as e:
        logger.error(f"Error listing workflow details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list workflow details: {str(e)}")

@router.get("/performance")
async def get_workflow_performance(request: Request, workflow_name: str, workflow_id: str):
    """
    특정 워크플로우의 성능 통계를 반환합니다.
    node_id와 node_name별로 평균 성능 지표를 계산합니다.

    Args:
        workflow_name: 워크플로우 이름
        workflow_id: 워크플로우 ID

    Returns:
        노드별 성능 통계와 전체 워크플로우 통계
    """
    try:
        user_id = extract_user_id_from_request(request)
        app_db = get_db_manager(request)

        # SQL 쿼리 작성
        query = """
        SELECT
            node_id,
            node_name,
            AVG(processing_time_ms) as avg_processing_time_ms,
            AVG(cpu_usage_percent) as avg_cpu_usage_percent,
            AVG(ram_usage_mb) as avg_ram_usage_mb,
            AVG(CASE WHEN gpu_usage_percent IS NOT NULL THEN gpu_usage_percent END) as avg_gpu_usage_percent,
            AVG(CASE WHEN gpu_memory_mb IS NOT NULL THEN gpu_memory_mb END) as avg_gpu_memory_mb,
            COUNT(*) as execution_count,
            COUNT(CASE WHEN gpu_usage_percent IS NOT NULL THEN 1 END) as gpu_execution_count
        FROM node_performance
        WHERE workflow_name = %s AND workflow_id = %s AND user_id = %s
        GROUP BY node_id, node_name
        ORDER BY node_id
        """

        # SQLite인 경우 파라미터 플레이스홀더 변경
        if app_db.config_db_manager.db_type == "sqlite":
            query = query.replace("%s", "?")

        # 쿼리 실행
        result = app_db.config_db_manager.execute_query(query, (workflow_name, workflow_id, user_id))

        if not result:
            logger.info(f"No performance data found for workflow: {workflow_name} ({workflow_id})")
            return JSONResponse(content={
                "workflow_name": workflow_name,
                "workflow_id": workflow_id,
                "performance_stats": [],
                "message": "No performance data found for this workflow"
            })

        # Decimal 타입을 float로 변환하는 헬퍼 함수
        def safe_round_float(value, decimal_places=4):
            if value is None:
                return None
            try:
                # Decimal, float, int, str 모든 타입을 float로 변환
                if hasattr(value, '__float__'):  # Decimal 포함
                    return round(float(value), decimal_places)
                elif isinstance(value, (int, float)):
                    return round(float(value), decimal_places)
                elif isinstance(value, str):
                    return round(float(value), decimal_places)
                else:
                    return float(value) if value else 0.0
            except (ValueError, TypeError):
                return 0.0

        # 결과 포맷팅
        performance_stats = []
        for row in result:
            stats = {
                "node_id": row['node_id'],
                "node_name": row['node_name'],
                "avg_processing_time_ms": safe_round_float(row['avg_processing_time_ms']) if row['avg_processing_time_ms'] else 0.0,
                "avg_cpu_usage_percent": safe_round_float(row['avg_cpu_usage_percent']) if row['avg_cpu_usage_percent'] else 0.0,
                "avg_ram_usage_mb": safe_round_float(row['avg_ram_usage_mb']) if row['avg_ram_usage_mb'] else 0.0,
                "avg_gpu_usage_percent": safe_round_float(row['avg_gpu_usage_percent']) if row['avg_gpu_usage_percent'] else None,
                "avg_gpu_memory_mb": safe_round_float(row['avg_gpu_memory_mb']) if row['avg_gpu_memory_mb'] else None,
                "execution_count": int(row['execution_count']) if row['execution_count'] else 0,
                "gpu_execution_count": int(row['gpu_execution_count']) if row['gpu_execution_count'] else 0
            }
            performance_stats.append(stats)

        # 전체 워크플로우 통계 계산
        total_executions = sum(stat['execution_count'] for stat in performance_stats)
        avg_total_processing_time = sum(float(stat['avg_processing_time_ms']) * stat['execution_count'] for stat in performance_stats) / total_executions if total_executions > 0 else 0.0
        avg_total_cpu_usage = sum(float(stat['avg_cpu_usage_percent']) * stat['execution_count'] for stat in performance_stats) / total_executions if total_executions > 0 else 0.0
        avg_total_ram_usage = sum(float(stat['avg_ram_usage_mb']) * stat['execution_count'] for stat in performance_stats) / total_executions if total_executions > 0 else 0.0

        # GPU 통계 (GPU 데이터가 있는 경우만)
        gpu_stats = None
        total_gpu_executions = sum(stat['gpu_execution_count'] for stat in performance_stats)
        if total_gpu_executions > 0:
            gpu_processing_time_sum = sum(float(stat['avg_processing_time_ms']) * stat['gpu_execution_count'] for stat in performance_stats if stat['avg_gpu_usage_percent'] is not None)
            gpu_usage_sum = sum(float(stat['avg_gpu_usage_percent']) * stat['gpu_execution_count'] for stat in performance_stats if stat['avg_gpu_usage_percent'] is not None)
            gpu_memory_sum = sum(float(stat['avg_gpu_memory_mb']) * stat['gpu_execution_count'] for stat in performance_stats if stat['avg_gpu_memory_mb'] is not None)

            gpu_stats = {
                "avg_gpu_usage_percent": round(float(gpu_usage_sum / total_gpu_executions), 2) if total_gpu_executions > 0 else None,
                "avg_gpu_memory_mb": round(float(gpu_memory_sum / total_gpu_executions), 2) if total_gpu_executions > 0 else None,
                "gpu_execution_count": total_gpu_executions
            }

        response_data = {
            "workflow_name": workflow_name,
            "workflow_id": workflow_id,
            "summary": {
                "total_executions": total_executions,
                "avg_total_processing_time_ms": round(float(avg_total_processing_time), 2),
                "avg_total_cpu_usage_percent": round(float(avg_total_cpu_usage), 2),
                "avg_total_ram_usage_mb": round(float(avg_total_ram_usage), 2),
                "gpu_stats": gpu_stats
            },
            "performance_stats": performance_stats
        }

        logger.info(f"Performance stats retrieved for workflow: {workflow_name} ({workflow_id})")
        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"Error retrieving workflow performance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve performance data: {str(e)}")

@router.delete("/performance")
async def delete_workflow_performance(request: Request, workflow_name: str, workflow_id: str):
    """
    특정 워크플로우의 성능 데이터를 삭제합니다.

    Args:
        workflow_name: 워크플로우 이름
        workflow_id: 워크플로우 ID

    Returns:
        삭제된 성능 데이터 개수와 성공 메시지
    """
    try:
        user_id = extract_user_id_from_request(request)
        app_db = get_db_manager(request)
        existing_data = app_db.find_by_condition(
            NodePerformance,
            {
                "user_id": user_id,
                "workflow_name": workflow_name,
                "workflow_id": workflow_id
            },
            limit=1000000
        )

        delete_count = len(existing_data) if existing_data else 0

        if delete_count == 0:
            logger.info(f"No performance data found to delete for workflow: {workflow_name} ({workflow_id})")
            return JSONResponse(content={
                "workflow_name": workflow_name,
                "workflow_id": workflow_id,
                "deleted_count": 0,
                "message": "No performance data found to delete"
            })

        app_db.delete_by_condition(
            NodePerformance,
            {
                "user_id": user_id,
                "workflow_name": workflow_name,
                "workflow_id": workflow_id
            }
        )

        response_data = {
            "user_id": user_id,
            "workflow_name": workflow_name,
            "workflow_id": workflow_id,
            "deleted_count": delete_count,
            "message": f"Successfully deleted {delete_count} performance records"
        }

        logger.info(f"Deleted {delete_count} performance records for workflow: {workflow_name} ({workflow_id})")
        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"Error deleting performance data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete performance data: {str(e)}")

def parse_input_data(input_data_str: str) -> str:
    """
    input_data에서 실제 입력 텍스트만 파싱합니다.

    Args:
        input_data_str: 파싱할 입력 데이터 문자열

    Returns:
        파싱된 입력 텍스트
    """
    if not input_data_str or not isinstance(input_data_str, str):
        return input_data_str

    # "Input: " 패턴으로 시작하는지 확인
    if input_data_str.startswith("Input: "):
        # "Input: " 이후의 텍스트 추출
        after_input = input_data_str[7:]  # "Input: " 길이만큼 자르기

        # "\n\nparameters:" 또는 "\n\nAdditional Parameters:" 패턴 찾기
        patterns = ["\n\nparameters:", "\n\nAdditional Parameters:", "\n\nValidation Error:"]

        for pattern in patterns:
            if pattern in after_input:
                # 패턴 앞까지의 텍스트 반환
                return after_input.split(pattern)[0].strip()

        # 패턴이 없으면 전체 텍스트 반환 (Input: 이후)
        return after_input.strip()

    # "Input: " 패턴이 없으면 원본 반환
    return input_data_str

@router.get("/io_logs")
async def get_workflow_io_logs(request: Request, workflow_name: str, workflow_id: str, interaction_id: str = 'default'):
    """
    특정 워크플로우의 ExecutionIO 로그를 반환합니다.

    Args:
        workflow_name: 워크플로우 이름
        workflow_id: 워크플로우 ID
        interaction_id: 상호작용 ID (선택적, 제공되지 않으면 default만 반환)

    Returns:
        ExecutionIO 로그 리스트
    """
    try:
        user_id = extract_user_id_from_request(request)
        app_db = get_db_manager(request)
        result = app_db.find_by_condition(
            ExecutionIO,
            {
                "user_id": user_id,
                "workflow_name": workflow_name,
                # "workflow_id": workflow_id, # workflow_id 로직 삭제
                "interaction_id": interaction_id
            },
            limit=1000000,  # 필요에 따라 조정 가능
            orderby="updated_at",
            orderby_asc=True,
            return_list=True
        )

        if not result:
            logger.info(f"No performance data found for workflow: {workflow_name} ({workflow_id})")
            return JSONResponse(content={
                "workflow_name": workflow_name,
                "workflow_id": workflow_id,
                "in_out_logs": [],
                "message": "No in_out_logs data found for this workflow"
            })

        performance_stats = []
        for idx, row in enumerate(result):
            # input_data 파싱
            raw_input_data = json.loads(row['input_data']).get('result', None) if row['input_data'] else None
            parsed_input_data = parse_input_data(raw_input_data) if raw_input_data else None

            log_entry = {
                "log_id": idx + 1,
                "interaction_id": row['interaction_id'],
                "workflow_name": row['workflow_name'],
                "workflow_id": row['workflow_id'],
                "input_data": parsed_input_data,
                "output_data": json.loads(row['output_data']).get('result', None) if row['output_data'] else None,
                "updated_at": row['updated_at'].isoformat() if isinstance(row['updated_at'], datetime) else row['updated_at']
            }
            performance_stats.append(log_entry)

        response_data = {
            "workflow_name": workflow_name,
            "workflow_id": workflow_id,
            "in_out_logs": performance_stats,
            "message": "In/Out logs retrieved successfully"
        }

        logger.info(f"Performance stats retrieved for workflow: {workflow_name} ({workflow_id})")
        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"Error retrieving workflow performance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve performance data: {str(e)}")


@router.delete("/io_logs")
async def delete_workflow_io_logs(request: Request, workflow_name: str, workflow_id: str, interaction_id: str = "default"):
    """
    특정 워크플로우의 ExecutionIO 로그를 삭제합니다.

    Args:
        workflow_name: 워크플로우 이름
        workflow_id: 워크플로우 ID
        interaction_id: 상호작용 ID (기본값: "default")

    Returns:
        삭제된 로그 개수와 성공 메시지
    """
    try:
        user_id = extract_user_id_from_request(request)
        app_db = get_db_manager(request)

        existing_data = app_db.find_by_condition(
            ExecutionIO,
            {
                "user_id": user_id,
                "workflow_name": workflow_name,
                # "workflow_id": workflow_id, # workflow_id 로직 삭제
                "interaction_id": interaction_id
            },
            limit=1000000
        )

        delete_count = len(existing_data) if existing_data else 0

        if delete_count == 0:
            logger.info(f"No logs found to delete for workflow: {workflow_name} ({workflow_id}), interaction_id: {interaction_id}")
            return JSONResponse(content={
                "workflow_name": workflow_name,
                # "workflow_id": workflow_id,
                "interaction_id": interaction_id,
                "deleted_count": 0,
                "message": "No logs found to delete"
            })

        app_db.delete_by_condition(
            ExecutionIO,
            {
                "user_id": user_id,
                "workflow_name": workflow_name,
                # "workflow_id": workflow_id, # workflow_id 로직 삭제
                "interaction_id": interaction_id
            }
        )
        app_db.delete_by_condition(
            ExecutionMeta,
            {
                "user_id": user_id,
                "workflow_name": workflow_name,
                # "workflow_id": workflow_id, # workflow_id 로직 삭제
                "interaction_id": interaction_id
            }
        )

        logger.info(f"Successfully deleted {delete_count} logs for workflow: {workflow_name} ({workflow_id}), interaction_id: {interaction_id}")

        return JSONResponse(content={
            "workflow_name": workflow_name,
            # "workflow_id": workflow_id,
            "interaction_id": interaction_id,
            "deleted_count": delete_count,
            "message": f"Successfully deleted {delete_count} execution logs"
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting workflow logs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete workflow logs: {str(e)}")

@router.post("/execute/based_id", response_model=Dict[str, Any])
async def execute_workflow_with_id(request: Request, request_body: WorkflowRequest):
    """
    주어진 노드와 엣지 정보로 워크플로우를 실행합니다.
    """
    try:
        user_id = extract_user_id_from_request(request)
        app_db = get_db_manager(request)
        extracted_user_id = workflow_user_id_extractor(app_db, user_id, request_body.user_id, request_body.workflow_name)
        ## 일반채팅인 경우 미리 정의된 워크플로우를 이용하여 일반 채팅에 사용.
        if request_body.workflow_name == 'default_mode':
            default_mode_workflow_folder = os.path.join(os.getcwd(), "constants")
            file_path = os.path.join(default_mode_workflow_folder, "base_chat_workflow.json")
            with open(file_path, 'r', encoding='utf-8') as f:
                workflow_data = json.load(f)
            workflow_data = await _default_workflow_parameter_helper(request, request_body, workflow_data)

        ## 워크플로우 실행인 경우, 해당하는 워크플로우 파일을 찾아서 사용.
        else:
            downloads_path = os.path.join(os.getcwd(), "downloads")
            download_path_id = os.path.join(downloads_path, extracted_user_id)

            if not request_body.workflow_name.endswith('.json'):
                filename = f"{request_body.workflow_name}.json"
            else:
                filename = request_body.workflow_name
            file_path = os.path.join(download_path_id, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                workflow_data = json.load(f)
            workflow_data = await _workflow_parameter_helper(request_body, workflow_data)

        ## ========== 워크플로우 데이터 검증 ==========
        ## TODO 워크플로우 아이디 정합성 관련 로직 생각해볼 것
        # if workflow_data.get('workflow_id') != request_body.workflow_id:
        #     raise ValueError(f"워크플로우 ID가 일치하지 않습니다: {workflow_data.get('workflow_id')} != {request_body.workflow_id}")

        if not workflow_data or 'nodes' not in workflow_data or 'edges' not in workflow_data:
            raise ValueError(f"워크플로우 데이터가 유효하지 않습니다: {file_path}")

        print("--- 워크플로우 실행 시작 ---")
        print(f"실행 워크플로우: {request_body.workflow_name} ({request_body.workflow_id})")
        print(f"입력 데이터: {request_body.input_data}")

        ## 모든 워크플로우는 startnode가 있어야 하며, 입력 데이터는 startnode의 첫 번째 파라미터로 설정되어야 함.
        ## 사용자의 인풋은 여기에 입력되고, 워크플로우가 실행됨.
        if request_body.input_data is not None and request_body.input_data != "" and len(request_body.input_data) > 0:
            for node in workflow_data.get('nodes', []):
                if node.get('data', {}).get('functionId') == 'startnode':
                    parameters = node.get('data', {}).get('parameters', [])
                    if parameters and isinstance(parameters, list):
                        parameters[0]['value'] = request_body.input_data
                        break

        ## app에 저장된 db_manager를 가져옴. 이걸 통해 DB에 접근할 수 있음.
        ## DB에 접근하여 execution 데이터를 활용하여, 기록된 대화를 가져올지 말지 결정.
        app_db = get_db_manager(request)

        ## 일반적인 실행(execution)이 아닌 경우, 즉 대화형 실행(conversation execution)인 경우
        ## interaction_id가 "default"가 아닌 경우, 대화형 실행을 위한 메타데이터를 가져오거나 생성
        ## interaction_id가 "default"인 경우, execution_meta는 None으로 설정
        execution_meta = None
        if request_body.interaction_id != "default" and app_db:
            execution_meta = await get_or_create_execution_meta(
                app_db,
                user_id,
                request_body.interaction_id,
                request_body.workflow_id,
                request_body.workflow_name,
                request_body.input_data
            )

        ## 워크플로우를 실질적으로 비동기 실행 (가장중요)
        ## 비동기 워크플로우 실행 관련 로직은 AsyncWorkflowExecutor 클래스에 정의되어 있음.
        ## AsyncWorkflowExecutor 클래스는 워크플로우의 노드와 엣지를 기반으로 워크플로우를 백그라운드에서 실행하는 역할을 함.
        ## 워크플로우 실행 시, interaction_id를 전달하여 대화형 실행을 지원함. (이렇게 되는 경우, interaction_id는 대화형 실행의 ID로 사용되어 DB에 저장됨)
        ## 워크플로우 실행 결과는 final_outputs에 저장됨.
        executor = execution_manager.create_executor(
            workflow_data=workflow_data,
            db_manager=app_db,
            interaction_id=request_body.interaction_id,
            user_id=user_id
        )

        # 비동기 실행 및 결과 수집
        final_outputs = []
        async for output in executor.execute_workflow_async():
            final_outputs.append(output)

        ## 대화형 실행인 경우 execution_meta의 값을 가지고, 이 경우에는 대화 count를 증가.
        if execution_meta:
            await update_execution_meta_count(app_db, execution_meta)

        response_data = {"status": "success", "message": "워크플로우 실행 완료", "outputs": final_outputs}

        if execution_meta:
            response_data["execution_meta"] = {
                "interaction_id": execution_meta.interaction_id,
                "interaction_count": execution_meta.interaction_count + 1,
                "workflow_id": execution_meta.workflow_id,
                "workflow_name": execution_meta.workflow_name
            }

        return response_data

    except ValueError as e:
        logging.error(f"Workflow execution error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    finally:
        # 완료된 실행들 정리
        execution_manager.cleanup_completed_executions()

@router.post("/execute/based_id/stream")
async def execute_workflow_with_id_stream(request: Request, request_body: WorkflowRequest):
    """
    주어진 ID를 기반으로 워크플로우를 스트리밍 방식으로 실행합니다.
    """
    user_id = extract_user_id_from_request(request)
    app_db = get_db_manager(request)
    extracted_user_id = workflow_user_id_extractor(app_db, user_id, request_body.user_id, request_body.workflow_name)

    async def stream_generator(async_result_generator, db_manager, user_id, workflow_req):
        """결과 제너레이터를 SSE 형식으로 변환하는 비동기 제너레이터"""
        full_response_chunks = []
        try:
            async for chunk in async_result_generator:
                # 클라이언트에 보낼 데이터 형식 정의 (JSON)
                full_response_chunks.append(str(chunk))
                response_chunk = {"type": "data", "content": chunk}
                yield f"data: {json.dumps(response_chunk, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.01) # 짧은 딜레이로 이벤트 스트림 안정화

            end_message = {"type": "end", "message": "Stream finished"}
            yield f"data: {json.dumps(end_message)}\n\n"

        except Exception as e:
            logger.error(f"스트리밍 중 오류 발생: {e}", exc_info=True)
            error_message = {"type": "error", "detail": f"스트리밍 중 오류가 발생했습니다: {str(e)}"}
            yield f"data: {json.dumps(error_message)}\n\n"
        finally:
            # ✨ 2. 스트림이 모두 끝난 후, 수집된 내용으로 DB 로그 업데이트
            final_text = "".join(full_response_chunks)
            if not final_text:
                logger.info("스트림 결과가 비어있어 로그를 업데이트하지 않습니다.")
                return

            try:
                logger.info(f"스트림 완료. Interaction ID [{workflow_req.interaction_id}]의 로그 업데이트 시작.")

                # 가장 최근에 생성된 로그 레코드를 찾습니다.
                log_to_update_list = db_manager.find_by_condition(
                    ExecutionIO,
                    {
                        "user_id": user_id,
                        "interaction_id": workflow_req.interaction_id,
                        # "workflow_id": workflow_req.workflow_id, # 워크플로우 ID 로직 삭제
                    },
                    limit=1,
                    orderby="created_at",
                    orderby_asc=False
                )

                if not log_to_update_list:
                    logger.warning(f"업데이트할 ExecutionIO 로그를 찾지 못했습니다. Interaction ID: {workflow_req.interaction_id}")
                    return

                log_to_update = log_to_update_list[0]

                # output_data 필드의 JSON을 실제 결과로 수정
                output_data_dict = json.loads(log_to_update.output_data)
                output_data_dict['result'] = final_text # placeholder를 최종 텍스트로 교체

                # inputs 필드에 있던 generator placeholder도 업데이트 (선택적)
                if 'inputs' in output_data_dict and isinstance(output_data_dict['inputs'], dict):
                    for key, value in output_data_dict['inputs'].items():
                        if value == "<generator_output>":
                            output_data_dict['inputs'][key] = final_text

                # 수정된 JSON으로 레코드를 업데이트
                log_to_update.output_data = json.dumps(output_data_dict, ensure_ascii=False)
                db_manager.update(log_to_update)

                logger.info(f"Interaction ID [{workflow_req.interaction_id}]의 로그가 최종 스트림 결과로 업데이트되었습니다.")

            except Exception as db_error:
                logger.error(f"ExecutionIO 로그 업데이트 중 DB 오류 발생: {db_error}", exc_info=True)
            finally:
                # 완료된 실행들 정리
                execution_manager.cleanup_completed_executions()


    try:
        if request_body.workflow_name == 'default_mode':
            default_mode_workflow_folder = os.path.join(os.getcwd(), "constants")
            file_path = os.path.join(default_mode_workflow_folder, "base_chat_workflow.json")
            with open(file_path, 'r', encoding='utf-8') as f:
                workflow_data = json.load(f)
            workflow_data = await _default_workflow_parameter_helper(request, request_body, workflow_data)
        else:
            downloads_path = os.path.join(os.getcwd(), "downloads")
            download_path_id = os.path.join(downloads_path, extracted_user_id)
            filename = f"{request_body.workflow_name}.json" if not request_body.workflow_name.endswith('.json') else request_body.workflow_name
            file_path = os.path.join(download_path_id, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                workflow_data = json.load(f)
            workflow_data = await _workflow_parameter_helper(request_body, workflow_data)

        ## TODO 워크플로우 아이디 정합성 관련 로직 생각해볼 것
        # if workflow_data.get('workflow_id') != request_body.workflow_id:
        #     raise ValueError(f"워크플로우 ID가 일치하지 않습니다.")
        if not workflow_data or 'nodes' not in workflow_data or 'edges' not in workflow_data:
            raise ValueError(f"워크플로우 데이터가 유효하지 않습니다: {file_path}")

        if request_body.input_data is not None and request_body.input_data != "" and len(request_body.input_data) > 0:
            for node in workflow_data.get('nodes', []):
                if node.get('data', {}).get('functionId') == 'startnode':
                    parameters = node.get('data', {}).get('parameters', [])
                    if parameters:
                        parameters[0]['value'] = request_body.input_data
                        break

        app_db = get_db_manager(request)
        execution_meta = None
        if request_body.interaction_id != "default" and app_db:
            execution_meta = await get_or_create_execution_meta(
                app_db, user_id, request_body.interaction_id,
                request_body.workflow_id, request_body.workflow_name, request_body.input_data
            )

        if execution_meta:
            await update_execution_meta_count(app_db, execution_meta)

        # 비동기 실행기 생성
        executor = execution_manager.create_executor(
            workflow_data=workflow_data,
            db_manager=app_db,
            interaction_id=request_body.interaction_id,
            user_id=user_id
        )

        # 비동기 제너레이터 시작 (스트리밍용)
        result_generator = executor.execute_workflow_async_streaming()

        # StreamingResponse를 사용하여 SSE 스트림 반환
        return StreamingResponse(
            stream_generator(result_generator, app_db, user_id, request_body),
            media_type="text/event-stream"
        )

    except ValueError as e:
        logger.error(f"Workflow execution error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"An unexpected error occurred during workflow setup: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

class TesterTestCase(BaseModel):
    """테스터 테스트 케이스 모델"""
    id: int
    input: str
    expected_output: Optional[str] = None

class TesterExecuteRequest(BaseModel):
    """테스터 실행 요청 모델"""
    workflow_name: str
    workflow_id: str
    test_cases: List[TesterTestCase]
    batch_size: int = 5
    interaction_id: str = "batch_test"
    selected_collections: Optional[List[str]] = None
    llm_eval_enabled: Optional[bool] = False
    llm_eval_type: Optional[str] = None
    llm_eval_model: Optional[str] = None

class TesterTestResult(BaseModel):
    """테스터 테스트 결과 모델"""
    id: int
    input: str
    expected_output: Optional[str]
    actual_output: Optional[str]
    status: str  # 'success', 'error'
    execution_time: Optional[int]  # milliseconds
    error: Optional[str]
    llm_eval_score: Optional[float] = None  # LLM 평가 점수 (0.0 ~ 1.0)

class ScoreModelParser(BaseModel):
    llm_eval_score: float = Field(description="주어진 데이터를 평가하여 0~1점의 점수로 반환합니다. 소수점 2째 자리까지 표현하십시오 (0.00 ~ 1.00)", ge=0.00, le=1.00)

# 테스터 작업 상태 저장용 (메모리 기반)
tester_status_storage = {}

async def evaluate_with_llm(
    unique_interaction_id: str,
    input_data: str,
    expected_output: Optional[str],
    actual_output: str,
    llm_eval_type: str,
    llm_eval_model: str,
    app_db,
    config_composer
) -> float:
    """
    LLM을 사용하여 실제 출력과 예상 출력을 비교하고 점수를 반환합니다.

    Args:
        unique_interaction_id: 고유 상호작용 ID
        input_data: 입력 데이터
        expected_output: 예상 출력 (레퍼런스)
        actual_output: 실제 출력
        llm_eval_type: LLM 평가 타입
        llm_eval_model: 사용할 LLM 모델
        app_db: 데이터베이스 매니저

    Returns:
        평가 점수 (0.0 ~ 1.0)
    """
    logger.info(f"LLM 평가 시작: unique_interaction_id={unique_interaction_id}")

    if '<think>' in actual_output and '</think>' in actual_output:
        actual_output = re.sub(r'<think>.*?</think>', '', actual_output, flags=re.DOTALL).strip()

    if '[Cite.' in actual_output and '}}]' in actual_output:
        actual_output = re.sub(r'\[Cite\.\s*\{\{.*?\}\}\]', '', actual_output, flags=re.DOTALL).strip()

    if '<TOOLUSELOG>' in actual_output and '</TOOLUSELOG>' in actual_output:
        actual_output = re.sub(r'<TOOLUSELOG>.*?</TOOLUSELOG>', '', actual_output, flags=re.DOTALL).strip()

    if '<TOOLOUTPUTLOG>' in actual_output and '</TOOLOUTPUTLOG>' in actual_output:
        actual_output = re.sub(r'<TOOLOUTPUTLOG>.*?</TOOLOUTPUTLOG>', '', actual_output, flags=re.DOTALL).strip()

    try:
        if llm_eval_type == "OpenAI":
            api_key = config_composer.get_config_by_name("OPENAI_API_KEY").value
            base_url = "https://api.openai.com/v1"
            model_name = llm_eval_model
            if not api_key:
                logger.error(f"[LLM_EVAL] OpenAI API 키가 설정되지 않았습니다")
                raise ValueError("OpenAI API 키가 설정되지 않았습니다.")

        elif llm_eval_type == "vLLM":
            api_key = None
            base_url = config_composer.get_config_by_name("VLLM_API_BASE_URL").value
            model_name = config_composer.get_config_by_name("VLLM_MODEL_NAME").value

        else:
            raise ValueError(f"지원되지 않는 LLM 평가 타입입니다: {llm_eval_type}")

        temperature = 0.1
        if llm_eval_model == "gpt-5" or llm_eval_model == "gpt-5-nano" or llm_eval_model == "gpt-5-mini":
            temperature = 1

        llm_client = ChatOpenAI(
            api_key=api_key,
            model=model_name,
            temperature=temperature,
            max_tokens=1000,
            base_url=base_url
        )

        parser = JsonOutputParser(pydantic_object=ScoreModelParser)

        system_msg = SystemMessage(
            content="""당신은 정확한 답변 평가 전문가입니다.
주어진 입력에 대해 실제 생성된 답변이 레퍼런스 정답과 얼마나 일치하는지 평가해주세요.

평가 기준:
1. 레퍼런스 정답에서 요구하는 핵심 정보나 값이 정확히 포함되어 있는가?
2. 답변이 적절해 보여도 레퍼런스가 지정하는 정확한 값과 다르면 낮은 점수를 주어야 합니다.
3. 부분적으로 맞더라도 핵심 내용이 틀리면 낮은 점수를 주어야 합니다.
4. 완전히 정확한 경우에만 높은 점수(0.9-1.0)를 주세요.

점수 기준:
- 1.0: 레퍼런스와 완전히 일치하거나 동등한 정확성
- 0.7-0.9: 대부분 정확하지만 일부 세부사항이 다름
- 0.4-0.6: 부분적으로 맞지만 중요한 부분이 틀림
- 0.1-0.3: 대부분 틀렸지만 일부 관련성 있음
- 0.0: 완전히 틀렸거나 관련성 없음

응답은 반드시 JSON 형식으로 소수점 2자리까지 정확하게 제공해주세요."""
        )

        evaluation_prompt = f"""다음 내용을 평가해주세요:

**입력 질문/요청:**
{input_data}

**레퍼런스 정답 (기준):**
{expected_output or "없음"}

**실제 생성된 답변:**
{actual_output}

위 실제 답변이 레퍼런스 정답과 얼마나 정확히 일치하는지 0.00~1.00 사이의 점수로 평가해주세요.
**답변 형식**
{parser.get_format_instructions()}"""
        human_msg = HumanMessage(content=evaluation_prompt)

        # LLM 호출
        response = await llm_client.ainvoke([system_msg, human_msg])
        content = response.content.strip()

        # JSON 파싱
        parsed_result = parser.parse(content)
        try:
            score = parsed_result.llm_eval_score
        except:
            score = parsed_result.get('llm_eval_score', 0.0)

        score = max(0.0, min(1.0, round(score, 2)))

        # DB 업데이트
        existing_data = app_db.find_by_condition(
            ExecutionIO,
            {
                "interaction_id": unique_interaction_id,
            },
            limit=1
        )

        if existing_data:
            updated_data = existing_data[0]
            updated_data.llm_eval_score = score
            app_db.update(updated_data)
        else:
            logger.warning(f"해당 interaction_id로 레코드를 찾을 수 없습니다: {unique_interaction_id}")

        return score

    except Exception as e:
        logger.error(f"LLM 평가 중 오류 발생: {str(e)}", exc_info=True)
        fallback_score = 0.0
        return fallback_score


async def process_batch_group(
    user_id: str,
    workflow_name: str,
    workflow_id: str,
    test_cases: List[TesterTestCase],
    interaction_id: str,
    selected_collections: Optional[List[str]],
    batch_id: str,
    app_db,
    individual_result_callback=None,
) -> List[TesterTestResult]:
    """
    배치 그룹을 병렬로 처리하며 개별 완료 시마다 콜백 호출
    """
    results = []

    # asyncio.gather를 사용해서 병렬 실행
    tasks = []
    for test_case in test_cases:
        unique_interaction_id = f"{interaction_id}____{workflow_name}____{batch_id}____{test_case.id}"
        task = execute_single_workflow_for_tester_with_callback(
            user_id=user_id,
            workflow_name=workflow_name,
            workflow_id=workflow_id,
            input_data=test_case.input,
            interaction_id=unique_interaction_id,
            selected_collections=selected_collections,
            app_db=app_db,
            test_case=test_case,
            callback=individual_result_callback,
            expected_output=test_case.expected_output,
        )
        tasks.append(task)

    # 모든 태스크를 병렬로 실행
    execution_results = await asyncio.gather(*tasks, return_exceptions=True)

    # 결과 처리 (콜백에서 이미 처리되었지만 최종 반환용)
    for test_case, exec_result in zip(test_cases, execution_results):
        if isinstance(exec_result, Exception):
            result = TesterTestResult(
                id=test_case.id,
                input=test_case.input,
                expected_output=test_case.expected_output,
                actual_output=None,
                status="error",
                execution_time=0,
                error=str(exec_result),
                llm_eval_score=None
            )
        elif exec_result.get("success"):
            # outputs 처리 - 다양한 형태의 결과를 문자열로 변환
            outputs = exec_result.get("outputs", "결과 없음")
            if isinstance(outputs, list):
                actual_output = outputs[0] if outputs else "결과 없음"
            else:
                actual_output = str(outputs)

            result = TesterTestResult(
                id=test_case.id,
                input=test_case.input,
                expected_output=test_case.expected_output,
                actual_output=actual_output,
                status="success",
                execution_time=exec_result.get("execution_time", 0),
                error=None,
                llm_eval_score=None
            )
        else:
            result = TesterTestResult(
                id=test_case.id,
                input=test_case.input,
                expected_output=test_case.expected_output,
                actual_output=None,
                status="error",
                execution_time=exec_result.get("execution_time", 0),
                error=exec_result.get("error", "알 수 없는 오류"),
                llm_eval_score=None
            )

        results.append(result)

    # 진행 상황 업데이트 (배치 그룹 처리 완료 후 한 번에 업데이트)
    if batch_id in tester_status_storage:
        tester_status_storage[batch_id]["completed_count"] += len(test_cases)
        progress = (tester_status_storage[batch_id]["completed_count"] /
                   tester_status_storage[batch_id]["total_count"]) * 100
        tester_status_storage[batch_id]["progress"] = progress

    return results

async def execute_single_workflow_for_tester_with_callback(
    user_id: str,
    workflow_name: str,
    workflow_id: str,
    input_data: str,
    interaction_id: str,
    selected_collections: Optional[List[str]],
    app_db,
    test_case: TesterTestCase,
    callback=None,
    expected_output=None
) -> Dict[str, Any]:
    """
    개별 워크플로우 실행 후 콜백 호출
    """
    start_time = time.time()

    try:
        downloads_path = os.path.join(os.getcwd(), "downloads")
        download_path_id = os.path.join(downloads_path, user_id)

        filename = f"{workflow_name}.json" if not workflow_name.endswith('.json') else workflow_name
        file_path = os.path.join(download_path_id, filename)

        if not os.path.exists(file_path):
            raise ValueError(f"워크플로우 파일을 찾을 수 없습니다: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            workflow_data = json.load(f)

        temp_request = WorkflowRequest(
            workflow_name=workflow_name,
            workflow_id=workflow_id,
            input_data=input_data,
            interaction_id=interaction_id,
            selected_collections=selected_collections
        )
        workflow_data = await _workflow_parameter_helper(temp_request, workflow_data)

        if not workflow_data or 'nodes' not in workflow_data or 'edges' not in workflow_data:
            raise ValueError(f"워크플로우 데이터가 유효하지 않습니다")

        if input_data is not None:
            for node in workflow_data.get('nodes', []):
                if node.get('data', {}).get('functionId') == 'startnode':
                    parameters = node.get('data', {}).get('parameters', [])
                    if parameters and isinstance(parameters, list):
                        parameters[0]['value'] = input_data
                        break

        executor = execution_manager.create_executor(
            workflow_data=workflow_data,
            db_manager=app_db,
            interaction_id=interaction_id,
            user_id=user_id,
            expected_output=expected_output,
            test_mode=True
        )

        final_outputs = []
        async for chunk in executor.execute_workflow_async():
            final_outputs.append(chunk)

        if len(final_outputs) == 1:
            processed_output = final_outputs[0]
        elif len(final_outputs) > 1:
            if all(isinstance(item, str) for item in final_outputs):
                processed_output = ''.join(final_outputs)
            else:
                processed_output = final_outputs[-1]
        else:
            processed_output = "결과 없음"

        execution_time = int((time.time() - start_time) * 1000)

        result = {
            "success": True,
            "outputs": processed_output,
            "execution_time": execution_time
        }

        # 콜백 호출 - 개별 테스트 케이스 완료 시
        if callback:
            tester_result = TesterTestResult(
                id=test_case.id,
                input=test_case.input,
                expected_output=test_case.expected_output,
                actual_output=str(processed_output),
                status="success",
                execution_time=execution_time,
                error=None,
                llm_eval_score=None
            )
            await callback(tester_result)

        return result

    except Exception as e:
        execution_time = int((time.time() - start_time) * 1000)
        logger.error(f"테스터 워크플로우 실행 실패: {str(e)}")

        result = {
            "success": False,
            "error": str(e),
            "execution_time": execution_time
        }

        # 콜백 호출 - 에러 발생 시
        if callback:
            tester_result = TesterTestResult(
                id=test_case.id,
                input=test_case.input,
                expected_output=test_case.expected_output,
                actual_output=None,
                status="error",
                execution_time=execution_time,
                error=str(e),
                llm_eval_score=None
            )
            await callback(tester_result)

        return result

# ==================================================
# 테스터 실행 API 엔드포인트 / tester로 만듦.
# ==================================================
@router.get("/tester/io_logs")
async def get_workflow_io_logs_for_tester(request: Request, workflow_name: str):
    """
    특정 워크플로우의 ExecutionIO 로그를 interaction_batch_id별로 그룹화하여 반환합니다.

    Args:
        workflow_name: 워크플로우 이름

    Returns:
        interaction_batch_id별로 그룹화된 ExecutionIO 로그 리스트
    """
    try:
        user_id = extract_user_id_from_request(request)
        app_db = get_db_manager(request)
        result = app_db.find_by_condition(
            ExecutionIO,
            {
                "user_id": user_id,
                "workflow_name": workflow_name,
                "test_mode": True
            },
            limit=1000000,  # 필요에 따라 조정 가능
            orderby="updated_at",
            orderby_asc=True,
            return_list=True
        )

        if not result:
            logger.info(f"No performance data found for workflow: {workflow_name}")
            return JSONResponse(content={
                "workflow_name": workflow_name,
                "response_data_list": [],
                "message": "No in_out_logs data found for this workflow"
            })

        # interaction_batch_id별로 그룹화
        tester_groups = {}

        for idx, row in enumerate(result):
            interaction_id = row['interaction_id']

            # interaction_id에서 마지막 숫자를 제외한 배치 ID 추출
            # 예: batch_test____Workflow____a4e5af3a-c975-4356-8a72-6dcf74695c9f____4
            # -> batch_test____Workflow____a4e5af3a-c975-4356-8a72-6dcf74695c9f
            parts = interaction_id.split('____')
            if len(parts) >= 4:
                interaction_batch_id = '____'.join(parts[:-1])
            else:
                interaction_batch_id = interaction_id

            if interaction_batch_id not in tester_groups:
                tester_groups[interaction_batch_id] = []

            # input_data 파싱
            raw_input_data = json.loads(row['input_data']).get('result', None) if row['input_data'] else None
            parsed_input_data = parse_input_data(raw_input_data) if raw_input_data else None

            # interaction_id에서 마지막 번호를 추출하여 log_id로 사용
            # 예: batch_test____Workflow____a4e5af3a-c975-4356-8a72-6dcf74695c9f____4 -> 4
            parts = interaction_id.split('____')
            if len(parts) >= 4 and parts[-1].isdigit():
                log_id = int(parts[-1])
            else:
                log_id = len(tester_groups[interaction_batch_id]) + 1

            log_entry = {
                "log_id": log_id,
                "interaction_id": row['interaction_id'],
                "workflow_name": row['workflow_name'],
                "workflow_id": row['workflow_id'],
                "input_data": parsed_input_data,
                "output_data": json.loads(row['output_data']).get('result', None) if row['output_data'] else None,
                "expected_output": row['expected_output'],
                "llm_eval_score": row['llm_eval_score'],
                "updated_at": row['updated_at'].isoformat() if isinstance(row['updated_at'], datetime) else row['updated_at']
            }
            tester_groups[interaction_batch_id].append(log_entry)

        # 각 테스터 그룹을 response_data 형태로 변환
        response_data_list = []
        for interaction_batch_id, performance_stats in tester_groups.items():
            response_data = {
                "workflow_name": workflow_name,
                "interaction_batch_id": interaction_batch_id,
                "in_out_logs": performance_stats,
                "message": "In/Out logs retrieved successfully"
            }
            response_data_list.append(response_data)

        final_response = {
            "workflow_name": workflow_name,
            "response_data_list": response_data_list,
            "message": f"In/Out logs retrieved successfully for {len(response_data_list)} tester groups"
        }

        logger.info(f"Performance stats retrieved for workflow: {workflow_name}, {len(response_data_list)} tester groups")
        return JSONResponse(content=final_response)

    except Exception as e:
        logger.error(f"Error retrieving workflow performance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve performance data: {str(e)}")

@router.delete("/tester/io_logs")
async def delete_workflow_io_logs_for_tester(request: Request, workflow_name: str, interaction_batch_id: str):
    """
    특정 워크플로우의 ExecutionIO 로그를 삭제합니다.

    Args:
        workflow_name: 워크플로우 이름
        workflow_id: 워크플로우 ID
        interaction_id: 상호작용 ID (기본값: "default")

    Returns:
        삭제된 로그 개수와 성공 메시지
    """
    try:
        user_id = extract_user_id_from_request(request)
        app_db = get_db_manager(request)

        existing_data = app_db.find_by_condition(
            ExecutionIO,
            {
                "user_id": user_id,
                "workflow_name": workflow_name,
                "interaction_id__like__": interaction_batch_id,
                "test_mode": True
            },
            limit=1000000
        )

        delete_count = len(existing_data) if existing_data else 0

        if delete_count == 0:
            logger.info(f"No logs found to delete for workflow: {workflow_name}")
            return JSONResponse(content={
                "workflow_name": workflow_name,
                # "workflow_id": workflow_id,
                "deleted_count": 0,
                "message": "No logs found to delete"
            })

        app_db.delete_by_condition(
            ExecutionIO,
            {
                "user_id": user_id,
                "workflow_name": workflow_name,
                # "workflow_id": workflow_id, # workflow_id 로직 삭제,
                "interaction_id__like__": interaction_batch_id,
                "test_mode": True,
            }
        )

        logger.info(f"Successfully deleted {delete_count} logs for workflow: {workflow_name}")

        return JSONResponse(content={
            "workflow_name": workflow_name,
            "deleted_count": delete_count,
            "interaction_batch_id": interaction_batch_id,
            "message": f"Successfully deleted {delete_count} execution logs"
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting workflow logs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete workflow logs: {str(e)}")

@router.post("/execute/tester/stream")
async def execute_workflow_tester_stream(request: Request, tester_request: TesterExecuteRequest):
    """
    워크플로우 테스터 실행 스트리밍 엔드포인트
    여러 테스트 케이스를 배치로 처리하며 개별 완료 시마다 실시간 진행 상황을 SSE로 스트리밍합니다.

    Args:
        tester_request: 테스터 실행 요청 데이터

    Returns:
        StreamingResponse: SSE 형식의 실시간 테스터 실행 결과
    """
    user_id = extract_user_id_from_request(request)
    app_db = get_db_manager(request)
    config_composer = get_config_composer(request=request)

    async def tester_stream_generator():
        batch_id = str(uuid.uuid4())
        start_time = time.time()
        completed_count = 0
        result_queue = asyncio.Queue()

        try:
            tester_status_storage[batch_id] = {
                "status": "running",
                "total_count": len(tester_request.test_cases),
                "completed_count": 0,
                "progress": 0.0,
                "start_time": start_time
            }

            initial_message = {
                "type": "tester_start",
                "batch_id": batch_id,
                "total_count": len(tester_request.test_cases),
                "batch_size": tester_request.batch_size,
                "workflow_name": tester_request.workflow_name
            }
            yield f"data: {json.dumps(initial_message, ensure_ascii=False)}\n\n"

            logger.info(f"테스터 스트림 {batch_id} 시작: 워크플로우={tester_request.workflow_name}, "
                       f"테스트 케이스={len(tester_request.test_cases)}개, 배치 크기={tester_request.batch_size}")

            all_results = []

            async def individual_completion_callback(result: TesterTestResult):
                await result_queue.put(result)

            async def batch_processor():
                nonlocal all_results
                try:
                    for i in range(0, len(tester_request.test_cases), tester_request.batch_size):
                        batch_group = tester_request.test_cases[i:i + tester_request.batch_size]
                        group_number = i // tester_request.batch_size + 1

                        logger.info(f"배치 그룹 {group_number} 처리 중: {len(batch_group)}개 병렬 실행")

                        group_results = await process_batch_group(
                            user_id=user_id,
                            workflow_name=tester_request.workflow_name,
                            workflow_id=tester_request.workflow_id,
                            test_cases=batch_group,
                            interaction_id=tester_request.interaction_id,
                            selected_collections=tester_request.selected_collections,
                            batch_id=batch_id,
                            app_db=app_db,
                            individual_result_callback=individual_completion_callback,
                        )

                        all_results.extend(group_results)

                        if i + tester_request.batch_size < len(tester_request.test_cases):
                            await asyncio.sleep(0.5)

                    await result_queue.put("TESTER_COMPLETE")
                except Exception as e:
                    await result_queue.put(f"ERROR:{str(e)}")

            batch_task = asyncio.create_task(batch_processor())

            while True:
                try:
                    result = await asyncio.wait_for(result_queue.get(), timeout=1.0)

                    if result == "TESTER_COMPLETE":
                        break
                    elif isinstance(result, str) and result.startswith("ERROR:"):
                        error_message = {
                            "type": "error",
                            "batch_id": batch_id,
                            "error": result[6:],  # "ERROR:" 제거
                            "message": "테스터 실행 중 오류가 발생했습니다"
                        }
                        yield f"data: {json.dumps(error_message, ensure_ascii=False)}\n\n"
                        break
                    elif isinstance(result, TesterTestResult):
                        completed_count += 1
                        result_message = {
                            "type": "test_result",
                            "batch_id": batch_id,
                            "result": result.dict()
                        }
                        yield f"data: {json.dumps(result_message, ensure_ascii=False)}\n\n"

                        progress = (completed_count / len(tester_request.test_cases)) * 100

                        progress_message = {
                            "type": "progress",
                            "batch_id": batch_id,
                            "completed_count": completed_count,
                            "total_count": len(tester_request.test_cases),
                            "progress": round(progress, 2),
                            "elapsed_time": int((time.time() - start_time) * 1000)
                        }
                        yield f"data: {json.dumps(progress_message, ensure_ascii=False)}\n\n"

                        if batch_id in tester_status_storage:
                            tester_status_storage[batch_id]["completed_count"] = completed_count
                            tester_status_storage[batch_id]["progress"] = progress

                except asyncio.TimeoutError:
                    continue

            # 배치 태스크 완료 대기
            await batch_task

            # LLM 평가 처리 (만약 llm_eval_enabled가 true인 경우)
            if tester_request.llm_eval_enabled and all_results:
                config_composer = get_config_composer(request=request)
                logger.info(f"LLM 평가 시작: {len(all_results)}개 결과 평가")

                eval_progress_message = {
                    "type": "eval_start",
                    "batch_id": batch_id,
                    "message": "LLM 평가를 시작합니다..."
                }
                yield f"data: {json.dumps(eval_progress_message, ensure_ascii=False)}\n\n"

                # 각 결과에 대해 LLM 평가 수행
                for idx, result in enumerate(all_results):
                    if result.status == "success" and result.actual_output:
                        try:
                            # result.id를 사용하여 unique_interaction_id 생성
                            unique_interaction_id = f"{tester_request.interaction_id}____{tester_request.workflow_name}____{batch_id}____{result.id}"

                            llm_score = await evaluate_with_llm(
                                unique_interaction_id=unique_interaction_id,
                                input_data=result.input,
                                expected_output=result.expected_output,
                                actual_output=result.actual_output,
                                llm_eval_type=tester_request.llm_eval_type,
                                llm_eval_model=tester_request.llm_eval_model,
                                app_db=app_db,
                                config_composer=config_composer
                            )

                            # LLM 평가 결과를 result 객체에 추가
                            result_dict = result.dict()
                            result_dict["llm_eval_score"] = llm_score

                            # LLM 평가 결과 SSE 전송
                            eval_result_message = {
                                "type": "eval_result",
                                "batch_id": batch_id,
                                "test_id": result.id,
                                "llm_eval_score": llm_score,
                                "progress": f"{idx + 1}/{len(all_results)}"
                            }
                            yield f"data: {json.dumps(eval_result_message, ensure_ascii=False)}\n\n"

                            # TODO: DB에 LLM 평가 점수 저장 로직 추가
                            # unique_interaction_id를 사용하여 해당 ExecutionIO 레코드를 찾아 llm_eval_score 업데이트
                            logger.info(f"테스트 케이스 {result.id} LLM 평가 완료: 점수={llm_score}, interaction_id={unique_interaction_id}")

                        except Exception as eval_error:
                            logger.error(f"테스트 케이스 {result.id} LLM 평가 실패: {str(eval_error)}")

                            eval_error_message = {
                                "type": "eval_error",
                                "batch_id": batch_id,
                                "test_id": result.id,
                                "error": str(eval_error)
                            }
                            yield f"data: {json.dumps(eval_error_message, ensure_ascii=False)}\n\n"

                # LLM 평가 완료 메시지
                eval_complete_message = {
                    "type": "eval_complete",
                    "batch_id": batch_id,
                    "message": "LLM 평가가 완료되었습니다"
                }
                yield f"data: {json.dumps(eval_complete_message, ensure_ascii=False)}\n\n"

            # 최종 결과 계산
            total_execution_time = int((time.time() - start_time) * 1000)
            success_count = sum(1 for r in all_results if r.status == "success")
            error_count = len(all_results) - success_count

            # 테스터 상태 완료로 업데이트
            tester_status_storage[batch_id]["status"] = "completed"
            tester_status_storage[batch_id]["progress"] = 100.0

            # 최종 완료 메시지
            final_message = {
                "type": "tester_complete",
                "batch_id": batch_id,
                "total_count": len(all_results),
                "success_count": success_count,
                "error_count": error_count,
                "total_execution_time": total_execution_time,
                "message": f"테스터 처리 완료: 성공={success_count}개, 실패={error_count}개"
            }
            yield f"data: {json.dumps(final_message, ensure_ascii=False)}\n\n"

            logger.info(f"테스터 스트림 {batch_id} 완료: 성공={success_count}개, 실패={error_count}개, "
                       f"총 소요시간={total_execution_time}ms")

        except Exception as e:
            logger.error(f"테스터 스트림 실행 중 오류: {str(e)}", exc_info=True)

            if 'batch_id' in locals() and batch_id in tester_status_storage:
                tester_status_storage[batch_id]["status"] = "error"
                tester_status_storage[batch_id]["error"] = str(e)

            error_message = {
                "type": "error",
                "batch_id": batch_id if 'batch_id' in locals() else "unknown",
                "error": str(e),
                "message": "테스터 실행 중 오류가 발생했습니다"
            }
            yield f"data: {json.dumps(error_message, ensure_ascii=False)}\n\n"

    return StreamingResponse(tester_stream_generator(), media_type="text/event-stream")

@router.get("/execution/status")
async def get_all_execution_status():
    """
    현재 실행 중인 모든 워크플로우의 상태를 반환합니다.

    Returns:
        모든 활성 워크플로우 실행 상태
    """
    try:
        status_data = execution_manager.get_all_execution_status()
        return JSONResponse(content={
            "active_executions": len(status_data),
            "executions": status_data
        })
    except Exception as e:
        logger.error(f"실행 상태 조회 중 오류: {e}")
        raise HTTPException(status_code=500, detail="실행 상태 조회 실패")

@router.get("/execution/status/{execution_id}")
async def get_execution_status(execution_id: str):
    """
    특정 워크플로우 실행의 상태를 반환합니다.

    Args:
        execution_id: 실행 ID (interaction_id_workflow_id_user_id 형태)

    Returns:
        워크플로우 실행 상태
    """
    try:
        status = execution_manager.get_execution_status(execution_id)
        if status is None:
            raise HTTPException(status_code=404, detail="실행을 찾을 수 없습니다")
        return JSONResponse(content=status)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"실행 상태 조회 중 오류: {e}")
        raise HTTPException(status_code=500, detail="실행 상태 조회 실패")

@router.post("/execution/cleanup")
async def cleanup_completed_executions():
    """
    완료된 워크플로우 실행들을 정리합니다.

    Returns:
        정리 결과
    """
    try:
        before_count = len(execution_manager.get_all_execution_status())
        execution_manager.cleanup_completed_executions()
        after_count = len(execution_manager.get_all_execution_status())
        cleaned_count = before_count - after_count

        return JSONResponse(content={
            "message": f"{cleaned_count}개의 완료된 실행이 정리되었습니다",
            "before_count": before_count,
            "after_count": after_count,
            "cleaned_count": cleaned_count
        })
    except Exception as e:
        logger.error(f"실행 정리 중 오류: {e}")
        raise HTTPException(status_code=500, detail="실행 정리 실패")

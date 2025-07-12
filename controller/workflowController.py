from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import json
import logging
from datetime import datetime
from src.node_composer import get_node_registry, get_node_class_registry

logger = logging.getLogger("workflow-controller")
router = APIRouter(prefix="/workflow", tags=["workflow"])

class WorkflowData(BaseModel):
    id: str
    view: Dict[str, Any]
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]

class SaveWorkflowRequest(BaseModel):
    workflow_id: str
    content: WorkflowData

@router.get("/list")
async def list_workflows():
    """
    downloads 폴더에 있는 모든 workflow 파일들의 이름을 반환합니다.
    """
    try:
        downloads_path = os.path.join(os.getcwd(), "downloads")
        
        # downloads 폴더가 존재하지 않으면 생성
        if not os.path.exists(downloads_path):
            os.makedirs(downloads_path)
            return JSONResponse(content={"workflows": []})
        
        # .json 확장자를 가진 파일들만 필터링
        workflow_files = []
        for file in os.listdir(downloads_path):
            if file.endswith('.json'):
                workflow_files.append(file)
        
        logger.info(f"Found {len(workflow_files)} workflow files")
        return JSONResponse(content={"workflows": workflow_files})
        
    except Exception as e:
        logger.error(f"Error listing workflows: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list workflows: {str(e)}")

@router.post("/save")
async def save_workflow(request: SaveWorkflowRequest):
    """
    Frontend에서 받은 workflow 정보를 파일로 저장합니다.
    파일명: {workflow_id}.json
    """
    try:
        downloads_path = os.path.join(os.getcwd(), "downloads")
        
        # downloads 폴더가 존재하지 않으면 생성
        if not os.path.exists(downloads_path):
            os.makedirs(downloads_path)
        
        # 파일명 생성 (workflow_id + .json)
        filename = f"{request.workflow_id}.json"
        file_path = os.path.join(downloads_path, filename)
        
        # workflow content를 JSON 파일로 저장
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(request.content.dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Workflow saved successfully: {filename}")
        return JSONResponse(content={
            "success": True,
            "message": f"Workflow '{request.workflow_id}' saved successfully",
            "filename": filename
        })
        
    except Exception as e:
        logger.error(f"Error saving workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save workflow: {str(e)}")

@router.get("/load/{workflow_id}")
async def load_workflow(workflow_id: str):
    """
    특정 workflow를 로드합니다.
    """
    try:
        downloads_path = os.path.join(os.getcwd(), "downloads")
        filename = f"{workflow_id}.json"
        file_path = os.path.join(downloads_path, filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            workflow_data = json.load(f)
        
        logger.info(f"Workflow loaded successfully: {filename}")
        return JSONResponse(content=workflow_data)
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")
    except Exception as e:
        logger.error(f"Error loading workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load workflow: {str(e)}")

@router.delete("/delete/{workflow_id}")
async def delete_workflow(workflow_id: str):
    """
    특정 workflow를 삭제합니다.
    """
    try:
        downloads_path = os.path.join(os.getcwd(), "downloads")
        filename = f"{workflow_id}.json"
        file_path = os.path.join(downloads_path, filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")
        
        os.remove(file_path)
        
        logger.info(f"Workflow deleted successfully: {filename}")
        return JSONResponse(content={
            "success": True,
            "message": f"Workflow '{workflow_id}' deleted successfully"
        })
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")
    except Exception as e:
        logger.error(f"Error deleting workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete workflow: {str(e)}")

@router.get("/list/detail")
async def list_workflows_detail():
    """
    downloads 폴더에 있는 모든 workflow 파일들의 상세 정보를 반환합니다.
    각 워크플로우에 대해 파일명, workflow_id, 노드 수, 마지막 수정일자를 포함합니다.
    """
    try:
        downloads_path = os.path.join(os.getcwd(), "downloads")
        
        # downloads 폴더가 존재하지 않으면 생성
        if not os.path.exists(downloads_path):
            os.makedirs(downloads_path)
            return JSONResponse(content={"workflows": []})
        
        workflow_details = []
        
        # .json 확장자를 가진 파일들만 처리
        for file in os.listdir(downloads_path):
            if not file.endswith('.json'):
                continue
                
            file_path = os.path.join(downloads_path, file)
            
            try:
                # 파일 메타데이터 수집
                file_stat = os.stat(file_path)
                last_modified = datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                
                # JSON 파일 읽기
                with open(file_path, 'r', encoding='utf-8') as f:
                    workflow_data = json.load(f)
                
                # workflow_id 추출 (최상위 id 필드)
                workflow_id = workflow_data.get('id', 'unknown')
                
                # nodes 수 계산
                nodes = workflow_data.get('nodes', [])
                node_count = len(nodes) if isinstance(nodes, list) else 0
                
                # 상세 정보 추가
                workflow_detail = {
                    "filename": file,
                    "workflow_id": workflow_id,
                    "node_count": node_count,
                    "last_modified": last_modified
                }
                
                workflow_details.append(workflow_detail)
                logger.debug(f"Processed workflow file: {file} (ID: {workflow_id}, Nodes: {node_count})")
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON file {file}: {str(e)}")
                # 파싱 실패한 파일도 기본 정보로 포함
                workflow_details.append({
                    "filename": file,
                    "workflow_id": "invalid_json",
                    "node_count": 0,
                    "last_modified": datetime.fromtimestamp(os.stat(file_path).st_mtime).isoformat(),
                    "error": "Invalid JSON format"
                })
            except Exception as e:
                logger.warning(f"Failed to process workflow file {file}: {str(e)}")
                # 처리 실패한 파일도 기본 정보로 포함
                workflow_details.append({
                    "filename": file,
                    "workflow_id": "error",
                    "node_count": 0,
                    "last_modified": datetime.fromtimestamp(os.stat(file_path).st_mtime).isoformat(),
                    "error": str(e)
                })
        
        logger.info(f"Found {len(workflow_details)} workflow files with detailed information")
        return JSONResponse(content={"workflows": workflow_details})
        
    except Exception as e:
        logger.error(f"Error listing workflow details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list workflow details: {str(e)}")

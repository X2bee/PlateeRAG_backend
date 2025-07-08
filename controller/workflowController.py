from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import json
import logging

logger = logging.getLogger("workflow-controller")
router = APIRouter(prefix="/workflow", tags=["workflow"])

# Pydantic models for request/response
class WorkflowData(BaseModel):
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

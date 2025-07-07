from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Union, List
import threading
import logging
import os
import signal
import subprocess
import json
from datetime import datetime
import glob
from pathlib import Path

from src.workflow_executor import WorkflowExecutor
from src.node_composer import (
    run_discovery,
    generate_json_spec,
)

router = APIRouter(
    prefix="/node",
    tags=["node"],
    responses={404: {"description": "Not found"}},
)

class Workflow(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    view: Dict[str, Any]

def get_node_list():
    try:
        exported_nodes_path = "./constants/exported_nodes.json"
        if not os.path.exists(exported_nodes_path):
            raise HTTPException(status_code=404, detail="No nodes available. Please run discovery first.")
        with open(exported_nodes_path, 'r') as file:
            nodes = json.load(file)

        if not nodes:
            raise HTTPException(status_code=404, detail="No nodes available. Please run discovery first.")
        
        return nodes

    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.get("/get", response_model=List[Dict[str, Any]])
async def list_nodes():
    try:
        nodes = get_node_list()
        return nodes
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Error listing nodes: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.get("/export", response_model=Dict[str, Any])
async def export_nodes():
    """
    Refesh and export the list of nodes to a JSON file.
    """
    try:
        from src.node_composer import NODE_REGISTRY
        run_discovery()
        output_filename = "./constants/exported_nodes.json"
        generate_json_spec(output_path=output_filename)
        if not os.path.exists(output_filename):
            raise HTTPException(status_code=500, detail="Failed to generate nodes JSON file.")
        
        else:
            return {"status": "success", "message": "Nodes exported successfully", "file": output_filename}
        
    except Exception as e:
        logging.error(f"Error listing nodes: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.post("/execute", response_model=Dict[str, Any])
async def execute_workflow(workflow: Workflow):
    """
    주어진 노드와 엣지 정보로 워크플로우를 실행합니다.
    """
    try:
        # Pydantic 모델을 dict로 변환
        workflow_data = workflow.dict()
        
        executor = WorkflowExecutor(workflow_data)
        final_outputs = executor.execute_workflow()
        
        return {"status": "success", "message": "워크플로우 실행 완료", "outputs": final_outputs}

    except ValueError as e:
        logging.error(f"Workflow execution error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
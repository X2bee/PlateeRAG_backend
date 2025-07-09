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
    get_node_registry,
    get_node_class_registry
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
    
    # print("DEBUG: 워크플로우 실행 요청\n", workflow)
    
    try:
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

@router.get("/registry", response_model=Dict[str, Any])
async def get_node_registry_info():
    """
    전역 NODE_REGISTRY에서 노드 정보를 가져옵니다.
    """
    try:
        node_registry = get_node_registry()
        node_class_registry = get_node_class_registry()
        
        return {
            "status": "success",
            "node_count": len(node_registry),
            "available_nodes": [node["id"] for node in node_registry],
            "registry_data": node_registry,
            "class_registry_keys": list(node_class_registry.keys())
        }
        
    except Exception as e:
        logging.error(f"Error getting node registry: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.get("/registry/nodes", response_model=List[Dict[str, Any]])
async def get_all_nodes():
    """
    등록된 모든 노드의 정보를 반환합니다.
    """
    try:
        node_registry = get_node_registry()
        return node_registry
        
    except Exception as e:
        logging.error(f"Error getting all nodes: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.get("/registry/node/{node_id}", response_model=Dict[str, Any])
async def get_node_by_id(node_id: str):
    """
    특정 ID의 노드 정보를 반환합니다.
    """
    try:
        node_registry = get_node_registry()
        for node in node_registry:
            if node["id"] == node_id:
                return node
        
        raise HTTPException(status_code=404, detail=f"Node with id '{node_id}' not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting node by id: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
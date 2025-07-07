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
from src.node_composer import (
    run_discovery,
    generate_json_spec,
)

router = APIRouter(
    prefix="/node",
    tags=["node"],
    responses={404: {"description": "Not found"}},
)

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

@router.get("/get/nodes", response_model=List[Dict[str, Any]])
async def list_nodes():
    try:
        nodes = get_node_list()
        return nodes
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Error listing nodes: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.get("/export/nodes", response_model=Dict[str, Any])
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

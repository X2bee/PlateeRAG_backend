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


router = APIRouter(
    prefix="/node",
    tags=["node"],
    responses={404: {"description": "Not found"}},
)

@router.get("/get_nodes", response_model=List[Dict[str, Any]])
async def list_nodes():
    """
    List all available nodes.
    """
    try:
        # Assuming NODE_REGISTRY is a global variable containing registered nodes
        from src.node_composer import NODE_REGISTRY
        return NODE_REGISTRY
    except Exception as e:
        logging.error(f"Error listing nodes: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
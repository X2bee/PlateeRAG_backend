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
from config.persistent_config import (
    get_all_persistent_configs, 
    refresh_all_configs, 
    save_all_configs, 
    export_config_summary,
    PersistentConfig
)

router = APIRouter(
    prefix="/config",
    tags=["config"],
    responses={404: {"description": "Not found"}},
)

class ConfigUpdateRequest(BaseModel):
    value: Any
    save_to_db: bool = True

# PersistentConfig 관련 엔드포인트들
@router.get("/persistent/summary")
async def get_persistent_config_summary():
    """모든 PersistentConfig의 요약 정보 반환"""
    try:
        return export_config_summary()
    except Exception as e:
        logging.error("Error getting config summary: %s", e)
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.get("/persistent/all")
async def get_all_persistent_configs_info():
    """등록된 모든 PersistentConfig 객체의 상세 정보 반환"""
    try:
        configs = get_all_persistent_configs()
        return [
            {
                "env_name": config.env_name,
                "config_path": config.config_path,
                "current_value": config.value,
                "default_value": config.env_value,
                "is_saved": config.config_value is not None,
                "type": type(config.value).__name__
            }
            for config in configs
        ]
    except Exception as e:
        logging.error("Error getting all configs: %s", e)
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.get("/persistent/{env_name}")
async def get_persistent_config_by_name(env_name: str):
    """특정 이름의 PersistentConfig 정보 반환"""
    try:
        configs = get_all_persistent_configs()
        for config in configs:
            if config.env_name == env_name:
                return {
                    "env_name": config.env_name,
                    "config_path": config.config_path,
                    "current_value": config.value,
                    "default_value": config.env_value,
                    "is_saved": config.config_value is not None,
                    "type": type(config.value).__name__
                }
        
        raise HTTPException(status_code=404, detail=f"Config '{env_name}' not found")
    except HTTPException:
        raise
    except Exception as e:
        logging.error("Error getting config by name: %s", e)
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.put("/persistent/{env_name}")
async def update_persistent_config(env_name: str, request: ConfigUpdateRequest):
    """특정 PersistentConfig 값 업데이트"""
    try:
        configs = get_all_persistent_configs()
        for config in configs:
            if config.env_name == env_name:
                old_value = config.value
                
                # 값 타입 변환
                try:
                    if isinstance(config.env_value, bool):
                        config.value = bool(request.value)
                    elif isinstance(config.env_value, int):
                        config.value = int(request.value)
                    elif isinstance(config.env_value, float):
                        config.value = float(request.value)
                    elif isinstance(config.env_value, list):
                        config.value = list(request.value) if isinstance(request.value, (list, tuple)) else [request.value]
                    elif isinstance(config.env_value, dict):
                        config.value = dict(request.value) if isinstance(request.value, dict) else {"value": request.value}
                    else:
                        config.value = str(request.value)
                except (ValueError, TypeError) as e:
                    raise HTTPException(status_code=400, detail=f"Invalid value type: {e}")
                
                if request.save_to_db:
                    config.save()
                
                return {
                    "message": f"Config '{env_name}' updated successfully",
                    "old_value": old_value,
                    "new_value": config.value,
                    "saved_to_db": request.save_to_db
                }
        
        raise HTTPException(status_code=404, detail=f"Config '{env_name}' not found")
    except HTTPException:
        raise
    except Exception as e:
        logging.error("Error updating config: %s", e)
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.post("/persistent/refresh")
async def refresh_all_persistent_configs():
    """모든 PersistentConfig를 데이터베이스에서 다시 로드"""
    try:
        refresh_all_configs()
        return {
            "message": "All configs refreshed successfully",
            "config_count": len(get_all_persistent_configs())
        }
    except Exception as e:
        logging.error("Error refreshing configs: %s", e)
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.post("/persistent/save")
async def save_all_persistent_configs():
    """모든 PersistentConfig를 데이터베이스에 저장"""
    try:
        save_all_configs()
        return {
            "message": "All configs saved successfully",
            "config_count": len(get_all_persistent_configs())
        }
    except Exception as e:
        logging.error("Error saving configs: %s", e)
        raise HTTPException(status_code=500, detail="Internal Server Error")
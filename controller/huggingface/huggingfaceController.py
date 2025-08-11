from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
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
from controller.controller_helper import extract_user_id_from_request

router = APIRouter(
    prefix="/api/huggingface",
    tags=["node"],
    responses={404: {"description": "Not found"}},
)

def get_config_composer(request: Request):
    """request.app.state에서 config_composer 가져오기"""
    if hasattr(request.app.state, 'config_composer') and request.app.state.config_composer:
        return request.app.state.config_composer
    else:
        from config.config_composer import config_composer
        return config_composer

@router.get("/models")
async def get_models(request: Request):
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")
    config_composer = get_config_composer(request)
    if not config_composer:
        raise HTTPException(status_code=500, detail="Config composer not found in request state")

    hugging_face_user_id = config_composer.get_config_by_name("HUGGING_FACE_USER_ID").value
    hugging_face_hub_token = config_composer.get_config_by_name("HUGGING_FACE_HUB_TOKEN").value

    if not hugging_face_user_id or not hugging_face_hub_token:
        raise HTTPException(status_code=500, detail="Hugging Face user ID or token not configured")

    # 실제 Hugging Face Hub API를 통해 모델 리스트 조회
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise HTTPException(status_code=500, detail="huggingface_hub 라이브러리가 설치되어 있지 않습니다.")

    api = HfApi(token=hugging_face_hub_token)
    try:
        models = api.list_models(author=hugging_face_user_id)
        model_ids = [model.id for model in models]
        logging.info(f"[INFO] SUCCESS: Load Model List from UserID: {hugging_face_user_id}")
    except Exception as e:
        logging.warning(f"[WARNING] FAIL: Load Model List from UserID: {hugging_face_user_id}")
        logging.warning(f"[WARNING] RETURN Empty List. Error: {e}")
        model_ids = []

    return {"models": model_ids}


@router.get("/datasets")
async def get_datasets(request: Request):
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    try:
        datasets = []
        dataset_files = glob.glob(f"datasets/{user_id}/*.json")
        for file_path in dataset_files:
            with open(file_path, 'r') as file:
                dataset_data = json.load(file)
                datasets.append(dataset_data)
        return {"datasets": datasets}
    except Exception as e:
        logging.error(f"Error retrieving datasets: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

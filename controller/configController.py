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
import requests
from requests.exceptions import RequestException
import asyncio
from service.llm.llm_service import LLMService

router = APIRouter(
    prefix="/api/config",
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
                
                # OpenAI API 키가 업데이트된 경우 자동으로 임베딩 제공자 전환 시도
                if env_name == "OPENAI_API_KEY" and request.value and str(request.value).strip():
                    try:
                        from fastapi import Request
                        import asyncio
                        
                        # app.state에서 config 가져오기 (비동기 처리를 위해 별도 함수로 분리)
                        asyncio.create_task(_auto_switch_embedding_provider_after_delay())
                        
                        logging.info("Scheduled auto-switch for embedding provider after OpenAI API key update")
                    except Exception as e:
                        logging.warning(f"Failed to schedule auto-switch for embedding provider: {e}")
                
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

async def _auto_switch_embedding_provider_after_delay():
    """OpenAI API 키 업데이트 후 잠시 대기 후 임베딩 제공자 자동 전환"""
    try:
        # 1초 대기 (설정이 완전히 저장될 때까지)
        await asyncio.sleep(1)
        
        # app.state에서 config 가져오기
        from fastapi import applications
        import uvicorn
        
        # 현재 실행 중인 app 인스턴스 찾기 (간접적 방법)
        # 직접적인 방법이 없으므로 전역 설정에서 가져오기
        try:
            configs = get_all_persistent_configs()
            
            # vectordb config 찾기
            vectordb_configs = [c for c in configs if c.config_path.startswith("vectordb.")]
            if vectordb_configs:
                # 새로 고침해서 최신 설정 가져오기
                refresh_all_configs()
                
                # VectorDB 설정 재구성
                from config.config_composer import ConfigComposer
                composer = ConfigComposer()
                
                if "vectordb" in composer.config_categories and "openai" in composer.config_categories:
                    composer.config_categories["vectordb"].set_openai_config(composer.config_categories["openai"])
                    switched = composer.config_categories["vectordb"].check_and_switch_to_best_provider()
                    
                    if switched:
                        logging.info("Successfully auto-switched embedding provider after OpenAI API key update")
                    else:
                        logging.info("No embedding provider switch needed after OpenAI API key update")
                        
        except Exception as e:
            logging.warning(f"Error during auto-switch process: {e}")
            
    except Exception as e:
        logging.error(f"Failed to auto-switch embedding provider: {e}")

@router.get("/llm/status")
async def get_llm_status():
    """LLM 제공자 상태 정보 반환"""
    try:
        configs = get_all_persistent_configs()
        
        # 현재 기본 제공자 가져오기
        current_provider = "openai"  # 기본값
        for config in configs:
            if config.env_name == "DEFAULT_LLM_PROVIDER":
                current_provider = config.value
                break
        
        # 각 제공자별 설정 상태 확인
        providers_status = {}
        
        # OpenAI 상태 확인
        openai_config = {}
        for config in configs:
            if config.env_name == "OPENAI_API_KEY":
                openai_config['api_key'] = config.value
            elif config.env_name == "OPENAI_API_BASE_URL":
                openai_config['base_url'] = config.value
            elif config.env_name == "OPENAI_MODEL_DEFAULT":
                openai_config['model'] = config.value
        
        llm_service = LLMService()

        openai_validation = llm_service.validate_provider_config("openai", openai_config)
        providers_status["openai"] = {
            "configured": openai_validation["valid"],
            "available": openai_validation["valid"],
            "error": openai_validation.get("error")
        }
        
        # vLLM 상태 확인
        vllm_config = {}
        for config in configs:
            if config.env_name == "VLLM_API_BASE_URL":
                vllm_config['base_url'] = config.value
            elif config.env_name == "VLLM_API_KEY":
                vllm_config['api_key'] = config.value
            elif config.env_name == "VLLM_MODEL_NAME":
                vllm_config['model_name'] = config.value
        
        vllm_validation = llm_service.validate_provider_config("vllm", vllm_config)
        providers_status["vllm"] = {
            "configured": vllm_validation["valid"],
            "available": vllm_validation["valid"],
            "error": vllm_validation.get("error")
        }
        
        # SGL 상태 확인
        sgl_config = {}
        for config in configs:
            if config.env_name == "SGL_API_BASE_URL":
                sgl_config['base_url'] = config.value
            elif config.env_name == "SGL_API_KEY":
                sgl_config['api_key'] = config.value
            elif config.env_name == "SGL_MODEL_NAME":
                sgl_config['model_name'] = config.value
        
        sgl_validation = llm_service.validate_provider_config("sgl", sgl_config)
        providers_status["sgl"] = {
            "configured": sgl_validation["valid"],
            "available": sgl_validation["valid"],
            "error": sgl_validation.get("error", sgl_validation.get("errors")),
            "warnings": sgl_validation.get("warnings")
        }
        
        # 사용 가능한 제공자 목록
        available_providers = [provider for provider, status in providers_status.items() if status["available"]]
        
        return {
            "current_provider": current_provider,
            "available_providers": available_providers,
            "providers": providers_status
        }
        
    except Exception as e:
        logging.error(f"Error getting LLM status: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.post("/llm/switch-provider")
async def switch_llm_provider(request: dict):
    """LLM 기본 제공자 변경"""
    try:
        provider = request.get("provider")
        if not provider:
            raise HTTPException(status_code=400, detail="Provider is required")
        
        if provider not in ["openai", "vllm", "sgl"]:
            raise HTTPException(status_code=400, detail="Invalid provider. Must be 'openai', 'vllm', or 'sgl'")
        
        # DEFAULT_LLM_PROVIDER 설정 업데이트
        configs = get_all_persistent_configs()
        for config in configs:
            if config.env_name == "DEFAULT_LLM_PROVIDER":
                old_value = config.value
                config.value = provider
                config.save()
                
                return {
                    "status": "success",
                    "message": f"Default LLM provider switched to {provider}",
                    "old_provider": old_value,
                    "new_provider": provider
                }
        
        # DEFAULT_LLM_PROVIDER 설정이 없는 경우
        raise HTTPException(status_code=404, detail="DEFAULT_LLM_PROVIDER config not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error switching LLM provider: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.post("/llm/auto-switch")
async def auto_switch_llm_provider():
    """사용 가능한 LLM 제공자로 자동 전환"""
    try:
        configs = get_all_persistent_configs()
        
        # 각 제공자별 사용 가능 여부 확인
        available_providers = []
        
        # OpenAI 확인
        openai_config = {}
        for config in configs:
            if config.env_name == "OPENAI_API_KEY":
                openai_config['api_key'] = config.value
        
        llm_service = LLMService()

        if llm_service.validate_provider_config("openai", openai_config)["valid"]:
            available_providers.append("openai")
        
        # vLLM 확인
        vllm_config = {}
        for config in configs:
            if config.env_name == "VLLM_API_BASE_URL":
                vllm_config['base_url'] = config.value
        
        if llm_service.validate_provider_config("vllm", vllm_config)["valid"]:
            available_providers.append("vllm")
        
        # SGL 확인
        sgl_config = {}
        for config in configs:
            if config.env_name == "SGL_API_BASE_URL":
                sgl_config['base_url'] = config.value
            elif config.env_name == "SGL_MODEL_NAME":
                sgl_config['model_name'] = config.value
        
        if llm_service.validate_provider_config("sgl", sgl_config)["valid"]:
            available_providers.append("sgl")
        
        if not available_providers:
            raise HTTPException(status_code=400, detail="No LLM providers are available")
        
        # 첫 번째 사용 가능한 제공자로 설정
        selected_provider = available_providers[0]
        
        # DEFAULT_LLM_PROVIDER 업데이트
        for config in configs:
            if config.env_name == "DEFAULT_LLM_PROVIDER":
                old_value = config.value
                config.value = selected_provider
                config.save()
                
                return {
                    "status": "success",
                    "message": f"Auto-selected provider: {selected_provider}",
                    "old_provider": old_value,
                    "new_provider": selected_provider,
                    "available_providers": available_providers
                }
        
        raise HTTPException(status_code=404, detail="DEFAULT_LLM_PROVIDER config not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error auto-switching LLM provider: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.get("/llm/models/{provider}")
async def get_llm_models(provider: str):
    """특정 LLM 제공자의 사용 가능한 모델 목록 조회"""
    try:
        if provider not in ["openai", "vllm", "sgl"]:
            raise HTTPException(status_code=400, detail="Invalid provider. Must be 'openai', 'vllm', or 'sgl'")
        
        configs = get_all_persistent_configs()
        config_dict = {config.env_name: config.value for config in configs}
        
        llm_service = LLMService()
        
        if provider == "sgl":
            config_data = {
                'base_url': config_dict.get('SGL_API_BASE_URL'),
                'api_key': config_dict.get('SGL_API_KEY')
            }
        elif provider == "vllm":
            config_data = {
                'base_url': config_dict.get('VLLM_API_BASE_URL'),
                'api_key': config_dict.get('VLLM_API_KEY'),
                'model_name': config_dict.get('VLLM_MODEL_NAME')
            }
        else:
            # OpenAI는 모델 목록 조회가 제한적이므로 기본 모델 목록 반환
            return {
                "status": "success",
                "models": [
                    {"id": "gpt-4", "object": "model"},
                    {"id": "gpt-4-turbo", "object": "model"},
                    {"id": "gpt-3.5-turbo", "object": "model"},
                    {"id": "gpt-3.5-turbo-16k", "object": "model"}
                ],
                "count": 4,
                "note": "OpenAI models list is predefined"
            }
        
        result = await llm_service.get_provider_models(provider, config_data)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting models for {provider}: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# 기존 test_connection 함수 수정
@router.post("/test/{category}")
async def test_connection(category: str):
    """연결 테스트 API - OpenAI, vLLM, SGL, 현재 LLM 제공자 테스트"""
    try:
        configs = get_all_persistent_configs()
        config_dict = {config.env_name: config.value for config in configs}
        llm_service = LLMService()
        
        if category == "openai":
            config_data = {
                'api_key': config_dict.get('OPENAI_API_KEY'),
                'base_url': config_dict.get('OPENAI_API_BASE_URL', 'https://api.openai.com/v1'),
                'model': config_dict.get('OPENAI_MODEL_DEFAULT', 'gpt-3.5-turbo')
            }
            return await llm_service.test_openai_connection(config_data)
            
        elif category == "vllm":
            config_data = {
                'base_url': config_dict.get('VLLM_API_BASE_URL'),
                'api_key': config_dict.get('VLLM_API_KEY'),
                'model_name': config_dict.get('VLLM_MODEL_NAME')
            }
            return await llm_service.test_vllm_connection(config_data)
            
        elif category == "sgl":
            config_data = {
                'base_url': config_dict.get('SGL_API_BASE_URL'),
                'api_key': config_dict.get('SGL_API_KEY'),
                'model_name': config_dict.get('SGL_MODEL_NAME')
            }
            return await llm_service.test_sgl_connection(config_data)
            
        elif category == "llm":
            # 현재 기본 LLM 제공자 테스트
            current_provider = config_dict.get('DEFAULT_LLM_PROVIDER', 'openai')
            
            if current_provider == "openai":
                config_data = {
                    'api_key': config_dict.get('OPENAI_API_KEY'),
                    'base_url': config_dict.get('OPENAI_API_BASE_URL', 'https://api.openai.com/v1'),
                    'model': config_dict.get('OPENAI_MODEL_DEFAULT', 'gpt-3.5-turbo')
                }
                return await llm_service.test_openai_connection(config_data)
                
            elif current_provider == "vllm":
                config_data = {
                    'base_url': config_dict.get('VLLM_API_BASE_URL'),
                    'api_key': config_dict.get('VLLM_API_KEY'),
                    'model_name': config_dict.get('VLLM_MODEL_NAME')
                }
                return await llm_service.test_vllm_connection(config_data)
                
            elif current_provider == "sgl":
                config_data = {
                    'base_url': config_dict.get('SGL_API_BASE_URL'),
                    'api_key': config_dict.get('SGL_API_KEY'),
                    'model_name': config_dict.get('SGL_MODEL_NAME')
                }
                return await llm_service.test_sgl_connection(config_data)
                
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported LLM provider: {current_provider}")
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported category: {category}")
            
    except HTTPException:
        raise
    except (ValueError, ConnectionError, RequestException) as e:
        logging.error(f"Connection test failed for {category}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Connection test failed for {category}: {e}")
        raise HTTPException(status_code=500, detail=f"Connection test failed: {str(e)}")

@router.post("/llm/validate/{provider}")
async def validate_llm_provider_config(provider: str):
    """특정 LLM 제공자 설정 유효성 검사"""
    try:
        if provider not in ["openai", "vllm", "sgl"]:
            raise HTTPException(status_code=400, detail="Invalid provider. Must be 'openai', 'vllm', or 'sgl'")
        
        configs = get_all_persistent_configs()
        config_dict = {config.env_name: config.value for config in configs}
        
        llm_service = LLMService()
        
        if provider == "openai":
            config_data = {
                'api_key': config_dict.get('OPENAI_API_KEY'),
                'base_url': config_dict.get('OPENAI_API_BASE_URL'),
                'model': config_dict.get('OPENAI_MODEL_DEFAULT')
            }
        elif provider == "vllm":
            config_data = {
                'base_url': config_dict.get('VLLM_API_BASE_URL'),
                'api_key': config_dict.get('VLLM_API_KEY'),
                'model_name': config_dict.get('VLLM_MODEL_NAME')
            }
        elif provider == "sgl":
            config_data = {
                'base_url': config_dict.get('SGL_API_BASE_URL'),
                'api_key': config_dict.get('SGL_API_KEY'),
                'model_name': config_dict.get('SGL_MODEL_NAME')
            }
        
        validation_result = llm_service.validate_provider_config(provider, config_data)
        
        return {
            "provider": provider,
            "validation": validation_result,
            "config_data": {k: "***" if "key" in k.lower() and v else v for k, v in config_data.items()}
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error validating {provider} config: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
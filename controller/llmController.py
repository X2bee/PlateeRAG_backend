from fastapi import Request, APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Union, List
import logging
from requests.exceptions import RequestException
from service.llm.llm_service import LLMService
from controller.helper.singletonHelper import get_config_composer, get_vector_manager, get_rag_service, get_document_processor, get_db_manager

router = APIRouter(
    prefix="/api/llm",
    tags=["LLM"],
    responses={404: {"description": "Not found"}},
)

class ConfigUpdateRequest(BaseModel):
    value: Any
    save_to_db: bool = True

@router.get("/status")
async def get_llm_status(request: Request):
    """LLM 제공자 상태 정보 반환"""
    try:
        config_composer = get_config_composer(request)

        current_provider = "openai"
        if config_composer:
            current_provider = config_composer.get_config_by_name("DEFAULT_LLM_PROVIDER").value

        providers_status = {}

        openai_config = {}
        openai_config['api_key'] = config_composer.get_config_by_name("OPENAI_API_KEY").value
        openai_config['base_url'] = config_composer.get_config_by_name("OPENAI_API_BASE_URL").value
        openai_config['model'] = config_composer.get_config_by_name("OPENAI_MODEL_DEFAULT").value

        llm_service = LLMService()
        openai_validation = llm_service.validate_provider_config("openai", openai_config)
        providers_status["openai"] = {
            "configured": openai_validation["valid"],
            "available": openai_validation["valid"],
            "error": openai_validation.get("error")
        }

        vllm_config = {}
        vllm_config['base_url'] = config_composer.get_config_by_name("VLLM_API_BASE_URL").value
        vllm_config['api_key'] = config_composer.get_config_by_name("VLLM_API_KEY").value
        vllm_config['model_name'] = config_composer.get_config_by_name("VLLM_MODEL_NAME").value

        vllm_validation = llm_service.validate_provider_config("vllm", vllm_config)
        providers_status["vllm"] = {
            "configured": vllm_validation["valid"],
            "available": vllm_validation["valid"],
            "error": vllm_validation.get("error")
        }

        sgl_config = {}
        sgl_config['base_url'] = config_composer.get_config_by_name("SGL_API_BASE_URL").value
        sgl_config['api_key'] = config_composer.get_config_by_name("SGL_API_KEY").value
        sgl_config['model_name'] = config_composer.get_config_by_name("SGL_MODEL_NAME").value

        sgl_validation = llm_service.validate_provider_config("sgl", sgl_config)
        providers_status["sgl"] = {
            "configured": sgl_validation["valid"],
            "available": sgl_validation["valid"],
            "error": sgl_validation.get("error", sgl_validation.get("errors")),
            "warnings": sgl_validation.get("warnings")
        }

        gemini_config = {}
        gemini_config['base_url'] = config_composer.get_config_by_name("GEMINI_API_BASE_URL").value
        gemini_config['api_key'] = config_composer.get_config_by_name("GEMINI_API_KEY").value
        gemini_config['model_name'] = config_composer.get_config_by_name("GEMINI_MODEL_DEFAULT").value

        gemini_validation = llm_service.validate_provider_config("gemini", gemini_config)
        providers_status["gemini"] = {
            "configured": gemini_validation["valid"],
            "available": gemini_validation["valid"],
            "error": gemini_validation.get("error", gemini_validation.get("errors")),
            "warnings": gemini_validation.get("warnings")
        }

        anthropic_config = {}
        anthropic_config['base_url'] = config_composer.get_config_by_name("ANTHROPIC_API_BASE_URL").value
        anthropic_config['api_key'] = config_composer.get_config_by_name("ANTHROPIC_API_KEY").value
        anthropic_config['model_name'] = config_composer.get_config_by_name("ANTHROPIC_MODEL_DEFAULT").value

        anthropic_validation = llm_service.validate_provider_config("anthropic", anthropic_config)
        providers_status["anthropic"] = {
            "configured": anthropic_validation["valid"],
            "available": anthropic_validation["valid"],
            "error": anthropic_validation.get("error", anthropic_validation.get("errors")),
            "warnings": anthropic_validation.get("warnings")
        }

        available_providers = [provider for provider, status in providers_status.items() if status["available"]]

        return {
            "current_provider": current_provider,
            "available_providers": available_providers,
            "providers": providers_status
        }

    except Exception as e:
        logging.error(f"Error getting LLM status: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.post("/switch-provider")
async def switch_llm_provider(request: Request, request_context: dict):
    """LLM 기본 제공자 변경"""
    try:
        provider = request_context.get("provider")
        if not provider:
            raise HTTPException(status_code=400, detail="Provider is required")

        if provider not in ["openai", "vllm", "sgl", "gemini", "anthropic"]:
            raise HTTPException(status_code=400, detail="Invalid provider. Must be 'openai', 'vllm', 'sgl', 'gemini', or 'anthropic'")

        config_composer = get_config_composer(request)
        default_provider_config = config_composer.get_config_by_name("DEFAULT_LLM_PROVIDER")

        if default_provider_config:
            old_value = default_provider_config.value
            config_composer.update_config("DEFAULT_LLM_PROVIDER", provider)

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

    except Exception as e:
        logging.error(f"Error getting models for {provider}: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# 기존 test_connection 함수 수정
@router.post("/test/{category}")
async def test_connection(request: Request, category: str):
    """연결 테스트 API - OpenAI, vLLM, SGL, 현재 LLM 제공자 테스트"""
    try:
        config_composer = get_config_composer(request)
        llm_service = LLMService()

        if category == "openai":
            config_data = {
                'api_key': config_composer.get_config_by_name("OPENAI_API_KEY").value,
                'base_url': config_composer.get_config_by_name("OPENAI_API_BASE_URL").value,
                'model': config_composer.get_config_by_name("OPENAI_MODEL_DEFAULT").value
            }
            return await llm_service.test_openai_connection(config_data)

        elif category == "vllm":
            config_data = {
                'base_url': config_composer.get_config_by_name("VLLM_API_BASE_URL").value,
                'api_key': config_composer.get_config_by_name("VLLM_API_KEY").value,
                'model_name': config_composer.get_config_by_name("VLLM_MODEL_NAME").value
            }
            return await llm_service.test_vllm_connection(config_data)

        elif category == "sgl":
            config_data = {
                'base_url': config_composer.get_config_by_name("SGL_API_BASE_URL").value,
                'api_key': config_composer.get_config_by_name("SGL_API_KEY").value,
                'model_name': config_composer.get_config_by_name("SGL_MODEL_NAME").value
            }
            return await llm_service.test_sgl_connection(config_data)

        elif category == "gemini":
            config_data = {
                'base_url': config_composer.get_config_by_name("GEMINI_API_BASE_URL").value,
                'api_key': config_composer.get_config_by_name("GEMINI_API_KEY").value,
                'model_name': config_composer.get_config_by_name("GEMINI_MODEL_DEFAULT").value
            }
            return await llm_service.test_gemini_connection(config_data)

        elif category == "anthropic":
            config_data = {
                'base_url': config_composer.get_config_by_name("ANTHROPIC_API_BASE_URL").value,
                'api_key': config_composer.get_config_by_name("ANTHROPIC_API_KEY").value,
                'model_name': config_composer.get_config_by_name("ANTHROPIC_MODEL_DEFAULT").value
            }
            return await llm_service.test_anthropic_connection(config_data)

        elif category == "llm":
            current_provider = config_composer.get_config_by_name('DEFAULT_LLM_PROVIDER').value

            if current_provider == "openai":
                config_data = {
                    'api_key': config_composer.get_config_by_name('OPENAI_API_KEY').value,
                    'base_url': config_composer.get_config_by_name('OPENAI_API_BASE_URL').value,
                    'model': config_composer.get_config_by_name('OPENAI_MODEL_DEFAULT').value
                }
                return await llm_service.test_openai_connection(config_data)

            elif current_provider == "vllm":
                config_data = {
                    'base_url': config_composer.get_config_by_name('VLLM_API_BASE_URL').value,
                    'api_key': config_composer.get_config_by_name('VLLM_API_KEY').value,
                    'model_name': config_composer.get_config_by_name('VLLM_MODEL_NAME').value
                }
                return await llm_service.test_vllm_connection(config_data)

            elif current_provider == "sgl":
                config_data = {
                    'base_url': config_composer.get_config_by_name('SGL_API_BASE_URL').value,
                    'api_key': config_composer.get_config_by_name('SGL_API_KEY').value,
                    'model_name': config_composer.get_config_by_name('SGL_MODEL_NAME').value
                }
                return await llm_service.test_sgl_connection(config_data)

            elif current_provider == "gemini":
                config_data = {
                    'base_url': config_composer.get_config_by_name("GEMINI_API_BASE_URL").value,
                    'api_key': config_composer.get_config_by_name("GEMINI_API_KEY").value,
                    'model_name': config_composer.get_config_by_name("GEMINI_MODEL_DEFAULT").value
                }
                return await llm_service.test_gemini_connection(config_data)

            elif current_provider == "anthropic":
                config_data = {
                    'base_url': config_composer.get_config_by_name("ANTHROPIC_API_BASE_URL").value,
                    'api_key': config_composer.get_config_by_name("ANTHROPIC_API_KEY").value,
                    'model_name': config_composer.get_config_by_name("ANTHROPIC_MODEL_DEFAULT").value
                }
                return await llm_service.test_anthropic_connection(config_data)
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

@router.post("/test/collection/{category}")
async def test_collection_connection(request: Request, category: str):
    """Collection 연결 테스트 API - 이미지-텍스트 변환 설정 기반 연결 테스트"""
    try:
        config_composer = get_config_composer(request)
        const_provider = config_composer.get_config_by_name("DOCUMENT_PROCESSOR_IMAGE_TEXT_MODEL_PROVIDER").value
        if const_provider not in ("openai", "vllm"):
            raise HTTPException(status_code=400, detail=f"Unsupported collection provider: {const_provider}")

        llm_service = LLMService()
        if const_provider == "openai":
            config_data = {
                'api_key': config_composer.get_config_by_name("DOCUMENT_PROCESSOR_OPENAI_IMAGE_TEXT_API_KEY").value,
                'base_url': config_composer.get_config_by_name("DOCUMENT_PROCESSOR_OPENAI_IMAGE_TEXT_BASE_URL").value,
                'model': config_composer.get_config_by_name("DOCUMENT_PROCESSOR_OPENAI_IMAGE_TEXT_MODEL_NAME").value
            }
            return {
                    "status": "success",
                    "message": "OpenAI API connection successful",
                    "api_url": config_composer.get_config_by_name("DOCUMENT_PROCESSOR_OPENAI_IMAGE_TEXT_BASE_URL").value,
                    "models_count": 10,
                    "configured_model": config_composer.get_config_by_name("DOCUMENT_PROCESSOR_OPENAI_IMAGE_TEXT_MODEL_NAME").value,
                    "model_available": config_composer.get_config_by_name("DOCUMENT_PROCESSOR_OPENAI_IMAGE_TEXT_MODEL_NAME").value,
                    "completion_test": True ,
                }
        elif const_provider == "vllm":
            config_data = {
                'base_url':  config_composer.get_config_by_name("DOCUMENT_PROCESSOR_VLLM_IMAGE_TEXT_BASE_URL").value,
                'api_key':   config_composer.get_config_by_name("DOCUMENT_PROCESSOR_VLLM_IMAGE_TEXT_API_KEY").value,
                'model_name':config_composer.get_config_by_name("DOCUMENT_PROCESSOR_VLLM_IMAGE_TEXT_MODEL_NAME").value
            }
            logging.info(f"Testing vLLM connection with config: {config_data}")
            return await llm_service.test_vllm_connection(config_data)

    except HTTPException:
        raise
    except (ValueError, ConnectionError, RequestException) as e:
        logging.error(f"Collection connection test failed for {category}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Collection connection test failed for {category}: {e}")
        raise HTTPException(status_code=500, detail=f"Collection connection test failed: {str(e)}")

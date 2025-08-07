"""
Document Processor 설정 관리
"""

import logging
from typing import Dict, Any

logger = logging.getLogger("document-processor")

class ConfigManager:
    """설정 관리 클래스"""
    
    def get_current_image_text_config(self) -> Dict[str, Any]:
        """실시간으로 현재 IMAGE_TEXT 설정 가져오기"""
        try:
            from main import app
            if hasattr(app.state, 'config_composer'):
                collection_config = app.state.config_composer.get_config_by_category_name("collection")
                
                if hasattr(collection_config, 'IMAGE_TEXT_MODEL_PROVIDER'):
                    provider_obj = getattr(collection_config, 'IMAGE_TEXT_MODEL_PROVIDER')
                    base_url_obj = getattr(collection_config, 'IMAGE_TEXT_BASE_URL')
                    api_key_obj = getattr(collection_config, 'IMAGE_TEXT_API_KEY')
                    model_obj = getattr(collection_config, 'IMAGE_TEXT_MODEL_NAME')
                    temp_obj = getattr(collection_config, 'IMAGE_TEXT_TEMPERATURE')
                    batch_size_obj = getattr(collection_config, 'IMAGE_TEXT_BATCH_SIZE', None)
                    
                    config = {
                        'provider': str(provider_obj.value if hasattr(provider_obj, 'value') else provider_obj).lower(),
                        'base_url': str(base_url_obj.value if hasattr(base_url_obj, 'value') else base_url_obj),
                        'api_key': str(api_key_obj.value if hasattr(api_key_obj, 'value') else api_key_obj),
                        'model': str(model_obj.value if hasattr(model_obj, 'value') else model_obj),
                        'temperature': float(temp_obj.value if hasattr(temp_obj, 'value') else temp_obj),
                        'batch_size': int(batch_size_obj.value if batch_size_obj and hasattr(batch_size_obj, 'value') else 1)
                    }
                    
                    logger.info(f"🔄 Direct value access config: {config}")
                    return config
            
        except Exception as e:
            logger.error(f"🔍 Error in get_current_image_text_config: {e}")
            import traceback
            logger.error(f"🔍 Traceback: {traceback.format_exc()}")
        
        # fallback
        logger.warning("🔍 Using fallback config")
        return {'provider': 'no_model', 'batch_size': 1}

    def is_image_text_enabled(self, config: Dict[str, Any]) -> bool:
        """설정에 따라 OCR이 활성화되어 있는지 확인"""
        from .dependencies import LANGCHAIN_OPENAI_AVAILABLE
        
        provider = config.get('provider', 'no_model')
        if provider in ('openai', 'vllm'):
            if not LANGCHAIN_OPENAI_AVAILABLE:
                logger.warning("langchain_openai not available for OCR")
                return False
            return True
        return False
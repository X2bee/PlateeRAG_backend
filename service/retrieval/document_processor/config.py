# your_package/document_processor/config.py
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("document-processor")

class ConfigProvider:
    """이미지-텍스트(OCR) 관련 설정을 제공하는 인터페이스."""
    def get_current_image_text_config(self) -> Dict[str, Any]:
        raise NotImplementedError

class AppStateConfigProvider(ConfigProvider):
    """
    기존 코드와 '동일한 동작'을 최대한 유지:
    - main.app가 있고, app.state.config_composer.get_config_by_category_name("collection")가 있으면
      그 값을 그대로 꺼내서 반환
    - 실패/부재 시 fallback {'provider':'no_model', 'batch_size':1}
    """
    def get_current_image_text_config(self) -> Dict[str, Any]:
        try:
            from main import app  # 지연 임포트(순환 의존 최소화)
            if hasattr(app.state, 'config_composer'):
                collection_config = app.state.config_composer.get_config_by_category_name("collection")
                def gv(name, default=None):
                    if hasattr(collection_config, name):
                        obj = getattr(collection_config, name)
                        return obj.value if hasattr(obj, 'value') else obj
                    return default

                config = {
                    'provider': str(gv('IMAGE_TEXT_MODEL_PROVIDER', 'no_model')).lower(),
                    'base_url': str(gv('IMAGE_TEXT_BASE_URL', '')),
                    'api_key': str(gv('IMAGE_TEXT_API_KEY', '')),
                    'model': str(gv('IMAGE_TEXT_MODEL_NAME', '')),
                    'temperature': float(gv('IMAGE_TEXT_TEMPERATURE', 0.7)),
                    'batch_size': int(gv('IMAGE_TEXT_BATCH_SIZE', 1)),
                }
                logger.info(f"🔄 Direct value access config: {config}")
                return config
        except Exception as e:
            logger.error(f"🔍 Error in AppStateConfigProvider: {e}", exc_info=True)

        logger.warning("🔍 Using fallback config")
        return {'provider': 'no_model', 'batch_size': 1}

def is_image_text_enabled(config: Dict[str, Any], langchain_openai_available: bool) -> bool:
    provider = config.get('provider', 'no_model')
    if provider in ('openai', 'vllm'):
        if not langchain_openai_available:
            logger.warning("langchain_openai not available for OCR")
            return False
        return True
    return False

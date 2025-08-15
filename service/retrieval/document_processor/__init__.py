"""
Document Processor 패키지 초기화

이 패키지는 두 가지 구현이 공존할 수 있는 상황을 처리합니다:
- 최상위 모듈: service/retrieval/document_processor.py (풍부한 구현)
- 패키지 내부 모듈: service/retrieval/document_processor/document_processor.py (경량 구현)

우선적으로 최상위 파일 기반 구현이 존재하면 이를 로드하여 `DocumentProcessor`를 제공하고,
없거나 로드 실패하면 패키지 내부 구현으로 폴백합니다.
"""
from pathlib import Path
import importlib.util
import logging

logger = logging.getLogger(__name__)


def _load_preferred_document_processor():
    # 최상위 파일 경로: ../document_processor.py
    try:
        base_dir = Path(__file__).resolve().parents[1]  # service/retrieval
        top_level = base_dir / 'document_processor.py'
        if top_level.exists():
            try:
                spec = importlib.util.spec_from_file_location(
                    'plateerag.document_processor_file', str(top_level)
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                cls = getattr(module, 'DocumentProcessor', None)
                if cls:
                    logger.info('Loaded DocumentProcessor from top-level file: %s', top_level)
                    return cls
            except Exception as e:
                logger.warning('Failed to load top-level document_processor.py: %s', e)

    except Exception as e:
        logger.debug('Error while attempting to locate top-level document_processor: %s', e)

    # 폴백: 패키지 내부 모듈에서 로드
    try:
        from .document_processor import DocumentProcessor as _PkgDocProc
        logger.info('Using package-internal DocumentProcessor implementation')
        return _PkgDocProc
    except Exception as e:
        logger.error('Failed to load any DocumentProcessor implementation: %s', e)
        raise


DocumentProcessor = _load_preferred_document_processor()

__all__ = ['DocumentProcessor']
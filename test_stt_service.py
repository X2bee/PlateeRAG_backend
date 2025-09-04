"""
STT 서비스 테스트 스크립트
"""

import asyncio
import logging
from config.config_composer import config_composer
from service.stt.stt_factory import STTFactory

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_stt")

async def test_stt_service():
    """STT 서비스 테스트"""
    try:
        # config_composer 초기화
        logger.info("Initializing config composer...")
        config_composer.initialize_all_configs()

        # STT 클라이언트 생성
        logger.info("Creating STT client...")
        stt_client = STTFactory.create_stt_client(config_composer)

        # 서비스 정보 확인
        provider_info = stt_client.get_provider_info()
        logger.info(f"STT Provider Info: {provider_info}")

        # 서비스 사용 가능성 확인
        is_available = await stt_client.is_available()
        logger.info(f"STT Service Available: {is_available}")

        if is_available:
            logger.info("✅ STT service is ready!")

            # 실제 오디오 파일 테스트를 원하는 경우:
            # audio_file_path = "path/to/your/audio/file.wav"
            # if os.path.exists(audio_file_path):
            #     transcription = await stt_client.transcribe_audio(audio_file_path)
            #     logger.info(f"Transcription: {transcription}")
        else:
            logger.warning("❌ STT service is not available")

        # 정리
        await stt_client.cleanup()
        logger.info("STT client cleanup completed")

    except Exception as e:
        logger.error(f"Error during STT service test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_stt_service())

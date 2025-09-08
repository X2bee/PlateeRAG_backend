"""
Zonos TTS 클라이언트 (Zonos 모델 전용)
"""

import asyncio
from typing import Dict, Any, Optional
import logging
import os
import tempfile
from service.tts.base_tts import BaseTTS

logger = logging.getLogger("tts.zonos")

class ZonosTTS(BaseTTS):
    """Zonos TTS 클라이언트"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get("model_name", "Zyphra/Zonos-v0.1-transformer")
        self.model_device = config.get("model_device", "cpu")
        self.default_speaker = config.get("default_speaker", "female_sample3")
        self.model = None
        self.speaker_embedding = None

        self._initialize_model()

    def _initialize_model(self):
        """모델 및 화자 임베딩 초기화"""
        try:
            from service.tts.zonos.model import Zonos
            import torch

            logger.info("Loading Zonos TTS model: %s", self.model_name)
            device = torch.device(torch.cuda.current_device()) if self.model_device.lower() in ["gpu", "cuda"] and torch.cuda.is_available() else torch.device("cpu")
            logger.info("Using device: %s", device)

            self.model = Zonos.from_pretrained(self.model_name, device=device)
            self._load_default_speaker()

            logger.info("Zonos TTS model loaded successfully: %s", self.model_name)

        except ImportError as import_error:
            logger.error("Zonos package not available: %s", import_error)
            self.model = None
        except (RuntimeError, OSError) as e:
            logger.error("Failed to initialize Zonos TTS model: %s", e)
            self.model = None

    def _load_default_speaker(self):
        """기본 화자 임베딩 로드"""
        try:
            import torchaudio

            # assets 폴더에서 기본 화자 오디오 로드
            script_dir = os.path.dirname(os.path.abspath(__file__))
            audio_path = os.path.join(script_dir, "assets", f"{self.default_speaker}.mp3")

            if not os.path.exists(audio_path):
                logger.warning("Default speaker audio not found: %s", audio_path)
                return

            wav, sampling_rate = torchaudio.load(audio_path)
            self.speaker_embedding = self.model.make_speaker_embedding(wav, sampling_rate)

            logger.info("Default speaker embedding loaded: %s", self.default_speaker)

        except (ImportError, RuntimeError, OSError) as e:
            logger.error("Failed to load default speaker: %s", e)
            self.speaker_embedding = None

    async def generate_speech(
        self,
        text: str,
        speaker: Optional[str] = None,
        language: str = "ko",
        output_format: str = "wav",
        emotion: Optional[list[float]] = None
    ) -> bytes:
        """텍스트를 음성으로 변환"""
        try:
            # 모델과 스피커 임베딩 상태를 안전하게 체크
            if self.model is None:
                raise ValueError("Zonos TTS model not initialized")

            if self.speaker_embedding is None:
                raise ValueError("Speaker embedding not available")

            # CPU 집약적인 작업을 별도 스레드에서 실행
            audio_bytes = await asyncio.to_thread(
                self._generate_speech_sync, text, speaker, language, output_format, emotion
            )

            logger.info("Speech generation completed for text: %s...", text[:50])
            return audio_bytes

        except Exception as e:
            logger.error("Failed to generate speech: %s", e)
            raise

    def _generate_speech_sync(
        self,
        text: str,
        speaker: Optional[str],
        language: str,
        output_format: str,
        emotion: Optional[list[float]] = None
    ) -> bytes:
        """음성 생성 (동기 함수)"""
        try:
            import torch
            import torchaudio
            from service.tts.zonos.conditioning import make_cond_dict
            from service.tts.zonos.utils import DEFAULT_DEVICE

            _ = speaker
            logger.info("Starting speech generation for: %s", text[:50])

            # 시드 설정 (일관된 결과를 위해)
            torch.manual_seed(421)

            # 스피커 임베딩 안전 체크
            safe_speaker_embedding = self.speaker_embedding
            if hasattr(safe_speaker_embedding, 'clone'):
                safe_speaker_embedding = safe_speaker_embedding.clone().detach()

            # 조건 딕셔너리 생성
            try:
                cond_dict_kwargs = {
                    "text": text,
                    "speaker": safe_speaker_embedding,
                    "language": language,
                    "device": DEFAULT_DEVICE
                }

                cond_dict = make_cond_dict(**cond_dict_kwargs)
                logger.info("Condition dict created successfully")
            except Exception as e:
                logger.error("Error creating condition dict: %s", e)
                raise RuntimeError(f"Failed to create condition dict: {e}")

            # 조건부 인코딩 준비
            try:
                conditioning = self.model.prepare_conditioning(cond_dict)
                logger.info("Conditioning prepared successfully")
            except Exception as e:
                logger.error("Error preparing conditioning: %s", e)
                raise RuntimeError(f"Failed to prepare conditioning: {e}")

            # 오디오 코드 생성
            try:
                codes = self.model.generate(conditioning, progress_bar=False)
                logger.info("Audio codes generated successfully")
            except Exception as e:
                logger.error("Error generating audio codes: %s", e)
                raise RuntimeError(f"Failed to generate audio codes: {e}")

            # 오디오 파형으로 디코딩
            try:
                wavs = self.model.autoencoder.decode(codes).cpu()
                logger.info("Audio decoded successfully")
            except Exception as e:
                logger.error("Error decoding audio: %s", e)
                raise RuntimeError(f"Failed to decode audio: {e}")

            # 임시 파일로 저장하고 bytes로 읽기
            with tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False) as temp_file:
                temp_path = temp_file.name

            try:
                # torchaudio로 파일 저장
                torchaudio.save(temp_path, wavs[0], self.model.autoencoder.sampling_rate)

                # 파일을 bytes로 읽기
                with open(temp_path, 'rb') as f:
                    audio_bytes = f.read()

                logger.info("Audio saved and read successfully")
                return audio_bytes

            finally:
                # 임시 파일 정리
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        except (ImportError, RuntimeError, OSError) as e:
            logger.error("Error in speech generation: %s", e)
            raise

    async def is_available(self) -> bool:
        """Zonos TTS 서비스 사용 가능성 확인"""
        try:
            model_ok = self.model is not None
            if not model_ok:
                return False

            embedding_ok = self.speaker_embedding is not None
            if not embedding_ok:
                return False

            # 텐서인 경우 안전한 체크
            if hasattr(self.speaker_embedding, 'numel'):
                return self.speaker_embedding.numel() > 0

            return True
        except Exception as e:
            logger.error("Error in is_available: %s", e)
            return False

    def get_provider_info(self) -> Dict[str, Any]:
        """Zonos TTS 제공자 정보 반환"""
        try:
            # 안전한 가용성 체크
            model_ok = self.model is not None
            embedding_ok = self.speaker_embedding is not None

            available = False
            if model_ok and embedding_ok:
                if hasattr(self.speaker_embedding, 'numel'):
                    available = self.speaker_embedding.numel() > 0
                else:
                    available = True

            return {
                "provider": "zonos",
                "model": self.model_name,
                "default_speaker": self.default_speaker,
                "device": self.model_device,
                "available": available
            }
        except Exception as e:
            logger.error("Error in get_provider_info: %s", e)
            return {
                "provider": "zonos",
                "model": self.model_name,
                "default_speaker": self.default_speaker,
                "device": self.model_device,
                "available": False,
                "error": str(e)
            }

    async def cleanup(self):
        """Zonos TTS 모델 리소스 정리"""
        logger.info("Cleaning up Zonos TTS client: %s", self.model_name)

        if self.model:
            try:
                # 모델을 CPU로 이동 (GPU 메모리 해제)
                if hasattr(self.model, 'to'):
                    self.model.to('cpu')

                # PyTorch 캐시 정리
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    logger.info("PyTorch GPU cache cleared")
                except ImportError:
                    logger.info("PyTorch not available, skipping GPU cleanup")

                # 모델 객체 정리
                del self.model
                self.model = None

                # 화자 임베딩 정리
                del self.speaker_embedding
                self.speaker_embedding = None

                logger.info("Zonos TTS model cleanup completed")

            except (RuntimeError, AttributeError) as e:
                logger.warning("Error during Zonos TTS model cleanup: %s", e)

        # 부모 클래스 cleanup 호출
        await super().cleanup()

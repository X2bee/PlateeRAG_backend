"""
HuggingFace STT 클라이언트 (transformers 사용)
"""

import asyncio
from typing import Dict, Any, Union
import logging
import io
import os
from service.stt.base_stt import BaseSTT

logger = logging.getLogger("stt.huggingface")

class HuggingFaceSTT(BaseSTT):
    """HuggingFace transformers STT 클라이언트"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get("model_name", "openai/whisper-small")
        self.api_key = config.get("api_key", "")
        self.model_device = config.get("model_device", "cpu")
        self.processor = None
        self.model = None

        self._initialize_model()

    def _initialize_model(self):
        """모델 초기화"""
        try:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration

            # API key가 있으면 설정
            if self.api_key:
                os.environ["HUGGINGFACE_HUB_TOKEN"] = self.api_key

            logger.info("Loading HuggingFace STT model: %s", self.model_name)

            try:
                is_gpu_device = (
                    self.model_device == 'gpu' or
                    self.model_device == 'cuda'
                )
                logger.info("Using device: %s (GPU: %s)", self.model_device, is_gpu_device)
                device = 'cuda' if is_gpu_device else 'cpu'

                # 프로세서와 모델 로드
                self.processor = WhisperProcessor.from_pretrained(self.model_name)
                self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name)

                # 모델을 지정된 디바이스로 이동
                self.model.to(device)

                # forced_decoder_ids를 None으로 설정
                self.model.config.forced_decoder_ids = None

                logger.info("STT Model loaded on ===%s=== device", device)
                logger.info("HuggingFace STT model loaded successfully: %s", self.model_name)

            except (ImportError, OSError, RuntimeError) as model_error:
                raise RuntimeError("All HuggingFace STT models failed to load") from model_error

        except ImportError as import_error:
            logger.error("transformers package not installed. Run: pip install transformers")
            logger.error("Import error details: %s", import_error)
            self.processor = None
            self.model = None
        except (RuntimeError, OSError) as e:
            logger.error("Failed to initialize any HuggingFace STT model: %s", e)
            self.processor = None
            self.model = None

    async def transcribe_audio(self, audio_data: Union[bytes, str], audio_format: str = "wav") -> str:
        """오디오 데이터를 텍스트로 변환"""
        if not self.model or not self.processor:
            raise ValueError("HuggingFace STT model not initialized")

        try:
            # CPU 집약적인 작업을 별도 스레드에서 실행
            transcription = await asyncio.to_thread(
                self._transcribe_audio_sync, audio_data, audio_format
            )

            logger.info("Audio transcription completed: %s...", transcription[:50])
            return transcription

        except (ValueError, ImportError, RuntimeError) as e:
            logger.error("Failed to transcribe audio: %s", e)
            raise

    def _convert_audio_to_numpy(self, input_path: str, audio_format: str) -> tuple:
        """오디오 파일을 numpy array로 변환 (ffmpeg 사용)"""
        import ffmpeg
        import tempfile
        import os
        import numpy as np

        # 모든 형식을 16kHz 모노 PCM으로 변환
        logger.info("Converting %s audio to numpy array using ffmpeg", audio_format)

        try:
            # ffmpeg로 오디오를 16-bit PCM raw 데이터로 추출
            process = (
                ffmpeg
                .input(input_path)
                .output('pipe:', format='f32le', acodec='pcm_f32le', ac=1, ar=16000)
                .run_async(pipe_stdout=True, pipe_stderr=True, quiet=True)
            )

            out, err = process.communicate()

            if process.returncode != 0:
                error_msg = err.decode() if err else "Unknown ffmpeg error"
                raise RuntimeError(f"ffmpeg failed: {error_msg}")

            # bytes를 numpy array로 변환
            audio_array = np.frombuffer(out, np.float32)

            logger.info("Audio conversion completed: %d samples at 16kHz", len(audio_array))

            # 유효성 검사
            if len(audio_array) == 0:
                raise ValueError("Audio file appears to be empty or corrupted")

            return audio_array, 16000

        except Exception as e:
            logger.error("Failed to convert audio: %s", e)
            raise

    def _transcribe_audio_sync(self, audio_data: Union[bytes, str], audio_format: str) -> str:
        """오디오를 텍스트로 변환 (동기 함수)"""
        try:
            import tempfile
            import os
            import numpy as np

            # 오디오 데이터 준비
            if isinstance(audio_data, str):
                # 파일 경로인 경우
                input_path = audio_data
                temp_input_path = None
            else:
                # bytes 데이터인 경우 - 임시 파일로 저장
                with tempfile.NamedTemporaryFile(suffix=f".{audio_format}", delete=False) as temp_file:
                    temp_file.write(audio_data)
                    temp_file.flush()
                    input_path = temp_file.name
                    temp_input_path = temp_file.name

            try:
                # ffmpeg로 오디오를 numpy array로 변환
                audio_array, sampling_rate = self._convert_audio_to_numpy(input_path, audio_format)

            finally:
                # bytes에서 생성된 임시 파일 정리
                if temp_input_path:
                    os.unlink(temp_input_path)

            logger.info("Audio loaded successfully: %d samples, sample_rate=%d", len(audio_array), sampling_rate)

            # 입력 특성 추출
            input_features = self.processor(
                audio_array,
                sampling_rate=sampling_rate,
                return_tensors="pt"
            ).input_features

            # GPU를 사용하는 경우 텐서를 GPU로 이동
            device = next(self.model.parameters()).device
            input_features = input_features.to(device)

            # 토큰 ID 생성
            predicted_ids = self.model.generate(input_features)

            # 토큰 ID를 텍스트로 디코드
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)

            # 첫 번째 결과 반환 (배치 크기가 1이므로)
            return transcription[0] if transcription else ""

        except ImportError as import_error:
            if "ffmpeg" in str(import_error):
                logger.error("ffmpeg-python package not installed. Run: pip install ffmpeg-python")
                raise ValueError("Required package not installed: ffmpeg-python") from import_error
            else:
                logger.error("numpy package not installed. Run: pip install numpy")
                raise ValueError("Required package not installed: numpy") from import_error
        except FileNotFoundError as fnf_error:
            if "ffmpeg" in str(fnf_error):
                logger.error("ffmpeg binary not found. Please install ffmpeg: apt install ffmpeg")
                raise ValueError("ffmpeg is not installed. Please install ffmpeg to process audio files.") from fnf_error
            else:
                logger.error("File not found: %s", fnf_error)
                raise ValueError("Audio file not found or inaccessible.") from fnf_error
        except Exception as e:
            # ffmpeg.Error는 Exception의 하위 클래스이므로 여기서 처리
            if "ffmpeg" in str(type(e)) or "ffmpeg" in str(e):
                logger.error("ffmpeg conversion failed: %s", e)
                raise ValueError("Audio format conversion failed. Make sure the audio file is valid.") from e
            else:
                logger.error("Error in audio transcription: %s", e)
                raise

    async def is_available(self) -> bool:
        """HuggingFace STT 서비스 사용 가능성 확인"""
        if not self.model or not self.processor:
            return False

        try:
            # 간단한 테스트 (더미 오디오로 테스트하기 어려우므로 모델 존재만 확인)
            return hasattr(self.model, 'generate') and hasattr(self.processor, 'batch_decode')

        except (RuntimeError, AttributeError) as e:
            logger.warning("HuggingFace STT service not available: %s", e)
            return False

    def get_provider_info(self) -> Dict[str, Any]:
        """HuggingFace STT 제공자 정보 반환"""
        return {
            "provider": "huggingface",
            "model": self.model_name,
            "api_key_configured": bool(self.api_key),
            "available": bool(self.model and self.processor)
        }

    async def cleanup(self):
        """HuggingFace STT 모델 리소스 정리"""
        logger.info("Cleaning up HuggingFace STT client: %s", self.model_name)

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

                # 프로세서 객체 정리
                del self.processor
                self.processor = None

                logger.info("HuggingFace STT model cleanup completed")

            except (RuntimeError, AttributeError) as e:
                logger.warning("Error during HuggingFace STT model cleanup: %s", e)

        # 부모 클래스 cleanup 호출
        await super().cleanup()

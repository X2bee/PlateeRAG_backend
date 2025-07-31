"""
Vast 설정
"""

from typing import Dict, List
from config.base_config import (
    BaseConfig,
    PersistentConfig,
    convert_to_bool,
    convert_to_int,
)

class VastConfig(BaseConfig):
    """Vast 관련 설정을 한곳에서 관리"""

    def initialize(self) -> Dict[str, PersistentConfig]:
        """
        BaseConfig.initialize() 를 구현.
        create_persistent_config() 는
          ① env ➞ ② config_path ➞ ③ default_value ➞ ④ file_path
        순으로 값을 결정(추정)합니다.
        """
        # ‣ API / TOKEN 류 ────────────────────────────────
        self.VAST_API_KEY = self.create_persistent_config(
            env_name="VAST_API_KEY",
            config_path="vast.api_key",
            default_value="5bc0e602079ce2d0a54e2ba300f7cf4b6f802cd244af1d6a3d2dd6c05d8bf50e",
            file_path="vast_api_key.txt",
        )
        # ‣ 인스턴스 템플릿(컨테이너) ─────────────────────
        self.IMAGE_NAME = self.create_persistent_config(
            env_name="VAST_IMAGE_NAME",
            config_path="vast.image.name",
            default_value="cocorof/vllm-openai-praque",
        )
        self.IMAGE_TAG = self.create_persistent_config(
            env_name="VAST_IMAGE_TAG",
            config_path="vast.image.tag",
            default_value="v0.10.0",
        )

        # ‣ 자원/가격 한계 ───────────────────────────────
        self.MAX_PRICE = self.create_persistent_config(
            env_name="VAST_MAX_PRICE",
            config_path="vast.resource.max_price",
            default_value="1.0",
        )
        self.DISK_SIZE_GB = self.create_persistent_config(
            env_name="VAST_DISK_SIZE",
            config_path="vast.resource.disk_size_gb",
            default_value="256",
        )
        self.MIN_GPU_RAM_GB = self.create_persistent_config(
            env_name="VAST_MIN_GPU_RAM",
            config_path="vast.resource.min_gpu_ram",
            default_value="8",
        )
        self.MIN_DISK_GB = self.create_persistent_config(
            env_name="VAST_MIN_DISK",
            config_path="vast.resource.min_disk",
            default_value="20",
        )

        # ‣ 검색 쿼리 기본값 ──────────────────────────────
        self.SEARCH_QUERY = self.create_persistent_config(
            env_name="VAST_SEARCH_QUERY",
            config_path="vast.search.query",
            default_value=(
                "gpu_name=A100_SXM4 "
                "cuda_max_good=12.8 "
                "num_gpus=1 "
                "inet_down>=5000 inet_up>=5000 "
                "disk_space>=200"
            ),
        )

        # ‣ 네트워크 및 포트 설정 ──────────────────────────
        self.DEFAULT_PORTS = self.create_persistent_config(
            env_name="VAST_DEFAULT_PORTS",
            config_path="vast.network.default_ports",
            default_value="1111,6006,8080,8384,72299,11479,11480",
        )
        self.VLLM_HOST_IP = self.create_persistent_config(
            env_name="VLLM_HOST_IP",
            config_path="vast.vllm.host_ip",
            default_value="0.0.0.0",
        )
        self.VLLM_PORT = self.create_persistent_config(
            env_name="VLLM_PORT",
            config_path="vast.vllm.port",
            default_value="11479",
        )
        self.VLLM_CONTROLLER_PORT = self.create_persistent_config(
            env_name="VLLM_CONTROLLER_PORT",
            config_path="vast.vllm.controller_port",
            default_value="11480",
        )

        # ‣ vLLM 모델 및 실행 설정 ─────────────────────────
        self.VLLM_MODEL_NAME = self.create_persistent_config(
            env_name="VLLM_MODEL_NAME",
            config_path="vast.vllm.model_name",
            default_value="Qwen/Qwen3-1.7B",
        )
        self.VLLM_MAX_MODEL_LEN = self.create_persistent_config(
            env_name="VLLM_MAX_MODEL_LEN",
            config_path="vast.vllm.max_model_len",
            default_value="2048",
        )
        self.VLLM_GPU_MEMORY_UTILIZATION = self.create_persistent_config(
            env_name="VLLM_GPU_MEMORY_UTILIZATION",
            config_path="vast.vllm.gpu_memory_utilization",
            default_value="0.5",
        )
        self.VLLM_PIPELINE_PARALLEL_SIZE = self.create_persistent_config(
            env_name="VLLM_PIPELINE_PARALLEL_SIZE",
            config_path="vast.vllm.pipeline_parallel_size",
            default_value="1",
        )
        self.VLLM_TENSOR_PARALLEL_SIZE = self.create_persistent_config(
            env_name="VLLM_TENSOR_PARALLEL_SIZE",
            config_path="vast.vllm.tensor_parallel_size",
            default_value="1",
        )
        self.VLLM_DTYPE = self.create_persistent_config(
            env_name="VLLM_DTYPE",
            config_path="vast.vllm.dtype",
            default_value="bfloat16",
        )
        self.VLLM_TOOL_CALL_PARSER = self.create_persistent_config(
            env_name="VLLM_TOOL_CALL_PARSER",
            config_path="vast.vllm.tool_call_parser",
            default_value="hermes",
        )

        # ‣ 런타임 플래그 ────────────────────────────────
        self.DEBUG = self.create_persistent_config(
            env_name="VAST_DEBUG",
            config_path="vast.debug",
            default_value="false",
        )
        self.AUTO_DESTROY = self.create_persistent_config(
            env_name="VAST_AUTO_DESTROY",
            config_path="vast.auto_destroy",
            default_value="false",
        )
        self.TIMEOUT = self.create_persistent_config(
            env_name="VAST_TIMEOUT",
            config_path="vast.timeout",
            default_value="600",
        )

        # ‣ onstart 명령어 설정 ──────────────────────────
        self.ONSTART_SCRIPT = self.create_persistent_config(
            env_name="VAST_ONSTART_SCRIPT",
            config_path="vast.onstart.script",
            default_value="",
        )

        # API 키 상태 로깅
        if self.VAST_API_KEY.value and self.VAST_API_KEY.value.strip():
            self.logger.info("Vast API key configured")
        else:
            self.logger.warning("Vast API key not configured")

        return self.configs

    # # ────────────────────────────────────────────────────────────
    # # 2)  Getter 헬퍼
    # # ────────────────────────────────────────────────────────────

    # # 컨테이너 이미지 (이름:태그)
    # def image_name(self) -> str:
    #     name = self.IMAGE_NAME.value
    #     tag = self.IMAGE_TAG.value
    #     return f"{name}:{tag}" if tag else name

    # # HF / Vast 토큰
    # def vast_api_key(self) -> str:
    #     return self.VAST_API_KEY.value

    # def hf_hub_token(self) -> str:
    #     return self.HUGGING_FACE_HUB_TOKEN.value

    # # 숫자 / bool 변환
    # def max_price(self) -> float:
    #     return float(self.MAX_PRICE.value)

    # def disk_size(self) -> int:
    #     return convert_to_int(self.DISK_SIZE_GB.value)

    # def min_gpu_ram(self) -> int:
    #     return convert_to_int(self.MIN_GPU_RAM_GB.value)

    # def min_disk(self) -> int:
    #     return convert_to_int(self.MIN_DISK_GB.value)

    # # 검색 조건
    # def search_query(self) -> str:
    #     return self.SEARCH_QUERY.value

    # # 네트워크 설정
    # def default_ports(self) -> List[int]:
    #     """기본 포트 목록 반환"""
    #     ports_str = self.DEFAULT_PORTS.value
    #     return [int(p.strip()) for p in ports_str.split(",") if p.strip().isdigit()]

    # def vllm_host_ip(self) -> str:
    #     return self.VLLM_HOST_IP.value

    # def vllm_port(self) -> int:
    #     return convert_to_int(self.VLLM_PORT.value)

    # def vllm_controller_port(self) -> int:
    #     return convert_to_int(self.VLLM_CONTROLLER_PORT.value)

    # # vLLM 모델 및 실행 설정
    # def vllm_model_name(self) -> str:
    #     return self.VLLM_MODEL_NAME.value

    # def vllm_max_model_len(self) -> int:
    #     return convert_to_int(self.VLLM_MAX_MODEL_LEN.value)

    # def vllm_gpu_memory_utilization(self) -> float:
    #     return float(self.VLLM_GPU_MEMORY_UTILIZATION.value)

    # def vllm_pipeline_parallel_size(self) -> int:
    #     return convert_to_int(self.VLLM_PIPELINE_PARALLEL_SIZE.value)

    # def vllm_tensor_parallel_size(self) -> int:
    #     return convert_to_int(self.VLLM_TENSOR_PARALLEL_SIZE.value)

    # def vllm_dtype(self) -> str:
    #     return self.VLLM_DTYPE.value

    # def vllm_tool_call_parser(self) -> str:
    #     return self.VLLM_TOOL_CALL_PARSER.value

    # # 런타임 플래그
    # def debug(self) -> bool:
    #     return convert_to_bool(self.DEBUG.value)

    # def auto_destroy(self) -> bool:
    #     return convert_to_bool(self.AUTO_DESTROY.value)

    # def timeout(self) -> int:
    #     return convert_to_int(self.TIMEOUT.value)

    # def onstart_script(self) -> str:
    #     return self.ONSTART_SCRIPT.value

    # def generate_onstart_command(self) -> str:
    #     """표준 onstart 명령어 생성"""
    #     vllm_host = self.vllm_host_ip()
    #     vllm_port = self.vllm_port()
    #     vllm_controller_port = self.vllm_controller_port()

    #     # 기본 onstart 명령어 (HF 로그인 제거)
    #     base_cmd = f"""
    #     sleep 30 &&
    #     cd /home/vllm-script &&
    #     export VLLM_HOST_IP={vllm_host} &&
    #     export VLLM_PORT={vllm_port} &&
    #     export VLLM_CONTROLLER_PORT={vllm_controller_port} &&
    #     nohup python3 /home/vllm-script/main.py > /tmp/vllm.log 2>&1 &
    #     """.strip().replace('\n', ' ')

    #     # 사용자 정의 스크립트가 있으면 추가
    #     custom_script = self.onstart_script()
    #     if custom_script:
    #         return f"{base_cmd} && {custom_script}"

    #     return base_cmd

"""
Vast 설정
"""

from typing import Dict, List
from config.base_config import (
    BaseConfig,
    PersistentConfig,
    convert_to_str,
    convert_to_bool,
    convert_to_int,
    convert_to_float,
    convert_to_int_list
)

class VastConfig(BaseConfig):
    def initialize(self) -> Dict[str, PersistentConfig]:
        # ‣ API / TOKEN 류 ────────────────────────────────
        self.VAST_API_KEY = self.create_persistent_config(
            env_name="VAST_API_KEY",
            config_path="vast.api_key",
            default_value="5bc0e602079ce2d0a54e2ba300f7cf4b6f802cd244af1d6a3d2dd6c05d8bf50e",
            file_path="vast_api_key.txt",
        )
        # ‣ 인스턴스 템플릿(컨테이너) ─────────────────────
        self.VAST_IMAGE_NAME = self.create_persistent_config(
            env_name="VAST_IMAGE_NAME",
            config_path="vast.image.name",
            default_value="cocorof/vllm-openai-xgen",
        )
        self.VAST_IMAGE_TAG = self.create_persistent_config(
            env_name="VAST_IMAGE_TAG",
            config_path="vast.image.tag",
            default_value="v0.10.2",
        )
        # ‣ 트레인 인스턴스 템플릿(컨테이너) ─────────────────────
        self.VAST_TRAIN_IMAGE_NAME = self.create_persistent_config(
            env_name="VAST_TRAIN_IMAGE_NAME",
            config_path="vast.train.image.name",
            default_value="",
        )
        self.VAST_TRAIN_IMAGE_TAG = self.create_persistent_config(
            env_name="VAST_TRAIN_IMAGE_TAG",
            config_path="vast.train.image.tag",
            default_value="",
        )

        # ‣ 자원/가격 한계 ───────────────────────────────
        self.VAST_MAX_PRICE = self.create_persistent_config(
            env_name="VAST_MAX_PRICE",
            config_path="vast.resource.max_price",
            default_value="1.0",
            type_converter=convert_to_float,
        )
        self.VAST_DISK_SIZE_GB = self.create_persistent_config(
            env_name="VAST_DISK_SIZE",
            config_path="vast.resource.disk_size_gb",
            default_value="256",
            type_converter=convert_to_int,
        )
        self.VAST_MIN_GPU_RAM_GB = self.create_persistent_config(
            env_name="VAST_MIN_GPU_RAM",
            config_path="vast.resource.min_gpu_ram",
            default_value="8",
            type_converter=convert_to_int,
        )
        self.VAST_MIN_DISK_GB = self.create_persistent_config(
            env_name="VAST_MIN_DISK",
            config_path="vast.resource.min_disk",
            default_value="200",
            type_converter=convert_to_int,
        )
        self.VAST_SEARCH_QUERY = self.create_persistent_config(
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
        self.VAST_DEFAULT_PORTS = self.create_persistent_config(
            env_name="VAST_DEFAULT_PORTS",
            config_path="vast.network.default_ports",
            default_value=[1111, 6006, 8080, 8384, 72299, 12434, 12435],  # 리스트로 저장
            type_converter=convert_to_int_list,
        )
        self.VAST_DEFAULT_TRAIN_PORTS = self.create_persistent_config(
            env_name="VAST_DEFAULT_TRAIN_PORTS",
            config_path="vast.network.default_train_ports",
            default_value=[1111, 6006, 8080, 8384, 72299, 8010],  # 리스트로 저장
            type_converter=convert_to_int_list,
        )

        self.VLLM_HOST_IP = self.create_persistent_config(
            env_name="VLLM_HOST_IP",
            config_path="vast.vllm.host_ip",
            default_value="0.0.0.0",
        )
        self.VLLM_PORT = self.create_persistent_config(
            env_name="VLLM_PORT",
            config_path="vast.vllm.port",
            default_value="12434",
            type_converter=convert_to_int,
        )
        self.VLLM_CONTROLLER_PORT = self.create_persistent_config(
            env_name="VLLM_CONTROLLER_PORT",
            config_path="vast.vllm.controller_port",
            default_value="12435",
            type_converter=convert_to_int,
        )

        self.VLLM_SERVE_MODEL_NAME = self.create_persistent_config(
            env_name="VLLM_SERVE_MODEL_NAME",
            config_path="vast.vllm.serve_model_name",
            default_value="x2bee/Polar-14B",
            type_converter=convert_to_str,
        )
        self.VLLM_MAX_MODEL_LEN = self.create_persistent_config(
            env_name="VLLM_MAX_MODEL_LEN",
            config_path="vast.vllm.max_model_len",
            default_value="2048",
            type_converter=convert_to_int,
        )
        self.VLLM_GPU_MEMORY_UTILIZATION = self.create_persistent_config(
            env_name="VLLM_GPU_MEMORY_UTILIZATION",
            config_path="vast.vllm.gpu_memory_utilization",
            default_value="0.5",
            type_converter=convert_to_float,
        )
        self.VLLM_PIPELINE_PARALLEL_SIZE = self.create_persistent_config(
            env_name="VLLM_PIPELINE_PARALLEL_SIZE",
            config_path="vast.vllm.pipeline_parallel_size",
            default_value="1",
            type_converter=convert_to_int,
        )
        self.VLLM_TENSOR_PARALLEL_SIZE = self.create_persistent_config(
            env_name="VLLM_TENSOR_PARALLEL_SIZE",
            config_path="vast.vllm.tensor_parallel_size",
            default_value="1",
            type_converter=convert_to_int,
        )
        self.VLLM_DTYPE = self.create_persistent_config(
            env_name="VLLM_DTYPE",
            config_path="vast.vllm.dtype",
            default_value="bfloat16",
            type_converter=convert_to_str,
        )
        self.VLLM_TOOL_CALL_PARSER = self.create_persistent_config(
            env_name="VLLM_TOOL_CALL_PARSER",
            config_path="vast.vllm.tool_call_parser",
            default_value="hermes",
            type_converter=convert_to_str,
        )

        # ‣ 런타임 플래그 ────────────────────────────────
        self.VAST_DEBUG = self.create_persistent_config(
            env_name="VAST_DEBUG",
            config_path="vast.debug",
            default_value="false",
            type_converter=convert_to_bool,
        )
        self.VAST_AUTO_DESTROY = self.create_persistent_config(
            env_name="VAST_AUTO_DESTROY",
            config_path="vast.auto_destroy",
            default_value="false",
            type_converter=convert_to_bool,
        )
        self.VAST_TIMEOUT = self.create_persistent_config(
            env_name="VAST_TIMEOUT",
            config_path="vast.timeout",
            default_value="600",
            type_converter=convert_to_int,
        )

        # ‣ 프록시 설정 ────────────────────────────────
        self.VAST_PROXY_MODE = self.create_persistent_config(
            env_name="VAST_PROXY_MODE",
            config_path="vast.proxy.mode",
            default_value="local",
            type_converter=convert_to_str,
        )
        self.VAST_PROXY_BASE_URL = self.create_persistent_config(
            env_name="VAST_PROXY_BASE_URL",
            config_path="vast.proxy.base_url",
            default_value="",
            type_converter=convert_to_str,
        )
        self.VAST_PROXY_TIMEOUT = self.create_persistent_config(
            env_name="VAST_PROXY_TIMEOUT",
            config_path="vast.proxy.timeout",
            default_value="30",
            type_converter=convert_to_int,
        )
        self.VAST_PROXY_API_TOKEN = self.create_persistent_config(
            env_name="VAST_PROXY_API_TOKEN",
            config_path="vast.proxy.api_token",
            default_value="",
        )

        # ‣ onstart 명령어 설정 ──────────────────────────
        self.VAST_ONSTART_SCRIPT = self.create_persistent_config(
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

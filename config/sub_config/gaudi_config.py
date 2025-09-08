"""
LLM 제공자 관련 설정
"""
from typing import Dict
from config.base_config import (
    BaseConfig,
    PersistentConfig,
    convert_to_str,
    convert_to_bool,
    convert_to_int,
    convert_to_float,
    convert_to_int_list
)

class GaudiConfig(BaseConfig):
   """LLM 제공자 관련 설정 관리"""

   def initialize(self) -> Dict[str, PersistentConfig]:
       """LLM 관련 설정들을 초기화"""
       # LLM 자동 전환 설정
       self.GAUDI_VLLM_HOST_IP = self.create_persistent_config(
           env_name="GAUDI_VLLM_HOST_IP",
           config_path="gaudi.vllm.host_ip",
           default_value="0.0.0.0",
       )
       self.GAUDI_VLLM_PORT = self.create_persistent_config(
           env_name="GAUDI_VLLM_PORT",
           config_path="gaudi.vllm.port",
           default_value="12434",
           type_converter=convert_to_int,
       )
       self.GAUDI_VLLM_CONTROLLER_PORT = self.create_persistent_config(
           env_name="GAUDI_VLLM_CONTROLLER_PORT",
           config_path="gaudi.vllm.controller_port",
           default_value="12435",
           type_converter=convert_to_int,
       )

       self.GAUDI_VLLM_SERVE_MODEL_NAME = self.create_persistent_config(
           env_name="GAUDI_VLLM_SERVE_MODEL_NAME",
           config_path="gaudi.vllm.serve_model_name",
           default_value="x2bee/Polar-14B",
           type_converter=convert_to_str,
       )
       self.GAUDI_VLLM_MAX_MODEL_LEN = self.create_persistent_config(
           env_name="GAUDI_VLLM_MAX_MODEL_LEN",
           config_path="gaudi.vllm.max_model_len",
           default_value="2048",
           type_converter=convert_to_int,
       )
       self.GAUDI_VLLM_GPU_MEMORY_UTILIZATION = self.create_persistent_config(
           env_name="GAUDI_VLLM_GPU_MEMORY_UTILIZATION",
           config_path="gaudi.vllm.gpu_memory_utilization",
           default_value="0.5",
           type_converter=convert_to_float,
       )
       self.GAUDI_VLLM_PIPELINE_PARALLEL_SIZE = self.create_persistent_config(
           env_name="GAUDI_VLLM_PIPELINE_PARALLEL_SIZE",
           config_path="gaudi.vllm.pipeline_parallel_size",
           default_value="1",
           type_converter=convert_to_int,
       )
       self.GAUDI_VLLM_TENSOR_PARALLEL_SIZE = self.create_persistent_config(
           env_name="GAUDI_VLLM_TENSOR_PARALLEL_SIZE",
           config_path="gaudi.vllm.tensor_parallel_size",
           default_value="1",
           type_converter=convert_to_int,
       )
       self.GAUDI_VLLM_DTYPE = self.create_persistent_config(
           env_name="GAUDI_VLLM_DTYPE",
           config_path="gaudi.vllm.dtype",
           default_value="bfloat16",
           type_converter=convert_to_str,
       )
       self.GAUDI_VLLM_TOOL_CALL_PARSER = self.create_persistent_config(
           env_name="GAUDI_VLLM_TOOL_CALL_PARSER",
           config_path="gaudi.vllm.tool_call_parser",
           default_value="hermes",
           type_converter=convert_to_str,
       )
       return self.configs
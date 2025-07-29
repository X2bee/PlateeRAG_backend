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

    # ────────────────────────────────────────────────────────────
    # 1)  Persistent + Env 변수 정의
    # ────────────────────────────────────────────────────────────
    def initialize(self) -> Dict[str, PersistentConfig]:
        """
        BaseConfig.initialize() 를 구현.
        create_persistent_config() 는
          ① env ➞ ② config_path ➞ ③ default_value ➞ ④ file_path
        순으로 값을 결정(추정)합니다.
        """
        return {
            # ‣ API / TOKEN 류 ────────────────────────────────
            "VAST_API_KEY": self.create_persistent_config(
                env_name="VAST_API_KEY",
                config_path="vast.api_key",
                default_value="",
                file_path="vast_api_key.txt",
            ),
            "HF_TOKEN": self.create_persistent_config(
                env_name="HF_TOKEN",
                config_path="vast.hf_token",
                default_value="",
                file_path="hf_token.txt",
            ),

            # ‣ 인스턴스 템플릿(컨테이너) ─────────────────────
            "IMAGE_NAME": self.create_persistent_config(
                env_name="VAST_IMAGE_NAME",
                config_path="vast.image.name",
                default_value="cocorof/polarops-vllm",
            ),
            "IMAGE_TAG": self.create_persistent_config(
                env_name="VAST_IMAGE_TAG",
                config_path="vast.image.tag",
                default_value="1.0.1",
            ),

            # ‣ 자원/가격 한계 ───────────────────────────────
            "MAX_PRICE": self.create_persistent_config(
                env_name="VAST_MAX_PRICE",
                config_path="vast.resource.max_price",
                default_value="1.0",          # string → float 변환 필요
            ),
            "DISK_SIZE_GB": self.create_persistent_config(
                env_name="VAST_DISK_SIZE",
                config_path="vast.resource.disk_size_gb",
                default_value="256",          # string → int 변환 필요
            ),
            "MIN_GPU_RAM_GB": self.create_persistent_config(
                env_name="VAST_MIN_GPU_RAM",
                config_path="vast.resource.min_gpu_ram",
                default_value="8",
            ),
            "MIN_DISK_GB": self.create_persistent_config(
                env_name="VAST_MIN_DISK",
                config_path="vast.resource.min_disk",
                default_value="20",
            ),

            # ‣ 검색 쿼리 기본값 ──────────────────────────────
            "SEARCH_QUERY": self.create_persistent_config(
                env_name="VAST_SEARCH_QUERY",
                config_path="vast.search.query",
                default_value=(
                    "gpu_name=A100_SXM4 "
                    "cuda_max_good=12.8 "
                    "num_gpus=1 "
                    "inet_down>=5000 inet_up>=5000 "
                    "disk_space>=200"
                ),
            ),

            # ‣ 네트워크 및 포트 설정 ──────────────────────────
            "DEFAULT_PORTS": self.create_persistent_config(
                env_name="VAST_DEFAULT_PORTS",
                config_path="vast.network.default_ports",
                default_value="1111,6006,8080,8384,72299,11479,11480",
            ),
            "VLLM_HOST_IP": self.create_persistent_config(
                env_name="VLLM_HOST_IP",
                config_path="vast.vllm.host_ip",
                default_value="0.0.0.0",
            ),
            "VLLM_PORT": self.create_persistent_config(
                env_name="VLLM_PORT",
                config_path="vast.vllm.port",
                default_value="11479",
            ),
            "VLLM_CONTROLLER_PORT": self.create_persistent_config(
                env_name="VLLM_CONTROLLER_PORT",
                config_path="vast.vllm.controller_port",
                default_value="11480",
            ),

            # ‣ 런타임 플래그 ────────────────────────────────
            "DEBUG": self.create_persistent_config(
                env_name="VAST_DEBUG",
                config_path="vast.debug",
                default_value="false",
            ),
            "AUTO_DESTROY": self.create_persistent_config(
                env_name="VAST_AUTO_DESTROY",
                config_path="vast.auto_destroy",
                default_value="false",
            ),
            "TIMEOUT": self.create_persistent_config(
                env_name="VAST_TIMEOUT",
                config_path="vast.timeout",
                default_value="600",
            ),

            # ‣ onstart 명령어 설정 ──────────────────────────
            "ONSTART_SCRIPT": self.create_persistent_config(
                env_name="VAST_ONSTART_SCRIPT",
                config_path="vast.onstart.script",
                default_value="",
            ),
        }

    # ────────────────────────────────────────────────────────────
    # 2)  Getter 헬퍼
    # ────────────────────────────────────────────────────────────
    #  (필요한 항목만 노출; 나머진 상위 코드가
    #   self["KEY"].value 형태로 직접 접근해도 무방)
    # -----------------------------------------------------------

    # 컨테이너 이미지 (이름:태그)
    def image_name(self) -> str:
        name = self["IMAGE_NAME"].value
        tag  = self["IMAGE_TAG"].value
        return f"{name}:{tag}" if tag else name

    # HF / Vast 토큰
    def vast_api_key(self) -> str:
        return self["VAST_API_KEY"].value

    def hf_token(self) -> str:
        return self["HF_TOKEN"].value

    # 숫자 / bool 변환
    def max_price(self) -> float:
        return float(self["MAX_PRICE"].value)

    def disk_size(self) -> int:
        return convert_to_int(self["DISK_SIZE_GB"].value)

    def min_gpu_ram(self) -> int:
        return convert_to_int(self["MIN_GPU_RAM_GB"].value)

    def min_disk(self) -> int:
        return convert_to_int(self["MIN_DISK_GB"].value)

    # 검색 조건
    def search_query(self) -> str:
        return self["SEARCH_QUERY"].value

    # 네트워크 설정
    def default_ports(self) -> List[int]:
        """기본 포트 목록 반환"""
        ports_str = self["DEFAULT_PORTS"].value
        return [int(p.strip()) for p in ports_str.split(",") if p.strip().isdigit()]

    def vllm_host_ip(self) -> str:
        return self["VLLM_HOST_IP"].value

    def vllm_port(self) -> int:
        return convert_to_int(self["VLLM_PORT"].value)

    def vllm_controller_port(self) -> int:
        return convert_to_int(self["VLLM_CONTROLLER_PORT"].value)

    # 런타임 플래그
    def debug(self) -> bool:
        return convert_to_bool(self["DEBUG"].value)

    def auto_destroy(self) -> bool:
        return convert_to_bool(self["AUTO_DESTROY"].value)

    def timeout(self) -> int:
        return convert_to_int(self["TIMEOUT"].value)

    def onstart_script(self) -> str:
        return self["ONSTART_SCRIPT"].value

    def generate_onstart_command(self) -> str:
        """표준 onstart 명령어 생성"""
        vllm_host = self.vllm_host_ip()
        vllm_port = self.vllm_port()
        vllm_controller_port = self.vllm_controller_port()
        
        # 기본 onstart 명령어 (HF 로그인 제거)
        base_cmd = f"""
        sleep 30 && 
        cd /home/vllm-script && 
        export VLLM_HOST_IP={vllm_host} && 
        export VLLM_PORT={vllm_port} && 
        export VLLM_CONTROLLER_PORT={vllm_controller_port} && 
        nohup python3 /home/vllm-script/main.py > /tmp/vllm.log 2>&1 &
        """.strip().replace('\n', ' ')
        
        # 사용자 정의 스크립트가 있으면 추가
        custom_script = self.onstart_script()
        if custom_script:
            return f"{base_cmd} && {custom_script}"
        
        return base_cmd

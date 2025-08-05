"""
VastAI 인스턴스 관리자

Vast.ai CLI를 래핑하여 인스턴스 생성, 삭제, 상태 관리 등의 기능을 제공합니다.
"""

import subprocess
import json
import time
import logging
import re
import asyncio
import shlex
import urllib.parse
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger("vast-manager")

class VastAIManager:
    """Vast.ai CLI 래핑 클래스"""

    def __init__(self, config, db_manager=None):
        """VastAIManager 초기화

        Args:
            config: VastConfig 인스턴스
            db_manager: 데이터베이스 매니저 (선택사항)
        """
        self.config = config
        self.db_manager = db_manager
        self.timeout = 600  # 기본 타임아웃 10분

        # VastAI 실행 방식 설정 (환경에 따라 조정 가능)
        self.vastai_prefix = ["python", "-m", "uv", "run"]

    def run_command(self, cmd: List[str], parse_json: bool = True, timeout: int = None) -> Dict[str, Any]:
        """CLI 명령 실행 및 결과 파싱

        Args:
            cmd: 실행할 명령어 리스트
            parse_json: JSON 파싱 여부
            timeout: 타임아웃 (초)

        Returns:
            명령 실행 결과
        """
        timeout = timeout or self.timeout

        # vastai 명령어인 경우 먼저 prefix와 함께 시도
        if cmd and cmd[0] == "vastai":
            prefixed_cmd = self.vastai_prefix + cmd

            # prefix와 함께 실행 시도
            result = self._execute_command(prefixed_cmd, parse_json, timeout)

            # prefix와 함께 실행이 실패한 경우 prefix 없이 재시도
            if not result["success"]:
                logger.info(f"🔄 prefix 실행 실패, fallback으로 재시도: {' '.join(cmd)}")
                result = self._execute_command(cmd, parse_json, timeout)

            return result
        else:
            return self._execute_command(cmd, parse_json, timeout)

    def _execute_command(self, cmd: List[str], parse_json: bool = True, timeout: int = None) -> Dict[str, Any]:
        """실제 명령어 실행"""
        try:
            if self.config.debug():
                logger.debug(f"실행 명령: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if self.config.debug():
                logger.debug(f"stdout: {result.stdout}")
                logger.debug(f"stderr: {result.stderr}")

            if result.returncode != 0:
                error_msg = f"명령 실행 실패: {result.stderr}"

                # VastAI CLI의 특정 오류들을 더 자세히 분석
                stderr_lower = result.stderr.lower()
                if any(err in stderr_lower for err in ['nonetype', 'subscriptable', 'traceback']):
                    logger.debug(f"VastAI CLI 내부 오류 감지: {result.stderr}")
                else:
                    logger.error(error_msg)

                return {
                    "success": False,
                    "error": error_msg,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }

            output = result.stdout.strip()

            if parse_json and output:
                try:
                    data = json.loads(output)
                    return {"success": True, "data": data}
                except json.JSONDecodeError:
                    # JSON 파싱 실패 시 텍스트로 반환
                    return {"success": True, "data": output}

            return {"success": True, "data": output}

        except subprocess.TimeoutExpired:
            logger.error(f"명령 타임아웃: {' '.join(cmd)}")
            return {"success": False, "error": "Command timeout"}
        except Exception as e:
            logger.error(f"명령 실행 중 오류: {e}")
            return {"success": False, "error": str(e)}

    def setup_api_key(self) -> bool:
        """API 키 설정 및 확인"""
        api_key = self.config.vast_api_key()

        # API 키 없이도 일부 기능이 동작하도록 허용
        if not api_key or api_key == "your_api_key_here":
            logger.warning("⚠️ API 키가 설정되지 않았습니다.")
            logger.info("Vast.ai 웹사이트 (https://cloud.vast.ai/)에서 계정을 생성하고 API 키를 발급받아 주세요.")
            logger.info("발급받은 후 다음 명령어로 설정하세요: vastai set api-key YOUR_API_KEY")
            return self._handle_missing_api_key()

        try:
            # API 키 설정
            result = self._run_command_without_api_key(["vastai", "set", "api-key", api_key])
            logger.info("✅ API 키가 설정되었습니다.")

            # API 키 검증
            try:
                test_result = self._run_command_without_api_key(["vastai", "show", "user"], capture_json=True)
                logger.info("✅ API 키 검증 완료")
                if isinstance(test_result, dict):
                    user_info = test_result
                    logger.info(f"사용자: {user_info.get('username', 'Unknown')}")
                    if 'credit' in user_info:
                        logger.info(f"잔액: ${user_info['credit']:.2f}")
                return True
            except Exception as e:
                logger.warning(f"API 키 검증 실패: {e}")
                return False

        except Exception as e:
            logger.warning(f"API 키 설정 실패: {e}")
            return False

    def _run_command_without_api_key(self, cmd: List[str], capture_json: bool = False) -> Any:
        """API 키 없이 명령어 실행 (API 키 관리용)"""
        # vastai 명령어인 경우 먼저 prefix와 함께 시도
        if cmd and cmd[0] == "vastai":
            prefixed_cmd = self.vastai_prefix + cmd

            # prefix와 함께 실행 시도
            try:
                return self._execute_command_without_api_key(prefixed_cmd, capture_json)
            except Exception as e:
                logger.info(f"🔄 prefix API 키 명령 실패, fallback으로 재시도: {' '.join(cmd)} (오류: {e})")
                # prefix 없이 재시도
                return self._execute_command_without_api_key(cmd, capture_json)
        else:
            return self._execute_command_without_api_key(cmd, capture_json)

    def _execute_command_without_api_key(self, cmd: List[str], capture_json: bool = False) -> Any:
        """실제 명령어 실행 (API 키 관리용)"""
        try:
            logger.debug(f"명령어 실행: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")

            output = result.stdout.strip()

            if capture_json and output:
                try:
                    return json.loads(output)
                except json.JSONDecodeError:
                    logger.debug(f"JSON 파싱 실패, 텍스트로 반환: {output[:100]}...")
                    return output

            return output

        except subprocess.TimeoutExpired:
            raise TimeoutError("명령어 실행 시간 초과")

    def _handle_missing_api_key(self) -> bool:
        """API 키 부재 처리"""
        try:
            # 저장된 API 키 확인
            result = self._run_command_without_api_key(["vastai", "show", "api-keys"], capture_json=True)

            # API 키 목록 확인 및 선택
            api_keys = []

            # 응답 형식에 따른 처리
            if isinstance(result, dict) and 'apikeys' in result:
                api_keys = result['apikeys']
            elif isinstance(result, list):
                api_keys = result
            elif isinstance(result, str):
                logger.debug("API 키 목록이 텍스트 형식으로 반환되었습니다. 파싱 시도...")
                try:
                    import ast
                    parsed = ast.literal_eval(result)
                    if isinstance(parsed, dict) and 'apikeys' in parsed:
                        api_keys = parsed['apikeys']
                    elif isinstance(parsed, list):
                        api_keys = parsed
                except:
                    logger.debug("API 키 목록 파싱 실패")

            # 유효한 API 키 필터링
            valid_keys = []
            for key_info in api_keys:
                if key_info.get('deleted_at') is None and key_info.get('key') is not None:
                    valid_keys.append(key_info)

            if valid_keys:
                # 우선순위: key_type이 'api'인 키 > 'primary' 키 > 나머지
                api_type_keys = [k for k in valid_keys if k.get('key_type') == 'api']
                primary_keys = [k for k in valid_keys if k.get('key_type') == 'primary']

                selected_key = None
                if api_type_keys:
                    selected_key = api_type_keys[0]
                elif primary_keys:
                    selected_key = primary_keys[0]
                elif valid_keys:
                    selected_key = valid_keys[0]

                if selected_key:
                    logger.info(f"기존 API 키를 선택했습니다. (ID: {selected_key['id']}, 유형: {selected_key.get('key_type', 'unknown')})")
                    return True

            logger.info("사용 가능한 API 키를 찾을 수 없습니다. 새로운 API 키 생성이 필요합니다.")
            return False

        except Exception as e:
            logger.info(f"API 키 확인 중 오류 발생: {e}")
            return False

    def search_offers(self, custom_query: Optional[str] = None) -> List[Dict[str, Any]]:
        """오퍼 검색 (JSON 형식으로 개선)"""
        logger.info("🔍 사용 가능한 인스턴스 검색 중...")

        # 기본 명령어 구성 - --raw 옵션으로 JSON 출력 받기
        cmd = ["vastai", "search", "offers", "--raw"]

        # 검색 쿼리 결정
        query = custom_query if custom_query else ""

        if query.strip():
            # 쿼리를 개별 파라미터로 분리해서 추가
            query_parts = query.strip().split()
            cmd.extend(query_parts)
            logger.info(f"🔍 검색 쿼리: {' '.join(query_parts)}")
        else:
            logger.info("🔍 모든 오퍼 검색 (필터 없음)")

        try:
            logger.debug(f"실행할 명령어: {' '.join(cmd)}")

            # 명령어 실행 - JSON 파싱 활성화
            result = self.run_command(cmd, parse_json=True, timeout=30)

            if not result["success"]:
                logger.error(f"검색 실행 실패: {result.get('error')}")
                return []

            # JSON 응답 처리
            data = result.get("data", [])
            if not data:
                logger.warning("검색 결과가 비어있습니다")
                return []

            # JSON 데이터를 직접 사용 (이미 올바른 형식)
            if isinstance(data, list):
                offers = []
                for offer in data:
                    # 필요한 필드들을 정규화
                    normalized_offer = {
                        "id": str(offer.get("id", "")),
                        "gpu_name": offer.get("gpu_name", "Unknown"),
                        "gpu_ram": float(offer.get("gpu_ram", 0)),
                        "dph_total": float(offer.get("dph_total", 0)),
                        "num_gpus": int(offer.get("num_gpus", 1)),
                        "rentable": offer.get("rentable", True),
                        "verified": offer.get("verified", False),
                        "public_ipaddr": offer.get("public_ipaddr"),
                        "reliability": offer.get("reliability", 0.0),
                        "score": offer.get("score", 0.0),
                        "geolocation": offer.get("geolocation", "Unknown"),
                        "cpu_cores": offer.get("cpu_cores", 1),
                        "cpu_name": offer.get("cpu_name", "Unknown"),
                        "ram": offer.get("cpu_ram", 1),
                        "disk_space": offer.get("disk_space", 10),
                        "inet_down": offer.get("inet_down", 100),
                        "inet_up": offer.get("inet_up", 100),
                        "cuda_max_good": offer.get("cuda_max_good", "11.0"),
                        "hostname": offer.get("hostname", "unknown-host")
                    }
                    offers.append(normalized_offer)

                logger.info(f"✅ 검색 성공: {len(offers)}개 인스턴스 발견")
                return offers
            else:
                logger.warning("예상치 못한 데이터 형식")
                return []

        except Exception as e:
            logger.error(f"검색 실행 중 오류: {e}")
            # 백업으로 텍스트 파싱 시도
            logger.info("백업 방식으로 텍스트 파싱 시도...")
            return self._fallback_search_offers(custom_query)

    def _parse_offers(self, data) -> List[Dict[str, Any]]:
        """오퍼 데이터 파싱"""
        if isinstance(data, list):
            return data
        elif isinstance(data, str):
            # 텍스트 파싱 시도
            return self._parse_text_offers(data)
        return []

    def _parse_text_offers(self, text: str) -> List[Dict[str, Any]]:
        """텍스트 형태의 오퍼 파싱 (개선된 버전)"""
        offers = []
        lines = text.strip().split('\n')

        # 다양한 출력 형식 처리
        logger.debug(f"파싱할 텍스트: {text[:200]}...")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 헤더나 에러 메시지 건너뛰기
            if any(word in line.upper() for word in ['ID', 'GPU', 'ERROR', 'FAILED', 'COMMAND']):
                continue

            # 기본 파싱: ID 가격 GPU 정보 추출
            # 예시: "123456  RTX4090  24GB  $1.50/hr  Available"
            offer_data = self._extract_offer_info(line)
            if offer_data:
                offers.append(offer_data)

        # 파싱 결과가 없으면 더 간단한 방식 시도
        if not offers:
            logger.debug("기본 파싱 실패, 간단한 파싱 시도")
            offers = self._simple_parse_offers(text)

        return offers

    def _extract_offer_info(self, line: str) -> Optional[Dict[str, Any]]:
        """단일 라인에서 오퍼 정보 추출"""
        try:
            # 다양한 패턴 시도
            patterns = [
                # 패턴 1: ID GPU RAM 가격
                r'(\d+)\s+([A-Z0-9_]+)\s+(\d+(?:\.\d+)?)\s*GB?\s+\$(\d+\.?\d*)',
                # 패턴 2: ID 가격 (단순)
                r'(\d+).*?\$(\d+\.?\d*)',
                # 패턴 3: ID 정보들 가격
                r'(\d+)\s+.*?\$(\d+\.?\d*)'
            ]

            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    offer_id = groups[0]

                    if len(groups) >= 4:  # 풀 패턴
                        gpu_name = groups[1]
                        gpu_ram = float(groups[2])
                        price = float(groups[3])
                    else:  # 간단한 패턴
                        gpu_name = "Unknown"
                        gpu_ram = 0
                        price = float(groups[1])

                    return {
                        "id": offer_id,
                        "gpu_name": gpu_name,
                        "gpu_ram": gpu_ram,
                        "dph_total": price,
                        "num_gpus": 1,
                        "rentable": True,
                        "verified": False
                    }
        except (ValueError, IndexError) as e:
            logger.debug(f"라인 파싱 실패: {line[:50]}... - {e}")

        return None

    def _fallback_search_offers(self, custom_query: Optional[str] = None) -> List[Dict[str, Any]]:
        """백업용 텍스트 파싱 방식 검색"""
        logger.info("🔄 백업 방식으로 텍스트 파싱 시도 중...")

        # 기본 명령어 구성 - raw 옵션 없이
        cmd = ["vastai", "search", "offers"]

        # 검색 쿼리 결정
        query = custom_query if custom_query else ""

        if query.strip():
            query_parts = query.strip().split()
            cmd.extend(query_parts)

        try:
            # 명령어 실행 - JSON 파싱 비활성화로 raw 출력 받기
            result = self.run_command(cmd, parse_json=False, timeout=30)

            if not result["success"]:
                logger.error(f"백업 검색 실행 실패: {result.get('error')}")
                return []

            # 텍스트 응답을 직접 파싱
            raw_output = result.get("data", "")
            if not raw_output:
                logger.warning("백업 검색 결과가 비어있습니다")
                return []

            # 텍스트 기반 파싱
            offers = self._parse_text_offers(raw_output)

            if offers:
                logger.info(f"✅ 백업 검색 성공: {len(offers)}개 인스턴스 발견")
                return offers
            else:
                logger.warning("백업 파싱된 오퍼가 없습니다")
                return []

        except Exception as e:
            logger.error(f"백업 검색 실행 중 오류: {e}")
            return []

    def _simple_parse_offers(self, text: str) -> List[Dict[str, Any]]:
        """개선된 간단한 대안 파싱 (실제 가격 파싱 포함)"""
        offers = []

        # 숫자로 시작하는 라인 찾기
        lines = text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if re.match(r'^\d+', line):
                # 기본 정보 추출
                offer_data = {
                    "id": "",
                    "gpu_name": "Unknown",
                    "gpu_ram": 0,
                    "dph_total": 1.0,  # 기본값
                    "num_gpus": 1,
                    "rentable": True,
                    "verified": False
                }

                # ID 추출
                id_match = re.match(r'^(\d+)', line)
                if id_match:
                    offer_data["id"] = id_match.group(1)

                # 가격 추출 - 다양한 패턴 시도
                price_patterns = [
                    r'\$(\d+\.?\d*)',  # $1.50 형태
                    r'(\d+\.?\d*)\s*\$/h',  # 1.50 $/h 형태
                    r'(\d+\.?\d*)\s*USD',  # 1.50 USD 형태
                    r'(\d+\.?\d*)\s*dph',  # 1.50 dph 형태
                ]

                for pattern in price_patterns:
                    price_match = re.search(pattern, line, re.IGNORECASE)
                    if price_match:
                        try:
                            price = float(price_match.group(1))
                            offer_data["dph_total"] = price
                            break
                        except ValueError:
                            continue

                # GPU 이름 추출
                gpu_patterns = [
                    r'(RTX\s*\d+\w*)',  # RTX4090, RTX 3090 등
                    r'(GTX\s*\d+\w*)',  # GTX1080 등
                    r'(Tesla\s*\w+)',   # Tesla V100 등
                    r'(A\d+\w*)',       # A100, A6000 등
                    r'(V\d+\w*)',       # V100 등
                ]

                for pattern in gpu_patterns:
                    gpu_match = re.search(pattern, line, re.IGNORECASE)
                    if gpu_match:
                        offer_data["gpu_name"] = gpu_match.group(1).replace(' ', '')
                        break

                # GPU RAM 추출
                ram_patterns = [
                    r'(\d+)\s*GB',  # 24GB 형태
                    r'(\d+)\s*G',   # 24G 형태
                ]

                for pattern in ram_patterns:
                    ram_match = re.search(pattern, line, re.IGNORECASE)
                    if ram_match:
                        try:
                            ram = float(ram_match.group(1))
                            offer_data["gpu_ram"] = ram
                            break
                        except ValueError:
                            continue

                # GPU 개수 추출
                gpu_count_patterns = [
                    r'(\d+)x\s*' + offer_data["gpu_name"],  # 2x RTX4090
                    r'(\d+)\s*GPUs?',  # 2 GPU
                ]

                for pattern in gpu_count_patterns:
                    count_match = re.search(pattern, line, re.IGNORECASE)
                    if count_match:
                        try:
                            count = int(count_match.group(1))
                            offer_data["num_gpus"] = count
                            break
                        except ValueError:
                            continue

                offers.append(offer_data)

        if offers:
            logger.info(f"📋 간단 파싱으로 {len(offers)}개 오퍼 추출 완료")
            for i, offer in enumerate(offers[:3]):  # 처음 3개만 로그 출력
                logger.debug(f"  오퍼 {i+1}: ID={offer['id']}, GPU={offer['gpu_name']}, RAM={offer['gpu_ram']}GB, 가격=${offer['dph_total']}/h")

        return offers

    def select_offer(self, offers: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """오퍼 선택 (중간 가격대)"""
        if not offers:
            return None

        # 가격순 정렬
        sorted_offers = sorted(offers, key=lambda x: x.get("dph_total", 999))

        # 가격 필터링
        max_price = self.config.max_price()
        filtered_offers = [o for o in sorted_offers if o.get("dph_total", 999) <= max_price]

        if not filtered_offers:
            logger.warning(f"최대 가격 ${max_price}/h 이하의 오퍼가 없습니다")
            return None

        # 중간 가격대 선택
        mid_index = len(filtered_offers) // 2
        selected = filtered_offers[mid_index]

        logger.info(f"선택된 오퍼: ID={selected.get('id')}, 가격=${selected.get('dph_total')}/h")
        return selected

    def create_instance(self, offer_id: str) -> Optional[str]:
        logger.info(f"📦 인스턴스를 생성 중... (Offer ID: {offer_id})")

        image_name = self.config.image_name()
        disk_size = self.config.disk_size()
        default_ports = self.config.default_ports()

        # 기본 명령어 구성
        cmd = ["vastai", "create", "instance", str(offer_id)]
        cmd.extend(["--image", image_name])
        cmd.extend(["--disk", str(disk_size)])

        # 포트 설정 (간소화된 버전)
        ports_to_expose = sorted(default_ports)
        env_params = []

        for port in ports_to_expose:
            env_params.append(f"-p {port}:{port}")

        # 환경변수 설정
        vllm_host = self.config.vllm_host_ip()
        vllm_port = self.config.vllm_port()
        vllm_controller_port = self.config.vllm_controller_port()

        env_params.extend([
            "-e OPEN_BUTTON_PORT=1111",
            "-e OPEN_BUTTON_TOKEN=1",
            "-e JUPYTER_DIR=/",
            "-e DATA_DIRECTORY=/vllm/",
            f"-e PORTAL_CONFIG=\"localhost:1111:11111:/:Instance Portal|localhost:8080:18080:/:Jupyter|localhost:8080:8080:/terminals/1:Jupyter Terminal|localhost:8384:18384:/:Syncthing|localhost:6006:16006:/:Tensorboard\"",
            "-e NVIDIA_VISIBLE_DEVICES=all",
            f"-e VLLM_MODEL_NAME={self.config.vllm_model_name()}",
            f"-e VLLM_PORT={vllm_port}",
            f"-e VLLM_HOST_IP={vllm_host}",
            f"-e VLLM_CONTROLLER_PORT={vllm_controller_port}",
            f"-e VLLM_MAX_MODEL_LEN={self.config.vllm_max_model_len()}",
            f"-e VLLM_GPU_MEMORY_UTILIZATION={self.config.vllm_gpu_memory_utilization()}",
            f"-e VLLM_PIPELINE_PARALLEL_SIZE={self.config.vllm_pipeline_parallel_size()}",
            f"-e VLLM_TENSOR_PARALLEL_SIZE={self.config.vllm_tensor_parallel_size()}",
            f"-e VLLM_DTYPE={self.config.vllm_dtype()}",
            f"-e VLLM_TOOL_CALL_PARSER={self.config.vllm_tool_call_parser()}",
        ])

        # 환경 변수 문자열로 결합
        env_string = " ".join(env_params).strip()
        cmd.extend(["--env", env_string])

        # onstart 명령어
        onstart_cmd = "/vllm/entrypoint.sh"
        cmd.extend(["--onstart-cmd", onstart_cmd])

        # 기본 옵션들
        cmd.append("--jupyter")
        cmd.append("--ssh")
        cmd.append("--direct")

        logger.info(f"실행할 명령어: {' '.join(cmd)}")

        try:
            result = self.run_command(cmd, parse_json=False)
            if result["success"]:
                output = result["data"]
                instance_id = self._extract_instance_id_from_output(output)

                if instance_id:
                    logger.info(f"✅ 인스턴스 생성 성공: ID = {instance_id}")
                    return instance_id
                else:
                    logger.warning("⚠️ 인스턴스 ID를 찾을 수 없습니다.")
                    logger.info(f"CLI 출력: {output}")
            else:
                logger.error(f"❌ 인스턴스 생성 실패: {result.get('error')}")

        except Exception as e:
            logger.error(f"❌ 인스턴스 생성 중 오류: {e}")

        return None

    def _extract_instance_id_from_output(self, output: str) -> Optional[str]:
        """CLI 출력에서 인스턴스 ID 추출"""
        if isinstance(output, dict):
            # 딕셔너리 응답에서 ID 추출
            for key in ("new_instance_id", "instance_id", "id", "InstanceID", "created_instance_id"):
                val = output.get(key)
                if val is not None and str(val).isdigit():
                    return str(val)

        if isinstance(output, str):
            # 문자열 응답에서 패턴 매칭으로 ID 추출
            patterns = [
                r"Created instance (\d+)",
                r"instance[_\s]*id[\"':\s]*(\d+)",
                r"new[_\s]*instance[_\s]*id[\"':\s]*(\d+)",
                r"id[\"':\s]*(\d+)"
            ]

            for pattern in patterns:
                match = re.search(pattern, output, re.IGNORECASE)
                if match:
                    return match.group(1)

            # 6자리 이상의 숫자 ID 찾기
            numeric_ids = re.findall(r"\b\d{6,}\b", output)
            if numeric_ids:
                return max(numeric_ids, key=int)

        return None

    def create_instance_fallback(self, offer_id: str) -> Optional[str]:
        instance_id = self.create_instance(offer_id)

        if not instance_id:
            # 간단한 버전으로 재시도
            logger.info("기본 인스턴스 생성 실패, 간단한 버전으로 재시도")

            # 필수 환경변수만 포함한 간단한 버전
            vllm_host = self.config.vllm_host_ip()
            vllm_port = self.config.vllm_port()
            vllm_controller_port = self.config.vllm_controller_port()

            env_params = [
                f"-e VLLM_HOST_IP={vllm_host}",
                f"-e VLLM_PORT={vllm_port}",
                f"-e VLLM_CONTROLLER_PORT={vllm_controller_port}",
                f"-e VLLM_MODEL_NAME={self.config.vllm_model_name()}",
                f"-e VLLM_MAX_MODEL_LEN={self.config.vllm_max_model_len()}",
                f"-e VLLM_GPU_MEMORY_UTILIZATION={self.config.vllm_gpu_memory_utilization()}",
                f"-e VLLM_PIPELINE_PARALLEL_SIZE={self.config.vllm_pipeline_parallel_size()}",
                f"-e VLLM_TENSOR_PARALLEL_SIZE={self.config.vllm_tensor_parallel_size()}",
                f"-e VLLM_DTYPE={self.config.vllm_dtype()}",
                f"-e VLLM_TOOL_CALL_PARSER={self.config.vllm_tool_call_parser()}",
            ]
            env_string = " ".join(env_params).strip()

            cmd = [
                "vastai", "create", "instance",
                str(offer_id),
                "--image", self.config.image_name(),
                "--disk", str(self.config.disk_size()),
                "--env", env_string
            ]

            result = self.run_command(cmd, parse_json=False)

            if result["success"]:
                output = result["data"]
                match = re.search(r'(\d+)', output)
                if match:
                    instance_id = match.group(1)
                    logger.info(f"fallback 인스턴스 생성 성공: {instance_id}")
                    return instance_id

        return instance_id

    def wait_for_running(self, instance_id: str, max_wait: int = 1500) -> bool:
        """인스턴스 실행 상태 대기"""
        logger.info(f"인스턴스 {instance_id} 실행 대기 중...")

        start_time = time.time()

        while time.time() - start_time < max_wait:
            status = self.get_instance_status(instance_id)

            if status == "running":
                logger.info(f"인스턴스 {instance_id} 실행 중")
                return True
            elif status == "failed" or status == "destroyed" or status == "deleted":
                logger.error(f"인스턴스 {instance_id} 실행 실패")
                return False

            logger.info(f"현재 상태: {status}, 대기 중...")
            time.sleep(10)

        logger.error(f"인스턴스 {instance_id} 실행 대기 타임아웃")
        return False

    def wait_for_vllm_running(self, controller_url: str, max_wait: int = 1500) -> bool:
        """인스턴스 실행 상태 대기"""
        logger.info(f"인스턴스 {controller_url} 실행 대기 중...")

        start_time = time.time()

        while time.time() - start_time < max_wait:
            status = self.get_vllm_instance_status(controller_url)

            if status == "running":
                logger.info(f"인스턴스 {controller_url} 실행 중")
                return True
            elif status == "failed" or status == "destroyed" or status == "deleted":
                logger.error(f"인스턴스 {controller_url} 실행 실패")
                return False

            logger.info(f"현재 상태: {status}, 대기 중...")
            time.sleep(10)

        logger.error(f"인스턴스 {controller_url} 실행 대기 타임아웃")
        return False

    def get_instance_status(self, instance_id: str) -> str:
        """인스턴스 상태 확인"""
        # 3단계 파싱 시도
        strategies = [
            ("raw", ["vastai", "show", "instance", instance_id, "--raw"]),
        ]

        for strategy_name, cmd in strategies:
            try:
                result = self.run_command(cmd, parse_json=True)

                if result["success"] and result["data"]:
                    status = self._extract_status(result["data"], instance_id, strategy_name)
                    if status:
                        return status

                elif result["success"] == False:
                    error_msg = result.get('error', '').lower()
                    # VastAI CLI에서 삭제된 인스턴스 조회 시 발생하는 특정 오류들 처리
                    if any(err in error_msg for err in ['nonetype', 'subscriptable', 'not found']):
                        logger.debug(f"인스턴스 {instance_id}가 삭제된 것으로 보임: {error_msg}")
                        return "destroyed"
                    logger.error(f"상태 조회 실패 ({strategy_name}): {result.get('error')}")
                    return "failed"
            except Exception as e:
                error_msg = str(e).lower()
                # VastAI CLI 내부 오류 시 삭제된 것으로 간주
                if any(err in error_msg for err in ['nonetype', 'subscriptable', 'not found']):
                    logger.debug(f"인스턴스 {instance_id}가 삭제된 것으로 보임 (예외): {e}")
                    return "destroyed"
                logger.debug(f"상태 조회 오류 ({strategy_name}): {e}")
                return "failed"

        return "unknown"

    def get_vllm_instance_status(self, controller_url: str) -> str:
        """VLLM 인스턴스 상태 확인"""
        import urllib.request
        from urllib.request import Request as UrlRequest

        # 10초 타임아웃으로 요청
        try:
            req = UrlRequest(controller_url, headers={'Content-Type': 'application/json'})
            response = urllib.request.urlopen(req, timeout=10)

            if response.getcode() == 200:
                return "running"
        except:
            return "unknown"

    def _extract_status(self, data, instance_id: str, strategy: str) -> Optional[str]:
        """데이터에서 상태 추출"""
        try:
            if strategy == "raw":
                # --raw 옵션은 단일 딕셔너리를 반환
                if isinstance(data, dict):
                    if str(data.get("id")) == str(instance_id):
                        return data.get("actual_status", "unknown")
                # 리스트일 수도 있음 (예외 케이스)
                elif isinstance(data, list):
                    for instance in data:
                        if str(instance.get("id")) == str(instance_id):
                            return instance.get("actual_status", "unknown")

            elif strategy == "json" and isinstance(data, dict):
                return data.get("actual_status", "unknown")

            elif strategy == "list" and isinstance(data, list):
                for instance in data:
                    if str(instance.get("id")) == str(instance_id):
                        return instance.get("actual_status", "unknown")

        except Exception as e:
            logger.debug(f"상태 추출 오류 ({strategy}): {e}")

        return None

    def get_instance_info(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """인스턴스 상세 정보 조회"""
        result = self.run_command(["vastai", "show", "instance", instance_id, "--raw"], parse_json=True)

        if result["success"] and result["data"]:
            data = result["data"]

            if isinstance(data, dict):
                return data

            elif isinstance(data, str):
                logger.debug(f"인스턴스 정보가 텍스트 형식으로 반환됨: {data[:100]}...")
                # 텍스트에서 기본 정보 추출 시도
                parsed_info = self._parse_instance_info_from_text(data, instance_id)
                return parsed_info if parsed_info else None

        return None

    def _parse_instance_info_from_text(self, text: str, instance_id: str) -> Optional[Dict[str, Any]]:
        """텍스트 형식의 인스턴스 정보에서 딕셔너리 추출"""
        try:
            lines = text.strip().split('\n')

            # 인스턴스 ID가 포함된 라인 찾기
            instance_line = None
            for line in lines:
                if str(instance_id) in line:
                    instance_line = line
                    break

            if not instance_line:
                logger.warning(f"인스턴스 ID {instance_id}가 포함된 라인을 찾을 수 없음")
                return None

            # 라인을 공백으로 분리하여 파싱
            parts = instance_line.split()

            if len(parts) < 8:
                logger.warning(f"인스턴스 정보 파싱 실패: 충분한 필드가 없음")
                return None

            # 기본 정보 구조 생성
            info = {
                "id": instance_id,
                "actual_status": parts[2] if len(parts) > 2 else "unknown",
                "gpu_name": parts[4] if len(parts) > 4 else "Unknown",
                "num_gpus": 1,  # 기본값
                "cpu_cores": float(parts[5]) if len(parts) > 5 and parts[5].replace('.', '').isdigit() else 0,
                "cpu_ram": float(parts[6]) if len(parts) > 6 and parts[6].replace('.', '').isdigit() else 0,
                "dph_total": float(parts[9]) if len(parts) > 9 and parts[9].replace('.', '').isdigit() else 0,
            }

            # SSH 정보 추출
            if len(parts) > 8:
                ssh_addr = parts[7]
                ssh_port = parts[8]

                if ssh_addr and ssh_port.isdigit():
                    info["ssh_host"] = ssh_addr
                    info["ssh_port"] = int(ssh_port)
                    info["public_ipaddr"] = ssh_addr  # SSH 주소를 공인 IP로 사용

            # GPU 개수 파싱 (예: "2x" 형태)
            if len(parts) > 3:
                gpu_part = parts[3]
                if 'x' in gpu_part:
                    try:
                        num_gpus = int(gpu_part.split('x')[0])
                        info["num_gpus"] = num_gpus
                    except ValueError:
                        pass

            logger.debug(f"텍스트에서 파싱된 인스턴스 정보: {info}")
            return info

        except Exception as e:
            logger.warning(f"텍스트 인스턴스 정보 파싱 실패: {e}")
            return None

    def _parse_instances_from_text(self, text: str) -> List[Dict[str, Any]]:
        """vastai show instances 텍스트 출력에서 인스턴스 목록 파싱"""
        instances = []

        try:
            lines = text.strip().split('\n')

            # 헤더 라인 찾기 (ID로 시작하는 라인)
            header_line_idx = -1
            for i, line in enumerate(lines):
                if line.strip().startswith('ID') and 'Status' in line:
                    header_line_idx = i
                    break

            if header_line_idx == -1:
                logger.warning("인스턴스 목록에서 헤더를 찾을 수 없음")
                return instances

            # 헤더 다음 라인들 파싱
            for line in lines[header_line_idx + 1:]:
                line = line.strip()
                if not line:
                    continue

                # 라인을 공백으로 분리
                parts = line.split()

                if len(parts) < 8:  # 최소 필요한 필드 수
                    continue

                try:
                    instance_info = {
                        "id": parts[0],
                        "machine": parts[1] if len(parts) > 1 else "",
                        "actual_status": parts[2] if len(parts) > 2 else "unknown",
                        "num_gpus": parts[3] if len(parts) > 3 else "1x",
                        "gpu_name": parts[4] if len(parts) > 4 else "Unknown",
                        "util": float(parts[5]) if len(parts) > 5 and parts[5].replace('.', '').isdigit() else 0.0,
                        "cpu_cores": float(parts[6]) if len(parts) > 6 and parts[6].replace('.', '').isdigit() else 0,
                        "cpu_ram": float(parts[7]) if len(parts) > 7 and parts[7].replace('.', '').isdigit() else 0,
                        "storage": int(parts[8]) if len(parts) > 8 and parts[8].isdigit() else 0,
                        "ssh_host": parts[9] if len(parts) > 9 else "",
                        "ssh_port": int(parts[10]) if len(parts) > 10 and parts[10].isdigit() else 22,
                        "dph_total": float(parts[11]) if len(parts) > 11 and parts[11].replace('.', '').isdigit() else 0.0,
                    }

                    # GPU 개수 파싱 (예: "2x" -> 2)
                    if 'x' in instance_info["num_gpus"]:
                        try:
                            gpu_count = int(instance_info["num_gpus"].split('x')[0])
                            instance_info["gpu_count"] = gpu_count
                        except ValueError:
                            instance_info["gpu_count"] = 1
                    else:
                        instance_info["gpu_count"] = 1

                    instances.append(instance_info)

                except (ValueError, IndexError) as e:
                    logger.debug(f"인스턴스 라인 파싱 실패: {line[:50]}... - {e}")
                    continue

            logger.info(f"텍스트에서 {len(instances)}개 인스턴스 파싱 완료")

        except Exception as e:
            logger.error(f"인스턴스 텍스트 파싱 중 오류: {e}")

        return instances

    def _execute_stream_command(self, cmd: List[str]) -> Dict[str, Any]:
        """스트리밍 명령 실행"""
        # vastai 명령어인 경우 먼저 prefix와 함께 시도
        if cmd and cmd[0] == "vastai":
            prefixed_cmd = self.vastai_prefix + cmd

            # prefix와 함께 실행 시도
            result = self._execute_stream_command_internal(prefixed_cmd)

            # prefix와 함께 실행이 실패한 경우 prefix 없이 재시도
            if not result["success"]:
                logger.info(f"🔄 prefix 스트림 실행 실패, fallback으로 재시도: {' '.join(cmd)}")
                result = self._execute_stream_command_internal(cmd)

            return result
        else:
            return self._execute_stream_command_internal(cmd)

    def _execute_stream_command_internal(self, cmd: List[str]) -> Dict[str, Any]:
        """실제 스트리밍 명령 실행"""
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            stdout, stderr = process.communicate()

            return {
                "success": process.returncode == 0,
                "stdout": stdout,
                "stderr": stderr
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_port_mappings(self, instance_id: str) -> Dict[str, Any]:
        logger.info(f"🌐 포트 매핑 정보 조회 시작 (인스턴스 ID: {instance_id})")
        try:
            raw_mapping = self._get_port_mappings_from_raw_info(instance_id)
            if raw_mapping:
                logger.info(f"✅ --raw 방식으로 {len(raw_mapping)}개 포트 매핑 성공")
                return {"mappings": raw_mapping, "public_ip": self._extract_public_ip_from_mappings(raw_mapping)}
        except Exception as e:
            logger.warning(f"--raw 방식 실패: {e}")

        logger.error("❌ 포트 매핑 정보를 가져올 수 없습니다.")
        return {"mappings": {}, "public_ip": None}

    def _extract_public_ip_from_mappings(self, mappings: Dict[int, Tuple[str, int]]) -> Optional[str]:
        """매핑에서 공인 IP 추출"""
        for port, (ip, external_port) in mappings.items():
            if ip and ip != "0.0.0.0" and not ip.startswith("127."):
                return ip
        return None

    def _extract_public_ip_from_instance_info(self, instance_info: Dict[str, Any]) -> str:
        """인스턴스 정보에서 공인 IP 추출 (여러 필드명 시도)"""
        # 가능한 IP 필드들을 우선순위대로 확인
        ip_fields = [
            "public_ipaddr",     # 일반적인 VastAI 필드
            "external_ip",       # 외부 IP
            "host_ip",          # 호스트 IP
            "ssh_host",         # SSH 호스트
            "ip",               # 기본 IP
            "ipaddr",           # IP 주소
            "external_ipaddr",  # 외부 IP 주소
            "public_ip",        # 공개 IP
        ]

        for field in ip_fields:
            ip = instance_info.get(field)
            if ip and self._is_valid_public_ip(ip):
                logger.info(f"✅ {field} 필드에서 유효한 공인 IP 발견: {ip}")
                return ip

        # SSH URL에서 IP 추출 시도
        ssh_url = instance_info.get("ssh_url", "")
        if ssh_url:
            ip = self._extract_ip_from_ssh_url(ssh_url)
            if ip and self._is_valid_public_ip(ip):
                logger.info(f"✅ SSH URL에서 유효한 공인 IP 발견: {ip}")
                return ip

        # 포트 매핑에서 IP 추출 시도
        ports = instance_info.get("ports", {})
        if ports:
            for port_key, port_bindings in ports.items():
                if isinstance(port_bindings, list) and port_bindings:
                    binding = port_bindings[0]
                    if isinstance(binding, dict):
                        host_ip = binding.get("HostIp", "")
                        if host_ip and self._is_valid_public_ip(host_ip):
                            logger.info(f"✅ 포트 바인딩에서 유효한 공인 IP 발견: {host_ip}")
                            return host_ip

        logger.warning("⚠️ 유효한 공인 IP를 찾을 수 없습니다.")
        return "unknown"

    def _is_valid_public_ip(self, ip: str) -> bool:
        """유효한 공인 IP 주소인지 확인"""
        if not ip or not isinstance(ip, str):
            return False

        # 기본 유효성 검사
        if ip in ["0.0.0.0", "127.0.0.1", "localhost", "unknown", ""]:
            return False

        # 로컬 IP 대역 제외
        if ip.startswith(("127.", "10.", "172.", "192.168.", "169.254.")):
            return False

        # 기본 IP 형식 검사 (간단한 정규식)
        import re
        ip_pattern = r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
        return bool(re.match(ip_pattern, ip))

    def _extract_ip_from_ssh_url(self, ssh_url: str) -> Optional[str]:
        """SSH URL에서 IP 주소 추출"""
        import re

        # ssh://root@1.2.3.4:22 형태에서 IP 추출
        match = re.search(r"ssh://[^@]*@([^:]+)", ssh_url)
        if match:
            return match.group(1)

        # root@1.2.3.4 형태에서 IP 추출
        match = re.search(r"@([^:]+)", ssh_url)
        if match:
            return match.group(1)

        return None

    def _get_port_mappings_from_raw_info(self, instance_id: str) -> Dict[int, Tuple[str, int]]:
        try:
            # get_instance_info를 통해 --raw 정보 가져오기
            raw_info = self.get_instance_info(instance_id)
            if not raw_info or not isinstance(raw_info, dict):
                logger.warning("❌ 인스턴스 정보를 가져올 수 없습니다.")
                return {}

            logger.info(f"🔍 Raw info keys: {list(raw_info.keys())}")

            mapping: Dict[int, Tuple[str, int]] = {}

            # 공인 IP 추출 - 여러 필드명 시도
            public_ip = self._extract_public_ip_from_instance_info(raw_info)
            logger.info(f"🔍 추출된 공인 IP: {public_ip}")

            # 1. ports 필드에서 포트 매핑 정보 추출
            ports_data = raw_info.get("ports", {})

            if isinstance(ports_data, dict) and ports_data:
                logger.info(f"📊 포트 데이터 발견: {ports_data}")

                for port_key, port_bindings in ports_data.items():
                    try:
                        # 포트 키에서 컨테이너 포트 추출 ("1111/tcp" -> 1111)
                        container_port = int(port_key.split('/')[0])

                        # 포트 바인딩 정보 처리
                        if isinstance(port_bindings, list) and len(port_bindings) > 0:
                            # [{"HostIp": "0.0.0.0", "HostPort": "11346"}] 형태
                            first_binding = port_bindings[0]
                            if isinstance(first_binding, dict):
                                host_port = int(first_binding.get("HostPort", "0"))
                                host_ip = first_binding.get("HostIp", "0.0.0.0")

                                # 실제 공인 IP 사용 (개선된 로직)
                                if public_ip != "unknown" and self._is_valid_public_ip(public_ip):
                                    external_ip = public_ip
                                elif self._is_valid_public_ip(host_ip):
                                    external_ip = host_ip
                                    logger.info(f"   📌 HostIp를 외부 IP로 사용: {host_ip}")
                                else:
                                    external_ip = public_ip  # 최후의 수단
                                    logger.warning(f"   ⚠️ 유효한 공인 IP를 찾을 수 없음, {public_ip} 사용")

                                if container_port > 0 and host_port > 0:
                                    mapping[container_port] = (external_ip, host_port)
                                    logger.info(f"   ✅ 매핑 추가: {container_port} -> {external_ip}:{host_port}")

                        elif isinstance(port_bindings, str):
                            # "149.7.4.12:18773" 형태
                            if ":" in port_bindings:
                                ip, port = port_bindings.split(":")
                                mapping[container_port] = (ip, int(port))
                                logger.info(f"   ✅ 매핑 추가: {container_port} -> {ip}:{port}")

                    except (ValueError, TypeError, KeyError) as e:
                        logger.debug(f"포트 정보 파싱 실패: {port_key}={port_bindings}, 에러: {e}")
                        continue

            # 2. 다른 포트 관련 필드들도 확인
            port_fields_to_check = [
                "port_bindings", "port_map", "port_mappings", "exposed_ports"
            ]

            for field_name in port_fields_to_check:
                if field_name in raw_info and not mapping:
                    field_data = raw_info[field_name]
                    logger.info(f"🔍 {field_name} 필드 확인: {field_data}")

                    if isinstance(field_data, dict):
                        for key, value in field_data.items():
                            try:
                                container_port = int(key.split('/')[0]) if '/' in str(key) else int(key)

                                if isinstance(value, list) and len(value) > 0:
                                    binding = value[0]
                                    if isinstance(binding, dict):
                                        host_port = int(binding.get("HostPort", "0"))
                                        external_ip = public_ip if public_ip != "unknown" else "0.0.0.0"

                                        if container_port > 0 and host_port > 0:
                                            mapping[container_port] = (external_ip, host_port)
                                            logger.info(f"   ✅ {field_name}에서 매핑 추가: {container_port} -> {external_ip}:{host_port}")

                            except (ValueError, TypeError, KeyError) as e:
                                logger.debug(f"{field_name} 파싱 실패: {key}={value}, 에러: {e}")
                                continue

            # 3. 포트 매핑이 없는 경우 SSH 포트라도 추출 시도
            if not mapping:
                ssh_host = raw_info.get("ssh_host")
                ssh_port = raw_info.get("ssh_port", 22)

                if ssh_host and ssh_port:
                    mapping[22] = (ssh_host, int(ssh_port))
                    logger.info(f"   ✅ SSH 포트 매핑 추가: 22 -> {ssh_host}:{ssh_port}")

            if mapping:
                logger.info(f"✅ --raw 방식으로 {len(mapping)}개 포트 매핑 성공")
                return mapping
            else:
                logger.warning("⚠️ --raw 방식으로 포트 매핑을 찾을 수 없음")
                return {}

        except Exception as e:
            logger.warning(f"❌ --raw 방식 포트 매핑 수집 실패: {e}")
            return {}

    def destroy_instance(self, instance_id: str) -> bool:
        """인스턴스 삭제"""
        logger.info(f"인스턴스 {instance_id} 삭제 중...")

        result = self.run_command(["vastai", "destroy", "instance", instance_id], parse_json=False)

        if result["success"]:
            # 삭제 확인 - VastAI CLI 오류를 방지하기 위해 안전하게 처리
            time.sleep(5)
            try:
                status = self.get_instance_status(instance_id)
                if status in ["destroyed", "unknown", "failed"]:
                    logger.info(f"인스턴스 {instance_id} 삭제 완료 (상태: {status})")
                    return True
                else:
                    logger.warning(f"인스턴스 {instance_id} 삭제 확인 실패, 현재 상태: {status}")
                    return False
            except Exception as e:
                # VastAI CLI 오류 발생 시 (삭제된 인스턴스 조회 시 발생 가능)
                logger.info(f"인스턴스 {instance_id} 상태 조회 실패 (삭제 완료로 간주): {e}")
                return True

        logger.error(f"인스턴스 {instance_id} 삭제 실패: {result.get('error')}")
        return False

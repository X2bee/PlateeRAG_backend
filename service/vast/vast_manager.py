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
                        "dph_total": float(offer.get("dph_total", 0)),  # dph_total 필드를 직접 사용
                        "num_gpus": int(offer.get("num_gpus", 1)),
                        "rentable": offer.get("rentable", True),
                        "verified": offer.get("verified", False),
                        "public_ipaddr": offer.get("public_ipaddr"),
                        "reliability": offer.get("reliability", 0.0),
                        "score": offer.get("score", 0.0),
                        "geolocation": offer.get("geolocation", "Unknown"),
                        "cpu_cores": offer.get("cpu_cores", 1),
                        "ram": offer.get("cpu_ram", 1),
                        "disk_space": offer.get("disk_space", 10),
                        "inet_down": offer.get("inet_down", 100),
                        "inet_up": offer.get("inet_up", 100),
                        "cuda_max_good": offer.get("cuda_vers", "11.0"),
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

    def _normalize_offer(self, offer: Dict[str, Any]) -> Dict[str, Any]:
        """오퍼 데이터 정규화"""
        normalized = {
            'id': offer.get('id'),
            'gpu_name': offer.get('gpu_name', 'Unknown'),
            'gpu_ram': offer.get('gpu_ram', 0),
            'dph_total': float(offer.get('dph_total', 999.0)),
            'num_gpus': offer.get('num_gpus', 1),
            'verified': offer.get('verified', False),
            'rentable': offer.get('rentable', True),
            'cuda_max_good': offer.get('cuda_max_good', '11.0'),
            'cpu_cores': offer.get('cpu_cores', 1),
            'ram': offer.get('ram', 1),
            'disk_space': offer.get('disk_space', 10),
            'inet_down': offer.get('inet_down', 100),
            'inet_up': offer.get('inet_up', 100),
            'score': offer.get('score', 0.0),
            'reliability': offer.get('reliability', 0.0),
            'geolocation': offer.get('geolocation', 'Unknown'),
            'hostname': offer.get('hostname', 'unknown-host'),
        }

        # 추가 필드들도 보존
        for key, value in offer.items():
            if key not in normalized:
                normalized[key] = value

        return normalized

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
        """인스턴스 생성 (개선된 버전)"""
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
            "-e DATA_DIRECTORY=/workspace/",
            f"-e PORTAL_CONFIG=\"localhost:1111:11111:/:Instance Portal|localhost:8080:18080:/:Jupyter|localhost:8080:8080:/terminals/1:Jupyter Terminal|localhost:8384:18384:/:Syncthing|localhost:6006:16006:/:Tensorboard\"",
            "-e NVIDIA_VISIBLE_DEVICES=all",
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
        ])

        # 환경 변수 문자열로 결합
        env_string = " ".join(env_params).strip()
        cmd.extend(["--env", env_string])

        # onstart 명령어
        onstart_cmd = self.config.generate_onstart_command()
        cmd.extend(["--onstart-cmd", onstart_cmd])

        # 기본 옵션들
        cmd.append("--jupyter")
        cmd.append("--ssh")
        cmd.append("--direct")

        logger.debug(f"실행할 명령어: {' '.join(cmd)}")

        try:
            result = self.run_command(cmd, parse_json=False)

            if result["success"]:
                # 인스턴스 ID 추출
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
        """인스턴스 생성 (fallback 전략)"""
        # 기본 생성 시도
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

    def wait_for_running(self, instance_id: str, max_wait: int = 300) -> bool:
        """인스턴스 실행 상태 대기"""
        logger.info(f"인스턴스 {instance_id} 실행 대기 중...")

        start_time = time.time()

        while time.time() - start_time < max_wait:
            status = self.get_instance_status(instance_id)

            if status == "running":
                logger.info(f"인스턴스 {instance_id} 실행 중")
                return True
            elif status == "failed":
                logger.error(f"인스턴스 {instance_id} 실행 실패")
                return False

            logger.info(f"현재 상태: {status}, 대기 중...")
            time.sleep(10)

        logger.error(f"인스턴스 {instance_id} 실행 대기 타임아웃")
        return False

    def get_instance_status(self, instance_id: str) -> str:
        """인스턴스 상태 확인"""
        # 3단계 파싱 시도
        strategies = [
            ("raw", ["vastai", "show", "instance", "--raw"]),
            ("json", ["vastai", "show", "instance", instance_id]),
            ("list", ["vastai", "show", "instances"])
        ]

        for strategy_name, cmd in strategies:
            result = self.run_command(cmd, parse_json=True)

            if result["success"] and result["data"]:
                status = self._extract_status(result["data"], instance_id, strategy_name)
                if status:
                    return status

        return "unknown"

    def _extract_status(self, data, instance_id: str, strategy: str) -> Optional[str]:
        """데이터에서 상태 추출"""
        try:
            if strategy == "raw" and isinstance(data, list):
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
        result = self.run_command(["vastai", "show", "instance", instance_id], parse_json=True)

        if result["success"] and result["data"]:
            return result["data"]

        return None

    def execute_ssh_command(self, instance_id: str, command: str, stream: bool = False) -> Dict[str, Any]:
        """인스턴스에서 명령어 실행 (개선된 SSH 실행)"""
        logger.info(f"🔧 명령어 실행 중(SSH): {command[:80]}...")

        try:
            ssh_info = self.get_ssh_info(instance_id)
            user, host, port, key_path = self._parse_ssh_url(ssh_info)

            if not all([user, host]):
                return {"success": False, "error": "SSH URL 파싱 실패, 명령어 실행 불가"}

            ssh_base = [
                "ssh",
                "-p", str(port),
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "ConnectTimeout=30",
            ]

            if key_path:
                ssh_base.extend(["-i", key_path])

            ssh_base.append(f"{user}@{host}")
            ssh_base.append(command)

            if stream:
                return self._execute_stream_command(ssh_base)
            else:
                result = self.run_command(ssh_base, parse_json=False)
                if result["success"]:
                    logger.debug("✅ 명령어 실행 완료(SSH)")
                    return {"success": True, "stdout": result["data"], "stderr": ""}
                else:
                    return {"success": False, "error": result["error"], "stdout": "", "stderr": result["error"]}

        except Exception as e:
            # SSH 연결 실패 구분하여 처리
            error_msg = str(e).lower()
            if "connect failed" in error_msg or "connection refused" in error_msg:
                logger.debug(f"SSH connection failed for instance {instance_id}: {e}")
                return {"success": False, "error": f"SSH connection failed - instance may not be ready: {e}"}
            else:
                logger.error(f"❌ SSH 명령어 실행 실패: {e}")
                return {"success": False, "error": str(e)}

    def _parse_ssh_url(self, ssh_cmd: str) -> Tuple[Optional[str], Optional[str], int, Optional[str]]:
        """ssh-url 명령 결과에서 (user, host, port, key_path) 추출"""
        import shlex
        import urllib.parse

        if ssh_cmd.startswith("ssh://"):
            parsed = urllib.parse.urlparse(ssh_cmd)
            user = parsed.username
            host = parsed.hostname
            port = parsed.port or 22
            key_path = None
            return user, host, port, key_path

        parts = shlex.split(ssh_cmd)
        user = host = key_path = None
        port = 22
        i = 0

        while i < len(parts):
            token = parts[i]
            if token == "ssh":
                i += 1
                continue
            if token in ("-p", "--port") and i + 1 < len(parts):
                port = int(parts[i + 1])
                i += 2
                continue
            if token == "-i" and i + 1 < len(parts):
                key_path = parts[i + 1]
                i += 2
                continue

            # user@host 패턴 찾기
            m = re.match(r"([^@]+)@([\w.\-]+)(?::(\d+))?", token)
            if m:
                user = m.group(1)
                host = m.group(2)
                if m.group(3):
                    port = int(m.group(3))
            i += 1

        return user, host, port, key_path

    def get_ssh_info(self, instance_id: str) -> str:
        """SSH 연결 정보 조회"""
        cmd = ["vastai", "ssh-url", str(instance_id)]

        try:
            result = self.run_command(cmd, parse_json=False)
            if result["success"]:
                ssh_url = result["data"].strip()
                logger.info(f"🔐 SSH 연결 정보: {ssh_url}")
                return ssh_url
            else:
                logger.error(f"❌ SSH 정보 조회 실패: {result['error']}")
                raise RuntimeError(f"SSH 정보 조회 실패: {result['error']}")
        except Exception as e:
            logger.error(f"❌ SSH 정보 조회 실패: {e}")
            raise

    def _execute_stream_command(self, cmd: List[str]) -> Dict[str, Any]:
        """스트리밍 명령 실행"""
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
        """인스턴스의 {컨테이너 포트: (외부 IP, 외부 포트)} 매핑 반환"""
        logger.info(f"🌐 포트 매핑 정보 조회 시작 (인스턴스 ID: {instance_id})")

        # 1️⃣ 우선 방식: --raw 옵션을 활용한 정확한 정보 수집
        try:
            logger.debug("1차 시도: --raw 옵션을 활용한 포트 매핑 수집")
            raw_mapping = self._get_port_mappings_from_raw_info(instance_id)
            if raw_mapping:
                logger.info(f"✅ --raw 방식으로 {len(raw_mapping)}개 포트 매핑 성공")
                return {"mappings": raw_mapping, "public_ip": self._extract_public_ip_from_mappings(raw_mapping)}
            else:
                logger.info("⚠️ --raw 방식에서 포트 매핑을 찾지 못함, 다른 방법 시도")
        except Exception as e:
            logger.warning(f"--raw 방식 실패: {e}")

        # 2️⃣ 폴백 방식: vast show instances --raw 사용
        try:
            logger.debug("시도: vast show instances --raw 방식")
            instances_mapping = self._get_port_mappings_from_instances_list(instance_id)
            if instances_mapping:
                logger.info(f"✅ instances 목록 방식으로 {len(instances_mapping)}개 포트 매핑 성공")
                return {"mappings": instances_mapping, "public_ip": self._extract_public_ip_from_mappings(instances_mapping)}
        except Exception as e:
            logger.warning(f"instances 목록 방식 실패: {e}")

        # 3️⃣ 텍스트 파싱 방식 (최후 폴백)
        try:
            logger.debug("최후 폴백: 텍스트 파싱 방식")
            text_mapping = self._get_port_mappings_from_text_parsing(instance_id)
            if text_mapping:
                logger.info(f"✅ 텍스트 파싱으로 {len(text_mapping)}개 포트 매핑 성공")
                return {"mappings": text_mapping, "public_ip": self._extract_public_ip_from_mappings(text_mapping)}
        except Exception as e:
            logger.warning(f"텍스트 파싱 실패: {e}")

        logger.error("❌ 포트 매핑 정보를 가져올 수 없습니다.")
        return {"mappings": {}, "public_ip": None}

    def _extract_public_ip_from_mappings(self, mappings: Dict[int, Tuple[str, int]]) -> Optional[str]:
        """매핑에서 공인 IP 추출"""
        for port, (ip, external_port) in mappings.items():
            if ip and ip != "0.0.0.0" and not ip.startswith("127."):
                return ip
        return None

    def _get_port_mappings_from_instance_info(self, instance_id: str) -> Dict[str, Any]:
        """인스턴스 정보에서 포트 매핑 추출"""
        info = self.get_instance_info(instance_id)
        if not info:
            return {}

        mappings = {}
        public_ip = info.get("public_ipaddr")

        # 포트 정보 추출
        if "ports" in info:
            for port_info in info["ports"]:
                internal = port_info.get("internal_port")
                external = port_info.get("external_port")
                if internal and external:
                    mappings[str(internal)] = {
                        "external_port": external,
                        "url": f"http://{public_ip}:{external}" if public_ip else None
                    }

        return {"mappings": mappings, "public_ip": public_ip}

    def _get_port_mappings_from_raw_info(self, instance_id: str) -> Dict[int, Tuple[str, int]]:
        """--raw 옵션을 활용한 포트 매핑 정보 수집 (개선된 방법)"""
        logger.debug(f"🌐 --raw 옵션으로 포트 매핑 정보 수집 (인스턴스 ID: {instance_id})")

        try:
            # get_instance_info를 통해 --raw 정보 가져오기
            raw_info = self.get_instance_info(instance_id)

            if not raw_info or not isinstance(raw_info, dict):
                logger.warning("❌ 인스턴스 정보를 가져올 수 없습니다.")
                return {}

            logger.info(f"🔍 Raw info keys: {list(raw_info.keys())}")

            mapping: Dict[int, Tuple[str, int]] = {}
            public_ip = raw_info.get("public_ipaddr", "unknown")

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

                                # 실제 공인 IP 사용
                                external_ip = public_ip if public_ip != "unknown" else host_ip

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

    def _get_port_mappings_from_instances_list(self, instance_id: str) -> Dict[int, Tuple[str, int]]:
        """vast show instances --raw에서 포트 매핑 추출"""
        try:
            result = self.run_command(["vastai", "show", "instances", "--raw"], parse_json=True)

            if not result["success"] or not result["data"]:
                return {}

            instances_data = result["data"]
            if isinstance(instances_data, str):
                # 문자열 응답을 JSON으로 파싱 시도
                try:
                    instances_data = json.loads(instances_data)
                except json.JSONDecodeError:
                    logger.warning("❌ JSON 파싱 실패")
                    return self._parse_string_response_for_ports(instances_data, instance_id)

            mapping: Dict[int, Tuple[str, int]] = {}

            # 응답은 인스턴스 배열
            if isinstance(instances_data, list):
                # 해당 인스턴스 찾기
                target_instance = None
                for inst in instances_data:
                    if str(inst.get("id")) == str(instance_id):
                        target_instance = inst
                        break

                if target_instance:
                    logger.info(f"✅ 인스턴스 찾음 (ID: {target_instance.get('id')})")

                    # 공인 IP 가져오기
                    public_ip = target_instance.get("public_ipaddr", "unknown")

                    # 포트 정보 파싱
                    ports_dict = target_instance.get("ports", {})

                    for port_key, port_mappings in ports_dict.items():
                        try:
                            container_port = int(port_key.split('/')[0])

                            if port_mappings and len(port_mappings) > 0:
                                host_port = int(port_mappings[0].get("HostPort", "0"))

                                if container_port > 0 and host_port > 0:
                                    mapping[container_port] = (public_ip, host_port)
                                    logger.info(f"   ✅ 매핑 추가: {container_port} -> {public_ip}:{host_port}")

                        except (ValueError, TypeError, KeyError) as e:
                            logger.debug(f"포트 정보 파싱 실패: {port_key}={port_mappings}, 에러: {e}")
                            continue

                    return mapping

        except Exception as e:
            logger.warning(f"instances 목록 방식 실패: {e}")

        return {}

    def _get_port_mappings_from_text_parsing(self, instance_id: str) -> Dict[int, Tuple[str, int]]:
        """텍스트 파싱을 통한 포트 매핑 수집"""
        mapping: Dict[int, Tuple[str, int]] = {}

        try:
            # 일반 show instance 명령어 시도
            result = self.run_command(["vastai", "show", "instance", str(instance_id)], parse_json=False)

            if result["success"]:
                output = result["data"]

                # 포트 패턴 매칭
                patterns = [
                    # 패턴 1: IP:PORT -> CONTAINER_PORT/tcp
                    re.compile(r"(?P<ip>\d+\.\d+\.\d+\.\d+):(?P<host_port>\d+)\s*->\s*(?P<container_port>\d+)/tcp"),
                    # 패턴 2: PORT -> IP:HOST_PORT
                    re.compile(r"(?P<container_port>\d+)\s*->\s*(?P<ip>\d+\.\d+\.\d+\.\d+):(?P<host_port>\d+)"),
                ]

                for pattern in patterns:
                    for line in output.splitlines():
                        match = pattern.search(line)
                        if match:
                            try:
                                ip = match.group("ip")
                                host_port = int(match.group("host_port"))
                                container_port = int(match.group("container_port"))
                                mapping[container_port] = (ip, host_port)
                                logger.info(f"   패턴 매칭: {container_port} -> {ip}:{host_port}")
                            except Exception as e:
                                logger.debug(f"패턴 매칭 실패: {line}, 에러: {e}")

            # show instances로도 시도
            if not mapping:
                instances_result = self.run_command(["vastai", "show", "instances"], parse_json=False)
                if instances_result["success"]:
                    return self._parse_string_response_for_ports(instances_result["data"], instance_id)

        except Exception as e:
            logger.warning(f"텍스트 파싱 실패: {e}")

        return mapping

    def _parse_string_response_for_ports(self, response_str: str, instance_id: str) -> Dict[int, Tuple[str, int]]:
        """문자열 응답에서 포트 매핑 정보 추출"""
        logger.info(f"🔍 문자열 응답에서 포트 정보 추출 시도 (인스턴스 ID: {instance_id})")

        mapping: Dict[int, Tuple[str, int]] = {}

        try:
            lines = response_str.strip().split('\n')

            # 인스턴스 ID가 포함된 라인 찾기
            instance_line = None
            for line in lines:
                if str(instance_id) in line:
                    instance_line = line
                    logger.info(f"🔍 인스턴스 라인 발견: {line}")
                    break

            if not instance_line:
                logger.warning(f"⚠️ 인스턴스 ID {instance_id}가 포함된 라인을 찾을 수 없음")
                return mapping

            # 라인에서 IP:PORT 패턴 찾기
            ip_port_pattern = re.compile(r'(\d+\.\d+\.\d+\.\d+):(\d+)')
            matches = ip_port_pattern.findall(instance_line)

            logger.info(f"🔍 발견된 IP:PORT 패턴: {matches}")

            for ip, port_str in matches:
                try:
                    external_port = int(port_str)

                    # 포트 번호로 컨테이너 포트 추정
                    container_port = self._estimate_container_port(external_port)

                    if container_port:
                        mapping[container_port] = (ip, external_port)
                        logger.info(f"   ✅ 매핑 추가: {container_port} -> {ip}:{external_port}")
                    else:
                        logger.debug(f"   ❓ 컨테이너 포트 추정 불가: {external_port}")

                except ValueError as e:
                    logger.debug(f"포트 파싱 실패: {port_str}, 에러: {e}")
                    continue

            if mapping:
                logger.info(f"✅ 문자열 파싱으로 {len(mapping)}개 포트 매핑 성공")
            else:
                logger.warning("⚠️ 문자열에서 포트 매핑 정보를 찾을 수 없음")

        except Exception as e:
            logger.warning(f"❌ 문자열 파싱 중 오류: {e}")

        return mapping

    def _estimate_container_port(self, external_port: int) -> Optional[int]:
        """외부 포트 번호를 통해 컨테이너 포트 추정"""
        port_suffix = str(external_port)[-3:]  # 마지막 3자리

        port_mapping = {
            "111": 1111,    # xxxxx1111 -> 1111
            "080": 8080,    # xxxxx8080 -> 8080
            "006": 6006,    # xxxxx6006 -> 6006
            "384": 8384,    # xxxxx8384 -> 8384
            "479": 11479,   # xxxxx1479 -> 11479
            "480": 11480,   # xxxxx1480 -> 11480
        }

        if port_suffix in port_mapping:
            return port_mapping[port_suffix]
        elif external_port == 22:  # SSH
            return 22

        return None

    def _extract_port_info(self, instance_data: Dict[str, Any]) -> Dict[str, Any]:
        """인스턴스 데이터에서 포트 정보 추출"""
        mappings = {}
        public_ip = instance_data.get("public_ipaddr")

        # 다양한 포트 필드 확인
        port_fields = ["ports", "port_mappings", "exposed_ports"]

        for field in port_fields:
            if field in instance_data:
                port_data = instance_data[field]
                if isinstance(port_data, dict):
                    for internal, external in port_data.items():
                        mappings[str(internal)] = {
                            "external_port": external,
                            "url": f"http://{public_ip}:{external}" if public_ip else None
                        }
                elif isinstance(port_data, list):
                    for port_info in port_data:
                        if isinstance(port_info, dict):
                            internal = port_info.get("internal_port") or port_info.get("internal")
                            external = port_info.get("external_port") or port_info.get("external")
                            if internal and external:
                                mappings[str(internal)] = {
                                    "external_port": external,
                                    "url": f"http://{public_ip}:{external}" if public_ip else None
                                }

        return {"mappings": mappings, "public_ip": public_ip}

    def _get_port_mappings_from_ssh(self, instance_id: str, command: str) -> Dict[str, Any]:
        """SSH를 통해 포트 매핑 수집"""
        result = self.execute_ssh_command(instance_id, command)

        if not result["success"]:
            return {}

        # netstat/ss 출력 파싱
        mappings = {}
        for line in result["stdout"].split('\n'):
            if ':8000' in line or ':22' in line:
                # 간단한 포트 감지
                if ':8000' in line:
                    mappings["8000"] = {"external_port": "8000", "url": None}
                if ':22' in line:
                    mappings["22"] = {"external_port": "22", "url": None}

        return {"mappings": mappings, "public_ip": None}

    def _get_port_mappings_from_api(self, instance_id: str) -> Dict[str, Any]:
        """API를 통한 직접 포트 매핑 수집"""
        # 이는 향후 Vast.ai API 직접 호출로 구현 가능
        return {}

    def _get_default_port_mappings(self) -> Dict[str, Any]:
        """기본 포트 매핑 반환"""
        return {
            "mappings": {
                "8000": {"external_port": "8000", "url": None},
                "22": {"external_port": "22", "url": None}
            },
            "public_ip": None
        }

    def display_port_mappings(self, instance_id: str) -> Dict[str, Any]:
        """포트 매핑 정보를 보기 좋게 출력"""
        port_info = self.get_port_mappings(instance_id)
        port_mappings = port_info.get("mappings", {})

        if not port_mappings:
            logger.warning("포트 매핑 정보를 가져올 수 없습니다.")
            return port_info

        # 포트별 서비스 이름 매핑
        port_services = {
            1111: "Instance Portal",
            6006: "Tensorboard",
            8080: "Jupyter",
            8384: "Syncthing",
            11479: "vLLM Main",
            11480: "vLLM Controller",
            22: "SSH",
            72299: "Custom Service"
        }

        logger.info("\n🌐 포트 매핑 정보:")
        logger.info("=" * 50)

        # 포트 번호 순으로 정렬하여 출력
        for container_port in sorted(port_mappings.keys()):
            external_ip, external_port = port_mappings[container_port]
            service_name = port_services.get(container_port, "Unknown Service")

            logger.info(f"   {container_port:5d} ({service_name:16s}) → {external_ip}:{external_port}")

        logger.info("=" * 50)

        # 주요 서비스 URL 생성
        main_services = []
        if 1111 in port_mappings:
            ip, port = port_mappings[1111]
            main_services.append(f"🏠 Instance Portal: http://{ip}:{port}")

        if 8080 in port_mappings:
            ip, port = port_mappings[8080]
            main_services.append(f"📓 Jupyter: http://{ip}:{port}")

        if 11479 in port_mappings:
            ip, port = port_mappings[11479]
            main_services.append(f"🤖 vLLM Main: http://{ip}:{port}")

        if 11480 in port_mappings:
            ip, port = port_mappings[11480]
            main_services.append(f"🎛️ vLLM Controller: http://{ip}:{port}")

        if 6006 in port_mappings:
            ip, port = port_mappings[6006]
            main_services.append(f"📊 Tensorboard: http://{ip}:{port}")

        if main_services:
            logger.info("\n🔗 주요 서비스 URL:")
            for service in main_services:
                logger.info(f"   {service}")

        return port_info

    def destroy_instance(self, instance_id: str) -> bool:
        """인스턴스 삭제"""
        logger.info(f"인스턴스 {instance_id} 삭제 중...")

        result = self.run_command(["vastai", "destroy", "instance", instance_id], parse_json=False)

        if result["success"]:
            # 삭제 확인
            time.sleep(5)
            status = self.get_instance_status(instance_id)

            if status in ["destroyed", "unknown"]:
                logger.info(f"인스턴스 {instance_id} 삭제 완료")
                return True
            else:
                logger.warning(f"인스턴스 {instance_id} 삭제 확인 실패, 현재 상태: {status}")
                return False

        logger.error(f"인스턴스 {instance_id} 삭제 실패: {result.get('error')}")
        return False

    def setup_and_run_vllm(self, instance_id: str) -> bool:
        """vLLM 설정 및 실행 (HF 로그인 제거)"""
        logger.info("vLLM 설정 및 실행 중...")

        # vLLM 실행 명령
        commands = [
            "cd /home/vllm-script",
            "nohup python3 main.py > /tmp/vllm.log 2>&1 &"
        ]

        for cmd in commands:
            result = self.execute_ssh_command(instance_id, cmd)

            if not result["success"]:
                logger.error(f"명령 실행 실패: {cmd}")
                logger.error(f"오류: {result.get('error')}")
                return False

            logger.info(f"명령 완료: {cmd}")

        logger.info("vLLM 실행 완료")
        return True

    def check_vllm_status(self, instance_id: str) -> Dict[str, Any]:
        """vLLM 상태 확인"""
        # 로그 확인
        log_result = self.execute_ssh_command(instance_id, "tail -n 20 /tmp/vllm.log")

        # 프로세스 확인
        process_result = self.execute_ssh_command(instance_id, "ps aux | grep python")

        return {
            "log_output": log_result.get("stdout", ""),
            "process_info": process_result.get("stdout", ""),
            "log_success": log_result.get("success", False),
            "process_success": process_result.get("success", False)
        }

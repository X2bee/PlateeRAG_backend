"""
VastAI 서비스

VastAI 인스턴스 관리를 위한 고수준 서비스 클래스입니다.
"""

import logging
import asyncio
import time
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

from service.vast.vast_manager import VastAIManager
from service.database.models.vast import VastInstance, VastExecutionLog

logger = logging.getLogger("vast-service")

class InstanceTemplate:
    """인스턴스 생성 템플릿"""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config

    def apply_to_config(self, base_config, custom_overrides: Optional[Dict[str, Any]] = None):
        """템플릿을 기본 설정에 적용"""
        # 템플릿 설정 적용
        for key, value in self.config.items():
            # env_name을 통해 PersistentConfig 객체 찾기
            env_name = f"VLLM_{key.upper()}" if key.startswith('vllm_') else key.upper()

            # all_configs에서 env_name으로 찾기
            if hasattr(base_config, 'configs'):
                for config_obj in base_config.configs.values():
                    if hasattr(config_obj, 'env_name') and config_obj.env_name == env_name:
                        config_obj.value = value
                        logger.debug("템플릿 설정 적용: %s = %s", key, value)
                        break

        # 사용자 정의 오버라이드 적용
        if custom_overrides:
            for key, value in custom_overrides.items():
                # env_name을 통해 PersistentConfig 객체 찾기
                env_name = f"VLLM_{key.upper()}" if key.startswith('vllm_') else key.upper()

                # all_configs에서 env_name으로 찾기
                if hasattr(base_config, 'configs'):
                    for config_obj in base_config.configs.values():
                        if hasattr(config_obj, 'env_name') and config_obj.env_name == env_name:
                            config_obj.value = value
                            logger.debug("사용자 설정 적용: %s = %s", key, value)
                            break

        return base_config

class VastService:
    """VastAI 서비스 클래스"""

    def __init__(self, config, db_manager=None):
        """VastService 초기화

        Args:
            config: VastConfig 인스턴스
            db_manager: 데이터베이스 매니저 (선택사항)
        """
        self.config = config
        self.db_manager = db_manager
        self.vast_manager = VastAIManager(config, db_manager)
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, InstanceTemplate]:
        """인스턴스 템플릿 로드"""
        templates = {}

        # 기본 템플릿들
        templates["default_vllm"] = InstanceTemplate("default_vllm", {
            "vllm_model_name": "Qwen/Qwen2.5-Coder-32B-Instruct",
            "vllm_max_model_len": 4096,
            "vllm_gpu_memory_utilization": 0.9,
            "vllm_pipeline_parallel_size": 1,
            "vllm_tensor_parallel_size": 1
        })

        templates["high_performance"] = InstanceTemplate("high_performance", {
            "vllm_model_name": "Qwen/Qwen2.5-Coder-32B-Instruct",
            "vllm_max_model_len": 8192,
            "vllm_gpu_memory_utilization": 0.95,
            "vllm_pipeline_parallel_size": 2,
            "vllm_tensor_parallel_size": 2,
            "min_gpu_ram": 24,
            "max_price": 2.0
        })

        templates["budget"] = InstanceTemplate("budget", {
            "vllm_model_name": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "vllm_max_model_len": 2048,
            "vllm_gpu_memory_utilization": 0.8,
            "min_gpu_ram": 8,
            "max_price": 0.5
        })

        # 파일에서 사용자 정의 템플릿 로드
        try:
            template_dir = "constants/instance_templates"
            if os.path.exists(template_dir):
                for filename in os.listdir(template_dir):
                    if filename.endswith('.json'):
                        template_path = os.path.join(template_dir, filename)
                        with open(template_path, 'r', encoding='utf-8') as f:
                            template_data = json.load(f)
                            template_name = template_data.get('name', filename.replace('.json', ''))
                            template_config = template_data.get('config', {})
                            templates[template_name] = InstanceTemplate(template_name, template_config)
        except Exception as e:
            logger.warning(f"사용자 정의 템플릿 로드 실패: {e}")

        return templates

    def get_available_templates(self) -> List[Dict[str, Any]]:
        """사용 가능한 템플릿 목록 반환"""
        return [
            {
                "name": template.name,
                "config": template.config
            }
            for template in self.templates.values()
        ]

    def apply_template(self, template_name: str, custom_config: Optional[Dict[str, Any]] = None):
        """템플릿을 현재 설정에 적용"""
        if template_name not in self.templates:
            raise ValueError(f"템플릿 '{template_name}'을 찾을 수 없습니다")

        template = self.templates[template_name]
        template.apply_to_config(self.config, custom_config)
        logger.info(f"템플릿 '{template_name}' 적용 완료")

    def _log_execution(self, instance_id: str, operation: str, command: str,
                      result: str, success: bool, execution_time: float = 0.0,
                      error_message: str = None, metadata: Optional[Dict[str, Any]] = None):
        """실행 로그 저장 (향상된 버전)"""
        if not self.db_manager:
            return

        try:
            log_data = {
                "instance_id": instance_id,
                "operation": operation,
                "command": command,
                "result": result,
                "success": success,
                "execution_time": execution_time,
                "error_message": error_message
            }

            # 메타데이터가 있으면 JSON으로 저장
            if metadata:
                log_data["metadata"] = json.dumps(metadata)

            log = VastExecutionLog(**log_data)
            self.db_manager.insert(log)
        except Exception as e:
            logger.warning(f"실행 로그 저장 실패: {e}")

    def _save_instance(self, instance_data: Dict[str, Any]) -> bool:
        """인스턴스 정보 저장 (향상된 버전)"""
        if not self.db_manager:
            return True

        try:
            # 기본 필드 검증
            required_fields = ["instance_id"]
            for field in required_fields:
                if field not in instance_data:
                    logger.error(f"필수 필드 '{field}'가 누락되었습니다")
                    return False

            # JSON 필드 처리
            if "gpu_info" in instance_data and isinstance(instance_data["gpu_info"], dict):
                instance_data["gpu_info"] = json.dumps(instance_data["gpu_info"])

            if "port_mappings" in instance_data and isinstance(instance_data["port_mappings"], dict):
                instance_data["port_mappings"] = json.dumps(instance_data["port_mappings"])

            instance = VastInstance(**instance_data)
            result = self.db_manager.insert(instance)

            if result:
                logger.info(f"인스턴스 정보 저장 완료: {instance_data['instance_id']}")
                return True
            else:
                logger.error("인스턴스 정보 저장 실패")
                return False

        except Exception as e:
            logger.error(f"인스턴스 정보 저장 실패: {e}")
            return False

    def _update_instance(self, instance_id: str, updates: Dict[str, Any]) -> bool:
        """인스턴스 정보 업데이트 (향상된 버전)"""
        if not self.db_manager:
            return True

        try:
            # 기존 인스턴스 조회
            conditions = {"instance_id": instance_id}
            existing = self.db_manager.find_by_condition(VastInstance, conditions=conditions)

            if existing:
                instance = existing[0]

                # JSON 필드 처리
                if "gpu_info" in updates and isinstance(updates["gpu_info"], dict):
                    updates["gpu_info"] = json.dumps(updates["gpu_info"])

                if "port_mappings" in updates and isinstance(updates["port_mappings"], dict):
                    updates["port_mappings"] = json.dumps(updates["port_mappings"])

                # 업데이트 수행
                for key, value in updates.items():
                    if hasattr(instance, key):
                        setattr(instance, key, value)

                result = self.db_manager.update(instance)

                if result:
                    logger.info(f"인스턴스 정보 업데이트 완료: {instance_id}")
                    return True
                else:
                    logger.error(f"인스턴스 정보 업데이트 실패: {instance_id}")
                    return False
            else:
                logger.warning(f"업데이트할 인스턴스를 찾을 수 없음: {instance_id}")
                return False

        except Exception as e:
            logger.error(f"인스턴스 정보 업데이트 실패: {e}")
            return False

    def get_instance_from_db(self, instance_id: str) -> Optional[VastInstance]:
        """DB에서 인스턴스 정보 조회"""
        if not self.db_manager:
            return None

        try:
            conditions = {"instance_id": instance_id}
            instances = self.db_manager.find_by_condition(VastInstance, conditions=conditions)
            return instances[0] if instances else None
        except Exception as e:
            logger.error(f"DB 인스턴스 조회 실패: {e}")
            return None

    def get_enhanced_instance_status(self, instance_id: str) -> Dict[str, Any]:
        """향상된 인스턴스 상태 정보 반환"""
        try:
            # 기본 상태 정보
            status_info = self.get_instance_status_info(instance_id)

            # DB에서 추가 정보
            db_instance = self.get_instance_from_db(instance_id)

            # vLLM 상태
            vllm_status = None
            try:
                vllm_status = self.vast_manager.check_vllm_status(instance_id)
            except Exception as e:
                logger.warning(f"vLLM 상태 확인 실패: {e}")

            # 통합 상태 정보 구성
            enhanced_status = {
                "instance_id": instance_id,
                "basic_status": status_info,
                "vllm_status": vllm_status,
                "db_info": {}
            }

            if db_instance:
                enhanced_status["db_info"] = {
                    "status": db_instance.status,
                    "cost_per_hour": db_instance.cost_per_hour,
                    "gpu_info": db_instance.get_gpu_info_dict(),
                    "port_mappings": db_instance.get_port_mappings_dict(),
                    "auto_destroy": db_instance.auto_destroy,
                    "created_at": db_instance.created_at,
                    "updated_at": db_instance.updated_at,
                    "uptime": str(datetime.now() - db_instance.created_at) if db_instance.created_at else None
                }

            return enhanced_status

        except Exception as e:
            logger.error(f"향상된 인스턴스 상태 조회 실패: {e}")
            return {"error": str(e)}

    def search_and_select_offer(self) -> Optional[Dict[str, Any]]:
        """오퍼 검색 및 선택"""
        logger.info("VastAI 오퍼 검색 시작")

        start_time = time.time()

        # API 키 설정
        if not self.vast_manager.setup_api_key():
            self._log_execution(
                instance_id="",
                operation="setup_api_key",
                command="vastai set api-key",
                result="",
                success=False,
                execution_time=time.time() - start_time,
                error_message="API 키 설정 실패"
            )
            return None

        # 오퍼 검색
        offers = self.vast_manager.search_offers()

        if not offers:
            self._log_execution(
                instance_id="",
                operation="search_offers",
                command="vastai search offers",
                result="No offers found",
                success=False,
                execution_time=time.time() - start_time,
                error_message="사용 가능한 오퍼가 없습니다"
            )
            return None

        # 오퍼 선택
        selected_offer = self.vast_manager.select_offer(offers)

        if not selected_offer:
            self._log_execution(
                instance_id="",
                operation="select_offer",
                command="offer selection",
                result="No suitable offer",
                success=False,
                execution_time=time.time() - start_time,
                error_message="적합한 오퍼가 없습니다"
            )
            return None

        execution_time = time.time() - start_time
        self._log_execution(
            instance_id="",
            operation="search_offers",
            command="vastai search offers",
            result=f"Found {len(offers)} offers, selected offer {selected_offer.get('id')}",
            success=True,
            execution_time=execution_time
        )

        logger.info(f"오퍼 선택 완료: {selected_offer.get('id')} (${selected_offer.get('dph_total')}/h)")
        return selected_offer

    def create_vllm_instance(self, offer_id: str = None, template_name: str = None, create_request = None) -> Optional[str]:
        """vLLM 인스턴스 생성 (템플릿 지원)"""
        logger.info("vLLM 인스턴스 생성 시작")

        start_time = time.time()

        # 인스턴스 생성
        instance_id = self.vast_manager.create_instance_fallback(offer_id)

        if not instance_id:
            self._log_execution(
                instance_id="",
                operation="create_instance",
                command=f"vastai create instance {offer_id}",
                result="",
                success=False,
                execution_time=time.time() - start_time,
                error_message="인스턴스 생성 실패",
                metadata={
                    "offer_id": offer_id,
                    "template_name": template_name
                }
            )
            return None
        gpu_info = {
            "gpu_name": create_request.offer_info.get("gpu_name"),
            "num_gpus": create_request.offer_info.get("num_gpus", 1),
            "gpu_ram": create_request.offer_info.get("gpu_ram")
        }
        instance_data = {
            "instance_id": instance_id,
            "offer_id": offer_id,
            "image_name": self.config.image_name(),
            "status": "creating",
            "auto_destroy": self.config.auto_destroy(),
            "gpu_info": json.dumps(gpu_info) if gpu_info else None,
            "dph_total": create_request.offer_info.get("dph_total", 0.0),
            "cpu_name": create_request.offer_info.get("cpu_name"),
            "cpu_cores": create_request.offer_info.get("cpu_cores"),
            "ram": create_request.offer_info.get("ram"),
            "cuda_max_good": create_request.offer_info.get("cuda_max_good", 0.0),
            "model_name": create_request.vllm_config.vllm_model_name,
            "max_model_length": create_request.vllm_config.vllm_max_model_len,
        }

        if template_name:
            instance_data["template_name"] = template_name

        self._save_instance(instance_data)

        execution_time = time.time() - start_time
        self._log_execution(
            instance_id=instance_id,
            operation="create_instance",
            command=f"vastai create instance {offer_id}",
            result=f"Instance created successfully: {instance_id}",
            success=True,
            execution_time=execution_time,
            metadata={
                "offer_id": offer_id,
                "template_name": template_name,
                "gpu_info": instance_data.get("gpu_info"),
                "cost_per_hour": instance_data.get("cost_per_hour")
            }
        )

        logger.info(f"vLLM 인스턴스 생성 완료: {instance_id}")
        return instance_id

    def wait_and_setup_instance(self, instance_id: str) -> bool:
        """인스턴스 실행 대기 및 설정"""
        logger.info(f"인스턴스 {instance_id} 설정 시작")

        start_time = time.time()

        # 실행 상태 대기
        if not self.vast_manager.wait_for_running(instance_id):
            self._log_execution(
                instance_id=instance_id,
                operation="wait_for_running",
                command=f"wait for instance {instance_id}",
                result="",
                success=False,
                execution_time=time.time() - start_time,
                error_message="인스턴스 실행 대기 타임아웃"
            )
            return False

        # 상태 업데이트
        self._update_instance(instance_id, {"status": "running"})

        # 인스턴스 정보 수집
        instance_info = self.vast_manager.get_instance_info(instance_id)
        if instance_info and isinstance(instance_info, dict):
            updates = {
                "public_ip": instance_info.get("public_ipaddr"),
                "ssh_port": instance_info.get("ssh_port", 22)
            }

            # GPU 정보 저장
            if "gpu_name" in instance_info:
                gpu_info = {
                    "gpu_name": instance_info.get("gpu_name"),
                    "num_gpus": instance_info.get("num_gpus", 1),
                    "gpu_ram": instance_info.get("gpu_ram")
                }
                updates["gpu_info"] = gpu_info

            if "dph_total" in instance_info:
                updates["dph_total"] = instance_info.get("dph_total")

            self._update_instance(instance_id, updates)

        port_info = self.vast_manager.get_port_mappings(instance_id)
        if port_info.get("mappings"):
            # 포트 매핑을 JSON 문자열로 저장
            import json
            self._update_instance(instance_id, {
                "port_mappings": json.dumps(port_info["mappings"]),
                "status": "running_vllm"
            })

        execution_time = time.time() - start_time
        self._log_execution(
            instance_id=instance_id,
            operation="setup_instance",
            command="full instance setup",
            result="Instance setup completed successfully",
            success=True,
            execution_time=execution_time
        )

        logger.info(f"인스턴스 {instance_id} 설정 완료")
        return True

    def get_instance_status_info(self, instance_id: str) -> Dict[str, Any]:
        """인스턴스 상태 정보 조회"""
        # 기본 상태 확인
        status = self.vast_manager.get_instance_status(instance_id)

        # vLLM 상태 확인
        vllm_status = self.vast_manager.check_vllm_status(instance_id)

        # 포트 매핑 정보
        port_info = self.vast_manager.get_port_mappings(instance_id)

        return {
            "instance_id": instance_id,
            "status": status,
            "vllm_status": vllm_status,
            "port_mappings": port_info,
            "urls": self._generate_access_urls(port_info)
        }

    def _generate_access_urls(self, port_info: Dict[str, Any]) -> Dict[str, str]:
        """접속 URL 생성"""
        urls = {}

        public_ip = port_info.get("public_ip")
        mappings = port_info.get("mappings", {})

        if public_ip and mappings:
            for internal_port, mapping in mappings.items():
                external_port = mapping.get("external_port")

                if internal_port == "8000":
                    urls["vllm_api"] = f"http://{public_ip}:{external_port}"
                    urls["vllm_docs"] = f"http://{public_ip}:{external_port}/docs"
                elif internal_port == "22":
                    urls["ssh"] = f"ssh://{public_ip}:{external_port}"

        return urls

    def destroy_instance(self, instance_id: str) -> bool:
        """인스턴스 삭제"""
        logger.info(f"인스턴스 {instance_id} 삭제 시작")

        start_time = time.time()

        success = self.vast_manager.destroy_instance(instance_id)

        if success:
            # 데이터베이스 업데이트
            self._update_instance(instance_id, {
                "status": "destroyed",
                "destroyed_at": datetime.now()
            })

        execution_time = time.time() - start_time
        self._log_execution(
            instance_id=instance_id,
            operation="destroy_instance",
            command=f"vastai destroy instance {instance_id}",
            result="Instance destroyed" if success else "Destroy failed",
            success=success,
            execution_time=execution_time,
            error_message=None if success else "인스턴스 삭제 실패"
        )

        return success

    def list_instances(self) -> List[Dict[str, Any]]:
        """인스턴스 목록 조회"""
        if not self.db_manager:
            return []

        try:
            instances = self.db_manager.find_all(VastInstance)
            return [instance.to_dict() for instance in instances]
        except Exception as e:
            logger.error(f"인스턴스 목록 조회 실패: {e}")
            return []

    async def auto_destroy_after_delay(self, instance_id: str, delay_seconds: int = 60):
        """지연 후 자동 삭제"""
        logger.info(f"인스턴스 {instance_id} {delay_seconds}초 후 자동 삭제 예약")

        await asyncio.sleep(delay_seconds)

        logger.info(f"인스턴스 {instance_id} 자동 삭제 실행")
        self.destroy_instance(instance_id)


def auto_run_vllm(config, db_manager=None) -> Dict[str, Any]:
    """
    end-to-end vLLM 자동 실행 파이프라인

    Args:
        config: VastConfig 인스턴스
        db_manager: 데이터베이스 매니저 (선택사항)

    Returns:
        실행 결과 정보
    """
    logger.info("=== vLLM 자동 실행 파이프라인 시작 ===")

    service = VastService(config, db_manager)

    try:
        # 1. 오퍼 검색 및 선택
        logger.info("1. 오퍼 검색 및 선택")
        offer = service.search_and_select_offer()
        if not offer:
            return {
                "success": False,
                "error": "적합한 오퍼를 찾을 수 없습니다",
                "step": "offer_search"
            }

        # 2. 인스턴스 생성
        logger.info("2. 인스턴스 생성")
        instance_id = service.create_vllm_instance(offer.get('id'))
        if not instance_id:
            return {
                "success": False,
                "error": "인스턴스 생성에 실패했습니다",
                "step": "instance_creation"
            }

        # 3. 인스턴스 설정 및 vLLM 실행
        logger.info("3. 인스턴스 설정 및 vLLM 실행")
        if not service.wait_and_setup_instance(instance_id):
            # 실패 시 정리
            service.destroy_instance(instance_id)
            return {
                "success": False,
                "error": "인스턴스 설정에 실패했습니다",
                "step": "instance_setup",
                "instance_id": instance_id
            }

        # 4. 상태 정보 수집
        logger.info("4. 최종 상태 확인")
        status_info = service.get_instance_status_info(instance_id)

        # 5. 포트 매핑 표시
        logger.info("5. 포트 매핑 정보 표시")
        port_info = service.vast_manager.display_port_mappings(instance_id)

        # 6. 자동 삭제 예약 (설정된 경우)
        if config.auto_destroy():
            logger.info("6. 자동 삭제 예약")
            asyncio.create_task(service.auto_destroy_after_delay(instance_id, 60))

        logger.info("=== vLLM 자동 실행 파이프라인 완료 ===")

        return {
            "success": True,
            "instance_id": instance_id,
            "offer_info": offer,
            "status_info": status_info,
            "port_mappings": port_info,
            "auto_destroy": config.auto_destroy()
        }

    except Exception as e:
        logger.error(f"vLLM 자동 실행 중 오류: {e}")
        return {
            "success": False,
            "error": str(e),
            "step": "unexpected_error"
        }

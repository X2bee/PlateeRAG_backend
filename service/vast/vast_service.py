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

    def __init__(self, db_manager=None, config_composer=None):
        """VastService 초기화

        Args:
            config: VastConfig 인스턴스
            db_manager: 데이터베이스 매니저 (선택사항)
        """
        self.db_manager = db_manager
        self.config_composer = config_composer
        self.vast_manager = VastAIManager(self.db_manager, self.config_composer)
        self.templates = self._load_templates()
        self.status_change_callback = None

    def set_status_change_callback(self, callback):
        """상태 변경 콜백 설정"""
        self.status_change_callback = callback

    def _load_templates(self) -> Dict[str, InstanceTemplate]:
        """인스턴스 템플릿 로드"""
        templates = {}

        # 기본 템플릿들
        templates["default_vllm"] = InstanceTemplate("default_vllm", {
            "vllm_serve_model_name": "Qwen/Qwen2.5-Coder-32B-Instruct",
            "vllm_max_model_len": 4096,
            "vllm_gpu_memory_utilization": 0.9,
            "vllm_pipeline_parallel_size": 1,
            "vllm_tensor_parallel_size": 1
        })

        templates["high_performance"] = InstanceTemplate("high_performance", {
            "vllm_serve_model_name": "Qwen/Qwen2.5-Coder-32B-Instruct",
            "vllm_max_model_len": 8192,
            "vllm_gpu_memory_utilization": 0.95,
            "vllm_pipeline_parallel_size": 2,
            "vllm_tensor_parallel_size": 2,
            "min_gpu_ram": 24,
            "max_price": 2.0
        })

        templates["budget"] = InstanceTemplate("budget", {
            "vllm_serve_model_name": "Qwen/Qwen2.5-Coder-7B-Instruct",
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

    def apply_template(self, template_name: str, custom_config: Optional[Dict[str, Any]] = None):
        """템플릿을 현재 설정에 적용"""
        if template_name not in self.templates:
            raise ValueError(f"템플릿 '{template_name}'을 찾을 수 없습니다")
        config = self.config_composer.get_config_by_category_name("vast")
        template = self.templates[template_name]
        template.apply_to_config(config, custom_config)
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

                    # 상태 변경 시 콜백 호출
                    if self.status_change_callback and "status" in updates:
                        try:
                            # 동기 콜백으로 호출하여 이벤트 루프 문제 해결
                            self.status_change_callback(instance_id, updates["status"])
                        except Exception as e:
                            logger.warning(f"상태 변경 콜백 호출 실패: {e}")

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

    def update_instance_port_mappings(self, instance_id: str) -> bool:
        """인스턴스의 포트 매핑 정보를 DB에 업데이트"""
        try:
            port_info = self.vast_manager.get_port_mappings(instance_id)

            # get_port_mappings는 mappings와 public_ip를 반
            mappings = port_info.get("mappings", {})
            public_ip = port_info.get("public_ip")

            # 매핑이나 공개 IP가 없으면 실패로 간주
            if not mappings and not public_ip:
                logger.warning(f"인스턴스 {instance_id}의 포트 매핑 정보를 가져올 수 없습니다")
                return False

            # DB 업데이트 준비
            updates = {}

            if public_ip:
                updates["public_ip"] = public_ip
                logger.info(f"인스턴스 {instance_id}의 공개 IP 업데이트: {public_ip}")

            if mappings:
                # 포트 매핑을 JSON 문자열로 저장
                # mappings는 {internal_port: (external_ip, external_port)} 형태
                # DB 저장용으로 {internal_port: {"external_ip": ip, "external_port": port}} 형태로 변환
                db_mappings = {}
                for internal_port, mapping_info in mappings.items():
                    if isinstance(mapping_info, tuple) and len(mapping_info) >= 2:
                        external_ip, external_port = mapping_info[0], mapping_info[1]
                        db_mappings[str(internal_port)] = {
                            "external_ip": external_ip,
                            "external_port": external_port
                        }
                    else:
                        # 이미 dict 형태인 경우
                        db_mappings[str(internal_port)] = mapping_info

                updates["port_mappings"] = json.dumps(db_mappings)
                logger.info(f"인스턴스 {instance_id}의 포트 매핑 업데이트: {len(db_mappings)}개 포트")

            if updates:
                success = self._update_instance(instance_id, updates)
                if success:
                    logger.info(f"인스턴스 {instance_id}의 포트 매핑 정보가 DB에 성공적으로 업데이트되었습니다")
                else:
                    logger.error(f"인스턴스 {instance_id}의 DB 업데이트 실패")
                return success
            else:
                logger.info(f"인스턴스 {instance_id}에 대한 업데이트할 정보가 없습니다")
                return True

        except Exception as e:
            logger.error(f"포트 매핑 업데이트 실패: {e}")
            return False

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

    def create_vllm_instance(self, offer_id: str = None, template_name: str = None, create_request = None, vast_config_request = None) -> Optional[str]:
        """vLLM 인스턴스 생성 (템플릿 지원)"""
        logger.info("vLLM 인스턴스 생성 시작")

        start_time = time.time()
        instance_id = self.vast_manager.create_instance_fallback(offer_id, vast_config_request)

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
        name = self.config_composer.get_config_by_name("VAST_IMAGE_NAME").value
        tag = self.config_composer.get_config_by_name("VAST_IMAGE_TAG").value
        image_name = f"{name}:{tag}" if tag else name

        instance_data = {
            "instance_id": instance_id,
            "offer_id": offer_id,
            "image_name": image_name,
            "status": "creating",
            "auto_destroy": self.config_composer.get_config_by_name("VAST_AUTO_DESTROY").value,
            "gpu_info": json.dumps(gpu_info) if gpu_info else None,
            "dph_total": create_request.offer_info.get("dph_total", 0.0),
            "cpu_name": create_request.offer_info.get("cpu_name"),
            "cpu_cores": create_request.offer_info.get("cpu_cores"),
            "ram": create_request.offer_info.get("ram"),
            "cuda_max_good": create_request.offer_info.get("cuda_max_good", 0.0),
            "model_name": create_request.vllm_config.vllm_serve_model_name,
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

    def create_trainer_instance(self, offer_id: str = None, create_request = None) -> Optional[str]:
        """Trainer 인스턴스 생성 (템플릿 지원)"""
        logger.info("Trainer 인스턴스 생성 시작")

        start_time = time.time()

        # 인스턴스 생성
        instance_id = self.vast_manager.create_train_instance(offer_id)

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
                }
            )
            return None
        gpu_info = {
            "gpu_name": create_request.offer_info.get("gpu_name"),
            "num_gpus": create_request.offer_info.get("num_gpus", 1),
            "gpu_ram": create_request.offer_info.get("gpu_ram")
        }
        name = self.config_composer.get_config_by_name("VAST_TRAIN_IMAGE_NAME").value
        tag = self.config_composer.get_config_by_name("VAST_TRAIN_IMAGE_TAG").value
        image_name = f"{name}:{tag}" if tag else name
        instance_data = {
            "instance_id": instance_id,
            "offer_id": offer_id,
            "image_name": image_name,
            "status": "creating",
            "auto_destroy": self.config_composer.get_config_by_name("AUTO_DESTROY").value,
            "gpu_info": json.dumps(gpu_info) if gpu_info else None,
            "dph_total": create_request.offer_info.get("dph_total", 0.0),
            "cpu_name": create_request.offer_info.get("cpu_name"),
            "cpu_cores": create_request.offer_info.get("cpu_cores"),
            "ram": create_request.offer_info.get("ram"),
            "cuda_max_good": create_request.offer_info.get("cuda_max_good", 0.0),
            "model_name": "Trainer",
            "max_model_length": 0,
        }

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
                "gpu_info": instance_data.get("gpu_info"),
                "cost_per_hour": instance_data.get("cost_per_hour")
            }
        )

        logger.info(f"Trainer 인스턴스 생성 완료: {instance_id}")
        return instance_id

    def wait_and_setup_instance(self, instance_id: str, is_valid_model:bool = False) -> bool:
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
            self._update_instance(instance_id, {"status": "failed"})
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

            if "cpu_name" in instance_info:
                updates["cpu_name"] = instance_info.get("cpu_name")
            if "cpu_cores" in instance_info:
                updates["cpu_cores"] = instance_info.get("cpu_cores")
            if "cpu_ram" in instance_info:
                updates["ram"] = instance_info.get("cpu_ram")
            if "cuda_max_good" in instance_info:
                updates["cuda_max_good"] = instance_info.get("cuda_max_good")

            self._update_instance(instance_id, updates)

        logger.info(f"인스턴스 {instance_id}의 포트 매핑 정보 수집 중...")
        port_update_success = self.update_instance_port_mappings(instance_id)

        # 최종 상태 결정 로직
        if is_valid_model and port_update_success:
            # 유효한 모델이고 포트 업데이트 성공 시 VLLM 헬스체크 수행
            db_instance = self.get_instance_from_db(instance_id)
            port_mappings = db_instance.get_port_mappings_dict()
            health_url = f"http://{port_mappings.get('12434', {}).get('external_ip')}:{port_mappings.get('12434', {}).get('external_port')}/health"

            if self.vast_manager.wait_for_vllm_running(health_url):
                self._update_instance(instance_id, {"status": "running_vllm"})
                logger.info(f"인스턴스 {instance_id} - VLLM 서비스 정상 동작 확인")
            else:
                self._update_instance(instance_id, {"status": "running"})
                logger.info(f"인스턴스 {instance_id} - VLLM 헬스체크 실패, 기본 실행 상태로 설정")

        elif is_valid_model and not port_update_success:
            # 유효한 모델이지만 포트 업데이트 실패 시
            self._update_instance(instance_id, {"status": "running"})
            logger.warning(f"인스턴스 {instance_id} - 포트 매핑 실패로 기본 실행 상태로 설정")

        else:
            self._update_instance(instance_id, {"status": "running"})
            logger.info(f"인스턴스 {instance_id} - 기본 실행 상태로 설정")

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

    def destroy_instance(self, instance_id: str) -> bool:
        """인스턴스 삭제"""
        logger.info(f"인스턴스 {instance_id} 삭제 시작")

        start_time = time.time()

        success = self.vast_manager.destroy_instance(instance_id)

        if success:
            # 데이터베이스 업데이트
            self._update_instance(instance_id, {
                "status": "deleted",
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

# /service/storage/redis_version_manager.py
import redis
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class RedisVersionManager:
    """Redis를 사용한 버전 메타데이터 관리 (Dataset-Centric)"""
    
    def __init__(self, host: str = "192.168.2.242", port: int = 6379,
                 db: int = 0, password: Optional[str] = 'redis_secure_password123!'):
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=True
        )
        # ⭐ Prefix 정의 (Dataset-Centric)
        self.dataset_prefix = "dataset"  # Dataset 메타데이터용
        self.manager_prefix = "manager"  # Manager 세션용
        self.user_prefix = "user"        # User 관련
        
        logger.info("Redis 연결 완료 (Dataset-Centric 구조)")
    
    # ========== 버전 관리 (Dataset 기준) ==========
    
    def save_version_metadata(self, dataset_id: str, version: int,
                         version_info: Dict[str, Any]):
        """
        버전 메타데이터 저장 (자동으로 다음 버전 준비)
        
        Args:
            dataset_id: Dataset ID
            version: 저장할 버전 번호 (예: 0)
            version_info: 버전 메타데이터
        """
        try:
            # 버전별 상세 정보 저장
            version_key = f"{self.dataset_prefix}:{dataset_id}:version:{version}"
            self.redis_client.set(version_key, json.dumps(version_info))
            
            # 버전 리스트에 추가
            versions_key = f"{self.dataset_prefix}:{dataset_id}:versions"
            self.redis_client.rpush(versions_key, json.dumps(version_info))
            
            # ⭐ 수정: current_version을 다음 버전(version + 1)으로 설정
            current_key = f"{self.dataset_prefix}:{dataset_id}:current_version"
            next_version = version + 1
            self.redis_client.set(current_key, next_version)
            
            logger.info(f"✅ 버전 메타데이터 저장: {dataset_id} v{version} (다음 버전: v{next_version})")
            
        except redis.RedisError as e:
            logger.error(f"❌ Redis 저장 실패: {e}")
    
    def increment_version(self, dataset_id: str) -> int:
        """데이터셋의 버전 번호를 1 증가"""
        try:
            current_key = f"{self.dataset_prefix}:{dataset_id}:current_version"
            new_version = self.redis_client.incr(current_key)
            
            logger.info(f"버전 증가: {dataset_id} → v{new_version}")
            return new_version
            
        except redis.RedisError as e:
            logger.error(f"버전 증가 실패: {dataset_id}, {e}")
            return 0
    
    def get_current_version(self, dataset_id: str) -> int:
        """현재 버전 번호 조회"""
        try:
            current_key = f"{self.dataset_prefix}:{dataset_id}:current_version"
            version = self.redis_client.get(current_key)
            return int(version) if version else 0
            
        except redis.RedisError as e:
            logger.error(f"현재 버전 조회 실패: {e}")
            return 0

    def get_user_managers(self, user_id: str) -> List[str]:
        """
        사용자의 모든 활성 Manager ID 조회
        
        Args:
            user_id: 사용자 ID
            
        Returns:
            List[str]: Manager ID 리스트
        """
        try:
            active_key = f"{self.user_prefix}:{user_id}:active_managers"
            managers = list(self.redis_client.smembers(active_key))
            
            logger.info(f"사용자 {user_id}의 활성 매니저: {len(managers)}개")
            return managers
            
        except redis.RedisError as e:
            logger.error(f"사용자 매니저 조회 실패: {e}")
            logger.info(f"사용자 {user_id}의 활성 매니저: 0개")
            return []

    def get_version_metadata(self, dataset_id: str, version: int) -> Optional[Dict]:
        """특정 버전의 메타데이터 조회"""
        try:
            version_key = f"{self.dataset_prefix}:{dataset_id}:version:{version}"
            data = self.redis_client.get(version_key)
            
            if data:
                if isinstance(data, bytes):
                    data = data.decode('utf-8')
                return json.loads(data)
            
            return None
            
        except (json.JSONDecodeError, redis.RedisError) as e:
            logger.error(f"버전 메타데이터 조회 실패: {dataset_id} v{version}, {e}")
            return None
    
    def get_all_versions(self, dataset_id: str) -> List[Dict[str, Any]]:
        """데이터셋의 모든 버전 이력 조회"""
        try:
            versions_key = f"{self.dataset_prefix}:{dataset_id}:versions"
            versions = self.redis_client.lrange(versions_key, 0, -1)
            
            return [json.loads(v) for v in versions]
            
        except (redis.RedisError, json.JSONDecodeError) as e:
            logger.error(f"버전 이력 조회 실패: {e}")
            return []
    
    # ========== Dataset 관리 ==========
    
    def register_dataset(self, dataset_id: str, user_id: str, metadata: Dict):
        """데이터셋 등록"""
        try:
            # 데이터셋 소유자 저장
            owner_key = f"{self.dataset_prefix}:{dataset_id}:owner"
            self.redis_client.set(owner_key, user_id)
            
            # 사용자의 데이터셋 목록에 추가
            user_datasets_key = f"{self.user_prefix}:{user_id}:datasets"
            self.redis_client.sadd(user_datasets_key, dataset_id)
            
            # 메타데이터 저장
            metadata_key = f"{self.dataset_prefix}:{dataset_id}:metadata"
            self.redis_client.set(metadata_key, json.dumps(metadata))
            
            # 버전 초기화 (0부터 시작)
            version_key = f"{self.dataset_prefix}:{dataset_id}:current_version"
            self.redis_client.set(version_key, 0)
            
            logger.info(f"✅ 데이터셋 등록: {dataset_id} (owner: {user_id})")
            
        except redis.RedisError as e:
            logger.error(f"데이터셋 등록 실패: {e}")
            raise
    
    def get_dataset_owner(self, dataset_id: str) -> Optional[str]:
        """데이터셋 소유자 조회"""
        try:
            owner_key = f"{self.dataset_prefix}:{dataset_id}:owner"
            owner = self.redis_client.get(owner_key)
            return str(owner) if owner else None
            
        except redis.RedisError as e:
            logger.error(f"소유자 조회 실패: {e}")
            return None
    
    def get_dataset_metadata(self, dataset_id: str) -> Optional[Dict]:
        """데이터셋 메타데이터 조회"""
        try:
            metadata_key = f"{self.dataset_prefix}:{dataset_id}:metadata"
            data = self.redis_client.get(metadata_key)
            
            if data:
                return json.loads(data)
            return None
            
        except (redis.RedisError, json.JSONDecodeError) as e:
            logger.error(f"메타데이터 조회 실패: {e}")
            return None
    
    def get_user_datasets(self, user_id: str) -> List[str]:
        """사용자의 모든 데이터셋 ID 조회"""
        try:
            user_datasets_key = f"{self.user_prefix}:{user_id}:datasets"
            datasets = list(self.redis_client.smembers(user_datasets_key))
            
            logger.info(f"사용자 {user_id}의 데이터셋: {len(datasets)}개")
            return datasets
            
        except redis.RedisError as e:
            logger.error(f"사용자 데이터셋 조회 실패: {e}")
            return []
    
    def verify_dataset_ownership(self, dataset_id: str, user_id: str) -> bool:
        """데이터셋 소유권 확인"""
        try:
            owner = self.get_dataset_owner(dataset_id)
            return owner == str(user_id)
        except Exception as e:
            logger.error(f"소유권 확인 실패: {e}")
            return False
    
    # ========== Manager 세션 관리 ==========
    
    def link_manager_to_dataset(self, manager_id: str, dataset_id: Optional[str], user_id: str):
        """Manager를 Dataset에 연결 (세션 관리)"""
        try:
            # manager → dataset 매핑 (dataset_id가 None일 수도 있음)
            if dataset_id:
                mapping_key = f"{self.manager_prefix}:{manager_id}:dataset_id"
                self.redis_client.set(mapping_key, dataset_id)
            
            # manager → user 매핑
            owner_key = f"{self.manager_prefix}:{manager_id}:owner"
            self.redis_client.set(owner_key, user_id)
            
            # 생성 시간
            created_key = f"{self.manager_prefix}:{manager_id}:created_at"
            self.redis_client.set(created_key, datetime.now().isoformat())
            
            # 사용자의 활성 매니저 목록
            active_key = f"{self.user_prefix}:{user_id}:active_managers"
            self.redis_client.sadd(active_key, manager_id)
            
            logger.info(f"✅ Manager-Dataset 연결: {manager_id} → {dataset_id or 'None'}")
            
        except redis.RedisError as e:
            logger.error(f"Manager 연결 실패: {e}")
            raise
    
    def get_manager_dataset_id(self, manager_id: str) -> Optional[str]:
        """Manager의 Dataset ID 조회"""
        try:
            mapping_key = f"{self.manager_prefix}:{manager_id}:dataset_id"
            dataset_id = self.redis_client.get(mapping_key)
            return str(dataset_id) if dataset_id else None
            
        except redis.RedisError as e:
            logger.error(f"Manager dataset 조회 실패: {e}")
            return None
    
    def get_manager_owner(self, manager_id: str) -> Optional[str]:
        """Manager의 소유자 조회"""
        try:
            owner_key = f"{self.manager_prefix}:{manager_id}:owner"
            owner = self.redis_client.get(owner_key)
            
            if owner:
                if isinstance(owner, bytes):
                    return owner.decode('utf-8')
                return str(owner)
            return None
            
        except redis.RedisError as e:
            logger.error(f"Manager 소유자 조회 실패: {e}")
            return None
    
    def get_user_active_managers(self, user_id: str) -> List[str]:
        """사용자의 활성 Manager 목록"""
        try:
            active_key = f"{self.user_prefix}:{user_id}:active_managers"
            managers = list(self.redis_client.smembers(active_key))
            return managers
            
        except redis.RedisError as e:
            logger.error(f"활성 Manager 조회 실패: {e}")
            return []
    
    def unlink_manager(self, manager_id: str, user_id: str):
        """Manager 세션 해제"""
        try:
            # Manager 관련 키 삭제
            self.redis_client.delete(f"{self.manager_prefix}:{manager_id}:dataset_id")
            self.redis_client.delete(f"{self.manager_prefix}:{manager_id}:owner")
            self.redis_client.delete(f"{self.manager_prefix}:{manager_id}:created_at")
            
            # 활성 목록에서 제거
            active_key = f"{self.user_prefix}:{user_id}:active_managers"
            self.redis_client.srem(active_key, manager_id)
            
            logger.info(f"Manager 세션 해제: {manager_id}")
            
        except redis.RedisError as e:
            logger.error(f"Manager 해제 실패: {e}")
    
    # ========== 소스 정보 관리 ==========
    
    def save_source_info(self, dataset_id: str, source_info: Dict[str, Any]):
        """데이터셋 소스 정보 저장"""
        try:
            source_key = f"{self.dataset_prefix}:{dataset_id}:source_info"
            self.redis_client.set(source_key, json.dumps(source_info))
            
            # 소스 이력에도 추가
            history_key = f"{self.dataset_prefix}:{dataset_id}:source_history"
            self.redis_client.rpush(history_key, json.dumps(source_info))
            
            logger.info(f"소스 정보 저장: {dataset_id}")
            
        except redis.RedisError as e:
            logger.error(f"소스 정보 저장 실패: {e}")
    
    def get_source_info(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """최신 소스 정보 조회"""
        try:
            source_key = f"{self.dataset_prefix}:{dataset_id}:source_info"
            data = self.redis_client.get(source_key)
            
            if data:
                return json.loads(data)
            return None
            
        except (redis.RedisError, json.JSONDecodeError) as e:
            logger.error(f"소스 정보 조회 실패: {e}")
            return None
    
    def get_all_source_info(self, dataset_id: str) -> Optional[List[Dict[str, Any]]]:
        """모든 소스 정보 이력 조회"""
        try:
            history_key = f"{self.dataset_prefix}:{dataset_id}:source_history"
            history = self.redis_client.lrange(history_key, 0, -1)
            
            if history:
                return [json.loads(h) for h in history]
            return None
            
        except (redis.RedisError, json.JSONDecodeError) as e:
            logger.error(f"소스 이력 조회 실패: {e}")
            return None
    
    # ========== 전체 사용자 관리 ==========
    
    def get_all_users(self) -> List[str]:
        """모든 사용자 ID 조회"""
        try:
            # 모든 user:*:datasets 키 조회
            cursor = 0
            users = set()
            
            while True:
                cursor, keys = self.redis_client.scan(
                    cursor=cursor,
                    match=f"{self.user_prefix}:*:datasets",
                    count=100
                )
                
                for key in keys:
                    # user:USER_ID:datasets에서 USER_ID 추출
                    parts = key.split(':')
                    if len(parts) >= 2:
                        users.add(parts[1])
                
                if cursor == 0:
                    break
            
            return list(users)
            
        except redis.RedisError as e:
            logger.error(f"사용자 목록 조회 실패: {e}")
            return []
    
    # ========== 데이터 삭제 ==========
    
    def delete_dataset_metadata(self, dataset_id: str, user_id: str) -> bool:
        """데이터셋 메타데이터 완전 삭제"""
        try:
            # 소유권 확인
            owner = self.get_dataset_owner(dataset_id)
            if owner != user_id:
                logger.warning(f"소유권 없음: {dataset_id}")
                return False
            
            # 모든 버전 메타데이터 삭제
            versions = self.get_all_versions(dataset_id)
            for idx, _ in enumerate(versions):
                version_key = f"{self.dataset_prefix}:{dataset_id}:version:{idx}"
                self.redis_client.delete(version_key)
            
            # 데이터셋 관련 키 삭제
            self.redis_client.delete(f"{self.dataset_prefix}:{dataset_id}:versions")
            self.redis_client.delete(f"{self.dataset_prefix}:{dataset_id}:current_version")
            self.redis_client.delete(f"{self.dataset_prefix}:{dataset_id}:metadata")
            self.redis_client.delete(f"{self.dataset_prefix}:{dataset_id}:owner")
            self.redis_client.delete(f"{self.dataset_prefix}:{dataset_id}:source_info")
            self.redis_client.delete(f"{self.dataset_prefix}:{dataset_id}:source_history")
            
            # 사용자 데이터셋 목록에서 제거
            user_datasets_key = f"{self.user_prefix}:{user_id}:datasets"
            self.redis_client.srem(user_datasets_key, dataset_id)
            
            logger.info(f"데이터셋 메타데이터 삭제: {dataset_id}")
            return True
            
        except redis.RedisError as e:
            logger.error(f"메타데이터 삭제 실패: {e}")
            return False
    
    def delete_manager_metadata(self, manager_id: str, user_id: str):
        """Manager 메타데이터 삭제 (하위 호환성)"""
        try:
            # Manager의 dataset_id 조회
            dataset_id = self.get_manager_dataset_id(manager_id)
            
            if dataset_id:
                # Dataset 메타데이터 삭제
                self.delete_dataset_metadata(dataset_id, user_id)
            
            # Manager 세션 해제
            self.unlink_manager(manager_id, user_id)
            
            logger.info(f"Manager 메타데이터 삭제: {manager_id}")
            
        except Exception as e:
            logger.error(f"Manager 메타데이터 삭제 실패: {e}")
    
    # ========== MLflow 계보 관리 ==========
    
    def save_lineage(self, dataset_id: str, lineage_info: Dict[str, Any]):
        """데이터 계보 정보 저장"""
        try:
            lineage_key = f"{self.dataset_prefix}:{dataset_id}:lineage"
            self.redis_client.set(lineage_key, json.dumps(lineage_info))
            
        except redis.RedisError as e:
            logger.error(f"계보 정보 저장 실패: {e}")
    
    def get_lineage(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """데이터 계보 정보 조회"""
        try:
            lineage_key = f"{self.dataset_prefix}:{dataset_id}:lineage"
            data = self.redis_client.get(lineage_key)
            
            if data:
                return json.loads(data)
            return None
            
        except redis.RedisError as e:
            logger.error(f"계보 정보 조회 실패: {e}")
            return None
    
    def add_mlflow_run(self, dataset_id: str, run_id: str, run_info: Dict[str, Any]):
        """MLflow Run 정보 추가"""
        try:
            lineage = self.get_lineage(dataset_id) or {
                "mlflow_runs": []
            }
            
            lineage["mlflow_runs"].append({
                "run_id": run_id,
                "uploaded_at": datetime.now().isoformat(),
                **run_info
            })
            
            self.save_lineage(dataset_id, lineage)
            
        except redis.RedisError as e:
            logger.error(f"MLflow Run 정보 추가 실패: {e}")
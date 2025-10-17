# /service/data_manager/data_manager_register.py
import threading
from typing import Dict, Any, Optional, List
import gc
import logging
from service.data_manager.data_manager import DataManager
from service.storage.minio_client import MinioDataStorage
from service.storage.redis_version_manager import RedisVersionManager
from datetime import datetime

logger = logging.getLogger("data-manager-registry")

class DataManagerRegistry:
    """
    DataManager 인스턴스들을 등록/관리하는 레지스트리 클래스 (Dataset-Centric)
    - Dataset ID 중심의 버전 관리
    - Manager ID는 세션 관리용
    - API 재시작 시 자동 복원 (Lazy Loading)
    """

    def __init__(self):
        self.managers: Dict[str, DataManager] = {}
        self._lock = threading.Lock()
        
        # ========== 스토리지 클라이언트 초기화 ==========
        try:
            self.minio_storage = MinioDataStorage(
                endpoint="minio.x2bee.com",
                access_key="minioadmin",
                secret_key="minioadmin123",
            )
            logger.info("✅ MinIO 스토리지 초기화 완료")
        except Exception as e:
            logger.exception("❌ MinIO 초기화 실패")
            self.minio_storage = None
        
        try:
            self.redis_manager = RedisVersionManager(
                host="192.168.2.242",
                port=6379,
                password='redis_secure_password123!',
                db=0
            )
            logger.info("✅ Redis 버전 관리자 초기화 완료")
        except Exception as e:
            logger.error(f"❌ Redis 초기화 실패: {e}")
            self.redis_manager = None
        
        logger.info("DataManagerRegistry initialized with Dataset-Centric architecture")
        
        # ========== API 시작 시 메타데이터 로드 (Lazy Loading) ==========
        self._load_metadata_on_startup()

    def _load_metadata_on_startup(self, max_autoload_per_user: int = 3):
        """
        API 시작 시 Redis에서 메타데이터만 로드하되,
        복원이 가능해 보이는 매니저는 최대 max_autoload_per_user 개수만 메모리로 자동 복원
        """
        if not self.redis_manager:
            logger.warning("⚠️  Redis가 없어 메타데이터를 로드할 수 없습니다")
            return

        try:
            all_users = self.redis_manager.get_all_users()
            total_datasets = 0
            total_managers = 0
            restored_count = 0

            logger.info(f"메타데이터 초기 로드 시작: {len(all_users)}명 사용자")

            for user_id in all_users:
                dataset_ids = self.redis_manager.get_user_datasets(user_id)
                total_datasets += len(dataset_ids)

                manager_ids = self.redis_manager.get_user_active_managers(user_id)
                total_managers += len(manager_ids)

                logger.info(f"사용자 {user_id}: {len(dataset_ids)}개 데이터셋, {len(manager_ids)}개 활성 매니저 (Redis)")

                if not self.minio_storage:
                    logger.debug("MinIO 미초기화: 자동 복원 스킵")
                    continue

                autoloaded = 0
                for manager_id in manager_ids:
                    if autoloaded >= max_autoload_per_user:
                        logger.info(f"자동 복원 제한({max_autoload_per_user}) 도달: user={user_id}")
                        break

                    if manager_id in self.managers:
                        continue

                    try:
                        manager = self.get_manager(manager_id, user_id)
                        if manager:
                            restored_count += 1
                            autoloaded += 1
                            logger.info(f"자동 복원 성공: {manager_id} (user: {user_id})")
                        else:
                            logger.debug(f"자동 복원 불가: {manager_id} (user: {user_id})")
                    except Exception as e:
                        logger.warning(f"자동 복원 중 오류: manager={manager_id}, error={e}")

            logger.info(f"✅ 메타데이터 로드 완료: users={len(all_users)}, datasets={total_datasets}, managers={total_managers}, restored={restored_count}")
            logger.info("💡 실제 데이터는 (제한된 범위 내에서) 자동으로 메모리에 복원되었습니다")

        except Exception as e:
            logger.error(f"❌ 메타데이터 로드 실패: {e}", exc_info=True)

    def create_manager(self, user_id: str, user_name: str = "Unknown") -> str:
        """
        새로운 DataManager 인스턴스 생성 및 등록
        
        Args:
            user_id: 사용자 ID
            user_name: 사용자 이름
            
        Returns:
            str: 생성된 매니저 ID
        """
        with self._lock:
            # ✅ manager_id 없이 생성 (새 ID 자동 생성)
            manager = DataManager(
                user_id, 
                user_name,
                minio_storage=self.minio_storage,
                redis_manager=self.redis_manager
            )
            self.managers[manager.manager_id] = manager

            logger.info(f"✅ DataManager {manager.manager_id} created for user {user_name} ({user_id})")
            return manager.manager_id

    def get_manager(self, manager_id: str, user_id: str) -> Optional[DataManager]:
        """
        매니저 조회 - 메모리에 없으면 Redis/MinIO에서 자동 복원 (Lazy Loading)
        
        Args:
            manager_id: Manager ID
            user_id: 사용자 ID
            
        Returns:
            DataManager 또는 None
        """
        # ========== 1. 메모리에서 확인 ==========
        if manager_id in self.managers:
            logger.debug(f"✅ 메모리에서 매니저 조회: {manager_id}")
            return self.managers[manager_id]
        
        # ========== 2. Redis/MinIO에서 복원 시도 ==========
        logger.info(f"📦 Redis/MinIO에서 매니저 복원 시도: {manager_id}")
        
        if not self.redis_manager or not self.minio_storage:
            logger.error("❌ Redis 또는 MinIO가 초기화되지 않음")
            return None
        
        try:
            # Manager 소유자 확인
            owner = self.redis_manager.get_manager_owner(manager_id)
            if not owner:
                logger.warning(f"⚠️  Manager {manager_id}의 소유자 정보 없음")
                return None
            
            if str(owner) != str(user_id):
                logger.warning(f"⚠️  소유권 불일치: owner={owner}, user_id={user_id}")
                return None
            
            # Manager → Dataset 매핑 조회
            dataset_id = self.redis_manager.get_manager_dataset_id(manager_id)
            
            if not dataset_id:
                logger.warning(f"⚠️  Manager {manager_id}에 연결된 dataset 없음")
                return None
            
            logger.info(f"  └─ Manager {manager_id} → Dataset {dataset_id}")
            
            # Dataset 메타데이터 조회
            current_version = self.redis_manager.get_current_version(dataset_id)
            
            if current_version == 0:
                logger.warning(f"⚠️  Dataset {dataset_id}에 버전 정보 없음")
                return None
            
            # 최신 버전 메타데이터
            version_info = self.redis_manager.get_version_metadata(
                dataset_id, 
                current_version - 1
            )
            
            if not version_info:
                logger.warning(f"⚠️  버전 메타데이터 없음: {dataset_id} v{current_version - 1}")
                return None
            
            logger.info(f"  └─ 버전 메타데이터 발견: v{current_version - 1}")
            
            # MinIO에서 데이터 로드
            operation = version_info.get("operation", "unknown")
            
            try:
                table = self.minio_storage.load_version_snapshot(
                    dataset_id,
                    current_version - 1,
                    operation
                )
                logger.info(f"  └─ ✅ MinIO 데이터 로드: {table.num_rows} rows × {table.num_columns} cols")
            except Exception as e:
                logger.error(f"  └─ ❌ MinIO 로드 실패: {e}")
                return None
            
            if table is None or table.num_rows == 0:
                logger.error("❌ MinIO에서 유효한 데이터를 로드할 수 없음")
                return None
            
            # Dataset 메타데이터 조회
            dataset_metadata = self.redis_manager.get_dataset_metadata(dataset_id)
            user_name = dataset_metadata.get('created_by', 'Unknown') if dataset_metadata else 'Unknown'
            
            # ✅ DataManager 생성 시 manager_id 주입
            manager = DataManager(
                user_id=user_id,
                user_name=user_name,
                minio_storage=self.minio_storage,
                redis_manager=self.redis_manager,
                manager_id=manager_id  # ✅ 기존 Manager ID 전달
            )
            
            # ========== 상태 복원 ==========
            manager.dataset_id = dataset_id
            manager.dataset = table
            manager.current_version = current_version
            manager.viewing_version = current_version - 1
            
            # 소스 정보로부터 load_count 복원
            source_history = self.redis_manager.get_all_source_info(dataset_id)
            manager.dataset_load_count = len(source_history) if source_history else 1
            
            # Manager-Dataset 재연결 확인 (이미 연결되어 있어야 함)
            try:
                # Redis 연결 상태 재확인 (필요시)
                existing_dataset = self.redis_manager.get_manager_dataset_id(manager_id)
                if existing_dataset != dataset_id:
                    logger.warning(f"⚠️  Redis 연결 불일치 감지, 재연결: {manager_id} → {dataset_id}")
                    self.redis_manager.link_manager_to_dataset(
                        manager_id,
                        dataset_id,
                        user_id
                    )
            except Exception as e:
                logger.warning(f"⚠️  Redis 연결 확인 실패: {e}")
            
            # 메모리에 등록
            with self._lock:
                self.managers[manager_id] = manager
            
            logger.info(f"✅ Manager 복원 완료: {manager_id} (dataset: {dataset_id}, version: {current_version - 1})")
            return manager
            
        except Exception as e:
            logger.error(f"❌ Manager 복원 실패: {manager_id}, {e}", exc_info=True)
            return None

    def remove_manager(self, manager_id: str, user_id: str) -> bool:
        """
        매니저 제거 (메모리 + Redis 모두 처리)
        
        Args:
            manager_id: Manager ID
            user_id: 사용자 ID
            
        Returns:
            bool: 성공 여부
        """
        try:
            # ========== 1. 소유권 확인 (Redis 우선) ==========
            if not self.redis_manager:
                logger.error("❌ Redis가 연결되지 않음")
                return False
            
            # Redis에서 소유자 확인
            owner = self.redis_manager.get_manager_owner(manager_id)
            if not owner:
                logger.warning(f"⚠️  Manager {manager_id}를 찾을 수 없습니다 (Redis)")
                return False
            
            if str(owner) != str(user_id):
                logger.warning(f"⚠️  소유권 없음: owner={owner}, user_id={user_id}")
                return False
            
            logger.info(f"✅ 소유권 확인: Manager {manager_id} (owner: {owner})")
            
            # ========== 2. 메모리에서 제거 (있으면) ==========
            if manager_id in self.managers:
                with self._lock:
                    manager = self.managers[manager_id]
                    
                    # Manager cleanup
                    try:
                        manager.is_active = False
                        manager._monitoring = False
                        manager.dataset = None
                    except Exception as e:
                        logger.warning(f"Manager cleanup 실패: {e}")
                    
                    del self.managers[manager_id]
                    logger.info(f"🗑️  메모리에서 Manager 제거: {manager_id}")
            else:
                logger.info(f"💡 Manager {manager_id}는 메모리에 없음 (Redis에만 존재)")
            
            # ========== 3. Redis에서 Manager 세션 해제 ==========
            try:
                self.redis_manager.unlink_manager(manager_id, user_id)
                logger.info(f"✅ Redis에서 Manager 세션 해제: {manager_id}")
            except Exception as e:
                logger.warning(f"⚠️  Redis Manager 해제 실패: {e}")
            
            logger.info(f"✅ Manager {manager_id} 제거 완료")
            return True
            
        except Exception as e:
            logger.error(f"❌ Manager 제거 실패: {manager_id}, {e}", exc_info=True)
            return False

    def list_managers(self, user_id: str = None) -> Dict[str, Any]:
        """
        매니저 목록 반환 (메타데이터만, 복원 안 함)
        
        Args:
            user_id: 특정 사용자의 매니저만 반환 (None이면 전체)
            
        Returns:
            Dict[str, Any]: 매니저 목록
        """
        with self._lock:
            result = []

            # ========== 1. 메모리에 있는 매니저 (활성) ==========
            for manager_id, manager in self.managers.items():
                if user_id is None or manager.user_id == user_id:
                    stats = manager.get_resource_stats()
                    stats['in_memory'] = True
                    stats['status'] = 'active'
                    result.append(stats)
            
            # ========== 2. Redis에만 있는 매니저 (비활성) ==========
            if self.redis_manager and user_id:
                try:
                    # 사용자의 활성 Manager 목록
                    active_managers = self.redis_manager.get_user_active_managers(user_id)
                    
                    for manager_id in active_managers:
                        # 이미 메모리에 있으면 스킵
                        if manager_id in self.managers:
                            continue
                        
                        # Manager → Dataset 매핑
                        dataset_id = self.redis_manager.get_manager_dataset_id(manager_id)
                        if not dataset_id:
                            continue
                        
                        # Dataset 정보만 조회 (복원 안 함)
                        current_version = self.redis_manager.get_current_version(dataset_id)
                        if current_version == 0:
                            continue
                        
                        version_info = self.redis_manager.get_version_metadata(
                            dataset_id, current_version - 1
                        )
                        
                        if version_info:
                            source_info = self.redis_manager.get_source_info(dataset_id)
                            
                            result.append({
                                'manager_id': manager_id,
                                'dataset_id': dataset_id,
                                'user_id': user_id,
                                'user_name': 'Unknown',
                                'in_memory': False,
                                'status': 'stored',
                                'has_dataset': True,
                                'dataset_rows': version_info.get('num_rows', 0),
                                'dataset_columns': version_info.get('num_columns', 0),
                                'current_version': current_version - 1,
                                'last_operation': version_info.get('operation'),
                                'last_updated': version_info.get('timestamp'),
                                'source_type': source_info.get('type') if source_info else None,
                                'storage_location': 'redis+minio'
                            })
                
                except Exception as e:
                    logger.warning(f"⚠️  Redis에서 매니저 목록 조회 실패: {e}")

            return {
                "success": True,
                "user_id": user_id,
                "managers": result,
                "total_count": len(result),
                "in_memory_count": sum(1 for m in result if m.get('in_memory')),
                "stored_count": sum(1 for m in result if not m.get('in_memory'))
            }

    def get_total_stats(self) -> Dict[str, Any]:
        """전체 매니저들의 통계 정보 반환"""
        with self._lock:
            total_managers_memory = len(self.managers)
            active_managers = sum(1 for m in self.managers.values() if m.is_active)
            total_datasets = sum(1 for m in self.managers.values() if m.dataset is not None)
            
            # Redis에서 전체 통계
            total_users = 0
            total_datasets_redis = 0
            total_managers_redis = 0
            
            if self.redis_manager:
                try:
                    all_users = self.redis_manager.get_all_users()
                    total_users = len(all_users)
                    
                    for user_id in all_users:
                        dataset_ids = self.redis_manager.get_user_datasets(user_id)
                        total_datasets_redis += len(dataset_ids)
                        
                        manager_ids = self.redis_manager.get_user_active_managers(user_id)
                        total_managers_redis += len(manager_ids)
                        
                except Exception as e:
                    logger.warning(f"Redis 통계 조회 실패: {e}")

            return {
                'total_managers_in_memory': total_managers_memory,
                'active_managers': active_managers,
                'total_datasets_in_memory': total_datasets,
                'total_users': total_users,
                'total_datasets_in_redis': total_datasets_redis,
                'total_managers_in_redis': total_managers_redis,
                'version_management_enabled': self.minio_storage is not None and self.redis_manager is not None,
                'storage_status': {
                    'redis': 'connected' if self.redis_manager else 'disconnected',
                    'minio': 'connected' if self.minio_storage else 'disconnected'
                }
            }

    def cleanup_inactive_managers(self, max_age_seconds: int = 3600):
        """
        비활성 매니저 정리 (메모리 절약)
        
        Args:
            max_age_seconds: 최대 유지 시간 (초)
        """
        now = datetime.now()
        to_remove = []
        
        with self._lock:
            for manager_id, manager in self.managers.items():
                age = (now - manager.created_at).total_seconds()
                
                # 오래된 비활성 매니저
                if age > max_age_seconds and not manager.is_active:
                    logger.info(f"🧹 비활성 매니저 정리: {manager_id} (나이: {age:.0f}초)")
                    
                    # Redis에 상태 저장
                    if self.redis_manager and manager.dataset_id:
                        try:
                            self.redis_manager.link_manager_to_dataset(
                                manager_id,
                                manager.dataset_id,
                                manager.user_id
                            )
                        except Exception as e:
                            logger.warning(f"Redis 저장 실패: {e}")
                    
                    to_remove.append(manager_id)
        
        # 메모리에서 제거
        for manager_id in to_remove:
            with self._lock:
                if manager_id in self.managers:
                    del self.managers[manager_id]
        
        if to_remove:
            gc.collect()
            logger.info(f"✅ {len(to_remove)}개 매니저 정리 완료")

    def cleanup(self):
        """레지스트리 정리 - 모든 매니저 정리 및 리소스 해제"""
        logger.info("🧹 DataManagerRegistry 정리 시작...")

        with self._lock:
            # 모든 매니저들을 안전하게 정리
            manager_ids = list(self.managers.keys())
            
            for manager_id in manager_ids:
                try:
                    manager = self.managers[manager_id]
                    if manager:
                        # Manager-Dataset 연결 유지 (Redis)
                        if self.redis_manager and manager.dataset_id:
                            try:
                                self.redis_manager.link_manager_to_dataset(
                                    manager_id,
                                    manager.dataset_id,
                                    manager.user_id
                                )
                            except Exception as e:
                                logger.warning(f"Redis 저장 실패: {e}")
                        
                        # Manager cleanup (메모리만 정리)
                        manager.is_active = False
                        manager._monitoring = False
                        manager.dataset = None
                        
                        logger.info(f"Manager {manager_id} cleaned up")
                        
                except Exception as e:
                    logger.error(f"Manager {manager_id} 정리 실패: {e}")

            # 매니저 딕셔너리 비우기
            self.managers.clear()

        # 강제 가비지 컬렉션
        gc.collect()

        logger.info("✅ DataManagerRegistry cleanup completed")

    def __del__(self):
        """소멸자"""
        if hasattr(self, 'managers') and self.managers:
            try:
                self.cleanup()
            except Exception as e:
                logger.error(f"소멸자에서 정리 실패: {e}")
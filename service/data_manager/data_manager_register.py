import threading
from typing import Dict, Any, Optional
import gc
import logging
from service.data_manager.data_manager import DataManager

logger = logging.getLogger("data-manager-registry")

class DataManagerRegistry:
    """
    DataManager 인스턴스들을 등록/관리하는 레지스트리 클래스
    """

    def __init__(self):
        self.managers: Dict[str, DataManager] = {}
        self._lock = threading.Lock()
        logger.info("DataManagerRegistry initialized")

    def create_manager(self, user_id: str, user_name: str = "Unknown") -> str:
        """
        새로운 DataManager 인스턴스 생성 및 등록

        Args:
            user_id (str): 사용자 ID
            user_name (str): 사용자 이름

        Returns:
            str: 생성된 매니저 ID
        """
        with self._lock:
            manager = DataManager(user_id, user_name)
            self.managers[manager.manager_id] = manager

            logger.info(f"DataManager {manager.manager_id} registered for user {user_name} ({user_id})")
            return manager.manager_id

    def get_manager(self, manager_id: str, user_id: str) -> Optional[DataManager]:
        """
        매니저 ID와 사용자 ID로 DataManager 인스턴스 반환

        Args:
            manager_id (str): 매니저 ID
            user_id (str): 사용자 ID

        Returns:
            Optional[DataManager]: DataManager 인스턴스 또는 None
        """
        with self._lock:
            manager = self.managers.get(manager_id)

            # 사용자 ID 검증
            if manager and manager.user_id != user_id:
                logger.warning(f"Access denied: user {user_id} tried to access manager {manager_id}")
                return None

            return manager

    def remove_manager(self, manager_id: str, user_id: str) -> bool:
        """
        DataManager 인스턴스 제거

        Args:
            manager_id (str): 매니저 ID
            user_id (str): 사용자 ID

        Returns:
            bool: 제거 성공 여부
        """
        with self._lock:
            manager = self.managers.get(manager_id)

            if not manager:
                return False

            # 사용자 ID 검증
            if manager.user_id != user_id:
                logger.warning(f"Access denied: user {user_id} tried to remove manager {manager_id}")
                return False

            # 매니저 정리
            manager.cleanup()
            del self.managers[manager_id]

            logger.info(f"DataManager {manager_id} removed successfully")
            return True

    def list_managers(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        매니저 목록 반환

        Args:
            user_id (str, optional): 특정 사용자의 매니저만 반환

        Returns:
            Dict[str, Any]: 매니저 목록
        """
        with self._lock:
            result = {}

            for manager_id, manager in self.managers.items():
                if user_id is None or manager.user_id == user_id:
                    result[manager_id] = manager.get_resource_stats()

            return result

    def get_total_stats(self) -> Dict[str, Any]:
        """전체 매니저들의 통계 정보 반환"""
        with self._lock:
            total_managers = len(self.managers)
            active_managers = sum(1 for m in self.managers.values() if m.is_active)
            total_datasets = sum(1 for m in self.managers.values() if m.dataset is not None)

            return {
                'total_managers': total_managers,
                'active_managers': active_managers,
                'total_datasets': total_datasets,
                'managers_by_user': {}
            }

    def cleanup(self):
        """레지스트리 정리 - 모든 매니저 정리 및 리소스 해제"""
        logger.info("Cleaning up DataManagerRegistry...")

        with self._lock:
            # 모든 매니저들을 안전하게 정리
            manager_ids = list(self.managers.keys())
            for manager_id in manager_ids:
                try:
                    manager = self.managers[manager_id]
                    if manager:
                        manager.cleanup()
                        logger.info(f"Manager {manager_id} cleaned up")
                except Exception as e:
                    logger.error(f"Error cleaning up manager {manager_id}: {e}")

            # 매니저 딕셔너리 비우기
            self.managers.clear()

        # 강제 가비지 컬렉션
        gc.collect()

        logger.info("DataManagerRegistry cleanup completed")

    def __del__(self):
        if hasattr(self, 'managers') and self.managers:
            self.cleanup()

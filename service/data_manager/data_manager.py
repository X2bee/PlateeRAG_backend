import uuid
import psutil
import os
import threading
import time
from typing import Dict, Any, Optional
from datetime import datetime
import gc
import weakref
from huggingface_hub import hf_hub_download, list_repo_files
import logging

logger = logging.getLogger(__name__)

class DataManager:
    """
    Data Manager Instance Class
    - 단일 인스턴스로 동작하며 CPU/Memory 사용량을 추적
    - UUID 기반 고유 ID 생성
    - 사용자 ID 기반 접근 제어
    - Huggingface 데이터 관리
    """

    def __init__(self, user_id: str):
        """
        DataManager 인스턴스 초기화

        Args:
            user_id (str): 사용자 ID
        """
        self.manager_id = str(uuid.uuid4())
        self.user_id = user_id
        self.created_at = datetime.now()
        self.is_active = True

        # 리소스 모니터링
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self.process.memory_info().rss
        self.initial_cpu_percent = self.process.cpu_percent()

        # 데이터 저장소
        self.datasets: Dict[str, Any] = {}
        self.models: Dict[str, Any] = {}
        self.files: Dict[str, Any] = {}

        # 모니터링 스레드
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self._resource_stats = {
            'cpu_usage': [],
            'memory_usage': [],
            'peak_memory': self.initial_memory,
            'average_cpu': 0.0
        }

        self._monitor_thread.start()

        logger.info(f"DataManager {self.manager_id} created for user {self.user_id}")

    def _monitor_resources(self):
        """리소스 사용량 모니터링 스레드"""
        cpu_samples = []

        while self._monitoring and self.is_active:
            try:
                # CPU 사용률 (백분율)
                cpu_percent = self.process.cpu_percent(interval=0.1)
                cpu_samples.append(cpu_percent)

                # 메모리 사용량 (바이트)
                memory_info = self.process.memory_info()
                current_memory = memory_info.rss

                self._resource_stats['cpu_usage'].append(cpu_percent)
                self._resource_stats['memory_usage'].append(current_memory)

                # 피크 메모리 업데이트
                if current_memory > self._resource_stats['peak_memory']:
                    self._resource_stats['peak_memory'] = current_memory

                # 평균 CPU 계산
                if cpu_samples:
                    self._resource_stats['average_cpu'] = sum(cpu_samples) / len(cpu_samples)

                # 최근 100개 샘플만 유지
                if len(self._resource_stats['cpu_usage']) > 100:
                    self._resource_stats['cpu_usage'] = self._resource_stats['cpu_usage'][-100:]
                    self._resource_stats['memory_usage'] = self._resource_stats['memory_usage'][-100:]
                    cpu_samples = cpu_samples[-100:]

                time.sleep(1)  # 1초마다 체크

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                logger.warning(f"Process monitoring failed for DataManager {self.manager_id}")
                break
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                break

    def get_resource_stats(self) -> Dict[str, Any]:
        """현재 리소스 사용량 통계 반환"""
        current_memory = self.process.memory_info().rss if self.is_active else 0
        current_cpu = self.process.cpu_percent() if self.is_active else 0

        return {
            'manager_id': self.manager_id,
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat(),
            'is_active': self.is_active,
            'current_memory_mb': current_memory / (1024 * 1024),
            'current_cpu_percent': current_cpu,
            'peak_memory_mb': self._resource_stats['peak_memory'] / (1024 * 1024),
            'average_cpu_percent': self._resource_stats['average_cpu'],
            'memory_delta_mb': (current_memory - self.initial_memory) / (1024 * 1024),
            'datasets_count': len(self.datasets),
            'models_count': len(self.models),
            'files_count': len(self.files)
        }

    def download_huggingface_file(self, repo_id: str, filename: str, **kwargs) -> str:
        """
        Huggingface Hub에서 파일 다운로드

        Args:
            repo_id (str): 리포지토리 ID
            filename (str): 파일명
            **kwargs: 추가 다운로드 옵션

        Returns:
            str: 다운로드된 파일 경로
        """
        if not self.is_active:
            raise RuntimeError("DataManager is not active")

        try:
            logger.info(f"Downloading file {filename} from {repo_id} for manager {self.manager_id}")

            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                **kwargs
            )

            file_key = f"{repo_id}_{filename}"

            self.files[file_key] = {
                'repo_id': repo_id,
                'filename': filename,
                'local_path': file_path,
                'downloaded_at': datetime.now(),
                'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0
            }

            logger.info(f"File {filename} downloaded successfully")
            return file_path

        except Exception as e:
            logger.error(f"Failed to download file {filename}: {e}")
            raise RuntimeError(f"File download failed: {str(e)}")

    def list_datasets(self) -> Dict[str, Any]:
        """로드된 데이터셋 목록 반환"""
        return {
            key: {
                'name': info['name'],
                'config': info['config'],
                'split': info['split'],
                'size': info['size'],
                'loaded_at': info['loaded_at'].isoformat()
            }
            for key, info in self.datasets.items()
        }

    def list_files(self) -> Dict[str, Any]:
        """다운로드된 파일 목록 반환"""
        return {
            key: {
                'repo_id': info['repo_id'],
                'filename': info['filename'],
                'local_path': info['local_path'],
                'file_size': info['file_size'],
                'downloaded_at': info['downloaded_at'].isoformat()
            }
            for key, info in self.files.items()
        }

    def get_dataset(self, dataset_key: str) -> Any:
        """데이터셋 객체 반환"""
        if not self.is_active:
            raise RuntimeError("DataManager is not active")

        if dataset_key not in self.datasets:
            raise KeyError(f"Dataset {dataset_key} not found")

        return self.datasets[dataset_key]['dataset']

    def remove_dataset(self, dataset_key: str) -> bool:
        """데이터셋 제거"""
        if dataset_key in self.datasets:
            del self.datasets[dataset_key]
            gc.collect()  # 가비지 컬렉션 강제 실행
            logger.info(f"Dataset {dataset_key} removed from manager {self.manager_id}")
            return True
        return False

    def remove_file(self, file_key: str) -> bool:
        """파일 정보 제거 (실제 파일은 삭제하지 않음)"""
        if file_key in self.files:
            del self.files[file_key]
            logger.info(f"File {file_key} removed from manager {self.manager_id}")
            return True
        return False

    def cleanup(self):
        """리소스 정리 및 매니저 종료"""
        logger.info(f"Cleaning up DataManager {self.manager_id}")

        self.is_active = False
        self._monitoring = False

        # 모니터링 스레드 종료 대기
        if self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2)

        # 데이터 정리
        self.datasets.clear()
        self.models.clear()
        self.files.clear()

        # 강제 가비지 컬렉션
        gc.collect()

        logger.info(f"DataManager {self.manager_id} cleaned up successfully")

    def __del__(self):
        """소멸자 - 자동 정리"""
        if hasattr(self, 'is_active') and self.is_active:
            self.cleanup()


class DataManagerRegistry:
    """
    DataManager 인스턴스들을 등록/관리하는 레지스트리 클래스
    """

    def __init__(self):
        self.managers: Dict[str, DataManager] = {}
        self._lock = threading.Lock()
        logger.info("DataManagerRegistry initialized")

    def create_manager(self, user_id: str) -> str:
        """
        새로운 DataManager 인스턴스 생성 및 등록

        Args:
            user_id (str): 사용자 ID

        Returns:
            str: 생성된 매니저 ID
        """
        with self._lock:
            manager = DataManager(user_id)
            self.managers[manager.manager_id] = manager

            logger.info(f"DataManager {manager.manager_id} registered for user {user_id}")
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
            total_datasets = sum(len(m.datasets) for m in self.managers.values())
            total_files = sum(len(m.files) for m in self.managers.values())

            return {
                'total_managers': total_managers,
                'active_managers': active_managers,
                'total_datasets': total_datasets,
                'total_files': total_files,
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

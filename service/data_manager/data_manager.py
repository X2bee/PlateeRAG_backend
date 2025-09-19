import uuid
import os
import threading
import time
import sys
from typing import Dict, Any, Optional
from datetime import datetime
import gc
from huggingface_hub import hf_hub_download, list_repo_files
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
import json
import logging

logger = logging.getLogger(__name__)

class DataManager:
    """
    Data Manager Instance Class
    - 단일 데이터셋 저장 및 메모리 사용량 추적
    - UUID 기반 고유 ID 생성
    - 사용자 ID 기반 접근 제어
    """

    def __init__(self, user_id: str, user_name: str = "Unknown"):
        """
        DataManager 인스턴스 초기화

        Args:
            user_id (str): 사용자 ID
            user_name (str): 사용자 이름
        """
        self.manager_id = str(uuid.uuid4())
        self.user_id = user_id
        self.user_name = user_name
        self.created_at = datetime.now()
        self.is_active = True

        # 리소스 모니터링 - DataManager 인스턴스 메모리 추적
        # 인스턴스 생성 전 메모리 측정
        gc.collect()
        self.initial_memory = self._get_object_memory_size()        # 데이터 저장소
        self.dataset: Any = None

        # 모니터링 스레드
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_instance_memory, daemon=True)
        self._resource_stats = {
            'instance_memory_usage': [],
            'dataset_memory': [],
            'peak_instance_memory': self.initial_memory,
            'current_instance_memory': self.initial_memory
        }

        self._monitor_thread.start()

        logger.info(f"DataManager {self.manager_id} created for user {self.user_name} ({self.user_id}) (Initial memory: {self.initial_memory / (1024 * 1024):.2f} MB)")

    def _get_object_memory_size(self) -> int:
        """DataManager 인스턴스의 메모리 사용량 계산"""
        total_size = 0

        # 인스턴스 자체의 크기
        total_size += sys.getsizeof(self)

        # 각 속성의 메모리 크기 계산
        for attr_name in dir(self):
            if not attr_name.startswith('_') or attr_name in ['_resource_stats']:
                try:
                    attr_value = getattr(self, attr_name)
                    total_size += self._calculate_deep_size(attr_value)
                except:
                    continue

        return total_size

    def _calculate_deep_size(self, obj, seen=None) -> int:
        """객체의 전체 메모리 크기를 재귀적으로 계산"""
        if seen is None:
            seen = set()

        obj_id = id(obj)
        if obj_id in seen:
            return 0

        seen.add(obj_id)
        size = sys.getsizeof(obj)

        if isinstance(obj, dict):
            for key, value in obj.items():
                size += self._calculate_deep_size(key, seen)
                size += self._calculate_deep_size(value, seen)
        elif isinstance(obj, (list, tuple, set, frozenset)):
            for item in obj:
                size += self._calculate_deep_size(item, seen)
        elif hasattr(obj, '__dict__'):
            for attr_value in obj.__dict__.values():
                size += self._calculate_deep_size(attr_value, seen)
        elif hasattr(obj, '__slots__'):
            for slot in obj.__slots__:
                if hasattr(obj, slot):
                    size += self._calculate_deep_size(getattr(obj, slot), seen)

        return size

    def _monitor_instance_memory(self):
        """DataManager 인스턴스 메모리 사용량 모니터링 스레드"""
        while self._monitoring and self.is_active:
            try:
                # 가비지 컬렉션 후 정확한 메모리 측정
                gc.collect()

                # 인스턴스 전체 메모리
                current_instance_memory = self._get_object_memory_size()

                # 개별 데이터 구조 메모리
                dataset_memory = self._calculate_deep_size(self.dataset) if self.dataset is not None else 0

                # 통계 업데이트
                self._resource_stats['instance_memory_usage'].append(current_instance_memory)
                self._resource_stats['dataset_memory'].append(dataset_memory)
                self._resource_stats['current_instance_memory'] = current_instance_memory

                # 피크 메모리 업데이트
                if current_instance_memory > self._resource_stats['peak_instance_memory']:
                    self._resource_stats['peak_instance_memory'] = current_instance_memory

                # 최근 100개 샘플만 유지
                for key in ['instance_memory_usage', 'dataset_memory']:
                    if len(self._resource_stats[key]) > 100:
                        self._resource_stats[key] = self._resource_stats[key][-100:]

                time.sleep(2)  # 2초마다 체크 (메모리 측정은 CPU보다 무거움)

            except Exception as e:
                logger.error(f"Instance memory monitoring error: {e}")
                break

    def get_resource_stats(self) -> Dict[str, Any]:
        """DataManager 인스턴스의 메모리 사용량 통계 반환"""
        current_instance_memory = self._resource_stats['current_instance_memory']

        # 데이터셋 메모리
        dataset_memory = self._calculate_deep_size(self.dataset) if self.dataset is not None else 0

        return {
            'manager_id': self.manager_id,
            'user_id': self.user_id,
            'user_name': self.user_name,
            'created_at': self.created_at.isoformat(),
            'is_active': self.is_active,

            # DataManager 인스턴스 메모리 정보
            'current_instance_memory_mb': current_instance_memory / (1024 * 1024),
            'initial_instance_memory_mb': self.initial_memory / (1024 * 1024),
            'peak_instance_memory_mb': self._resource_stats['peak_instance_memory'] / (1024 * 1024),
            'memory_growth_mb': (current_instance_memory - self.initial_memory) / (1024 * 1024),

            # 데이터셋 메모리
            'dataset_memory_mb': dataset_memory / (1024 * 1024),

            # 데이터 상태
            'has_dataset': self.dataset is not None,

            # 메모리 분포 (퍼센트)
            'memory_distribution': {
                'dataset_percent': (dataset_memory / current_instance_memory * 100) if current_instance_memory > 0 else 0,
                'other_percent': ((current_instance_memory - dataset_memory) / current_instance_memory * 100) if current_instance_memory > 0 else 0
            }
        }

    def get_dataset(self) -> Any:
        """데이터셋 반환"""
        if not self.is_active:
            raise RuntimeError("DataManager is not active")

        return self.dataset

    def get_dataset_sample(self, num_samples: int = 10) -> Dict[str, Any]:
        if not self.is_active:
            raise RuntimeError("DataManager is not active")

        if self.dataset is None:
            return {
                "success": True,
                "sample_data": [],
                "sample_count": 0,
                "total_rows": 0,
                "total_columns": 0,
                "columns": [],
                "column_info": "",
                "sampled_at": datetime.now().isoformat()
            }


        try:
            # pyarrow Table에서 상위 N개 행 가져오기
            actual_samples = min(num_samples, self.dataset.num_rows)
            sample_table = self.dataset.slice(0, actual_samples)

            # 각 컬럼의 데이터 타입 정보
            column_info = {}
            for i, col_name in enumerate(sample_table.column_names):
                col_type = str(sample_table.schema.field(i).type)
                column_info[col_name] = {
                    "type": col_type,
                    "nullable": sample_table.schema.field(i).nullable
                }

            # pyarrow Table을 Python 딕셔너리 리스트로 변환
            sample_records = []
            for row_idx in range(sample_table.num_rows):
                record = {}
                for col_idx, col_name in enumerate(sample_table.column_names):
                    value = sample_table.column(col_idx)[row_idx].as_py()
                    # None 값과 특수 값들을 JSON 호환 형태로 변환
                    if value is None:
                        record[col_name] = None
                    else:
                        record[col_name] = value
                sample_records.append(record)

            return {
                "success": True,
                "sample_data": sample_records,
                "sample_count": len(sample_records),
                "total_rows": self.dataset.num_rows,
                "total_columns": self.dataset.num_columns,
                "columns": sample_table.column_names,
                "column_info": column_info,
                "sampled_at": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get dataset sample: {e}")
            raise RuntimeError(f"Dataset sample retrieval failed: {str(e)}")

    def set_dataset(self, dataset: Any) -> None:
        """데이터셋 설정"""
        if not self.is_active:
            raise RuntimeError("DataManager is not active")

        self.dataset = dataset
        logger.info(f"Dataset set for manager {self.manager_id}")

    def remove_dataset(self) -> bool:
        """데이터셋 제거"""
        if self.dataset is not None:
            self.dataset = None
            gc.collect()  # 가비지 컬렉션 강제 실행
            logger.info(f"Dataset removed from manager {self.manager_id}")
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
        self.dataset = None

        # 강제 가비지 컬렉션
        gc.collect()

        logger.info(f"DataManager {self.manager_id} cleaned up successfully")

    def __del__(self):
        """소멸자 - 자동 정리"""
        if hasattr(self, 'is_active') and self.is_active:
            self.cleanup()


    def download_and_load_dataset(self, repo_id: str, filename: str = None, split: str = None) -> Dict[str, Any]:
        """
        Huggingface repo에서 데이터셋을 다운로드하고 pyarrow로 적재

        Args:
            repo_id (str): Huggingface 리포지토리 ID
            filename (str, optional): 특정 파일명. None이면 자동 탐색
            split (str, optional): 데이터 분할 (train, validation, test 등)

        Returns:
            Dict[str, Any]: 다운로드 및 로드 결과 정보
        """
        if not self.is_active:
            raise RuntimeError("DataManager is not active")

        try:
            # huggingface_cache 디렉토리 생성
            cache_dir = "/huggingface_cache"
            os.makedirs(cache_dir, exist_ok=True)

            logger.info(f"Starting dataset download from {repo_id} for manager {self.manager_id}")

            # 파일명이 지정되지 않은 경우 repo의 파일 목록에서 parquet 파일 찾기
            if filename is None:
                repo_files = list_repo_files(repo_id, repo_type='dataset')
                parquet_files = [f for f in repo_files if f.endswith('.parquet')]

                if not parquet_files:
                    raise RuntimeError(f"No parquet files found in repository {repo_id}")

                # split이 지정된 경우 해당 split 파일 찾기
                if split:
                    split_files = [f for f in parquet_files if split in f.lower()]
                    filename = split_files[0] if split_files else parquet_files[0]
                else:
                    filename = parquet_files[0]

            # 파일 다운로드
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type='dataset',
                cache_dir=cache_dir,
            )

            logger.info(f"File downloaded: {downloaded_path}")

            # pyarrow로 parquet 파일 읽기
            table = pq.read_table(downloaded_path)

            # self.dataset에 저장
            self.dataset = table

            logger.info(f"Dataset loaded successfully. Shape: {table.shape}, Columns: {table.column_names}")

            return {
                "success": True,
                "repo_id": repo_id,
                "filename": filename,
                "local_path": downloaded_path,
                "table_shape": table.shape,
                "columns": table.column_names,
                "num_rows": table.num_rows,
                "num_columns": table.num_columns,
                "loaded_at": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to download and load dataset from {repo_id}: {e}")
            raise RuntimeError(f"Dataset download/load failed: {str(e)}")

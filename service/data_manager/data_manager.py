import uuid
import os
import threading
import time
import sys
from typing import Dict, Any, Optional, List
from datetime import datetime
import gc
from huggingface_hub import hf_hub_download, list_repo_files
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as csv
from .data_manager_helper import (
    save_and_load_files,
    classify_dataset_files,
    download_and_read_file,
    process_multiple_files,
    combine_tables,
    determine_file_type_from_filename,
    create_result_info,
    generate_dataset_statistics,
    drop_columns_from_table,
    replace_column_values,
    apply_column_operation,
    remove_null_rows,
    upload_dataset_to_hf,
    copy_column,
    rename_column,
    format_columns_string,
    calculate_columns_operation,
    execute_safe_callback
)
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
        """DataManager 인스턴스의 메모리 사용량 계산 - 개선된 버전"""
        total_size = 0

        # 인스턴스 자체의 기본 크기
        total_size += sys.getsizeof(self)

        # 주요 속성들만 선별적으로 계산 (순환 참조 방지)
        safe_attrs = [
            'manager_id', 'user_id', 'user_name', 'created_at',
            'is_active', 'initial_memory', 'dataset'
        ]

        for attr_name in safe_attrs:
            if hasattr(self, attr_name):
                try:
                    attr_value = getattr(self, attr_name)
                    if attr_name == 'dataset' and attr_value is not None:
                        # 데이터셋은 별도로 정확하게 계산
                        total_size += self._calculate_dataset_memory_size(attr_value)
                    else:
                        total_size += self._calculate_deep_size(attr_value)
                except Exception as e:
                    logger.warning(f"메모리 계산 실패 ({attr_name}): {e}")
                    continue

        # _resource_stats는 별도로 간단하게 계산
        try:
            stats_size = sys.getsizeof(self._resource_stats)
            for key, value in self._resource_stats.items():
                stats_size += sys.getsizeof(key) + sys.getsizeof(value)
                if isinstance(value, list):
                    stats_size += sum(sys.getsizeof(item) for item in value[-10:])  # 최근 10개만
            total_size += stats_size
        except:
            pass

        return total_size

    def _calculate_dataset_memory_size(self, dataset) -> int:
        """데이터셋 전용 메모리 크기 계산"""
        if dataset is None:
            return 0

        try:
            # PyArrow Table의 실제 메모리 사용량 계산
            if hasattr(dataset, 'nbytes'):
                return dataset.nbytes
            elif hasattr(dataset, 'get_total_buffer_size'):
                return dataset.get_total_buffer_size()
            else:
                # fallback: 대략적인 크기 추정
                return sys.getsizeof(dataset)
        except Exception as e:
            logger.warning(f"데이터셋 메모리 계산 실패: {e}")
            return sys.getsizeof(dataset)

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
        """DataManager 인스턴스 메모리 사용량 모니터링 스레드 - 개선된 버전"""
        consecutive_errors = 0
        max_errors = 5

        while self._monitoring and self.is_active:
            try:
                # 가비지 컬렉션 후 정확한 메모리 측정 (빈도 줄임)
                if consecutive_errors == 0:  # 오류가 없을 때만 GC 실행
                    gc.collect()

                # 인스턴스 전체 메모리 (안전한 방법으로 계산)
                current_instance_memory = self._get_object_memory_size()

                # 데이터셋 메모리 (전용 함수 사용)
                dataset_memory = self._calculate_dataset_memory_size(self.dataset)

                # 스레드 안전성을 위한 락 (간단한 접근)
                try:
                    # 통계 업데이트
                    self._resource_stats['instance_memory_usage'].append(current_instance_memory)
                    self._resource_stats['dataset_memory'].append(dataset_memory)
                    self._resource_stats['current_instance_memory'] = current_instance_memory

                    # 피크 메모리 업데이트
                    if current_instance_memory > self._resource_stats['peak_instance_memory']:
                        self._resource_stats['peak_instance_memory'] = current_instance_memory

                    # 최근 50개 샘플만 유지 (메모리 절약)
                    for key in ['instance_memory_usage', 'dataset_memory']:
                        if len(self._resource_stats[key]) > 50:
                            self._resource_stats[key] = self._resource_stats[key][-50:]

                except Exception as e:
                    logger.warning(f"통계 업데이트 실패: {e}")

                # 오류 카운터 리셋
                consecutive_errors = 0

                # 적응적 대기 시간 (메모리 사용량에 따라 조정)
                if dataset_memory > 100 * 1024 * 1024:  # 100MB 이상
                    time.sleep(5)  # 더 자주 모니터링
                else:
                    time.sleep(10)  # 덜 자주 모니터링

            except Exception as e:
                consecutive_errors += 1
                logger.warning(f"메모리 모니터링 오류 ({consecutive_errors}/{max_errors}): {e}")

                if consecutive_errors >= max_errors:
                    logger.error("메모리 모니터링 연속 오류로 스레드 종료")
                    break

                time.sleep(15)  # 오류 시 더 긴 대기

    def get_resource_stats(self) -> Dict[str, Any]:
        """DataManager 인스턴스의 메모리 사용량 통계 반환 - 개선된 버전"""
        try:
            current_instance_memory = self._resource_stats.get('current_instance_memory', 0)

            # 데이터셋 메모리 (전용 함수 사용)
            dataset_memory = self._calculate_dataset_memory_size(self.dataset)

            # 메모리 히스토리 통계
            memory_history = self._resource_stats.get('instance_memory_usage', [])
            dataset_history = self._resource_stats.get('dataset_memory', [])

            # 평균 메모리 사용량 (최근 10개 샘플)
            recent_memory = memory_history[-10:] if memory_history else [current_instance_memory]
            recent_dataset = dataset_history[-10:] if dataset_history else [dataset_memory]

            avg_memory = sum(recent_memory) / len(recent_memory) if recent_memory else 0
            avg_dataset = sum(recent_dataset) / len(recent_dataset) if recent_dataset else 0

            return {
                'manager_id': self.manager_id,
                'user_id': self.user_id,
                'user_name': self.user_name,
                'created_at': self.created_at.isoformat(),
                'is_active': self.is_active,

                # DataManager 인스턴스 메모리 정보
                'current_instance_memory_mb': current_instance_memory / (1024 * 1024),
                'initial_instance_memory_mb': self.initial_memory / (1024 * 1024),
                'peak_instance_memory_mb': self._resource_stats.get('peak_instance_memory', 0) / (1024 * 1024),
                'memory_growth_mb': (current_instance_memory - self.initial_memory) / (1024 * 1024),

                # 데이터셋 메모리
                'dataset_memory_mb': dataset_memory / (1024 * 1024),

                # 평균 메모리 사용량
                'average_memory_mb': avg_memory / (1024 * 1024),
                'average_dataset_mb': avg_dataset / (1024 * 1024),

                # 데이터 상태
                'has_dataset': self.dataset is not None,
                'dataset_rows': self.dataset.num_rows if self.dataset is not None else 0,
                'dataset_columns': self.dataset.num_columns if self.dataset is not None else 0,

                # 메모리 분포 (퍼센트) - 안전한 계산
                'memory_distribution': {
                    'dataset_percent': (dataset_memory / current_instance_memory * 100) if current_instance_memory > 0 else 0,
                    'other_percent': ((current_instance_memory - dataset_memory) / current_instance_memory * 100) if current_instance_memory > 0 else 0
                },

                # 모니터링 상태
                'monitoring_active': self._monitoring,
                'memory_samples_count': len(memory_history)
            }

        except Exception as e:
            logger.error(f"리소스 통계 생성 실패: {e}")
            # 기본적인 정보만 반환
            return {
                'manager_id': self.manager_id,
                'user_id': self.user_id,
                'is_active': self.is_active,
                'error': f"통계 생성 실패: {str(e)}",
                'has_dataset': self.dataset is not None
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

    def hf_download_and_load_dataset(self, repo_id: str, filename: str = None, split: str = None) -> Dict[str, Any]:
        """
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
            cache_dir = "/plateerag_backend/downloads/huggingface_cache"
            os.makedirs(cache_dir, exist_ok=True)

            logger.info("Starting dataset download from %s for manager %s", repo_id, self.manager_id)

            if filename is None:
                # 다중 파일 처리
                repo_files = list_repo_files(repo_id, repo_type='dataset')
                target_files, file_type = classify_dataset_files(repo_files, split)

                tables, downloaded_paths = process_multiple_files(repo_id, target_files, file_type, cache_dir)
                combined_table = combine_tables(tables, target_files)

                result_info = create_result_info(
                    repo_id=repo_id,
                    file_type=file_type,
                    combined_table=combined_table,
                    files_processed=target_files,
                    local_paths=downloaded_paths
                )

            else:
                # 단일 파일 처리
                file_type = determine_file_type_from_filename(filename)
                combined_table, downloaded_path = download_and_read_file(repo_id, filename, file_type, cache_dir)

                result_info = create_result_info(
                    repo_id=repo_id,
                    file_type=file_type,
                    combined_table=combined_table,
                    filename=filename,
                    local_path=downloaded_path
                )

            # self.dataset에 저장
            self.dataset = combined_table
            logger.info("Dataset loaded successfully. Final shape: %s, Columns: %s", combined_table.shape, combined_table.column_names)

            return result_info

        except Exception as e:
            logger.error("Failed to download and load dataset from %s: %s", repo_id, e)
            raise RuntimeError(f"Dataset download/load failed: {str(e)}")

    def local_upload_and_load_dataset(self, uploaded_files, filenames: List[str]) -> Dict[str, Any]:
        """
        로컬 파일들을 업로드하고 자동 적재

        Args:
            uploaded_files: 업로드된 파일 객체들 (단일 또는 여러개)
            filenames: 파일명들

        Returns:
            Dict[str, Any]: 결과 정보
        """
        if not self.is_active:
            raise RuntimeError("DataManager is not active")

        try:
            # 단일 파일인 경우 리스트로 변환
            if not isinstance(uploaded_files, list):
                uploaded_files = [uploaded_files]
            if not isinstance(filenames, list):
                filenames = [filenames]

            logger.info("로컬 파일 업로드 시작: %d개 파일", len(uploaded_files))

            # 파일들 저장하고 로드
            combined_table, dataset_id = save_and_load_files(uploaded_files, filenames, self.manager_id)

            # 결과 정보
            result_info = {
                "success": True,
                "dataset_id": dataset_id,
                "num_files": len(filenames),
                "filenames": filenames,
                "num_rows": combined_table.num_rows,
                "num_columns": combined_table.num_columns,
                "columns": combined_table.column_names,
                "loaded_at": datetime.now().isoformat()
            }

            # 데이터셋 저장
            self.dataset = combined_table
            logger.info("로컬 데이터셋 로드 완료: %s", combined_table.shape)

            return result_info

        except Exception as e:
            logger.error("로컬 데이터셋 업로드 실패: %s", e)
            raise RuntimeError(f"로컬 데이터셋 업로드 실패: {str(e)}")

    def download_dataset_as_csv(self, output_path: str = None) -> str:
        """
        현재 로드된 데이터셋을 CSV 파일로 저장하고 경로 반환

        Args:
            output_path (str, optional): 출력 파일 경로. None이면 자동 생성

        Returns:
            str: 저장된 CSV 파일 경로
        """
        if not self.is_active:
            raise RuntimeError("DataManager is not active")

        if self.dataset is None:
            raise RuntimeError("No dataset loaded")

        try:
            # 출력 경로가 지정되지 않으면 자동 생성
            if output_path is None:
                download_dir = "/plateerag_backend/downloads/tmp/dataset_downloads"
                os.makedirs(download_dir, exist_ok=True)
                output_path = os.path.join(download_dir, f"dataset_{self.manager_id}.csv")

            # pyarrow Table을 CSV로 저장
            csv.write_csv(self.dataset, output_path)

            logger.info("Dataset exported to CSV: %s (rows: %d, columns: %d)",
                       output_path, self.dataset.num_rows, self.dataset.num_columns)

            return output_path

        except Exception as e:
            logger.error("Failed to export dataset to CSV: %s", e)
            raise RuntimeError(f"CSV export failed: {str(e)}")

    def download_dataset_as_parquet(self, output_path: str = None) -> str:
        """
        현재 로드된 데이터셋을 Parquet 파일로 저장하고 경로 반환

        Args:
            output_path (str, optional): 출력 파일 경로. None이면 자동 생성

        Returns:
            str: 저장된 Parquet 파일 경로
        """
        if not self.is_active:
            raise RuntimeError("DataManager is not active")

        if self.dataset is None:
            raise RuntimeError("No dataset loaded")

        try:
            # 출력 경로가 지정되지 않으면 자동 생성
            if output_path is None:
                download_dir = "/plateerag_backend/downloads/tmp/dataset_downloads"
                os.makedirs(download_dir, exist_ok=True)
                output_path = os.path.join(download_dir, f"dataset_{self.manager_id}.parquet")

            # pyarrow Table을 Parquet으로 저장
            pq.write_table(self.dataset, output_path)

            logger.info("Dataset exported to Parquet: %s (rows: %d, columns: %d)",
                       output_path, self.dataset.num_rows, self.dataset.num_columns)

            return output_path

        except Exception as e:
            logger.error("Failed to export dataset to Parquet: %s", e)
            raise RuntimeError(f"Parquet export failed: {str(e)}")

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        현재 로드된 데이터셋의 기술통계정보 반환

        Returns:
            Dict[str, Any]: 기술통계정보
        """
        if not self.is_active:
            raise RuntimeError("DataManager is not active")

        if self.dataset is None:
            raise RuntimeError("No dataset loaded")

        try:
            statistics = generate_dataset_statistics(self.dataset)
            logger.info("Dataset statistics generated for manager %s", self.manager_id)
            return statistics

        except Exception as e:
            logger.error("Failed to generate dataset statistics: %s", e)
            raise RuntimeError(f"Statistics generation failed: {str(e)}")

    def drop_dataset_columns(self, columns_to_drop: List[str]) -> Dict[str, Any]:
        """
        현재 로드된 데이터셋에서 지정된 컬럼들을 삭제

        Args:
            columns_to_drop (List[str]): 삭제할 컬럼명들

        Returns:
            Dict[str, Any]: 삭제 결과 정보
        """
        if not self.is_active:
            raise RuntimeError("DataManager is not active")

        if self.dataset is None:
            raise RuntimeError("No dataset loaded")

        # 메모리 사용량 추적 (삭제 전) - 개선된 방법
        initial_memory = self._calculate_dataset_memory_size(self.dataset)

        try:
            # 기존 테이블 참조 저장 (메모리 해제를 위해)
            old_table = self.dataset

            # 컬럼 삭제 실행
            new_table, result_info = drop_columns_from_table(self.dataset, columns_to_drop)

            # 데이터셋 업데이트
            self.dataset = new_table

            # 기존 테이블 명시적 해제
            del old_table

            # 강제 가비지 컬렉션 (메모리 즉시 해제)
            gc.collect()

            # 메모리 사용량 추적 (삭제 후) - 개선된 방법
            final_memory = self._calculate_dataset_memory_size(self.dataset)
            memory_reduced = initial_memory - final_memory

            # 결과에 메모리 정보 추가
            result_info["memory_info"] = {
                "initial_memory_mb": initial_memory / (1024 * 1024),
                "final_memory_mb": final_memory / (1024 * 1024),
                "memory_reduced_mb": memory_reduced / (1024 * 1024)
            }

            logger.info("Dataset columns dropped for manager %s: %s (메모리 절약: %.2f MB)",
                       self.manager_id, columns_to_drop, memory_reduced / (1024 * 1024))

            return result_info

        except Exception as e:
            # 오류 발생 시에도 가비지 컬렉션 실행
            gc.collect()
            logger.error("Failed to drop dataset columns: %s", e)
            raise RuntimeError(f"Column drop failed: {str(e)}")

    def replace_dataset_column_values(self, column_name: str, old_value: str, new_value: str) -> Dict[str, Any]:
        """
        데이터셋의 특정 컬럼에서 값을 교체

        Args:
            column_name (str): 대상 컬럼명
            old_value (str): 교체할 기존 값
            new_value (str): 새로운 값

        Returns:
            Dict[str, Any]: 교체 결과 정보
        """
        if not self.is_active:
            raise RuntimeError("DataManager is not active")

        if self.dataset is None:
            raise RuntimeError("No dataset loaded")

        try:
            # 값 교체 실행
            new_table, result_info = replace_column_values(self.dataset, column_name, old_value, new_value)

            # 데이터셋 업데이트
            self.dataset = new_table

            logger.info("Column values replaced for manager %s: %s",
                       self.manager_id, result_info)

            return result_info

        except Exception as e:
            logger.error("Failed to replace column values: %s", e)
            raise RuntimeError(f"Value replacement failed: {str(e)}")

    def apply_dataset_column_operation(self, column_name: str, operation: str) -> Dict[str, Any]:
        """
        데이터셋의 특정 컬럼에 수치 연산을 적용

        Args:
            column_name (str): 대상 컬럼명
            operation (str): 연산식 (예: "+4", "*3+4")

        Returns:
            Dict[str, Any]: 연산 적용 결과 정보
        """
        if not self.is_active:
            raise RuntimeError("DataManager is not active")

        if self.dataset is None:
            raise RuntimeError("No dataset loaded")

        try:
            # 기존 테이블 참조 저장
            old_table = self.dataset

            # 연산 적용
            new_table, result_info = apply_column_operation(self.dataset, column_name, operation)

            # 데이터셋 업데이트
            self.dataset = new_table

            # 메모리 정리
            del old_table
            gc.collect()

            return result_info

        except Exception as e:
            logger.error("컬럼 연산 적용 실패: %s", e)
            raise RuntimeError(f"컬럼 연산 적용 실패: {str(e)}")

    def remove_null_rows_from_dataset(self, column_name: str = None) -> Dict[str, Any]:
        """
        데이터셋에서 NULL 값이 있는 행을 제거

        Args:
            column_name (str, optional): 특정 컬럼명. None이면 전체 컬럼에서 NULL 체크

        Returns:
            Dict[str, Any]: NULL row 제거 결과 정보
        """
        if not self.is_active:
            raise RuntimeError("DataManager is not active")

        if self.dataset is None:
            raise RuntimeError("No dataset loaded")

        # 메모리 사용량 추적 (제거 전)
        initial_memory = self._calculate_dataset_memory_size(self.dataset)

        try:
            # 기존 테이블 참조 저장
            old_table = self.dataset

            # NULL row 제거 실행
            new_table, result_info = remove_null_rows(self.dataset, column_name)

            # 데이터셋 업데이트
            self.dataset = new_table

            # 기존 테이블 명시적 해제
            del old_table

            # 강제 가비지 컬렉션
            gc.collect()

            # 메모리 사용량 추적 (제거 후)
            final_memory = self._calculate_dataset_memory_size(self.dataset)
            memory_saved = initial_memory - final_memory

            # 메모리 정보 추가
            result_info["memory_info"] = {
                "initial_memory_mb": initial_memory / (1024 * 1024),
                "final_memory_mb": final_memory / (1024 * 1024),
                "memory_saved_mb": memory_saved / (1024 * 1024)
            }

            logger.info("NULL row 제거 완료: 매니저 %s에서 %d개 행 제거",
                       self.manager_id, result_info["removed_rows"])

            return result_info

        except Exception as e:
            logger.error("NULL row 제거 실패: %s", e)
            raise RuntimeError(f"NULL row 제거 실패: {str(e)}")

    def upload_dataset_to_hf_repo(self, repo_id: str, hf_user_id: str, hub_token: str,
                                 filename: str = None, private: bool = False) -> Dict[str, Any]:
        """
        현재 데이터셋을 HuggingFace Hub에 업로드

        Args:
            repo_id (str): HuggingFace 리포지토리 ID
            hf_user_id (str): HuggingFace 사용자 ID
            hub_token (str): HuggingFace Hub 토큰
            filename (str, optional): 업로드할 파일명. None이면 자동 생성
            private (bool): 프라이빗 리포지토리 여부

        Returns:
            Dict[str, Any]: 업로드 결과 정보
        """
        if not self.is_active:
            raise RuntimeError("DataManager is not active")

        if self.dataset is None:
            raise RuntimeError("No dataset loaded")

        try:
            # HuggingFace 업로드 실행
            result_info = upload_dataset_to_hf(
                self.dataset, repo_id, hf_user_id, hub_token, filename, private
            )

            logger.info("HuggingFace 업로드 완료: 매니저 %s → %s",
                       self.manager_id, result_info["repo_id"])

            return result_info

        except Exception as e:
            logger.error("HuggingFace 업로드 실패: %s", e)
            raise RuntimeError(f"HuggingFace 업로드 실패: {str(e)}")

    def copy_dataset_column(self, source_column: str, new_column: str) -> Dict[str, Any]:
        """
        데이터셋의 특정 컬럼을 복사하여 새로운 컬럼으로 추가

        Args:
            source_column (str): 복사할 원본 컬럼명
            new_column (str): 새로운 컬럼명

        Returns:
            Dict[str, Any]: 복사 결과 정보
        """
        if not self.is_active:
            raise RuntimeError("DataManager is not active")

        if self.dataset is None:
            raise RuntimeError("No dataset loaded")

        try:
            # 기존 테이블 참조 저장
            old_table = self.dataset

            # 컬럼 복사 실행
            new_table, result_info = copy_column(self.dataset, source_column, new_column)

            # 데이터셋 업데이트
            self.dataset = new_table

            # 메모리 정리
            del old_table
            gc.collect()

            logger.info("컬럼 복사 완료: 매니저 %s에서 '%s' → '%s'",
                       self.manager_id, source_column, new_column)

            return result_info

        except Exception as e:
            logger.error("컬럼 복사 실패: %s", e)
            raise RuntimeError(f"컬럼 복사 실패: {str(e)}")

    def rename_dataset_column(self, old_name: str, new_name: str) -> Dict[str, Any]:
        """
        데이터셋의 특정 컬럼 이름을 변경

        Args:
            old_name (str): 기존 컬럼명
            new_name (str): 새로운 컬럼명

        Returns:
            Dict[str, Any]: 이름 변경 결과 정보
        """
        if not self.is_active:
            raise RuntimeError("DataManager is not active")

        if self.dataset is None:
            raise RuntimeError("No dataset loaded")

        try:
            # 기존 테이블 참조 저장
            old_table = self.dataset

            # 컬럼 이름 변경 실행
            new_table, result_info = rename_column(self.dataset, old_name, new_name)

            # 데이터셋 업데이트
            self.dataset = new_table

            # 메모리 정리
            del old_table
            gc.collect()

            logger.info("컬럼 이름 변경 완료: 매니저 %s에서 '%s' → '%s'",
                       self.manager_id, old_name, new_name)

            return result_info

        except Exception as e:
            logger.error("컬럼 이름 변경 실패: %s", e)
            raise RuntimeError(f"컬럼 이름 변경 실패: {str(e)}")

    def format_columns_to_string(self, column_names: List[str], template: str, new_column: str) -> Dict[str, Any]:
        """
        여러 컬럼의 값들을 문자열 템플릿에 삽입하여 새로운 컬럼 생성

        Args:
            column_names (List[str]): 사용할 컬럼명들
            template (str): 문자열 템플릿 (예: "{col1}_aiaiaiai_{col2}")
            new_column (str): 새로운 컬럼명

        Returns:
            Dict[str, Any]: 문자열 포맷팅 결과 정보
        """
        if not self.is_active:
            raise RuntimeError("DataManager is not active")

        if self.dataset is None:
            raise RuntimeError("No dataset loaded")

        try:
            # 기존 테이블 참조 저장
            old_table = self.dataset

            # 문자열 포맷팅 실행
            new_table, result_info = format_columns_string(self.dataset, column_names, template, new_column)

            # 데이터셋 업데이트
            self.dataset = new_table

            # 메모리 정리
            del old_table
            gc.collect()

            logger.info("컬럼 문자열 포맷팅 완료: 매니저 %s에서 %s → '%s'",
                       self.manager_id, column_names, new_column)

            return result_info

        except Exception as e:
            logger.error("컬럼 문자열 포맷팅 실패: %s", e)
            raise RuntimeError(f"컬럼 문자열 포맷팅 실패: {str(e)}")

    def calculate_columns_to_new(self, col1: str, col2: str, operation: str, new_column: str) -> Dict[str, Any]:
        """
        두 컬럼 간 사칙연산을 수행하여 새로운 컬럼 생성

        Args:
            col1 (str): 첫 번째 컬럼명
            col2 (str): 두 번째 컬럼명
            operation (str): 연산자 (+, -, *, /)
            new_column (str): 새로운 컬럼명

        Returns:
            Dict[str, Any]: 연산 결과 정보
        """
        if not self.is_active:
            raise RuntimeError("DataManager is not active")

        if self.dataset is None:
            raise RuntimeError("No dataset loaded")

        try:
            # 기존 테이블 참조 저장
            old_table = self.dataset

            # 컬럼 연산 실행
            new_table, result_info = calculate_columns_operation(self.dataset, col1, col2, operation, new_column)

            # 데이터셋 업데이트
            self.dataset = new_table

            # 메모리 정리
            del old_table
            gc.collect()

            logger.info("컬럼 연산 완료: 매니저 %s에서 %s %s %s → '%s'",
                       self.manager_id, col1, operation, col2, new_column)

            return result_info

        except Exception as e:
            logger.error("컬럼 연산 실패: %s", e)
            raise RuntimeError(f"컬럼 연산 실패: {str(e)}")

    def execute_dataset_callback(self, callback_code: str) -> Dict[str, Any]:
        """
        사용자 정의 PyArrow 코드를 안전하게 실행하여 dataset을 조작

        Args:
            callback_code (str): 실행할 PyArrow 코드

        Returns:
            Dict[str, Any]: 콜백 실행 결과 정보
        """
        if not self.is_active:
            raise RuntimeError("DataManager is not active")

        if self.dataset is None:
            raise RuntimeError("No dataset loaded")

        try:
            # 기존 테이블 참조 저장
            old_table = self.dataset

            # 사용자 콜백 코드 실행
            new_table, result_info = execute_safe_callback(self.dataset, callback_code)

            # 데이터셋 업데이트
            self.dataset = new_table

            # 메모리 정리
            del old_table
            gc.collect()

            logger.info("사용자 콜백 실행 완료: 매니저 %s, %d행 → %d행, %d열 → %d열",
                       self.manager_id, result_info["original_rows"],
                       result_info["final_rows"], result_info["original_columns"],
                       result_info["final_columns"])

            return result_info

        except Exception as e:
            logger.error("사용자 콜백 실행 실패: %s", e)
            raise RuntimeError(f"사용자 콜백 실행 실패: {str(e)}")

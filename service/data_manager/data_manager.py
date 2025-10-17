# /service/data_manager/data_manager.py
import uuid
import os
import threading
import time
import sys
import io
import hashlib
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

# 다운로드 기본 경로 설정
downloads_path = os.path.join(os.getcwd(), "downloads")

class DataManager:
    """
    Data Manager Instance Class (Dataset-Centric 구조)
    - Dataset ID를 중심으로 데이터 관리
    - Manager ID는 세션 관리용
    - MinIO + Redis 기반 버전 관리
    """

    def __init__(self, user_id: str, user_name: str = "Unknown",
                 minio_storage=None, redis_manager=None,
                 manager_id: str = None):  # ✅ manager_id 매개변수 추가
        """
        DataManager 인스턴스 초기화

        Args:
            user_id (str): 사용자 ID
            user_name (str): 사용자 이름
            minio_storage: MinIO 스토리지 클라이언트
            redis_manager: Redis 버전 관리자
            manager_id (str, optional): 복원 시 사용할 Manager ID
        """
        # ========== 기본 식별자 ==========
        if manager_id:
            self.manager_id = manager_id  # ✅ 복원 시 기존 ID 사용
            self._is_restored = True
            logger.info(f"♻️ 기존 Manager ID로 복원: {manager_id}")
        else:
            self.manager_id = f"mgr_{uuid.uuid4().hex[:12]}"  # 새 생성 시에만 UUID
            self._is_restored = False
            logger.info(f"✨ 새 Manager ID 생성: {self.manager_id}")
        
        self.user_id = str(user_id)
        self.user_name = user_name
        self.created_at = datetime.now()
        self.is_active = True
        
        # ========== 데이터셋 관리 (Dataset-Centric) ==========
        self.dataset_id = None  # 실제 데이터의 ID (Primary Key)
        self.dataset = None  # PyArrow Table
        
        # ========== 스토리지 클라이언트 ==========
        self.minio_storage = minio_storage
        self.redis_manager = redis_manager
        
        # ========== 버전 관리 ==========
        self.current_version = 0
        self.viewing_version = 0  # 현재 보고 있는 버전
        
        self.dataset_load_count = 0  # 데이터셋 로드 횟수

        # ========== 메모리 모니터링 ==========
        gc.collect()
        self.initial_memory = self._get_object_memory_size()
        self._monitoring = True
        self._resource_stats = {
            'instance_memory_usage': [],
            'dataset_memory': [],
            'peak_instance_memory': self.initial_memory,
            'current_instance_memory': self.initial_memory
        }
        
        # 모니터링 스레드 시작
        self._monitor_thread = threading.Thread(
            target=self._monitor_instance_memory, 
            daemon=True
        )
        self._monitor_thread.start()
        
        # ========== Manager 세션 등록 ==========
        # ✅ 복원된 경우 Redis 재등록 하지 않음 (이미 연결되어 있음)
        if self.redis_manager and not self._is_restored:
            try:
                # 아직 dataset이 없으므로 None으로 등록
                self.redis_manager.link_manager_to_dataset(
                    self.manager_id, 
                    None,  # dataset_id는 첫 로드 시 설정
                    self.user_id
                )
                logger.info(f"✅ Manager 세션 등록: {self.manager_id}")
            except Exception as e:
                logger.warning(f"Manager 세션 등록 실패: {e}")
        
        logger.info(f"DataManager {'복원' if self._is_restored else '생성'}: {self.manager_id} (user: {self.user_id})")

    # ========== 메모리 관리 메서드 ==========

    def _get_object_memory_size(self) -> int:
        """DataManager 인스턴스의 메모리 사용량 계산"""
        total_size = 0

        # 인스턴스 자체의 기본 크기
        total_size += sys.getsizeof(self)

        # 주요 속성들만 선별적으로 계산
        safe_attrs = [
            'manager_id', 'user_id', 'user_name', 'created_at',
            'is_active', 'initial_memory', 'dataset', 'current_version', 'dataset_id'
        ]

        for attr_name in safe_attrs:
            if hasattr(self, attr_name):
                try:
                    attr_value = getattr(self, attr_name)
                    if attr_name == 'dataset' and attr_value is not None:
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
                    stats_size += sum(sys.getsizeof(item) for item in value[-10:])
            total_size += stats_size
        except:
            pass

        return total_size

    def _calculate_dataset_memory_size(self, dataset) -> int:
        """데이터셋 전용 메모리 크기 계산"""
        if dataset is None:
            return 0

        try:
            if hasattr(dataset, 'nbytes'):
                return dataset.nbytes
            elif hasattr(dataset, 'get_total_buffer_size'):
                return dataset.get_total_buffer_size()
            else:
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
        """DataManager 인스턴스 메모리 사용량 모니터링 스레드"""
        consecutive_errors = 0
        max_errors = 5

        while self._monitoring and self.is_active:
            try:
                if consecutive_errors == 0:
                    gc.collect()

                current_instance_memory = self._get_object_memory_size()
                dataset_memory = self._calculate_dataset_memory_size(self.dataset)

                try:
                    self._resource_stats['instance_memory_usage'].append(current_instance_memory)
                    self._resource_stats['dataset_memory'].append(dataset_memory)
                    self._resource_stats['current_instance_memory'] = current_instance_memory

                    if current_instance_memory > self._resource_stats['peak_instance_memory']:
                        self._resource_stats['peak_instance_memory'] = current_instance_memory

                    for key in ['instance_memory_usage', 'dataset_memory']:
                        if len(self._resource_stats[key]) > 50:
                            self._resource_stats[key] = self._resource_stats[key][-50:]

                except Exception as e:
                    logger.warning(f"통계 업데이트 실패: {e}")

                consecutive_errors = 0

                if dataset_memory > 100 * 1024 * 1024:
                    time.sleep(5)
                else:
                    time.sleep(10)

            except Exception as e:
                consecutive_errors += 1
                logger.warning(f"메모리 모니터링 오류 ({consecutive_errors}/{max_errors}): {e}")

                if consecutive_errors >= max_errors:
                    logger.error("메모리 모니터링 연속 오류로 스레드 종료")
                    break

                time.sleep(15)

    def _calculate_checksum(self, table: pa.Table) -> str:
        """데이터 체크섬 계산"""
        try:
            buffer = io.BytesIO()
            pq.write_table(table, buffer)
            data_bytes = buffer.getvalue()
            return hashlib.sha256(data_bytes).hexdigest()
        except Exception as e:
            logger.error(f"체크섬 계산 실패: {e}")
            return ""

    # ========== 버전 관리 메서드 ==========

    def _save_version(self, operation_name: str, metadata: Dict[str, Any] = None):
        """버전 저장 (Dataset ID 기준) - Redis 메타를 먼저 저장하고 MinIO는 후순위로 처리"""
        if self.dataset is None or self.dataset_id is None:
            logger.warning("⚠️  데이터셋 또는 dataset_id가 없어 버전 저장 건너뜀")
            return

        try:
            logger.info(f"📝 버전 저장 시작: dataset={self.dataset_id}, version={self.current_version}, operation={operation_name}")

            minio_path = None

            # 1) Redis에 메타데이터 준비/저장 (항상 시도)
            version_info = {
                "version": self.current_version,
                "operation": operation_name,
                "timestamp": datetime.now().isoformat(),
                "minio_path": None,
                "checksum": self._calculate_checksum(self.dataset),
                "num_rows": self.dataset.num_rows,
                "num_columns": self.dataset.num_columns,
                "columns": self.dataset.column_names,
                "manager_id": self.manager_id,
                "metadata": metadata or {}
            }

            if self.redis_manager:
                try:
                    self.redis_manager.save_version_metadata(self.dataset_id, self.current_version, version_info)
                    logger.info(f"  ✅ Redis 메타데이터 저장: {self.dataset_id} v{self.current_version}")
                except Exception as e:
                    logger.warning(f"  ⚠️ Redis 메타데이터 저장 실패(계속 진행): {e}")

            # 2) MinIO에 스냅샷 저장 (실패해도 메타는 유지)
            if self.minio_storage:
                try:
                    minio_path = self.minio_storage.save_version_snapshot(
                        self.dataset_id,
                        self.current_version,
                        self.dataset,
                        operation_name,
                        metadata or {}
                    )
                    logger.info(f"  ✅ MinIO 스냅샷 저장: {minio_path}")
                except Exception as e:
                    logger.warning(f"  ⚠️ MinIO 스냅샷 저장 실패(메타는 유지됨): {e}")

            # 3) Redis 메타에 minio_path가 있으면 업데이트 시도
            if self.redis_manager and minio_path:
                try:
                    existing = self.redis_manager.get_version_metadata(self.dataset_id, self.current_version)
                    if existing:
                        existing['minio_path'] = minio_path
                        key = f"{self.redis_manager.dataset_prefix}:{self.dataset_id}:version:{self.current_version}"
                        self.redis_manager.redis_client.set(key, json.dumps(existing))
                        logger.info(f"  ✅ Redis 버전 메타에 minio_path 업데이트")
                except Exception as e:
                    logger.warning(f"  ⚠️ Redis minio_path 업데이트 실패: {e}")

            # 4) 로컬 current_version 증가
            old_version = self.current_version
            self.current_version += 1
            logger.info(f"✅ 버전 저장 완료: {self.dataset_id} v{old_version} ({operation_name}) → 다음 버전: v{self.current_version}")

        except Exception as e:
            logger.error(f"❌ 버전 저장 실패: {e}", exc_info=True)

    def get_version_history(self) -> List[Dict[str, Any]]:
        """버전 이력 조회 (Dataset ID 기준)"""
        if self.redis_manager and self.dataset_id:
            return self.redis_manager.get_all_versions(self.dataset_id)
        return []

    def rollback_to_version(self, version: int) -> Dict[str, Any]:
        """특정 버전으로 롤백 (Dataset ID 기준)"""
        if not self.redis_manager or not self.minio_storage:
            raise RuntimeError("버전 관리 기능이 활성화되지 않았습니다")
        
        if not self.dataset_id:
            raise RuntimeError("dataset_id가 없습니다")

        try:
            # 버전 메타데이터 조회
            version_info = self.redis_manager.get_version_metadata(self.dataset_id, version)

            if not version_info:
                raise ValueError(f"버전 {version}을 찾을 수 없습니다")

            operation_name = version_info["operation"]
            
            # MinIO에서 해당 버전 로드
            self.dataset = self.minio_storage.load_version_snapshot(
                self.dataset_id,
                version,
                operation_name
            )

            self.current_version = version + 1
            self.viewing_version = version

            logger.info(f"✅ 버전 {version}으로 롤백 완료: {self.dataset_id}")

            return {
                "success": True,
                "dataset_id": self.dataset_id,
                "rolled_back_to_version": version,
                "operation": operation_name,
                "num_rows": self.dataset.num_rows,
                "num_columns": self.dataset.num_columns
            }

        except Exception as e:
            logger.error(f"롤백 실패: {e}")
            raise RuntimeError(f"롤백 실패: {str(e)}")

    # ========== 리소스 통계 ==========

    def get_resource_stats(self) -> Dict[str, Any]:
        """DataManager 인스턴스의 메모리 사용량 통계 반환"""
        try:
            current_instance_memory = self._resource_stats.get('current_instance_memory', 0)
            dataset_memory = self._calculate_dataset_memory_size(self.dataset)

            memory_history = self._resource_stats.get('instance_memory_usage', [])
            dataset_history = self._resource_stats.get('dataset_memory', [])

            recent_memory = memory_history[-10:] if memory_history else [current_instance_memory]
            recent_dataset = dataset_history[-10:] if dataset_history else [dataset_memory]

            avg_memory = sum(recent_memory) / len(recent_memory) if recent_memory else 0
            avg_dataset = sum(recent_dataset) / len(recent_dataset) if recent_dataset else 0

            return {
                'manager_id': self.manager_id,
                'dataset_id': self.dataset_id,
                'user_id': self.user_id,
                'user_name': self.user_name,
                'created_at': self.created_at.isoformat(),
                'is_active': self.is_active,

                # 메모리 정보
                'current_instance_memory_mb': current_instance_memory / (1024 * 1024),
                'initial_instance_memory_mb': self.initial_memory / (1024 * 1024),
                'peak_instance_memory_mb': self._resource_stats.get('peak_instance_memory', 0) / (1024 * 1024),
                'memory_growth_mb': (current_instance_memory - self.initial_memory) / (1024 * 1024),
                'dataset_memory_mb': dataset_memory / (1024 * 1024),
                'average_memory_mb': avg_memory / (1024 * 1024),
                'average_dataset_mb': avg_dataset / (1024 * 1024),

                # 데이터 상태
                'has_dataset': self.dataset is not None,
                'dataset_rows': self.dataset.num_rows if self.dataset is not None else 0,
                'dataset_columns': self.dataset.num_columns if self.dataset is not None else 0,

                # 버전 정보
                'current_version': self.current_version,
                'viewing_version': self.viewing_version,
                'version_management_enabled': self.minio_storage is not None and self.redis_manager is not None,

                # 메모리 분포
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
            return {
                'manager_id': self.manager_id,
                'dataset_id': self.dataset_id,
                'user_id': self.user_id,
                'is_active': self.is_active,
                'error': f"통계 생성 실패: {str(e)}",
                'has_dataset': self.dataset is not None,
                'current_version': self.current_version
            }

    # ========== 데이터셋 기본 조작 ==========

    def get_dataset(self) -> Any:
        """데이터셋 반환"""
        if not self.is_active:
            raise RuntimeError("DataManager is not active")
        return self.dataset

    def get_dataset_sample(self, num_samples: int = 10) -> Dict[str, Any]:
        """데이터셋 샘플 조회"""
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
            actual_samples = min(num_samples, self.dataset.num_rows)
            sample_table = self.dataset.slice(0, actual_samples)

            column_info = {}
            for i, col_name in enumerate(sample_table.column_names):
                col_type = str(sample_table.schema.field(i).type)
                column_info[col_name] = {
                    "type": col_type,
                    "nullable": sample_table.schema.field(i).nullable
                }

            sample_records = []
            for row_idx in range(sample_table.num_rows):
                record = {}
                for col_idx, col_name in enumerate(sample_table.column_names):
                    value = sample_table.column(col_idx)[row_idx].as_py()
                    record[col_name] = None if value is None else value
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
            gc.collect()
            logger.info(f"Dataset removed from manager {self.manager_id}")
            return True
        return False

    # ========== 데이터셋 로드 메서드 ==========

    def hf_download_and_load_dataset(self, repo_id: str, filename: str = None, 
                                split: str = None) -> Dict[str, Any]:
        if not self.is_active:
            raise RuntimeError("DataManager is not active")

        try:
            # 다운로드 준비...
            cache_dir = os.path.join(downloads_path, "huggingface_cache")
            os.makedirs(cache_dir, exist_ok=True)
            logger.info("HuggingFace 다운로드 시작: repo=%s, user=%s", repo_id, self.user_id)

            # 파일 다운로드 및 테이블 조합
            if filename is None:
                repo_files = list_repo_files(repo_id, repo_type='dataset')
                target_files, file_type = classify_dataset_files(repo_files, split)
                tables, downloaded_paths = process_multiple_files(repo_id, target_files, file_type, cache_dir)
                combined_table = combine_tables(tables, target_files)
                result_info = create_result_info(repo_id=repo_id, file_type=file_type, combined_table=combined_table, files_processed=target_files, local_paths=downloaded_paths)
            else:
                file_type = determine_file_type_from_filename(filename)
                combined_table, downloaded_path = download_and_read_file(repo_id, filename, file_type, cache_dir)
                result_info = create_result_info(repo_id=repo_id, file_type=file_type, combined_table=combined_table, filename=filename, local_path=downloaded_path)

            # 데이터셋 설정
            self.dataset = combined_table
            logger.info("테이블 로드 완료: %d행, %d열", combined_table.num_rows, combined_table.num_columns)

            # load_count 증가 (항상)
            self.dataset_load_count += 1
            is_first_load = (self.dataset_id is None)

            # Dataset ID 생성 및 Redis 등록 (원자적 보장)
            if is_first_load:
                repo_slug = repo_id.replace('/', '_')
                unique_id = uuid.uuid4().hex[:8]
                self.dataset_id = f"ds_hf_{repo_slug}_{unique_id}"
                logger.info(f"✨ 새 Dataset ID 생성: {self.dataset_id}")

                if self.redis_manager:
                    try:
                        dataset_metadata = {
                            "source_type": "huggingface",
                            "repo_id": repo_id,
                            "filename": filename,
                            "split": split,
                            "created_at": datetime.now().isoformat(),
                            "created_by": self.user_id,
                            "original_rows": combined_table.num_rows,
                            "original_columns": combined_table.num_columns
                        }
                        self.redis_manager.register_dataset(self.dataset_id, self.user_id, dataset_metadata)
                        self.redis_manager.link_manager_to_dataset(self.manager_id, self.dataset_id, self.user_id)
                        logger.info("✅ Redis 데이터셋 등록 및 Manager-Dataset 링크 완료")
                    except Exception as e:
                        logger.warning(f"Redis 등록 실패: {e}")

                # --- 추가: 원본(raw-datasets) 저장 ---
                if self.minio_storage:
                    try:
                        metadata_for_minio = dataset_metadata if 'dataset_metadata' in locals() else {}
                        self.minio_storage.save_original_dataset(self.user_id, self.dataset_id, self.dataset, metadata_for_minio)
                        logger.info(f"✅ MinIO 원본 저장 완료: raw-datasets/{self.user_id}/{self.dataset_id}/original.parquet")
                    except Exception as e:
                        logger.warning(f"MinIO 원본 저장 실패(계속): {e}")

            else:
                logger.info(f"♻️ 기존 Dataset 재로드: {self.dataset_id} (로드 {self.dataset_load_count}회차)")

            # 소스 정보 생성
            source_info = {
                "type": "huggingface",
                "repo_id": repo_id,
                "filename": filename,
                "split": split,
                "file_type": file_type,
                "loaded_at": datetime.now().isoformat(),
                "checksum": self._calculate_checksum(self.dataset),
                "num_rows": combined_table.num_rows,
                "num_columns": combined_table.num_columns,
                "columns": combined_table.column_names,
                "load_count": self.dataset_load_count,
                "is_reload": not is_first_load
            }

            # Redis에 소스 정보 저장 (항상 시도)
            if self.redis_manager:
                try:
                    self.redis_manager.save_source_info(self.dataset_id, source_info)
                    logger.info("✅ Redis 소스 정보 저장")
                except Exception as e:
                    logger.warning(f"Redis 소스 정보 저장 실패: {e}")

            # operation 이름 결정
            operation_name = "initial_load" if is_first_load else f"reload_{self.dataset_load_count}"
            logger.info(f"💾 버전 저장: operation={operation_name}, load_count={self.dataset_load_count}")

            # 안전한 버전 저장 (Redis 우선, MinIO 후순위)
            self._save_version(operation_name, source_info)

            # 결과 보강
            result_info.update({
                "dataset_id": self.dataset_id,
                "manager_id": self.manager_id,
                "user_id": self.user_id,
                "is_new_dataset": is_first_load,
                "current_version": self.current_version - 1,
                "load_count": self.dataset_load_count,
                "is_new_version": not is_first_load,
                "source_info": source_info
            })

            logger.info(f"✅ HuggingFace 다운로드 완료: dataset={self.dataset_id}, version={self.current_version - 1}, load_count={self.dataset_load_count}")
            return result_info

        except Exception as e:
            logger.error(f"HuggingFace 다운로드 실패: repo={repo_id}, error={e}", exc_info=True)
            raise RuntimeError(f"Dataset download/load failed: {str(e)}")
            
    def local_upload_and_load_dataset(self, uploaded_files, filenames: List[str]) -> Dict[str, Any]:
        if not self.is_active:
            raise RuntimeError("DataManager is not active")

        try:
            if not isinstance(uploaded_files, list):
                uploaded_files = [uploaded_files]
            if not isinstance(filenames, list):
                filenames = [filenames]

            logger.info("로컬 파일 업로드 시작: %d개 파일", len(uploaded_files))

            # 파일 저장 및 로드
            combined_table, base_dataset_id = save_and_load_files(uploaded_files, filenames, self.manager_id)

            # load count 증가
            self.dataset_load_count += 1
            is_first_load = (self.dataset_id is None)

            # Dataset ID 생성 및 Redis 등록
            if is_first_load:
                unique_id = uuid.uuid4().hex[:8]
                self.dataset_id = f"ds_local_{base_dataset_id}_{unique_id}"
                if self.redis_manager:
                    try:
                        dataset_metadata = {
                            "source_type": "local",
                            "filenames": filenames,
                            "created_at": datetime.now().isoformat(),
                            "created_by": self.user_id,
                            "original_rows": combined_table.num_rows,
                            "original_columns": combined_table.num_columns
                        }
                        self.redis_manager.register_dataset(self.dataset_id, self.user_id, dataset_metadata)
                        self.redis_manager.link_manager_to_dataset(self.manager_id, self.dataset_id, self.user_id)
                        logger.info("✅ Redis 데이터셋 등록 및 Manager-Dataset 링크 완료")
                    except Exception as e:
                        logger.warning(f"Redis 등록 실패: {e}")

                # --- 추가: 원본(raw-datasets) 저장 ---
                if self.minio_storage:
                    try:
                        metadata_for_minio = dataset_metadata if 'dataset_metadata' in locals() else {}
                        self.minio_storage.save_original_dataset(self.user_id, self.dataset_id, combined_table, metadata_for_minio)
                        logger.info(f"✅ MinIO 원본 저장 완료: raw-datasets/{self.user_id}/{self.dataset_id}/original.parquet")
                    except Exception as e:
                        logger.warning(f"MinIO 원본 저장 실패(계속): {e}")

            else:
                logger.info(f"♻️  기존 Dataset 재로드: {self.dataset_id} (로드 {self.dataset_load_count}회차)")

            # 데이터셋 설정
            self.dataset = combined_table

            # source_info
            source_info = {
                "type": "local",
                "filenames": filenames,
                "loaded_at": datetime.now().isoformat(),
                "checksum": self._calculate_checksum(self.dataset),
                "num_rows": self.dataset.num_rows,
                "num_columns": self.dataset.num_columns,
                "columns": self.dataset.column_names,
                "load_count": self.dataset_load_count,
                "is_reload": not is_first_load
            }

            # Redis에 소스 정보 저장
            if self.redis_manager:
                try:
                    self.redis_manager.save_source_info(self.dataset_id, source_info)
                    logger.info("✅ Redis 소스 정보 저장")
                except Exception as e:
                    logger.warning("Redis 소스 정보 저장 실패: %s", e)

            # 버전 저장
            operation_name = "initial_load" if is_first_load else f"reload_{self.dataset_load_count}"
            self._save_version(operation_name, source_info)

            result_info = {
                "success": True,
                "dataset_id": self.dataset_id,
                "manager_id": self.manager_id,
                "base_dataset_id": base_dataset_id,
                "num_files": len(filenames),
                "filenames": filenames,
                "num_rows": combined_table.num_rows,
                "num_columns": combined_table.num_columns,
                "columns": combined_table.column_names,
                "loaded_at": datetime.now().isoformat(),
                "load_count": self.dataset_load_count,
                "is_new_version": not is_first_load,
                "current_version": self.current_version - 1
            }

            logger.info(f"✅ 로컬 데이터셋 로드 완료: {self.dataset_id}, version={self.current_version - 1}, load_count={self.dataset_load_count}")
            return result_info

        except Exception as e:
            logger.error(f"로컬 업로드 실패: {e}", exc_info=True)
            raise RuntimeError(f"로컬 데이터셋 업로드 실패: {str(e)}")

    # ========== 데이터셋 내보내기 ==========

    def download_dataset_as_csv(self, output_path: str = None) -> str:
        """현재 로드된 데이터셋을 CSV 파일로 저장"""
        if not self.is_active:
            raise RuntimeError("DataManager is not active")

        if self.dataset is None:
            raise RuntimeError("No dataset loaded")

        try:
            if output_path is None:
                download_dir = os.path.join(downloads_path, "tmp", "dataset_downloads")
                os.makedirs(download_dir, exist_ok=True)
                output_path = os.path.join(download_dir, f"dataset_{self.manager_id}.csv")

            csv.write_csv(self.dataset, output_path)

            logger.info("Dataset exported to CSV: %s (rows: %d, columns: %d)",
                       output_path, self.dataset.num_rows, self.dataset.num_columns)

            return output_path

        except Exception as e:
            logger.error("Failed to export dataset to CSV: %s", e)
            raise RuntimeError(f"CSV export failed: {str(e)}")

    def download_dataset_as_parquet(self, output_path: str = None) -> str:
        """현재 로드된 데이터셋을 Parquet 파일로 저장"""
        if not self.is_active:
            raise RuntimeError("DataManager is not active")

        if self.dataset is None:
            raise RuntimeError("No dataset loaded")

        try:
            if output_path is None:
                download_dir = os.path.join(downloads_path, "tmp", "dataset_downloads")
                os.makedirs(download_dir, exist_ok=True)
                output_path = os.path.join(download_dir, f"dataset_{self.manager_id}.parquet")

            pq.write_table(self.dataset, output_path)

            logger.info("Dataset exported to Parquet: %s (rows: %d, columns: %d)",
                       output_path, self.dataset.num_rows, self.dataset.num_columns)

            return output_path

        except Exception as e:
            logger.error("Failed to export dataset to Parquet: %s", e)
            raise RuntimeError(f"Parquet export failed: {str(e)}")

    # ========== 데이터셋 통계 ==========

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """현재 로드된 데이터셋의 기술통계정보 반환"""
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

    # ========== 데이터셋 변환 메서드 (버전 저장 포함) ==========

    def drop_dataset_columns(self, columns_to_drop: List[str]) -> Dict[str, Any]:
        """컬럼 삭제 (버전 저장 포함)"""
        if not self.is_active:
            raise RuntimeError("DataManager is not active")

        if self.dataset is None:
            raise RuntimeError("No dataset loaded")

        initial_memory = self._calculate_dataset_memory_size(self.dataset)

        try:
            old_table = self.dataset
            new_table, result_info = drop_columns_from_table(self.dataset, columns_to_drop)
            self.dataset = new_table

            del old_table
            gc.collect()

            final_memory = self._calculate_dataset_memory_size(self.dataset)
            memory_reduced = initial_memory - final_memory

            result_info["memory_info"] = {
                "initial_memory_mb": initial_memory / (1024 * 1024),
                "final_memory_mb": final_memory / (1024 * 1024),
                "memory_reduced_mb": memory_reduced / (1024 * 1024)
            }

            # 버전 저장
            self._save_version("drop_columns", {
                "dropped_columns": columns_to_drop
            })

            logger.info("Dataset columns dropped for manager %s: %s (메모리 절약: %.2f MB)",
                       self.manager_id, columns_to_drop, memory_reduced / (1024 * 1024))

            return result_info

        except Exception as e:
            gc.collect()
            logger.error("Failed to drop dataset columns: %s", e)
            raise RuntimeError(f"Column drop failed: {str(e)}")

    def replace_dataset_column_values(self, column_name: str, old_value: str, 
                                     new_value: str) -> Dict[str, Any]:
        """값 교체 (버전 저장 포함)"""
        if not self.is_active:
            raise RuntimeError("DataManager is not active")

        if self.dataset is None:
            raise RuntimeError("No dataset loaded")

        try:
            new_table, result_info = replace_column_values(self.dataset, column_name, old_value, new_value)
            self.dataset = new_table

            # 버전 저장
            self._save_version("replace_values", {
                "column": column_name,
                "old_value": old_value,
                "new_value": new_value
            })

            logger.info("Column values replaced for manager %s: %s", self.manager_id, result_info)

            return result_info

        except Exception as e:
            logger.error("Failed to replace column values: %s", e)
            raise RuntimeError(f"Value replacement failed: {str(e)}")

    def apply_dataset_column_operation(self, column_name: str, operation: str) -> Dict[str, Any]:
        """컬럼 연산 적용 (버전 저장 포함)"""
        if not self.is_active:
            raise RuntimeError("DataManager is not active")

        if self.dataset is None:
            raise RuntimeError("No dataset loaded")

        try:
            old_table = self.dataset
            new_table, result_info = apply_column_operation(self.dataset, column_name, operation)
            self.dataset = new_table

            del old_table
            gc.collect()

            # 버전 저장
            self._save_version("apply_operation", {
                "column": column_name,
                "operation": operation
            })

            return result_info

        except Exception as e:
            logger.error("컬럼 연산 적용 실패: %s", e)
            raise RuntimeError(f"컬럼 연산 적용 실패: {str(e)}")

    def remove_null_rows_from_dataset(self, column_name: str = None) -> Dict[str, Any]:
        """NULL row 제거 (버전 저장 포함)"""
        if not self.is_active:
            raise RuntimeError("DataManager is not active")

        if self.dataset is None:
            raise RuntimeError("No dataset loaded")

        initial_memory = self._calculate_dataset_memory_size(self.dataset)

        try:
            old_table = self.dataset
            new_table, result_info = remove_null_rows(self.dataset, column_name)
            self.dataset = new_table

            del old_table
            gc.collect()

            final_memory = self._calculate_dataset_memory_size(self.dataset)
            memory_saved = initial_memory - final_memory

            result_info["memory_info"] = {
                "initial_memory_mb": initial_memory / (1024 * 1024),
                "final_memory_mb": final_memory / (1024 * 1024),
                "memory_saved_mb": memory_saved / (1024 * 1024)
            }

            # 버전 저장
            self._save_version("remove_null_rows", {
                "column": column_name,
                "rows_removed": result_info["removed_rows"]
            })

            logger.info("NULL row 제거 완료: 매니저 %s에서 %d개 행 제거",
                       self.manager_id, result_info["removed_rows"])

            return result_info

        except Exception as e:
            logger.error("NULL row 제거 실패: %s", e)
            raise RuntimeError(f"NULL row 제거 실패: {str(e)}")

    def upload_dataset_to_hf_repo(self, repo_id: str, hf_user_id: str, hub_token: str,
                                 filename: str = None, private: bool = False) -> Dict[str, Any]:
        """HuggingFace Hub 업로드"""
        if not self.is_active:
            raise RuntimeError("DataManager is not active")

        if self.dataset is None:
            raise RuntimeError("No dataset loaded")

        try:
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
        """컬럼 복사 (버전 저장 포함)"""
        if not self.is_active:
            raise RuntimeError("DataManager is not active")

        if self.dataset is None:
            raise RuntimeError("No dataset loaded")

        try:
            old_table = self.dataset
            new_table, result_info = copy_column(self.dataset, source_column, new_column)
            self.dataset = new_table

            del old_table
            gc.collect()

            # 버전 저장
            self._save_version("copy_column", {
                "source_column": source_column,
                "new_column": new_column
            })

            logger.info("컬럼 복사 완료: 매니저 %s에서 '%s' → '%s'",
                       self.manager_id, source_column, new_column)

            return result_info

        except Exception as e:
            logger.error("컬럼 복사 실패: %s", e)
            raise RuntimeError(f"컬럼 복사 실패: {str(e)}")

    def rename_dataset_column(self, old_name: str, new_name: str) -> Dict[str, Any]:
        """컬럼 이름 변경 (버전 저장 포함)"""
        if not self.is_active:
            raise RuntimeError("DataManager is not active")

        if self.dataset is None:
            raise RuntimeError("No dataset loaded")

        try:
            old_table = self.dataset
            new_table, result_info = rename_column(self.dataset, old_name, new_name)
            self.dataset = new_table

            del old_table
            gc.collect()

            # 버전 저장
            self._save_version("rename_column", {
                "old_name": old_name,
                "new_name": new_name
            })

            logger.info("컬럼 이름 변경 완료: 매니저 %s에서 '%s' → '%s'",
                       self.manager_id, old_name, new_name)

            return result_info

        except Exception as e:
            logger.error("컬럼 이름 변경 실패: %s", e)
            raise RuntimeError(f"컬럼 이름 변경 실패: {str(e)}")

    def format_columns_to_string(self, column_names: List[str], template: str, 
                                new_column: str) -> Dict[str, Any]:
        """컬럼 문자열 포맷팅 (버전 저장 포함)"""
        if not self.is_active:
            raise RuntimeError("DataManager is not active")

        if self.dataset is None:
            raise RuntimeError("No dataset loaded")

        try:
            old_table = self.dataset
            new_table, result_info = format_columns_string(
                self.dataset, column_names, template, new_column
            )
            self.dataset = new_table

            del old_table
            gc.collect()

            # 버전 저장
            self._save_version("format_columns", {
                "column_names": column_names,
                "template": template,
                "new_column": new_column
            })

            logger.info("컬럼 문자열 포맷팅 완료: 매니저 %s에서 %s → '%s'",
                       self.manager_id, column_names, new_column)

            return result_info

        except Exception as e:
            logger.error("컬럼 문자열 포맷팅 실패: %s", e)
            raise RuntimeError(f"컬럼 문자열 포맷팅 실패: {str(e)}")

    def calculate_columns_to_new(self, col1: str, col2: str, operation: str, 
                                 new_column: str) -> Dict[str, Any]:
        """컬럼 간 연산 (버전 저장 포함)"""
        if not self.is_active:
            raise RuntimeError("DataManager is not active")

        if self.dataset is None:
            raise RuntimeError("No dataset loaded")

        try:
            old_table = self.dataset
            new_table, result_info = calculate_columns_operation(
                self.dataset, col1, col2, operation, new_column
            )
            self.dataset = new_table

            del old_table
            gc.collect()

            # 버전 저장
            self._save_version("calculate_columns", {
                "col1": col1,
                "col2": col2,
                "operation": operation,
                "new_column": new_column
            })

            logger.info("컬럼 연산 완료: 매니저 %s에서 %s %s %s → '%s'",
                       self.manager_id, col1, operation, col2, new_column)

            return result_info

        except Exception as e:
            logger.error("컬럼 연산 실패: %s", e)
            raise RuntimeError(f"컬럼 연산 실패: {str(e)}")

    def execute_dataset_callback(self, callback_code: str) -> Dict[str, Any]:
        """사용자 콜백 실행 (버전 저장 포함)"""
        if not self.is_active:
            raise RuntimeError("DataManager is not active")

        if self.dataset is None:
            raise RuntimeError("No dataset loaded")

        try:
            old_table = self.dataset
            new_table, result_info = execute_safe_callback(self.dataset, callback_code)
            self.dataset = new_table

            del old_table
            gc.collect()

            # 버전 저장
            self._save_version("execute_callback", {
                "code_length": len(callback_code),
                "rows_changed": result_info["rows_changed"],
                "columns_changed": result_info["columns_changed"]
            })

            logger.info("사용자 콜백 실행 완료: 매니저 %s, %d행 → %d행, %d열 → %d열",
                       self.manager_id, result_info["original_rows"],
                       result_info["final_rows"], result_info["original_columns"],
                       result_info["final_columns"])

            return result_info

        except Exception as e:
            logger.error("사용자 콜백 실행 실패: %s", e)
            raise RuntimeError(f"사용자 콜백 실행 실패: {str(e)}")


    # /service/data_manager/data_manager.py에 추가

    # ========== 데이터셋 로드 메서드 ========== 섹션에 추가

    def db_load_dataset(self, 
                       db_config: Dict[str, Any],
                       query: str = None,
                       table_name: str = None,
                       chunk_size: int = None) -> Dict[str, Any]:
        """
        데이터베이스에서 데이터셋 로드
        
        Args:
            db_config: 데이터베이스 연결 설정
                {
                    'db_type': 'postgresql' | 'mysql' | 'sqlite',
                    'host': str,
                    'port': int,
                    'database': str,
                    'username': str,
                    'password': str
                }
            query: SQL 쿼리 (query 또는 table_name 중 하나 필수)
            table_name: 테이블명 (query 또는 table_name 중 하나 필수)
            chunk_size: 청크 크기 (대용량 데이터 처리용)
            
        Returns:
            Dict[str, Any]: 로드 결과 정보
        """
        if not self.is_active:
            raise RuntimeError("DataManager is not active")

        if not query and not table_name:
            raise RuntimeError("query 또는 table_name 중 하나는 필수입니다")

        try:
            import sqlalchemy
            from sqlalchemy import create_engine, text
            import pandas as pd
            
            logger.info("DB 데이터셋 로드 시작: db_type=%s, user=%s", 
                       db_config.get('db_type'), self.user_id)
            
            # ========== 1. DB 연결 문자열 생성 ==========
            db_type = db_config.get('db_type', 'postgresql').lower()
            
            if db_type == 'postgresql':
                connection_string = (
                    f"postgresql://{db_config['username']}:{db_config['password']}"
                    f"@{db_config['host']}:{db_config.get('port', 5432)}"
                    f"/{db_config['database']}"
                )
            elif db_type == 'mysql':
                connection_string = (
                    f"mysql+pymysql://{db_config['username']}:{db_config['password']}"
                    f"@{db_config['host']}:{db_config.get('port', 3306)}"
                    f"/{db_config['database']}"
                )
            elif db_type == 'sqlite':
                connection_string = f"sqlite:///{db_config['database']}"
            else:
                raise RuntimeError(f"지원되지 않는 DB 타입: {db_type}")
            
            # ========== 2. DB 연결 및 데이터 로드 ==========
            engine = create_engine(connection_string)
            
            # SQL 쿼리 결정
            if query:
                sql_query = query
                logger.info(f"  └─ 사용자 정의 쿼리 실행")
            else:
                sql_query = f"SELECT * FROM {table_name}"
                logger.info(f"  └─ 테이블 전체 조회: {table_name}")
            
            # 데이터 로드
            if chunk_size:
                # 청크 단위로 로드 (대용량 데이터)
                logger.info(f"  └─ 청크 크기: {chunk_size}")
                chunks = []
                for chunk_df in pd.read_sql(sql_query, engine, chunksize=chunk_size):
                    chunks.append(pa.Table.from_pandas(chunk_df))
                combined_table = pa.concat_tables(chunks)
                logger.info(f"  └─ {len(chunks)}개 청크 병합 완료")
            else:
                # 전체 로드
                df = pd.read_sql(sql_query, engine)
                combined_table = pa.Table.from_pandas(df)
            
            engine.dispose()
            
            logger.info("테이블 로드 완료: %d행, %d열", 
                       combined_table.num_rows, combined_table.num_columns)
            
            # ========== 3. 데이터셋 설정 ==========
            self.dataset = combined_table
            
            # load_count 증가
            self.dataset_load_count += 1
            is_first_load = (self.dataset_id is None)
            
            # ========== 4. Dataset ID 생성 및 Redis 등록 ==========
            if is_first_load:
                db_identifier = f"{db_type}_{db_config['database']}"
                if table_name:
                    db_identifier += f"_{table_name}"
                unique_id = uuid.uuid4().hex[:8]
                self.dataset_id = f"ds_db_{db_identifier}_{unique_id}"
                logger.info(f"✨ 새 Dataset ID 생성: {self.dataset_id}")
                
                if self.redis_manager:
                    try:
                        dataset_metadata = {
                            "source_type": "database",
                            "db_type": db_type,
                            "database": db_config['database'],
                            "table_name": table_name,
                            "query": query if query else f"SELECT * FROM {table_name}",
                            "created_at": datetime.now().isoformat(),
                            "created_by": self.user_id,
                            "original_rows": combined_table.num_rows,
                            "original_columns": combined_table.num_columns
                        }
                        self.redis_manager.register_dataset(
                            self.dataset_id, self.user_id, dataset_metadata
                        )
                        self.redis_manager.link_manager_to_dataset(
                            self.manager_id, self.dataset_id, self.user_id
                        )
                        logger.info("✅ Redis 데이터셋 등록 및 Manager-Dataset 링크 완료")
                    except Exception as e:
                        logger.warning(f"Redis 등록 실패: {e}")
                
                # MinIO 원본 저장
                if self.minio_storage:
                    try:
                        metadata_for_minio = dataset_metadata if 'dataset_metadata' in locals() else {}
                        self.minio_storage.save_original_dataset(
                            self.user_id, self.dataset_id, self.dataset, metadata_for_minio
                        )
                        logger.info(f"✅ MinIO 원본 저장 완료: raw-datasets/{self.user_id}/{self.dataset_id}/original.parquet")
                    except Exception as e:
                        logger.warning(f"MinIO 원본 저장 실패(계속): {e}")
            else:
                logger.info(f"♻️ 기존 Dataset 재로드: {self.dataset_id} (로드 {self.dataset_load_count}회차)")
            
            # ========== 5. 소스 정보 생성 ==========
            source_info = {
                "type": "database",
                "db_type": db_type,
                "database": db_config['database'],
                "table_name": table_name,
                "query": query if query else f"SELECT * FROM {table_name}",
                "loaded_at": datetime.now().isoformat(),
                "checksum": self._calculate_checksum(self.dataset),
                "num_rows": combined_table.num_rows,
                "num_columns": combined_table.num_columns,
                "columns": combined_table.column_names,
                "load_count": self.dataset_load_count,
                "is_reload": not is_first_load
            }
            
            # Redis에 소스 정보 저장
            if self.redis_manager:
                try:
                    self.redis_manager.save_source_info(self.dataset_id, source_info)
                    logger.info("✅ Redis 소스 정보 저장")
                except Exception as e:
                    logger.warning(f"Redis 소스 정보 저장 실패: {e}")
            
            # ========== 6. 버전 저장 ==========
            operation_name = "initial_load" if is_first_load else f"reload_{self.dataset_load_count}"
            logger.info(f"💾 버전 저장: operation={operation_name}, load_count={self.dataset_load_count}")
            
            self._save_version(operation_name, source_info)
            
            # ========== 7. 결과 반환 ==========
            result_info = {
                "success": True,
                "dataset_id": self.dataset_id,
                "manager_id": self.manager_id,
                "user_id": self.user_id,
                "db_type": db_type,
                "database": db_config['database'],
                "table_name": table_name,
                "query": query,
                "num_rows": combined_table.num_rows,
                "num_columns": combined_table.num_columns,
                "columns": combined_table.column_names,
                "loaded_at": datetime.now().isoformat(),
                "is_new_dataset": is_first_load,
                "current_version": self.current_version - 1,
                "load_count": self.dataset_load_count,
                "is_new_version": not is_first_load,
                "source_info": source_info
            }
            
            logger.info(f"✅ DB 데이터셋 로드 완료: dataset={self.dataset_id}, version={self.current_version - 1}")
            return result_info
            
        except ImportError as e:
            logger.error(f"필수 패키지 미설치: {e}")
            raise RuntimeError(f"DB 연결에 필요한 패키지가 설치되지 않았습니다: {str(e)}")
        except Exception as e:
            logger.error(f"DB 데이터셋 로드 실패: {e}", exc_info=True)
            raise RuntimeError(f"DB 데이터셋 로드 실패: {str(e)}")

    # ========== 정리 및 소멸자 ==========

    def cleanup(self):
        """리소스 정리 및 매니저 종료"""
        logger.info(f"Cleaning up DataManager {self.manager_id}")

        self.is_active = False
        self._monitoring = False

        # 모니터링 스레드 종료 대기
        if hasattr(self, '_monitor_thread') and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2)

        # 데이터 정리
        self.dataset = None

        # Redis Manager 세션 해제
        if self.redis_manager:
            try:
                self.redis_manager.unlink_manager(self.manager_id, self.user_id)
                logger.info(f"Manager 세션 해제: {self.manager_id}")
            except Exception as e:
                logger.warning(f"Manager 세션 해제 실패: {e}")

        # 강제 가비지 컬렉션
        gc.collect()

        logger.info(f"DataManager {self.manager_id} cleaned up successfully!")

    def __del__(self):
        """소멸자 - 자동 정리"""
        if hasattr(self, 'is_active') and self.is_active:
            try:
                self.cleanup()
            except Exception as e:
                logger.error(f"소멸자에서 정리 실패: {e}")
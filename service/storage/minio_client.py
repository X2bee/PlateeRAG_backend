# /service/storage/minio_client.py
from minio import Minio
from minio.error import S3Error
import io
import json
import os
from datetime import datetime
import logging
from typing import Dict, Any, Optional
import pyarrow.parquet as pq
import pyarrow as pa

logger = logging.getLogger(__name__)

class MinioDataStorage:
    def __init__(self, endpoint: Optional[str] = None,
                access_key: Optional[str] = None,
                secret_key: Optional[str] = None,
                secure: Optional[bool] = None):
        # 환경변수에서 읽기 (기본값 제공)
        endpoint = endpoint or os.getenv('MINIO_ENDPOINT', 'minio.x2bee.com')
        access_key = access_key or os.getenv('MINIO_DATA_ACCESS_KEY', 'minioadmin')
        secret_key = secret_key or os.getenv('MINIO_DATA_SECRET_KEY', 'minioadmin123')
        secure = secure if secure is not None else os.getenv('MINIO_SECURE', 'true').lower() == 'true'

        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )
        self.raw_datasets_bucket = "raw-datasets"
        self.versions_bucket = "dataset-versions"
        self.processed_bucket = "processed-datasets"  # <-- 추가
        self._init_buckets()

    def _init_buckets(self):
        """필요한 버킷들 생성"""
        buckets = [self.raw_datasets_bucket, self.versions_bucket, self.processed_bucket]
        
        for bucket in buckets:
            try:
                if not self.client.bucket_exists(bucket):
                    self.client.make_bucket(bucket)
                    logger.info(f"MinIO 버킷 생성: {bucket}")
            except S3Error as e:
                logger.error(f"버킷 생성 실패 {bucket}: {e}")
    
    def save_original_dataset(self, user_id: str, dataset_id: str, 
                             table: pa.Table, metadata: Dict) -> str:
        """원본 데이터셋 저장 (변경 없음 - 이미 dataset_id 사용)"""
        try:
            buffer = io.BytesIO()
            pq.write_table(table, buffer)
            buffer.seek(0)
            
            object_name = f"{user_id}/{dataset_id}/original.parquet"
            self.client.put_object(
                self.raw_datasets_bucket,
                object_name,
                buffer,
                length=buffer.getbuffer().nbytes,
                content_type="application/octet-stream"
            )
            
            logger.info(f"원본 저장: {object_name}")
            return object_name
            
        except S3Error as e:
            logger.error(f"MinIO 저장 실패: {e}")
            raise
    
    def save_version_snapshot(self, dataset_id: str, version: int,  # ⭐ manager_id → dataset_id
                             table: pa.Table, operation_name: str,
                             metadata: Dict) -> str:
        """버전 스냅샷 저장 (dataset_id 기준으로 변경)"""
        try:
            buffer = io.BytesIO()
            pq.write_table(table, buffer)
            buffer.seek(0)
            
            # ⭐ 핵심 변경: manager_id 대신 dataset_id 사용
            object_name = f"{dataset_id}/v{version}_{operation_name}.parquet"
            
            self.client.put_object(
                self.versions_bucket,
                object_name,
                buffer,
                length=buffer.getbuffer().nbytes,
                content_type="application/octet-stream",
                metadata={
                    "version": str(version),
                    "operation": operation_name,
                    "timestamp": datetime.now().isoformat(),
                    "rows": str(table.num_rows),
                    "columns": str(table.num_columns)
                }
            )
            
            logger.info(f"버전 저장: {object_name}")
            return object_name
            
        except S3Error as e:
            logger.error(f"버전 저장 실패: {e}")
            raise
    
    def load_version_snapshot(self, dataset_id: str, version: int,
                             operation_name: str) -> pa.Table:
        """버전 스냅샷 로드"""
        try:
            object_name = f"{dataset_id}/v{version}_{operation_name}.parquet"
            
            response = self.client.get_object(self.versions_bucket, object_name)
            buffer = io.BytesIO(response.read())
            response.close()
            response.release_conn()
            
            table = pq.read_table(buffer)
            logger.info(f"버전 로드: {object_name}")
            return table
            
        except S3Error as e:
            logger.error(f"버전 로드 실패: {e}")
            raise
    
    def list_versions(self, dataset_id: str) -> list:
        """데이터셋의 모든 버전 목록"""
        try:
            objects = self.client.list_objects(
                self.versions_bucket,
                prefix=f"{dataset_id}/",
                recursive=True
            )
            
            versions = []
            for obj in objects:
                versions.append({
                    "object_name": obj.object_name,
                    "size": obj.size,
                    "last_modified": obj.last_modified,
                    "metadata": obj.metadata
                })
            
            return versions
            
        except S3Error as e:
            logger.error(f"버전 목록 조회 실패: {e}")
            return []

    def delete_dataset(self, dataset_id: str, user_id: str) -> int:
        """
        데이터셋 완전 삭제 (원본 + 모든 버전)

        Args:
            dataset_id: 데이터셋 ID
            user_id: 사용자 ID (소유권 확인용)

        Returns:
            int: 삭제된 파일 개수
        """
        deleted_count = 0

        try:
            # 1. 원본 삭제 (raw-datasets bucket)
            original_path = f"{user_id}/{dataset_id}/original.parquet"
            try:
                self.client.remove_object(self.raw_datasets_bucket, original_path)
                deleted_count += 1
                logger.info(f"  ✅ 원본 삭제: {original_path}")
            except Exception as e:
                logger.debug(f"  원본 없음 또는 삭제 실패: {original_path}, {e}")

            # 2. 모든 버전 삭제 (dataset-versions bucket)
            try:
                objects = self.client.list_objects(
                    self.versions_bucket,
                    prefix=f"{dataset_id}/",
                    recursive=True
                )

                for obj in objects:
                    try:
                        self.client.remove_object(self.versions_bucket, obj.object_name)
                        deleted_count += 1
                        logger.debug(f"  ✅ 버전 삭제: {obj.object_name}")
                    except Exception as e:
                        logger.warning(f"  ⚠️  버전 삭제 실패: {obj.object_name}, {e}")

            except Exception as e:
                logger.warning(f"  ⚠️  버전 목록 조회 실패: {e}")

            logger.info(f"✅ 데이터셋 삭제 완료: {dataset_id} ({deleted_count}개 파일)")
            return deleted_count

        except S3Error as e:
            logger.error(f"❌ MinIO 삭제 실패: {dataset_id}, {e}")
            return deleted_count
        except Exception as e:
            logger.error(f"❌ 예상치 못한 삭제 오류: {dataset_id}, {e}")
            return deleted_count

    def delete_manager_data(self, manager_id: str):
        """매니저의 모든 버전 데이터 삭제"""
        try:
            objects = self.client.list_objects(
                self.versions_bucket,
                prefix=f"{manager_id}/",
                recursive=True,
            )
            
            for obj in objects:
                self.client.remove_object(self.versions_bucket, obj.object_name)
            
            logger.info(f"매니저 데이터 삭제 완료: {manager_id}")
            
        except S3Error as e:
            logger.error(f"데이터 삭제 실패: {e}")

    def load_original_dataset(self, user_id: str, dataset_id: str):
        """
        MinIO에서 원본 데이터셋 로드
        
        Args:
            user_id: 사용자 ID
            dataset_id: 데이터셋 ID (버전 포함)
            
        Returns:
            pyarrow.Table 또는 None
        """
        try:
            # save_original_dataset과 동일한 경로 사용
            object_name = f"{user_id}/{dataset_id}/original.parquet"
            
            # MinIO에서 파일 다운로드
            response = self.client.get_object(self.raw_datasets_bucket, object_name)
            data = response.read()
            response.close()
            response.release_conn()
            
            # Parquet 데이터를 PyArrow Table로 변환
            table = pq.read_table(io.BytesIO(data))
            
            logger.info(f"원본 데이터셋 로드 성공: {dataset_id} ({table.num_rows} rows)")
            return table
            
        except S3Error as e:
            if e.code == 'NoSuchKey':
                logger.warning(f"데이터셋을 찾을 수 없음: {object_name}")
                return None
            logger.error(f"MinIO에서 데이터셋 로드 실패: {e}")
            return None
        except Exception as e:
            logger.error(f"데이터셋 로드 중 오류: {e}")
            return None

# fix_existing_managers.py
import logging
import redis
import json
from service.storage.redis_version_manager import RedisVersionManager
from service.storage.minio_client import MinioDataStorage as MinIOStorage
from datetime import datetime
import os
import io
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("manager-recovery")

def recover_all_managers():
    """모든 매니저 복구 - Manager ID 보존"""
    
    # ===== Redis 클라이언트 초기화 =====
    redis_config = {
        'host': '192.168.2.242',
        'port': 6379,
        'db': 0,
        'password': 'redis_secure_password123!',
        'decode_responses': True,
        'socket_timeout': 5,
        'socket_connect_timeout': 5,
        'retry_on_timeout': True
    }
    
    logger.info(f"🔌 Redis 연결 시도: {redis_config['host']}:{redis_config['port']}")
    
    redis_client = redis.Redis(**redis_config)
    
    try:
        redis_client.ping()
        logger.info("✅ Redis 연결 성공")
    except Exception as e:
        logger.error(f"❌ Redis 연결 실패: {e}")
        return

    # MinIO Storage 초기화
    minio_storage = MinIOStorage(
        endpoint="minio.x2bee.com",
        access_key='minioadmin',
        secret_key='minioadmin123',
        secure=True
    )
    
    logger.info("🔧 매니저 복구 시작...")
    
    # Redis에서 모든 매니저 ID 조회
    all_manager_ids = set()
    cursor = 0
    
    while True:
        cursor, keys = redis_client.scan(
            cursor=cursor,
            match="manager:*:owner",
            count=100
        )
        
        for key in keys:
            if isinstance(key, bytes):
                key = key.decode('utf-8')
            
            parts = key.split(':')
            if len(parts) >= 3:
                manager_id = parts[1]
                all_manager_ids.add(manager_id)
        
        if cursor == 0:
            break
    
    logger.info(f"📋 발견된 매니저: {len(all_manager_ids)}개")
    
    if not all_manager_ids:
        logger.info("✅ 복구할 매니저 없음")
        redis_client.close()
        return
    
    recovered = 0
    failed = 0
    skipped = 0
    
    for manager_id in all_manager_ids:
        try:
            logger.info(f"\n🔄 복구 중: {manager_id}")
            
            owner_key = f"manager:{manager_id}:owner"
            owner = redis_client.get(owner_key)
            
            if not owner:
                logger.warning(f"  ⚠️ 소유자 정보 없음")
                failed += 1
                continue
            
            logger.info(f"  └─ 소유자: {owner}")
            
            # ✅ Manager → Dataset 매핑 확인 및 보존
            dataset_mapping_key = f"manager:{manager_id}:dataset"
            existing_dataset_id = redis_client.get(dataset_mapping_key)
            
            if existing_dataset_id:
                logger.info(f"  └─ 기존 Dataset 매핑 발견: {existing_dataset_id}")
                
                # Dataset의 현재 버전 확인
                dataset_version_key = f"dataset:{existing_dataset_id}:current_version"
                current_version = redis_client.get(dataset_version_key)
                current_version = int(current_version) if current_version else 0
                
                logger.info(f"  └─ Dataset 버전: {current_version}")
                
                # 버전 메타데이터 확인
                if current_version > 0:
                    version_meta_key = f"dataset:{existing_dataset_id}:version:{current_version - 1}"
                    meta_data = redis_client.get(version_meta_key)
                    
                    if meta_data:
                        try:
                            parsed = json.loads(meta_data)
                            if parsed.get('num_rows', 0) > 0:
                                logger.info(f"  ✅ 유효한 메타데이터 있음, Manager-Dataset 연결 확인")
                                
                                # ✅ Manager-Dataset 연결 보장
                                redis_client.set(dataset_mapping_key, existing_dataset_id)
                                
                                # ✅ User → Managers 연결 보장
                                user_managers_key = f"user:{owner}:managers"
                                redis_client.sadd(user_managers_key, manager_id)
                                
                                # ✅ Dataset → Owner 연결 보장
                                dataset_owner_key = f"dataset:{existing_dataset_id}:owner"
                                redis_client.set(dataset_owner_key, owner)
                                
                                logger.info(f"  ✅ Manager-Dataset 연결 보존: {manager_id} → {existing_dataset_id}")
                                recovered += 1
                                continue
                        except (json.JSONDecodeError, TypeError) as e:
                            logger.warning(f"  ⚠️ 메타데이터 파싱 실패: {e}")
            
            # ===== 기존 매핑이 없거나 유효하지 않은 경우 MinIO에서 복구 시도 =====
            logger.info(f"  🔍 MinIO에서 데이터 조회 중...")
            
            # MinIO에서 이 Manager와 연결된 Dataset 찾기
            # raw-datasets 버킷에서 owner의 모든 dataset 조회
            try:
                objects = minio_storage.client.list_objects(
                    minio_storage.raw_datasets_bucket,
                    prefix=f"{owner}/",
                    recursive=True
                )
                
                found_dataset = None
                for obj in objects:
                    # 경로 형식: owner/dataset_id/original.parquet
                    path_parts = obj.object_name.split('/')
                    if len(path_parts) >= 2:
                        potential_dataset_id = path_parts[1]
                        
                        # versions 버킷에서 해당 dataset의 버전 확인
                        version_objects = list(minio_storage.client.list_objects(
                            minio_storage.versions_bucket,
                            prefix=f"{potential_dataset_id}/",
                            recursive=False
                        ))
                        
                        if version_objects:
                            found_dataset = potential_dataset_id
                            logger.info(f"  └─ MinIO에서 Dataset 발견: {found_dataset}")
                            break
                
                if found_dataset:
                    # ✅ Manager-Dataset 연결 생성
                    redis_client.set(dataset_mapping_key, found_dataset)
                    
                    # ✅ User → Managers 연결
                    user_managers_key = f"user:{owner}:managers"
                    redis_client.sadd(user_managers_key, manager_id)
                    
                    # ✅ Dataset → Owner 연결
                    dataset_owner_key = f"dataset:{found_dataset}:owner"
                    redis_client.set(dataset_owner_key, owner)
                    
                    # ✅ User → Datasets 연결
                    user_datasets_key = f"user:{owner}:datasets"
                    redis_client.sadd(user_datasets_key, found_dataset)
                    
                    logger.info(f"  ✅ MinIO에서 복구 완료: {manager_id} → {found_dataset}")
                    recovered += 1
                    continue
                else:
                    logger.warning(f"  ⚠️ MinIO에서도 데이터를 찾을 수 없음")
                    failed += 1
                    
            except Exception as e:
                logger.error(f"  ❌ MinIO 조회 실패: {e}")
                failed += 1
            
        except Exception as e:
            logger.error(f"  ❌ 복구 실패: {manager_id}, {e}", exc_info=True)
            failed += 1
    
    logger.info(f"\n🎉 복구 완료!")
    logger.info(f"  ✅ 성공: {recovered}개")
    logger.info(f"  ⏭️  스킵: {skipped}개")
    logger.info(f"  ❌ 실패: {failed}개")
    logger.info("🔒 Redis 연결 종료")
    redis_client.close()

if __name__ == "__main__":
    recover_all_managers()
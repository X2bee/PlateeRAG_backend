"""
⚠️ 긴급 Manager 복구 스크립트 ⚠️

주의: 이 스크립트는 **긴급 상황에서만** 사용하세요!

정상 상황에서는 자동 복원 메커니즘이 작동합니다:
- 시작 시 자동 로드: main.py (Step 7.6)
- API 호출 시 Lazy Loading: DataManagerRegistry.get_manager()

이 스크립트가 필요한 경우:
1. ❌ Redis 데이터 손상 (Manager-Dataset 매핑 깨짐)
2. ❌ 시스템 마이그레이션 (데이터 구조 변경)
3. ❌ 비정상 종료 후 메타데이터 손실

사용법:
    python fix_existing_managers.py

자세한 내용은 README_RECOVERY.md를 참조하세요.

복구 작업:
- Manager ID 보존
- Manager ↔ Dataset ↔ User 연결 복원
- 소유자 정보 복구
"""
import logging
import redis
import json
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("manager-recovery")


class ManagerRecoveryTool:
    """Manager 복구 도구"""
    
    def __init__(self, redis_config: dict, minio_config: dict):
        """
        Args:
            redis_config: Redis 연결 설정
            minio_config: MinIO 연결 설정
        """
        self.redis_config = redis_config
        self.minio_config = minio_config
        self.redis_client = None
        self.minio_storage = None
        
    def connect(self) -> bool:
        """Redis 및 MinIO 연결"""
        try:
            # Redis 연결
            logger.info(
                f"🔌 Redis 연결 시도: "
                f"{self.redis_config['host']}:{self.redis_config['port']}"
            )
            
            self.redis_client = redis.Redis(**self.redis_config)
            self.redis_client.ping()
            logger.info("✅ Redis 연결 성공")
            
            # MinIO 연결
            from service.storage.minio_client import MinioDataStorage
            
            self.minio_storage = MinioDataStorage(
                endpoint=self.minio_config['endpoint'],
                access_key=self.minio_config['access_key'],
                secret_key=self.minio_config['secret_key'],
                secure=self.minio_config.get('secure', True)
            )
            logger.info("✅ MinIO 연결 성공")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 연결 실패: {e}", exc_info=True)
            return False
    
    def disconnect(self):
        """연결 종료"""
        if self.redis_client:
            self.redis_client.close()
            logger.info("🔒 Redis 연결 종료")
    
    def find_all_managers(self) -> set:
        """Redis에서 모든 Manager ID 찾기"""
        all_manager_ids = set()
        cursor = 0
        
        logger.info("🔍 Manager ID 검색 중...")
        
        while True:
            cursor, keys = self.redis_client.scan(
                cursor=cursor,
                match="manager:*:owner",
                count=100
            )
            
            for key in keys:
                if isinstance(key, bytes):
                    key = key.decode('utf-8')
                
                # manager:mgr_xxx:owner -> mgr_xxx 추출
                parts = key.split(':')
                if len(parts) >= 3:
                    manager_id = parts[1]
                    all_manager_ids.add(manager_id)
            
            if cursor == 0:
                break
        
        logger.info(f"📋 발견된 Manager: {len(all_manager_ids)}개")
        return all_manager_ids
    
    def get_manager_owner(self, manager_id: str) -> str:
        """Manager의 소유자 정보 가져오기"""
        owner_key = f"manager:{manager_id}:owner"
        owner = self.redis_client.get(owner_key)
        
        if isinstance(owner, bytes):
            owner = owner.decode('utf-8')
        
        return owner
    
    def get_existing_dataset(self, manager_id: str) -> tuple:
        """
        기존 Dataset 매핑 확인
        
        Returns:
            (dataset_id, is_valid) 튜플
        """
        dataset_mapping_key = f"manager:{manager_id}:dataset"
        existing_dataset_id = self.redis_client.get(dataset_mapping_key)
        
        if isinstance(existing_dataset_id, bytes):
            existing_dataset_id = existing_dataset_id.decode('utf-8')
        
        if not existing_dataset_id:
            return None, False
        
        # Dataset 유효성 검증
        dataset_version_key = f"dataset:{existing_dataset_id}:current_version"
        current_version = self.redis_client.get(dataset_version_key)
        current_version = int(current_version) if current_version else 0
        
        if current_version > 0:
            # 최신 버전 메타데이터 확인
            version_meta_key = f"dataset:{existing_dataset_id}:version:{current_version - 1}"
            meta_data = self.redis_client.get(version_meta_key)
            
            if meta_data:
                try:
                    if isinstance(meta_data, bytes):
                        meta_data = meta_data.decode('utf-8')
                    
                    parsed = json.loads(meta_data)
                    if parsed.get('num_rows', 0) > 0:
                        return existing_dataset_id, True
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"  ⚠️  메타데이터 파싱 실패: {e}")
        
        return existing_dataset_id, False
    
    def find_dataset_in_minio(self, owner: str) -> str:
        """MinIO에서 소유자의 Dataset 찾기"""
        try:
            objects = self.minio_storage.client.list_objects(
                self.minio_storage.raw_datasets_bucket,
                prefix=f"{owner}/",
                recursive=True
            )
            
            for obj in objects:
                # 경로 형식: owner/dataset_id/original.parquet
                path_parts = obj.object_name.split('/')
                if len(path_parts) >= 2:
                    potential_dataset_id = path_parts[1]
                    
                    # versions 버킷에서 해당 dataset의 버전 확인
                    version_objects = list(self.minio_storage.client.list_objects(
                        self.minio_storage.versions_bucket,
                        prefix=f"{potential_dataset_id}/",
                        recursive=False
                    ))
                    
                    if version_objects:
                        return potential_dataset_id
            
            return None
            
        except Exception as e:
            logger.error(f"  ❌ MinIO 조회 실패: {e}")
            return None
    
    def restore_manager_connections(
        self,
        manager_id: str,
        owner: str,
        dataset_id: str
    ) -> bool:
        """Manager와 Dataset의 모든 연결 복원"""
        try:
            # 1. Manager → Dataset 매핑
            dataset_mapping_key = f"manager:{manager_id}:dataset"
            self.redis_client.set(dataset_mapping_key, dataset_id)
            
            # 2. Manager → Owner 매핑 (이미 있지만 확인)
            owner_key = f"manager:{manager_id}:owner"
            self.redis_client.set(owner_key, owner)
            
            # 3. User → Managers 연결
            user_managers_key = f"user:{owner}:managers"
            self.redis_client.sadd(user_managers_key, manager_id)
            
            # 4. Dataset → Owner 매핑
            dataset_owner_key = f"dataset:{dataset_id}:owner"
            self.redis_client.set(dataset_owner_key, owner)
            
            # 5. User → Datasets 연결
            user_datasets_key = f"user:{owner}:datasets"
            self.redis_client.sadd(user_datasets_key, dataset_id)
            
            # 6. ✨ Manager 메타데이터 복원 (레지스트리 로드용)
            self.restore_manager_metadata(manager_id, owner, dataset_id)
            
            return True
            
        except Exception as e:
            logger.error(f"  ❌ 연결 복원 실패: {e}", exc_info=True)
            return False
    
    def restore_manager_metadata(
        self,
        manager_id: str,
        owner: str,
        dataset_id: str
    ):
        """Manager 메타데이터 복원 (레지스트리에서 사용)"""
        try:
            manager_meta_key = f"manager:{manager_id}:metadata"
            
            # 기존 메타데이터가 있는지 확인
            existing_meta = self.redis_client.get(manager_meta_key)
            
            if existing_meta:
                logger.debug(f"  └─ 기존 메타데이터 있음")
                return
            
            # 기본 메타데이터 생성
            metadata = {
                'manager_id': manager_id,
                'user_id': owner,
                'dataset_id': dataset_id,
                'created_at': datetime.now().isoformat(),
                'restored': True,
                'restored_at': datetime.now().isoformat()
            }
            
            self.redis_client.set(
                manager_meta_key,
                json.dumps(metadata)
            )
            
            logger.debug(f"  └─ 메타데이터 생성 완료")
            
        except Exception as e:
            logger.warning(f"  ⚠️  메타데이터 복원 실패: {e}")
    
    def recover_single_manager(self, manager_id: str) -> dict:
        """단일 Manager 복구"""
        result = {
            'manager_id': manager_id,
            'status': 'failed',
            'message': '',
            'owner': None,
            'dataset_id': None
        }
        
        try:
            logger.info(f"\n🔄 복구 중: {manager_id}")
            
            # 1. 소유자 정보 확인
            owner = self.get_manager_owner(manager_id)
            if not owner:
                result['message'] = '소유자 정보 없음'
                logger.warning(f"  ⚠️  {result['message']}")
                return result
            
            result['owner'] = owner
            logger.info(f"  └─ 소유자: {owner}")
            
            # 2. 기존 Dataset 매핑 확인
            dataset_id, is_valid = self.get_existing_dataset(manager_id)
            
            if dataset_id and is_valid:
                logger.info(f"  └─ 기존 Dataset 매핑 유효: {dataset_id}")
                result['dataset_id'] = dataset_id
                
                # 모든 연결 복원
                if self.restore_manager_connections(manager_id, owner, dataset_id):
                    result['status'] = 'success'
                    result['message'] = 'Redis에서 복원 (기존 매핑 사용)'
                    logger.info(f"  ✅ Manager-Dataset 연결 보존: {manager_id} → {dataset_id}")
                    return result
            
            # 3. MinIO에서 Dataset 찾기
            logger.info(f"  🔍 MinIO에서 데이터 조회 중...")
            found_dataset = self.find_dataset_in_minio(owner)
            
            if found_dataset:
                logger.info(f"  └─ MinIO에서 Dataset 발견: {found_dataset}")
                result['dataset_id'] = found_dataset
                
                # 모든 연결 복원
                if self.restore_manager_connections(manager_id, owner, found_dataset):
                    result['status'] = 'success'
                    result['message'] = 'MinIO에서 복원'
                    logger.info(f"  ✅ MinIO에서 복구 완료: {manager_id} → {found_dataset}")
                    return result
            else:
                result['message'] = 'MinIO에서도 데이터를 찾을 수 없음'
                logger.warning(f"  ⚠️  {result['message']}")
                return result
            
        except Exception as e:
            result['message'] = f'복구 중 에러: {str(e)}'
            logger.error(f"  ❌ {result['message']}", exc_info=True)
            return result
        
        return result
    
    def recover_all_managers(self) -> dict:
        """모든 Manager 복구"""
        logger.info("🔧 Manager 복구 시작...")
        
        # 모든 Manager ID 찾기
        all_manager_ids = self.find_all_managers()
        
        if not all_manager_ids:
            logger.info("✅ 복구할 Manager 없음")
            return {
                'total': 0,
                'success': 0,
                'failed': 0,
                'results': []
            }
        
        # 복구 실행
        results = []
        success_count = 0
        failed_count = 0
        
        for manager_id in sorted(all_manager_ids):
            result = self.recover_single_manager(manager_id)
            results.append(result)
            
            if result['status'] == 'success':
                success_count += 1
            else:
                failed_count += 1
        
        # 결과 요약
        summary = {
            'total': len(all_manager_ids),
            'success': success_count,
            'failed': failed_count,
            'results': results
        }
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🎉 복구 완료!")
        logger.info(f"  📊 전체: {summary['total']}개")
        logger.info(f"  ✅ 성공: {summary['success']}개")
        logger.info(f"  ❌ 실패: {summary['failed']}개")
        logger.info(f"{'='*60}")
        
        # 실패한 항목 상세 출력
        if failed_count > 0:
            logger.info("\n❌ 실패한 Manager 목록:")
            for result in results:
                if result['status'] == 'failed':
                    logger.info(
                        f"  - {result['manager_id']}: "
                        f"{result['message']} (owner: {result['owner']})"
                    )
        
        return summary
    
    def verify_recovery(self, manager_id: str) -> dict:
        """복구 검증"""
        try:
            logger.info(f"\n🔍 복구 검증: {manager_id}")
            
            checks = {
                'owner_exists': False,
                'dataset_mapping_exists': False,
                'user_managers_linked': False,
                'dataset_owner_linked': False,
                'metadata_exists': False
            }
            
            # 1. Owner 확인
            owner_key = f"manager:{manager_id}:owner"
            owner = self.redis_client.get(owner_key)
            checks['owner_exists'] = bool(owner)
            
            if not owner:
                logger.warning("  ❌ Owner 정보 없음")
                return checks
            
            if isinstance(owner, bytes):
                owner = owner.decode('utf-8')
            
            logger.info(f"  ✅ Owner: {owner}")
            
            # 2. Dataset 매핑 확인
            dataset_mapping_key = f"manager:{manager_id}:dataset"
            dataset_id = self.redis_client.get(dataset_mapping_key)
            checks['dataset_mapping_exists'] = bool(dataset_id)
            
            if dataset_id:
                if isinstance(dataset_id, bytes):
                    dataset_id = dataset_id.decode('utf-8')
                logger.info(f"  ✅ Dataset 매핑: {dataset_id}")
            else:
                logger.warning("  ❌ Dataset 매핑 없음")
                return checks
            
            # 3. User → Managers 연결 확인
            user_managers_key = f"user:{owner}:managers"
            is_member = self.redis_client.sismember(user_managers_key, manager_id)
            checks['user_managers_linked'] = bool(is_member)
            
            if is_member:
                logger.info(f"  ✅ User → Managers 연결됨")
            else:
                logger.warning("  ❌ User → Managers 연결 없음")
            
            # 4. Dataset → Owner 연결 확인
            dataset_owner_key = f"dataset:{dataset_id}:owner"
            dataset_owner = self.redis_client.get(dataset_owner_key)
            checks['dataset_owner_linked'] = bool(dataset_owner)
            
            if dataset_owner:
                logger.info(f"  ✅ Dataset → Owner 연결됨")
            else:
                logger.warning("  ❌ Dataset → Owner 연결 없음")
            
            # 5. 메타데이터 확인
            manager_meta_key = f"manager:{manager_id}:metadata"
            metadata = self.redis_client.get(manager_meta_key)
            checks['metadata_exists'] = bool(metadata)
            
            if metadata:
                logger.info(f"  ✅ 메타데이터 존재")
            else:
                logger.info(f"  ℹ️  메타데이터 없음 (선택사항)")
            
            # 전체 결과
            all_passed = all([
                checks['owner_exists'],
                checks['dataset_mapping_exists'],
                checks['user_managers_linked'],
                checks['dataset_owner_linked']
            ])
            
            if all_passed:
                logger.info(f"\n  ✅ 모든 검증 통과!")
            else:
                logger.warning(f"\n  ⚠️  일부 검증 실패")
            
            return checks
            
        except Exception as e:
            logger.error(f"  ❌ 검증 실패: {e}", exc_info=True)
            return {'error': str(e)}


def main():
    """메인 실행 함수"""
    
    # Redis 설정
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
    
    # MinIO 설정
    minio_config = {
        'endpoint': 'minio.x2bee.com',
        'access_key': 'minioadmin',
        'secret_key': 'minioadmin123',
        'secure': True
    }
    
    # 복구 도구 초기화
    recovery = ManagerRecoveryTool(redis_config, minio_config)
    
    try:
        # 연결
        if not recovery.connect():
            logger.error("❌ 연결 실패로 종료")
            return
        
        # 모든 Manager 복구
        summary = recovery.recover_all_managers()
        
        # 선택적: 특정 Manager 검증
        # if summary['success'] > 0:
        #     first_success = next(
        #         r for r in summary['results']
        #         if r['status'] == 'success'
        #     )
        #     recovery.verify_recovery(first_success['manager_id'])
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  사용자에 의해 중단됨")
    except Exception as e:
        logger.error(f"❌ 예상치 못한 에러: {e}", exc_info=True)
    finally:
        # 연결 종료
        recovery.disconnect()


if __name__ == "__main__":
    main()
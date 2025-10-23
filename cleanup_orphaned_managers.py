"""
고아 Manager 정리 스크립트

Redis와 DB Sync Config에서 owner 정보가 없는 Manager들을 찾아서 정리합니다.

사용법:
    python cleanup_orphaned_managers.py [--dry-run] [--force]

옵션:
    --dry-run: 실제로 삭제하지 않고 목록만 확인
    --force: 확인 없이 바로 삭제
"""
import redis
import argparse
import sys
import os
from typing import Set, List, Dict

# Redis 연결 정보
REDIS_HOST = "192.168.2.242"
REDIS_PORT = 6379
REDIS_PASSWORD = "redis_secure_password123!"
REDIS_DB = 0

# PostgreSQL 연결 정보 (환경변수 또는 기본값)
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_USER = os.getenv("POSTGRES_USER", "ailab")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "ailab123")
POSTGRES_DB = os.getenv("POSTGRES_DB", "plateerag")


def connect_redis() -> redis.Redis:
    """Redis 연결"""
    try:
        r = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=REDIS_PASSWORD,
            decode_responses=True
        )
        r.ping()
        print("✅ Redis 연결 성공")
        return r
    except Exception as e:
        print(f"❌ Redis 연결 실패: {e}")
        sys.exit(1)


def find_all_manager_ids(r: redis.Redis) -> Set[str]:
    """Redis에서 모든 Manager ID 찾기"""
    manager_ids = set()

    # 방법 1: manager:*:owner 패턴
    cursor = 0
    while True:
        cursor, keys = r.scan(cursor=cursor, match="manager:*:owner", count=100)
        for key in keys:
            parts = key.split(':')
            if len(parts) >= 3:
                manager_ids.add(parts[1])
        if cursor == 0:
            break

    # 방법 2: manager:*:dataset_id 패턴
    cursor = 0
    while True:
        cursor, keys = r.scan(cursor=cursor, match="manager:*:dataset_id", count=100)
        for key in keys:
            parts = key.split(':')
            if len(parts) >= 3:
                manager_ids.add(parts[1])
        if cursor == 0:
            break

    # 방법 3: manager:*:state 패턴
    cursor = 0
    while True:
        cursor, keys = r.scan(cursor=cursor, match="manager:*:state", count=100)
        for key in keys:
            parts = key.split(':')
            if len(parts) >= 3:
                manager_ids.add(parts[1])
        if cursor == 0:
            break

    # 방법 4: dataset:*:version:* 메타데이터에서 manager_id 추출
    cursor = 0
    while True:
        cursor, keys = r.scan(cursor=cursor, match="dataset:*:version:*", count=100)
        for key in keys:
            try:
                import json
                data = r.get(key)
                if data:
                    metadata = json.loads(data)
                    if 'metadata' in metadata and 'manager_id' in metadata['metadata']:
                        manager_ids.add(metadata['metadata']['manager_id'])
            except Exception:
                pass
        if cursor == 0:
            break

    return manager_ids


def check_manager_status(r: redis.Redis, manager_id: str) -> Dict[str, any]:
    """Manager의 상태 확인"""
    status = {
        'manager_id': manager_id,
        'has_owner': False,
        'owner': None,
        'has_dataset': False,
        'dataset_id': None,
        'has_state': False,
        'created_at': None,
        'orphaned': False
    }

    # Owner 확인
    owner_key = f"manager:{manager_id}:owner"
    owner = r.get(owner_key)
    if owner:
        status['has_owner'] = True
        status['owner'] = owner

    # Dataset 확인
    dataset_key = f"manager:{manager_id}:dataset_id"
    dataset_id = r.get(dataset_key)
    if dataset_id:
        status['has_dataset'] = True
        status['dataset_id'] = dataset_id

    # State 확인
    state_key = f"manager:{manager_id}:state"
    if r.exists(state_key):
        status['has_state'] = True

    # Created at 확인
    created_key = f"manager:{manager_id}:created_at"
    created_at = r.get(created_key)
    if created_at:
        status['created_at'] = created_at

    # 고아 판정: owner가 없으면 고아
    if not status['has_owner']:
        status['orphaned'] = True

    return status


def find_orphaned_managers(r: redis.Redis) -> List[Dict[str, any]]:
    """고아 Manager 찾기"""
    all_managers = find_all_manager_ids(r)
    print(f"\n📊 총 발견된 Manager: {len(all_managers)}개")

    orphaned = []
    for manager_id in all_managers:
        status = check_manager_status(r, manager_id)
        if status['orphaned']:
            orphaned.append(status)

    return orphaned


def delete_manager_keys(r: redis.Redis, manager_id: str) -> int:
    """Manager 관련 모든 Redis 키 삭제"""
    patterns = [
        f"manager:{manager_id}:owner",
        f"manager:{manager_id}:dataset_id",
        f"manager:{manager_id}:state",
        f"manager:{manager_id}:created_at",
        f"manager:{manager_id}:resource_stats",
    ]

    deleted_count = 0
    for pattern in patterns:
        if r.exists(pattern):
            r.delete(pattern)
            deleted_count += 1

    # user:*:active_managers에서도 제거
    cursor = 0
    while True:
        cursor, keys = r.scan(cursor=cursor, match="user:*:active_managers", count=100)
        for key in keys:
            if r.sismember(key, manager_id):
                r.srem(key, manager_id)
                deleted_count += 1
        if cursor == 0:
            break

    return deleted_count


def delete_db_sync_configs(manager_ids: List[str]) -> int:
    """DB Sync Config 테이블에서 고아 Manager 삭제"""
    try:
        import psycopg2

        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            database=POSTGRES_DB
        )
        cursor = conn.cursor()

        deleted_count = 0
        for manager_id in manager_ids:
            cursor.execute(
                "DELETE FROM db_sync_configs WHERE manager_id = %s",
                (manager_id,)
            )
            if cursor.rowcount > 0:
                deleted_count += cursor.rowcount

        conn.commit()
        cursor.close()
        conn.close()

        return deleted_count

    except Exception as e:
        print(f"⚠️  DB Sync Config 삭제 실패: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="고아 Manager 정리")
    parser.add_argument('--dry-run', action='store_true', help='실제로 삭제하지 않고 목록만 확인')
    parser.add_argument('--force', action='store_true', help='확인 없이 바로 삭제')
    args = parser.parse_args()

    print("=" * 70)
    print("🧹 고아 Manager 정리 스크립트")
    print("=" * 70)

    # Redis 연결
    r = connect_redis()

    # 고아 Manager 찾기
    print("\n🔍 고아 Manager 검색 중...")
    orphaned = find_orphaned_managers(r)

    if not orphaned:
        print("\n✅ 고아 Manager가 없습니다!")
        return

    print(f"\n⚠️  고아 Manager 발견: {len(orphaned)}개")
    print("-" * 70)

    # 목록 출력
    for i, status in enumerate(orphaned, 1):
        print(f"\n{i}. Manager ID: {status['manager_id']}")
        print(f"   └─ Owner: {status['owner'] or '❌ 없음'}")
        print(f"   └─ Dataset: {status['dataset_id'] or '❌ 없음'}")
        print(f"   └─ Created: {status['created_at'] or '❌ 없음'}")
        print(f"   └─ State: {'✅ 있음' if status['has_state'] else '❌ 없음'}")

    print("-" * 70)

    # Dry-run 모드
    if args.dry_run:
        print("\n💡 [DRY-RUN 모드] 실제로 삭제되지 않았습니다.")
        print(f"   삭제하려면 '--dry-run' 없이 실행하세요:")
        print(f"   python cleanup_orphaned_managers.py")
        return

    # 삭제 확인
    if not args.force:
        print(f"\n⚠️  {len(orphaned)}개의 고아 Manager를 삭제하시겠습니까?")
        print("   이 작업은 되돌릴 수 없습니다!")
        confirm = input("\n계속하려면 'yes'를 입력하세요: ")

        if confirm.lower() != 'yes':
            print("\n🚫 취소되었습니다.")
            return

    # 삭제 실행
    print("\n🗑️  삭제 중...")
    total_deleted_keys = 0

    for status in orphaned:
        manager_id = status['manager_id']
        deleted = delete_manager_keys(r, manager_id)
        total_deleted_keys += deleted
        print(f"   ✅ Redis: {manager_id}: {deleted}개 키 삭제")

    # DB Sync Config 정리
    print("\n🗑️  DB Sync Config 정리 중...")
    orphaned_ids = [s['manager_id'] for s in orphaned]
    deleted_db_configs = delete_db_sync_configs(orphaned_ids)

    print("\n" + "=" * 70)
    print(f"✅ 정리 완료!")
    print(f"   - 삭제된 Manager: {len(orphaned)}개")
    print(f"   - 삭제된 Redis 키: {total_deleted_keys}개")
    print(f"   - 삭제된 DB Sync Config: {deleted_db_configs}개")
    print("=" * 70)


if __name__ == "__main__":
    main()

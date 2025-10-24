# Manager 복구 가이드

## 자동 복원 메커니즘 (권장)

PlateeRAG 백엔드는 **자동 복원 기능**이 내장되어 있습니다.

### 정상 작동 방식

1. **시작 시 자동 로드** ([main.py:464-574](main.py#L464-L574))
   - Redis에서 모든 Manager 메타데이터 조회
   - MinIO에서 데이터 자동 복원
   - 메모리에 로드

2. **API 호출 시 Lazy Loading** ([data_manager_register.py:193-272](service/data_manager/data_manager_register.py#L193-L272))
   - 메모리에 없는 Manager 요청 시
   - Redis/MinIO에서 자동 복원
   - 투명하게 사용자에게 제공

### 정상 작동 확인

```bash
# 서버 시작 로그 확인
grep "자동 로드 완료" logs/app.log

# 또는 API 호출
curl http://localhost:8000/api/data-manager/stats
```

---

## 수동 복구 (긴급 상황만)

### fix_existing_managers.py 사용 시점

**다음 상황에서만** 수동 복구를 사용하세요:

1. ❌ **Redis 데이터 손상**
   - Manager-Dataset 매핑이 깨진 경우
   - `manager:xxx:owner` 키는 있지만 `manager:xxx:dataset` 키가 없는 경우

2. ❌ **시스템 마이그레이션**
   - Redis 데이터 구조 변경 시
   - 구버전 → 신버전 업그레이드 시

3. ❌ **비정상 종료 후 복구**
   - Redis 연결 끊김으로 인한 메타데이터 손실

### 복구 스크립트 실행

```bash
# 1. 가상환경 활성화
source venv/bin/activate  # 또는 conda activate

# 2. 복구 스크립트 실행
python fix_existing_managers.py

# 3. 결과 확인
# - ✅ 성공한 Manager 개수
# - ❌ 실패한 Manager 개수 및 원인
```

### 복구 스크립트가 하는 일

1. Redis에서 모든 `manager:*:owner` 키 스캔
2. 각 Manager의 Dataset 매핑 확인
3. 없으면 MinIO에서 Dataset 찾기
4. Manager ↔ Dataset ↔ User 연결 복원

---

## 문제 해결

### Q: "Manager를 찾을 수 없습니다" 에러

**자동 복원이 실패한 경우입니다.**

```bash
# 1. Redis 연결 확인
redis-cli -h 192.168.2.242 -p 6379 -a redis_secure_password123! ping

# 2. Manager 메타데이터 확인
redis-cli -h 192.168.2.242 -p 6379 -a redis_secure_password123!
> KEYS manager:*:owner
> GET manager:YOUR_MANAGER_ID:owner
> GET manager:YOUR_MANAGER_ID:dataset

# 3. Dataset 매핑이 없으면 수동 복구 실행
python fix_existing_managers.py
```

### Q: MinIO 연결 실패

```bash
# MinIO 연결 테스트
curl -I https://minio.x2bee.com

# MinIO에서 Dataset 확인
mc ls myminio/raw-datasets/
mc ls myminio/versions/
```

### Q: 복구 후에도 작동 안 함

```bash
# 1. 애플리케이션 재시작
pkill -f "uvicorn main:app"
uvicorn main:app --host 0.0.0.0 --port 8000

# 2. 로그 확인
tail -f logs/app.log | grep -E "자동 로드|복원"

# 3. Redis 데이터 수동 확인
redis-cli -h 192.168.2.242 -p 6379 -a redis_secure_password123!
> KEYS user:*:managers
> SMEMBERS user:YOUR_USER_ID:managers
```

---

## 예방 조치

### 1. Redis 백업 (권장)

```bash
# Redis RDB 스냅샷 생성
redis-cli -h 192.168.2.242 -p 6379 -a redis_secure_password123! BGSAVE

# 백업 파일 확인
ls -lh /var/lib/redis/dump.rdb
```

### 2. MinIO 백업 (선택)

```bash
# MinIO 버킷 미러링
mc mirror myminio/raw-datasets /backup/raw-datasets
mc mirror myminio/versions /backup/versions
```

### 3. 정기 헬스 체크

```bash
# 스크립트 작성: check_health.sh
#!/bin/bash

# Redis 연결 확인
redis-cli -h 192.168.2.242 -p 6379 -a redis_secure_password123! ping

# MinIO 연결 확인
curl -I https://minio.x2bee.com

# API 헬스 체크
curl http://localhost:8000/api/data-manager/stats

# 결과 로그
echo "Health check passed at $(date)" >> health_check.log
```

---

## 정리

- ✅ **정상 상황**: 자동 복원이 작동합니다. 별도 작업 불필요.
- ⚠️ **비정상 상황**: `fix_existing_managers.py` 실행
- 🔧 **예방**: Redis/MinIO 백업 및 헬스 체크

**대부분의 경우 자동 복원이 작동하므로, 수동 복구는 긴급 상황에서만 사용하세요.**

# VastAI 인스턴스 관리 API 가이드

이 문서는 VastAI 인스턴스를 검색, 생성, 관리하는 API 사용법을 설명합니다.

## 주요 기능

### 1. 인스턴스 검색 (오퍼 검색)
- **기본 검색**: `GET /api/vast/offers`
- **고급 검색**: `POST /api/vast/search-offers`

### 2. 템플릿 기반 인스턴스 생성
- **템플릿 목록**: `GET /api/vast/templates`
- **템플릿 상세**: `GET /api/vast/templates/{template_name}`
- **템플릿 기반 생성**: `POST /api/vast/instances/from-template`

### 3. 인스턴스 DB 저장 및 관리
- **자동 DB 저장**: 인스턴스 생성 시 자동으로 DB에 저장
- **상태 업데이트**: 인스턴스 상태 변경 시 자동 업데이트

### 4. 인스턴스 상태 조회
- **기본 상태**: `GET /api/vast/instances/{instance_id}`
- **상세 상태**: `GET /api/vast/instances/{instance_id}/enhanced`
- **비용 분석**: `GET /api/vast/instances/{instance_id}/cost-analysis`

## API 엔드포인트 상세

### 1. 인스턴스 검색

#### 기본 오퍼 검색
```bash
GET /api/vast/offers?gpu_name=RTX4090&max_price=1.5&rentable=true
```

#### 고급 오퍼 검색
```bash
POST /api/vast/search-offers
Content-Type: application/json

{
  "gpu_name": "RTX4090",
  "max_price": 2.0,
  "min_gpu_ram": 16,
  "num_gpus": 1,
  "rentable": true,
  "sort_by": "price",
  "limit": 10
}
```

**응답 예시:**
```json
{
  "offers": [
    {
      "id": "12345",
      "gpu_name": "RTX 4090",
      "num_gpus": 1,
      "gpu_ram": 24,
      "dph_total": 1.2,
      "rentable": true
    }
  ],
  "total": 50,
  "filtered_count": 10,
  "sort_info": {
    "sort_by": "price",
    "order": "ascending"
  }
}
```

### 2. 템플릿 관리

#### 사용 가능한 템플릿 목록 조회
```bash
GET /api/vast/templates
```

**응답:**
```json
{
  "templates": [
    {
      "name": "high_performance",
      "config": {
        "vllm_model_name": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "vllm_max_model_len": 8192,
        "min_gpu_ram": 24,
        "max_price": 2.0
      }
    },
    {
      "name": "budget",
      "config": {
        "vllm_model_name": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "min_gpu_ram": 8,
        "max_price": 0.5
      }
    }
  ]
}
```

#### 특정 템플릿 상세 정보
```bash
GET /api/vast/templates/high_performance
```

### 3. 인스턴스 생성

#### 템플릿 기반 인스턴스 생성
```bash
POST /api/vast/instances/from-template?template_name=high_performance&offer_id=12345
Content-Type: application/json

{
  "custom_config": {
    "vllm_max_model_len": 4096
  }
}
```

#### 일반 인스턴스 생성 (템플릿 지원)
```bash
POST /api/vast/instances
Content-Type: application/json

{
  "offer_id": "12345",
  "template_name": "budget",
  "custom_config": {
    "auto_destroy": true
  }
}
```

**응답:**
```json
{
  "success": true,
  "instance_id": "inst_67890",
  "template_name": "high_performance",
  "message": "템플릿 'high_performance'을 사용하여 인스턴스가 생성되었습니다.",
  "status": "creating"
}
```

### 4. 인스턴스 상태 조회

#### 기본 상태 조회
```bash
GET /api/vast/instances/inst_67890
```

#### 상세 상태 조회 (DB 정보 포함)
```bash
GET /api/vast/instances/inst_67890/enhanced
```

**응답:**
```json
{
  "success": true,
  "data": {
    "instance_id": "inst_67890",
    "basic_status": {
      "status": "running",
      "public_ip": "192.168.1.100"
    },
    "vllm_status": {
      "running": true,
      "port": 8000
    },
    "db_info": {
      "status": "running_vllm",
      "cost_per_hour": 1.2,
      "gpu_info": {
        "gpu_name": "RTX 4090",
        "num_gpus": 1,
        "gpu_ram": 24
      },
      "uptime": "2:30:45"
    }
  }
}
```

#### 비용 분석
```bash
GET /api/vast/instances/inst_67890/cost-analysis
```

**응답:**
```json
{
  "instance_id": "inst_67890",
  "cost_per_hour": 1.2,
  "total_cost": 3.0,
  "uptime_hours": 2.5,
  "cost_breakdown": {
    "hourly_rate": 1.2,
    "uptime_days": 0.1,
    "estimated_daily_cost": 28.8,
    "estimated_monthly_cost": 864.0
  }
}
```

### 5. 인스턴스 목록 및 통계

#### 향상된 인스턴스 목록 조회
```bash
GET /api/vast/instances?status_filter=running&sort_by=created_at
```

#### 전체 통계
```bash
GET /api/vast/statistics
```

**응답:**
```json
{
  "total_instances": 15,
  "active_instances": 8,
  "destroyed_instances": 7,
  "total_cost": 245.67,
  "average_cost_per_hour": 1.35,
  "gpu_distribution": {
    "RTX 4090": 5,
    "RTX 3090": 3
  },
  "status_distribution": {
    "running": 6,
    "creating": 2,
    "exited": 7
  }
}
```

## 사용 가능한 템플릿

### 1. high_performance
- **목적**: 고성능 작업, 대용량 모델
- **GPU**: RTX4090, A100 (24GB+ RAM)
- **모델**: Qwen2.5-Coder-32B-Instruct
- **최대 가격**: $2.0/hour

### 2. budget
- **목적**: 개발, 테스트, 소규모 작업
- **GPU**: RTX3080, RTX3090, GTX1080Ti (8GB+ RAM)
- **모델**: Qwen2.5-Coder-7B-Instruct
- **최대 가격**: $0.5/hour

### 3. research
- **목적**: 연구, 실험
- **GPU**: RTX4070, RTX4080, RTX3090 (16GB+ RAM)
- **모델**: Qwen2.5-Coder-14B-Instruct
- **최대 가격**: $1.0/hour

## 사용 예시

### 1. 예산형 인스턴스 빠른 생성
```bash
# 1. 템플릿 확인
curl -X GET "http://localhost:8001/api/vast/templates/budget"

# 2. 적합한 오퍼 검색
curl -X POST "http://localhost:8001/api/vast/search-offers" \
  -H "Content-Type: application/json" \
  -d '{"max_price": 0.5, "min_gpu_ram": 8, "limit": 5}'

# 3. 템플릿으로 인스턴스 생성
curl -X POST "http://localhost:8001/api/vast/instances/from-template?template_name=budget"
```

### 2. 커스텀 설정으로 고성능 인스턴스 생성
```bash
curl -X POST "http://localhost:8001/api/vast/instances" \
  -H "Content-Type: application/json" \
  -d '{
    "template_name": "high_performance",
    "custom_config": {
      "vllm_max_model_len": 16384,
      "vllm_gpu_memory_utilization": 0.9
    }
  }'
```

### 3. 인스턴스 모니터링
```bash
# 상세 상태 확인
curl -X GET "http://localhost:8001/api/vast/instances/inst_12345/enhanced"

# 비용 분석
curl -X GET "http://localhost:8001/api/vast/instances/inst_12345/cost-analysis"

# 전체 통계
curl -X GET "http://localhost:8001/api/vast/statistics"
```

## 데이터베이스 저장 정보

인스턴스 생성 시 다음 정보가 자동으로 DB에 저장됩니다:

- **기본 정보**: instance_id, offer_id, image_name, status
- **GPU 정보**: gpu_name, num_gpus, gpu_ram (JSON)
- **비용 정보**: cost_per_hour
- **네트워크 정보**: public_ip, ssh_port, port_mappings (JSON)
- **템플릿 정보**: template_name (사용된 템플릿)
- **시간 정보**: created_at, updated_at, destroyed_at
- **설정 정보**: start_command, auto_destroy

## 로그 시스템

모든 인스턴스 작업은 VastExecutionLog 테이블에 기록됩니다:

- **작업 정보**: operation, command, result
- **성능 정보**: execution_time, success
- **메타데이터**: template_name, gpu_info, cost_per_hour 등 (JSON)

## 에러 처리

API는 다음과 같은 HTTP 상태 코드를 반환합니다:

- **200**: 성공
- **400**: 잘못된 요청 (템플릿 없음, 파라미터 오류 등)
- **404**: 리소스 없음 (인스턴스 없음, 템플릿 없음 등)
- **500**: 서버 오류

에러 응답 예시:
```json
{
  "detail": "템플릿 'invalid_template'을 찾을 수 없습니다"
}
``` 
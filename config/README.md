# PlateeRAG Backend Configuration System

PlateeRAG Backend는 환경변수, 파일, 데이터베이스를 통한 계층적 설정 관리 시스템을 제공합니다. 이 문서는 설정 시스템의 구조, 사용법, 그리고 API를 통한 동적 관리 방법을 설명합니다.

## 📁 Directory Structure

```
config/
├── README.md                  # 이 문서
├── persistent_config.py       # 핵심 설정 관리 클래스
├── base_config.py            # 공통 설정 베이스 클래스
├── config_composer.py        # 설정 통합 관리자
└── sub_config/
    ├── __init__.py
    ├── openai_config.py      # OpenAI API 관련 설정
    ├── app_config.py         # 애플리케이션 기본 설정
    ├── workflow_config.py    # 워크플로우 관련 설정
    └── node_config.py        # 노드 시스템 관련 설정
```

## 🏗️ Configuration Hierarchy

설정값은 다음과 같은 우선순위로 결정됩니다:

1. **환경변수** (최고 우선순위)
2. **설정 파일** (예: `openai_api_key.txt`)
3. **데이터베이스** (PostgreSQL 또는 SQLite)
4. **기본값** (코드에 정의된 값)

## 🔧 Configuration Categories

### 1. OpenAI Configuration (`openai_config.py`)

OpenAI API와 관련된 모든 설정을 관리합니다.

| 환경변수 | 설정 파일 | 기본값 | 설명 |
|---------|---------|-------|------|
| `OPENAI_API_KEY` | `openai_api_key.txt` | `""` | OpenAI API 키 |
| `OPENAI_MODEL_DEFAULT` | - | `"gpt-3.5-turbo"` | 기본 AI 모델 |
| `OPENAI_API_BASE_URL` | - | `"https://api.openai.com/v1"` | API 엔드포인트 |
| `OPENAI_TEMPERATURE_DEFAULT` | - | `0.7` | 생성 온도 |
| `OPENAI_MAX_TOKENS_DEFAULT` | - | `1000` | 최대 토큰 수 |
| `OPENAI_REQUEST_TIMEOUT` | - | `30` | 요청 타임아웃 (초) |

### 2. Application Configuration (`app_config.py`)

애플리케이션의 기본 설정을 관리합니다.

| 환경변수 | 기본값 | 설명 |
|---------|-------|------|
| `APP_ENVIRONMENT` | `"development"` | 실행 환경 |
| `DEBUG_MODE` | `false` | 디버그 모드 활성화 |
| `LOG_LEVEL` | `"INFO"` | 로그 레벨 |
| `APP_HOST` | `"0.0.0.0"` | 서버 호스트 |
| `APP_PORT` | `8000` | 서버 포트 |
| `CORS_ORIGINS` | `["*"]` | CORS 허용 오리진 |
| `DATA_DIRECTORIES` | `["constants", "downloads"]` | 데이터 디렉토리 목록 |
| `LOG_FORMAT` | `"%(asctime)s - %(name)s - %(levelname)s - %(message)s"` | 로그 포맷 |

### 3. Workflow Configuration (`workflow_config.py`)

워크플로우 실행과 관련된 설정을 관리합니다.

| 환경변수 | 기본값 | 설명 |
|---------|-------|------|
| `WORKFLOW_TIMEOUT` | `300` | 워크플로우 타임아웃 (초) |
| `MAX_WORKFLOW_NODES` | `100` | 최대 노드 개수 |
| `WORKFLOW_ALLOW_PARALLEL` | `true` | 병렬 실행 허용 |
| `WORKFLOW_ENABLE_CACHING` | `true` | 결과 캐싱 활성화 |
| `MAX_CONCURRENT_WORKFLOWS` | `5` | 최대 동시 실행 수 |
| `WORKFLOW_SAVE_LOGS` | `true` | 실행 로그 저장 |

### 4. Node Configuration (`node_config.py`)

노드 시스템과 관련된 설정을 관리합니다.

| 환경변수 | 기본값 | 설명 |
|---------|-------|------|
| `NODE_CACHE_ENABLED` | `true` | 노드 캐싱 활성화 |
| `NODE_AUTO_DISCOVERY` | `true` | 자동 노드 발견 |
| `NODE_VALIDATION_ENABLED` | `true` | 노드 검증 활성화 |
| `NODE_EXECUTION_TIMEOUT` | `60` | 노드 실행 타임아웃 (초) |
| `NODE_REGISTRY_FILE_PATH` | `"constants/exported_nodes.json"` | 노드 레지스트리 파일 경로 |
| `NODE_DEBUG_MODE` | `false` | 노드 디버그 모드 |

## 💾 Database Configuration (구현 예정)

### Database Selection Priority

1. **PostgreSQL** (우선순위)
   - `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD` 환경변수가 모두 설정된 경우
   
2. **SQLite** (기본값)
   - PostgreSQL 접속 정보가 불완전하거나 없는 경우 자동으로 SQLite 사용
   - 기본 파일 경로: `constants/config.db`

### Database Environment Variables (구현 예정)

| 환경변수 | 기본값 | 설명 |
|---------|-------|------|
| `DATABASE_TYPE` | `"auto"` | 데이터베이스 타입 (`auto`, `sqlite`, `postgresql`) |
| `POSTGRES_HOST` | - | PostgreSQL 호스트 |
| `POSTGRES_PORT` | `5432` | PostgreSQL 포트 |
| `POSTGRES_DB` | `plateerag` | 데이터베이스 이름 |
| `POSTGRES_USER` | - | 사용자명 |
| `POSTGRES_PASSWORD` | - | 비밀번호 |
| `SQLITE_PATH` | `"constants/config.db"` | SQLite 파일 경로 |

## 🚀 Usage Examples

### 1. 환경변수를 통한 설정

```bash
# 개발 환경에서 포트 변경
export APP_PORT=8001
python main.py

# OpenAI API 키 설정
export OPENAI_API_KEY="sk-your-api-key-here"

# PostgreSQL 데이터베이스 사용 (구현 예정)
export POSTGRES_HOST="localhost"
export POSTGRES_USER="plateerag"
export POSTGRES_PASSWORD="your-password"
export POSTGRES_DB="plateerag_config"

# 프로덕션 환경 설정
export APP_ENVIRONMENT="production"
export DEBUG_MODE="false"
export LOG_LEVEL="WARNING"
```

### 2. 설정 파일을 통한 설정

```bash
# OpenAI API 키를 파일로 설정
echo "sk-your-api-key-here" > openai_api_key.txt
```

### 3. Python 코드에서 설정 접근

```python
from config.config_composer import config_composer

# 애플리케이션 시작 시 설정 초기화
configs = config_composer.initialize_all_configs()

# FastAPI app.state에 설정 저장
app.state.config = configs

# 설정값 사용
api_key = app.state.config["openai"].API_KEY.value
port = app.state.config["app"].PORT.value
debug_mode = app.state.config["app"].DEBUG_MODE.value
```

## 🌐 REST API Management

### 1. 설정 조회

#### 전체 설정 조회
```bash
GET /app/config
```

**응답 예시:**
```json
{
  "total_configs": 26,
  "categories": {
    "openai": {
      "class_name": "OpenAIConfig",
      "config_count": 6,
      "configs": {
        "OPENAI_API_KEY": {
          "current_value": "sk-proj-...",
          "default_value": "",
          "config_path": "openai.api_key"
        }
      }
    }
  }
}
```

#### 애플리케이션 상태 조회
```bash
GET /app/status
```

**응답 예시:**
```json
{
  "config": {
    "app_name": "PlateeRAG Backend",
    "version": "1.0.0",
    "environment": "development",
    "debug_mode": false
  },
  "node_count": 6,
  "available_nodes": ["chat/openai", "math/add_integers"],
  "status": "running"
}
```

### 2. 설정 수정

#### 개별 설정 업데이트
```bash
PUT /app/config/persistent/{config_name}
Content-Type: application/json

{
  "value": "new_value"
}
```

**예시:**
```bash
# OpenAI 최대 토큰 수 변경
curl -X PUT -H "Content-Type: application/json" \
  -d '{"value": 1500}' \
  http://localhost:8000/app/config/persistent/OPENAI_MAX_TOKENS_DEFAULT
```

**응답:**
```json
{
  "message": "Config 'OPENAI_MAX_TOKENS_DEFAULT' updated successfully",
  "old_value": 1000,
  "new_value": 1500
}
```

### 3. 설정 관리

#### 모든 설정 새로고침 (DB에서 다시 로드)
```bash
POST /app/config/persistent/refresh
```

#### 모든 설정 저장 (현재 값을 DB에 저장)
```bash
POST /app/config/persistent/save
```

## 🔄 Configuration Lifecycle

### 1. 시스템 시작 시 (main.py)

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. 설정 시스템 초기화
    configs = config_composer.initialize_all_configs()
    
    # 2. app.state에 설정 저장
    app.state.config = configs
    
    # 3. 설정 유효성 검증
    validation_result = config_composer.validate_critical_configs()
    
    # 4. 필요한 디렉토리 생성
    config_composer.ensure_directories()
    
    # 5. 노드 discovery (설정에 따라)
    if configs["node"].AUTO_DISCOVERY.value:
        run_discovery()
    
    yield  # 애플리케이션 실행
    
    # 6. 종료 시 설정 저장
    config_composer.save_all()
```

### 2. 설정 로드 순서

1. **BaseConfig.get_env_value()** 호출
2. 환경변수에서 값 확인
3. 설정 파일에서 값 확인 (있는 경우)
4. **PersistentConfig** 생성 시 데이터베이스에서 값 확인
5. 기본값 사용 (모든 소스에서 값을 찾지 못한 경우)

### 3. 설정 우선순위 예시

```python
# 예: OPENAI_MAX_TOKENS_DEFAULT 설정 결정 과정
# 1. 환경변수 확인: os.environ.get("OPENAI_MAX_TOKENS_DEFAULT")
# 2. 파일 확인: 해당 없음 (이 설정은 파일 소스 없음)
# 3. 데이터베이스 확인: constants/config.json에서 "openai.max_tokens_default" 경로
# 4. 기본값 사용: 1000
```

## 🔒 Security Considerations

### 1. 민감한 정보 보호

- **API 키**: 환경변수나 별도 파일에 저장, 데이터베이스에는 암호화된 형태로 저장 권장
- **데이터베이스 접속 정보**: 환경변수로만 설정, 코드에 하드코딩 금지

### 2. 설정 접근 제한

```python
# 민감한 설정은 API 응답에서 마스킹
if config_name.endswith("_KEY") or config_name.endswith("_PASSWORD"):
    display_value = "***masked***"
else:
    display_value = config.value
```

## 🐛 Troubleshooting

### 1. 설정이 반영되지 않는 경우

```bash
# 1. 설정 새로고침
curl -X POST http://localhost:8000/app/config/persistent/refresh

# 2. 환경변수 확인
env | grep OPENAI

# 3. 데이터베이스 상태 확인
curl http://localhost:8000/app/config | jq '.persistent_summary'
```

### 2. 데이터베이스 연결 문제 (구현 예정)

```bash
# SQLite 파일 권한 확인
ls -la constants/config.db

# PostgreSQL 연결 테스트
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB -c "SELECT 1;"
```

### 3. 로그 확인

```bash
# 설정 관련 로그 필터링
tail -f app.log | grep -E "(config|persistent)"
```

## 📝 Development Guide

### 1. 새로운 설정 카테고리 추가

1. `sub_config/` 폴더에 새 파일 생성:
```python
# sub_config/example_config.py
from config.base_config import BaseConfig, PersistentConfig

class ExampleConfig(BaseConfig):
    def initialize(self):
        self.EXAMPLE_SETTING = self.create_persistent_config(
            env_name="EXAMPLE_SETTING",
            config_path="example.setting",
            default_value="default_value"
        )
        return self.configs
```

2. `config_composer.py`에 추가:
```python
from config.sub_config.example_config import ExampleConfig

class ConfigComposer:
    def __init__(self):
        # ...existing code...
        self.example = ExampleConfig()
    
    def initialize_all_configs(self):
        # ...existing code...
        example_configs = self.example.initialize()
        self.all_configs.update(example_configs)
        
        return {
            # ...existing code...
            "example": self.example,
        }
```

### 2. 설정 타입 변환

```python
# 타입 변환 함수 사용
self.BOOLEAN_SETTING = self.create_persistent_config(
    env_name="BOOLEAN_SETTING",
    config_path="app.boolean_setting",
    default_value=False,
    type_converter=convert_to_bool
)

self.LIST_SETTING = self.create_persistent_config(
    env_name="LIST_SETTING",
    config_path="app.list_setting",
    default_value=["item1", "item2"],
    type_converter=lambda x: convert_to_list(x, separator=",")
)
```

## 📚 References

- [FastAPI Configuration](https://fastapi.tiangolo.com/advanced/settings/)
- [Python Logging](https://docs.python.org/3/library/logging.html)
- [Environment Variables Best Practices](https://12factor.net/config)
- [Database Migration Patterns](https://martinfowler.com/articles/evodb.html)

---

## 🚀 Future Enhancements

### 1. Database Integration (구현 예정)

- PostgreSQL과 SQLite 지원
- 자동 마이그레이션 시스템
- 설정 변경 이력 추적

### 2. Configuration Validation (구현 예정)

- JSON Schema 기반 설정 검증
- 설정 간 의존성 검사
- 런타임 설정 제약 조건 확인

### 3. Configuration Hot Reload (구현 예정)

- 설정 변경 시 자동 재로드
- 웹소켓을 통한 실시간 설정 동기화
- 설정 변경 알림 시스템

### 4. Configuration UI (구현 예정)

- 웹 기반 설정 관리 인터페이스
- 설정 변경 승인 워크플로우
- 설정 백업 및 복원 기능

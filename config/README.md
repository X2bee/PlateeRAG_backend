# PlateERAG Backend 설정 시스템 가이드

## 📖 개요

PlateERAG Backend는 **36개의 설정 매개변수**를 5개 카테고리로 관리하며, 환경변수와 데이터베이스를 통한 영속적 설정 관리를 지원합니다.

### 🎯 핵심 특징
- **환경변수 우선**: 환경변수로 모든 설정 덮어쓰기 가능
- **자동 DB 저장**: 변경된 설정은 SQLite/PostgreSQL에 자동 저장
- **실시간 변경**: API를 통한 런타임 설정 수정
- **타입 안전**: 강타입 검증 (문자열, 숫자, 불린, 리스트)

## 🗂️ 설정 카테고리 (총 36개)

### 1. 데이터베이스 설정 (DATABASE) - 10개

| 환경변수 | 기본값 | 타입 | 설명 |
|----------|--------|------|------|
| `DATABASE_TYPE` | `auto` | string | 데이터베이스 타입 (`auto`, `sqlite`, `postgresql`) |
| `POSTGRES_HOST` | `""` | string | PostgreSQL 호스트 주소 |
| `POSTGRES_PORT` | `5432` | integer | PostgreSQL 포트 번호 |
| `POSTGRES_DB` | `plateerag` | string | PostgreSQL 데이터베이스 이름 |
| `POSTGRES_USER` | `""` | string | PostgreSQL 사용자명 |
| `POSTGRES_PASSWORD` | `""` | string | PostgreSQL 비밀번호 |
| `SQLITE_PATH` | `constants/config.db` | string | SQLite 파일 경로 |
| `DB_POOL_SIZE` | `5` | integer | 데이터베이스 연결 풀 크기 |
| `DB_POOL_MAX_OVERFLOW` | `10` | integer | 연결 풀 최대 오버플로우 |
| `AUTO_MIGRATION` | `True` | boolean | 자동 마이그레이션 활성화 |

**사용 예시:**
```bash
# PostgreSQL 사용
export POSTGRES_HOST=localhost
export POSTGRES_USER=myuser
export POSTGRES_PASSWORD=mypass
export DATABASE_TYPE=postgresql

# SQLite 사용 (기본값)
export DATABASE_TYPE=sqlite
export SQLITE_PATH=data/my_config.db
```

### 2. OpenAI API 설정 (OPENAI) - 6개

| 환경변수 | 기본값 | 타입 | 설명 |
|----------|--------|------|------|
| `OPENAI_API_KEY` | `sk-proj-...` | string | OpenAI API 키 (환경변수 필수) |
| `OPENAI_MODEL_DEFAULT` | `gpt-3.5-turbo` | string | 기본 AI 모델 |
| `OPENAI_API_BASE_URL` | `https://api.openai.com/v1` | string | API 베이스 URL |
| `OPENAI_TEMPERATURE_DEFAULT` | `0.7` | float | 기본 temperature 값 |
| `OPENAI_MAX_TOKENS_DEFAULT` | `1000` | integer | 기본 최대 토큰 수 |
| `OPENAI_REQUEST_TIMEOUT` | `30` | integer | API 요청 타임아웃 (초) |

**사용 예시:**
```bash
# 개발 환경
export OPENAI_API_KEY=sk-test-...
export OPENAI_MODEL_DEFAULT=gpt-3.5-turbo
export OPENAI_TEMPERATURE_DEFAULT=0.8

# 프로덕션 환경
export OPENAI_API_KEY=sk-prod-...
export OPENAI_MODEL_DEFAULT=gpt-4
export OPENAI_TEMPERATURE_DEFAULT=0.2
```

### 3. 애플리케이션 설정 (APP) - 8개

| 환경변수 | 기본값 | 타입 | 설명 |
|----------|--------|------|------|
| `APP_ENVIRONMENT` | `development` | string | 실행 환경 (`development`, `staging`, `production`) |
| `DEBUG_MODE` | `False` | boolean | 디버그 모드 활성화 |
| `LOG_LEVEL` | `INFO` | string | 로그 레벨 (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `APP_HOST` | `0.0.0.0` | string | 서버 바인딩 호스트 |
| `APP_PORT` | `8000` | integer | 서버 포트 번호 |
| `CORS_ORIGINS` | `["*"]` | list | CORS 허용 오리진 목록 |
| `DATA_DIRECTORIES` | `["constants", "downloads"]` | list | 데이터 디렉토리 목록 |
| `LOG_FORMAT` | `%(asctime)s - %(name)s - %(levelname)s - %(message)s` | string | 로그 포맷 |

**사용 예시:**
```bash
# 개발 환경
export APP_ENVIRONMENT=development
export DEBUG_MODE=true
export LOG_LEVEL=DEBUG
export APP_PORT=3000

# 프로덕션 환경
export APP_ENVIRONMENT=production
export DEBUG_MODE=false
export LOG_LEVEL=WARNING
export APP_PORT=80
```

### 4. 워크플로우 설정 (WORKFLOW) - 6개

| 환경변수 | 기본값 | 타입 | 설명 |
|----------|--------|------|------|
| `WORKFLOW_TIMEOUT` | `300` | integer | 워크플로우 실행 타임아웃 (초) |
| `MAX_WORKFLOW_NODES` | `100` | integer | 워크플로우 최대 노드 수 |
| `WORKFLOW_ALLOW_PARALLEL` | `True` | boolean | 병렬 실행 허용 |
| `WORKFLOW_ENABLE_CACHING` | `True` | boolean | 워크플로우 캐싱 활성화 |
| `MAX_CONCURRENT_WORKFLOWS` | `5` | integer | 최대 동시 실행 워크플로우 수 |
| `WORKFLOW_SAVE_LOGS` | `True` | boolean | 워크플로우 로그 저장 |

**사용 예시:**
```bash
# 성능 최적화
export WORKFLOW_TIMEOUT=600
export MAX_WORKFLOW_NODES=200
export MAX_CONCURRENT_WORKFLOWS=10

# 리소스 절약
export WORKFLOW_ENABLE_CACHING=false
export MAX_CONCURRENT_WORKFLOWS=2
```

### 5. 노드 시스템 설정 (NODE) - 6개

| 환경변수 | 기본값 | 타입 | 설명 |
|----------|--------|------|------|
| `NODE_CACHE_ENABLED` | `True` | boolean | 노드 캐싱 활성화 |
| `NODE_AUTO_DISCOVERY` | `True` | boolean | 노드 자동 발견 |
| `NODE_VALIDATION_ENABLED` | `True` | boolean | 노드 유효성 검사 |
| `NODE_EXECUTION_TIMEOUT` | `60` | integer | 노드 실행 타임아웃 (초) |
| `NODE_REGISTRY_FILE_PATH` | `constants/exported_nodes.json` | string | 노드 레지스트리 파일 경로 |
| `NODE_DEBUG_MODE` | `False` | boolean | 노드 디버그 모드 |

**사용 예시:**
```bash
# 개발 환경 (디버깅)
export NODE_DEBUG_MODE=true
export NODE_VALIDATION_ENABLED=true
export NODE_EXECUTION_TIMEOUT=120

# 프로덕션 환경 (성능)
export NODE_CACHE_ENABLED=true
export NODE_AUTO_DISCOVERY=false
```

## 🚀 설정 사용법

### 1. 환경변수로 설정하기

```bash
# .env 파일 생성
cat > .env << EOF
# 애플리케이션 설정
APP_ENVIRONMENT=production
APP_PORT=8080
DEBUG_MODE=false

# OpenAI 설정
OPENAI_API_KEY=sk-your-api-key
OPENAI_MODEL_DEFAULT=gpt-4

# 데이터베이스 설정
DATABASE_TYPE=postgresql
POSTGRES_HOST=localhost
POSTGRES_USER=plateerag_user
POSTGRES_PASSWORD=secure_password
EOF

# 환경변수 로드
source .env
python main.py
```

### 2. 코드에서 설정 사용하기

```python
from config.config_composer import ConfigComposer

# 설정 초기화
composer = ConfigComposer()
configs = composer.initialize_all_configs()

# 설정 값 읽기
api_key = composer.openai.API_KEY.value
model = composer.openai.MODEL_DEFAULT.value
port = composer.app.PORT.value
debug = composer.app.DEBUG_MODE.value

print(f"서버가 포트 {port}에서 실행됩니다 (디버그: {debug})")
print(f"사용 모델: {model}")
```

### 3. 런타임에 설정 변경하기

```python
# 설정 값 변경
composer.openai.MODEL_DEFAULT.value = "gpt-4"
composer.app.DEBUG_MODE.value = True

# 데이터베이스에 저장
composer.openai.MODEL_DEFAULT.save()
composer.app.DEBUG_MODE.save()

# 또는 모든 설정 일괄 저장
composer.save_all()
```

## 🔌 API로 설정 관리하기

### 설정 조회

```bash
# 전체 설정 요약
curl http://localhost:8000/app/config

# 영속성 설정 상세 정보
curl http://localhost:8000/app/config/persistent
```

### 설정 변경

```bash
# OpenAI 모델 변경
curl -X PUT http://localhost:8000/app/config/persistent/OPENAI_MODEL_DEFAULT \
  -H "Content-Type: application/json" \
  -d '{"value": "gpt-4"}'

# 애플리케이션 포트 변경
curl -X PUT http://localhost:8000/app/config/persistent/APP_PORT \
  -H "Content-Type: application/json" \
  -d '{"value": 9000}'

# 디버그 모드 활성화
curl -X PUT http://localhost:8000/app/config/persistent/DEBUG_MODE \
  -H "Content-Type: application/json" \
  -d '{"value": true}'
```

### 설정 관리

```bash
# 모든 설정을 데이터베이스에 저장
curl -X POST http://localhost:8000/app/config/persistent/save

# 데이터베이스에서 설정 새로고침
curl -X POST http://localhost:8000/app/config/persistent/refresh
```

## ➕ 새로운 설정 추가하는 방법

### 1단계: 설정 카테고리 선택

기존 카테고리에 추가하거나 새 카테고리를 만듭니다:
- `openai_config.py` - OpenAI 관련 설정
- `app_config.py` - 애플리케이션 기본 설정
- `workflow_config.py` - 워크플로우 관련 설정
- `node_config.py` - 노드 시스템 설정
- `database_config.py` - 데이터베이스 설정

### 2단계: 설정 클래스에 추가

예시: `app_config.py`에 새로운 설정 `MAX_FILE_SIZE` 추가

```python
# config/sub_config/app_config.py

def initialize(self) -> Dict[str, PersistentConfig]:
    """애플리케이션 기본 설정들을 초기화"""
    
    # ...기존 코드...
    
    # 새로운 설정 추가
    self.MAX_FILE_SIZE = self.create_persistent_config(
        env_name="MAX_FILE_SIZE",           # 환경변수 이름
        config_path="app.max_file_size",    # DB 저장 경로
        default_value=10485760,             # 기본값 (10MB)
        type_converter=int                  # 타입 변환기
    )
    
    return self.configs
```

### 3단계: 타입 변환기 이해

```python
# 문자열 (기본값)
self.MY_STRING = self.create_persistent_config(
    env_name="MY_STRING",
    config_path="category.my_string",
    default_value="default_text"
    # type_converter 생략 시 문자열
)

# 정수
self.MY_INTEGER = self.create_persistent_config(
    env_name="MY_INTEGER",
    config_path="category.my_integer",
    default_value=100,
    type_converter=int
)

# 부동소수점
self.MY_FLOAT = self.create_persistent_config(
    env_name="MY_FLOAT",
    config_path="category.my_float",
    default_value=3.14,
    type_converter=float
)

# 불린
from config.base_config import convert_to_bool
self.MY_BOOLEAN = self.create_persistent_config(
    env_name="MY_BOOLEAN",
    config_path="category.my_boolean",
    default_value=True,
    type_converter=convert_to_bool
)

# 리스트/딕셔너리 (JSON)
self.MY_LIST = self.create_persistent_config(
    env_name="MY_LIST",
    config_path="category.my_list",
    default_value=["item1", "item2"]
    # JSON으로 자동 처리
)
```

### 4단계: 새 카테고리 만들기 (선택사항)

새로운 카테고리가 필요한 경우:

```python
# config/sub_config/email_config.py
"""
이메일 관련 설정
"""
from typing import Dict
from config.base_config import BaseConfig, PersistentConfig, convert_to_bool

class EmailConfig(BaseConfig):
    """이메일 시스템 설정 관리"""
    
    def initialize(self) -> Dict[str, PersistentConfig]:
        """이메일 관련 설정들을 초기화"""
        
        self.SMTP_HOST = self.create_persistent_config(
            env_name="SMTP_HOST",
            config_path="email.smtp_host",
            default_value="smtp.gmail.com"
        )
        
        self.SMTP_PORT = self.create_persistent_config(
            env_name="SMTP_PORT",
            config_path="email.smtp_port",
            default_value=587,
            type_converter=int
        )
        
        self.SMTP_USER = self.create_persistent_config(
            env_name="SMTP_USER",
            config_path="email.smtp_user",
            default_value=""
        )
        
        self.SMTP_PASSWORD = self.create_persistent_config(
            env_name="SMTP_PASSWORD",
            config_path="email.smtp_password",
            default_value=""
        )
        
        self.EMAIL_ENABLED = self.create_persistent_config(
            env_name="EMAIL_ENABLED",
            config_path="email.enabled",
            default_value=False,
            type_converter=convert_to_bool
        )
        
        return self.configs
```

### 5단계: ConfigComposer에 등록

```python
# config/config_composer.py

from config.sub_config.email_config import EmailConfig  # 새 카테고리 import

class ConfigComposer:
    def __init__(self):
        self.openai: OpenAIConfig = OpenAIConfig()
        self.app: AppConfig = AppConfig()
        self.workflow: WorkflowConfig = WorkflowConfig()
        self.node: NodeConfig = NodeConfig()
        self.database: DatabaseConfig = DatabaseConfig()
        self.email: EmailConfig = EmailConfig()  # 새 카테고리 추가
        
        self.all_configs: Dict[str, PersistentConfig] = {}
        self.logger = logger
    
    def initialize_all_configs(self) -> Dict[str, Any]:
        # ...기존 코드...
        
        email_configs = self.email.initialize()  # 새 카테고리 초기화
        
        # 모든 설정을 하나의 딕셔너리로 통합
        self.all_configs.update(openai_configs)
        self.all_configs.update(app_configs)
        self.all_configs.update(workflow_configs)
        self.all_configs.update(node_configs)
        self.all_configs.update(database_configs)
        self.all_configs.update(email_configs)  # 새 카테고리 추가
        
        # app.state에 저장할 구조화된 데이터 반환
        return {
            "openai": self.openai,
            "app": self.app,
            "workflow": self.workflow,
            "node": self.node,
            "database": self.database,
            "email": self.email,  # 새 카테고리 추가
            "all_configs": self.all_configs
        }
```

### 6단계: 사용하기

```python
# 새로운 설정 사용
composer = ConfigComposer()
configs = composer.initialize_all_configs()

# 새 설정 읽기
max_file_size = composer.app.MAX_FILE_SIZE.value
smtp_host = composer.email.SMTP_HOST.value

# 환경변수로 설정
export MAX_FILE_SIZE=20971520  # 20MB
export SMTP_HOST=smtp.company.com
export EMAIL_ENABLED=true
```

## 🛠️ 개발 팁

### 1. 설정 검증하기

```python
# 설정 유효성 검사
validation = composer.validate_critical_configs()
if not validation["valid"]:
    for error in validation["errors"]:
        print(f"❌ {error}")
```

### 2. 설정 백업/복원

```python
# 현재 설정 백업
from config.persistent_config import export_config_summary
backup = export_config_summary()

# 특정 설정만 백업
openai_settings = {
    "model": composer.openai.MODEL_DEFAULT.value,
    "temperature": composer.openai.TEMPERATURE_DEFAULT.value
}
```

### 3. 환경별 설정 파일

```bash
# 개발 환경 (.env.development)
APP_ENVIRONMENT=development
DEBUG_MODE=true
LOG_LEVEL=DEBUG
OPENAI_MODEL_DEFAULT=gpt-3.5-turbo

# 프로덕션 환경 (.env.production)
APP_ENVIRONMENT=production
DEBUG_MODE=false
LOG_LEVEL=WARNING
OPENAI_MODEL_DEFAULT=gpt-4
```

## 🔧 문제 해결

### 자주 발생하는 오류

1. **`AttributeError: 'ConfigClass' object has no attribute 'NEW_SETTING'`**
   - 새 설정을 추가했지만 `initialize()` 메서드에서 `self.NEW_SETTING = ...` 정의를 빠뜨린 경우
   - 해결: 설정 클래스의 `initialize()` 메서드에 설정 정의 추가

2. **타입 변환 오류**
   ```python
   # 잘못된 예
   export MY_NUMBER=abc  # 숫자가 아님
   
   # 올바른 예
   export MY_NUMBER=123
   ```

3. **JSON 형식 오류**
   ```python
   # 잘못된 예
   export MY_LIST=[item1, item2]  # 따옴표 없음
   
   # 올바른 예
   export MY_LIST='["item1", "item2"]'
   ```

### 디버깅 방법

```python
# 설정 로딩 과정 디버깅
import logging
logging.basicConfig(level=logging.DEBUG)

composer = ConfigComposer()
configs = composer.initialize_all_configs()

# 특정 설정의 상태 확인
config = composer.app.PORT
print(f"환경변수 이름: {config.env_name}")
print(f"현재 값: {config.value}")
print(f"기본값: {config.env_value}")
print(f"DB 저장값: {config.config_value}")
```

---

## 📚 참고 링크

- [FastAPI 통합 예시](../main.py)
- [데이터베이스 관리자](./database_manager.py)
- [영속성 설정 클래스](./persistent_config.py)
- [설정 통합 관리자](./config_composer.py)

**PlateERAG Backend Configuration System**  
*36개 설정, 5개 카테고리, 무한 확장성* 🚀

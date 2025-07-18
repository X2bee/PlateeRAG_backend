# PlateERAG Backend 설정 시스템 가이드

## 📖 개요

PlateERAG Backend는 **자동 발견 기능**을 통해 config 시스템을 관리하며, 환경변수와 데이터베이스를 통한 영속적 설정 관리를 지원합니다.

### 🎯 핵심 특징
- **자동 발견**: `sub_config/` 폴더의 `*_config.py` 파일들을 자동으로 로드
- **환경변수 우선**: 환경변수로 모든 설정 덮어쓰기 가능
- **자동 DB 저장**: 변경된 설정은 SQLite/PostgreSQL에 자동 저장
- **실시간 변경**: API를 통한 런타임 설정 수정
- **타입 안전**: 강타입 검증 (문자열, 숫자, 불린, 리스트, JSON)
- **플러그인 아키텍처**: 새로운 설정 카테고리를 쉽게 추가 가능

## 🏗️ 설정 시스템 아키텍처

### 폴더 구조
```
config/
├── README.md                    # 📖 이 문서
├── base_config.py              # 🔧 기본 설정 클래스 (모든 설정 클래스의 부모)
├── persistent_config.py        # 💾 영속성 설정 관리 (DB 연동)
├── config_composer.py          # 🎼 설정 통합 관리자 (자동 발견 기능)
├── database_manager.py         # 🗄️  데이터베이스 연결 및 마이그레이션 관리
└── sub_config/                 # 📂 설정 카테고리 폴더
    ├── __init__.py
    ├── openai_config.py        # 🤖 OpenAI API 설정
    ├── app_config.py           # 🖥️  애플리케이션 기본 설정
    ├── database_config.py      # 🗄️  데이터베이스 연결 설정
    ├── workflow_config.py      # 🔄 워크플로우 실행 설정
    ├── node_config.py          # 🔗 노드 시스템 설정
    └── vectordb_config.py      # 🔍 벡터 데이터베이스 설정
```

### 아키텍처 구성요소

#### 1. `BaseConfig` (base_config.py)
- **역할**: 모든 설정 클래스의 추상 기본 클래스
- **기능**: 환경변수 로딩, 타입 변환, PersistentConfig 생성
- **상속**: 모든 `*_config.py` 파일의 클래스가 상속

#### 2. `PersistentConfig` (persistent_config.py)
- **역할**: 개별 설정의 영속성 관리
- **기능**: 환경변수 우선순위, 데이터베이스 저장/로드, JSON 백업
- **특징**: 각 설정마다 하나씩 인스턴스 생성

#### 3. `ConfigComposer` (config_composer.py)
- **역할**: 설정 통합 관리자 (메인 컨트롤러)
- **기능**: 자동 발견, 설정 초기화, 통합 인터페이스 제공
- **자동 발견**: `sub_config/` 폴더의 `*_config.py` 파일들을 자동으로 스캔

#### 4. `DatabaseManager` (database_manager.py)
- **역할**: 데이터베이스 연결 및 마이그레이션 관리
- **기능**: SQLite/PostgreSQL 자동 선택, 스키마 관리
- **백업**: 데이터베이스 실패 시 JSON 파일로 백업

## 🗂️ 현재 설정 카테고리

### 1. OpenAI 설정 (openai_config.py)
| 환경변수 | 기본값 | 타입 | 설명 |
|----------|--------|------|------|
| `OPENAI_API_KEY` | `""` | string | OpenAI API 키 |
| `OPENAI_MODEL_DEFAULT` | `gpt-4o-2024-11-20` | string | 기본 AI 모델 |
| `OPENAI_API_BASE_URL` | `https://api.openai.com/v1` | string | API 베이스 URL |
| `OPENAI_TEMPERATURE_DEFAULT` | `0.7` | float | 기본 temperature 값 |
| `OPENAI_MAX_TOKENS_DEFAULT` | `1000` | integer | 기본 최대 토큰 수 |
| `OPENAI_REQUEST_TIMEOUT` | `30` | integer | API 요청 타임아웃 (초) |

### 2. 애플리케이션 설정 (app_config.py)
| 환경변수 | 기본값 | 타입 | 설명 |
|----------|--------|------|------|
| `APP_ENVIRONMENT` | `development` | string | 실행 환경 |
| `DEBUG_MODE` | `False` | boolean | 디버그 모드 |
| `LOG_LEVEL` | `INFO` | string | 로그 레벨 |
| `APP_HOST` | `0.0.0.0` | string | 서버 호스트 |
| `APP_PORT` | `8000` | integer | 서버 포트 |
| `LOG_FORMAT` | `%(asctime)s - %(name)s - %(levelname)s - %(message)s` | string | 로그 포맷 |

### 3. 데이터베이스 설정 (database_config.py)
| 환경변수 | 기본값 | 타입 | 설명 |
|----------|--------|------|------|
| `DATABASE_TYPE` | `auto` | string | 데이터베이스 타입 |
| `POSTGRES_HOST` | `0.0.0.0` | string | PostgreSQL 호스트 |
| `POSTGRES_PORT` | `5432` | integer | PostgreSQL 포트 |
| `POSTGRES_DB` | `plateerag` | string | 데이터베이스 이름 |
| `POSTGRES_USER` | `Unset` | string | PostgreSQL 사용자명 |
| `POSTGRES_PASSWORD` | `Unset` | string | PostgreSQL 비밀번호 |
| `SQLITE_PATH` | `constants/config.db` | string | SQLite 파일 경로 |

### 4. 워크플로우 설정 (workflow_config.py)
| 환경변수 | 기본값 | 타입 | 설명 |
|----------|--------|------|------|
| `WORKFLOW_TIMEOUT` | `300` | integer | 워크플로우 실행 타임아웃 (초) |
| `MAX_WORKFLOW_NODES` | `1000` | integer | 워크플로우 최대 노드 수 |
| `WORKFLOW_ALLOW_PARALLEL` | `True` | boolean | 병렬 실행 허용 |
| `WORKFLOW_ENABLE_CACHING` | `True` | boolean | 워크플로우 캐싱 활성화 |
| `MAX_CONCURRENT_WORKFLOWS` | `5` | integer | 최대 동시 실행 워크플로우 수 |

### 5. 노드 설정 (node_config.py)
| 환경변수 | 기본값 | 타입 | 설명 |
|----------|--------|------|------|
| `NODE_CACHE_ENABLED` | `True` | boolean | 노드 캐싱 활성화 |
| `NODE_AUTO_DISCOVERY` | `True` | boolean | 노드 자동 발견 |
| `NODE_VALIDATION_ENABLED` | `True` | boolean | 노드 유효성 검사 |
| `NODE_EXECUTION_TIMEOUT` | `60` | integer | 노드 실행 타임아웃 (초) |
| `NODE_REGISTRY_FILE_PATH` | `constants/exported_nodes.json` | string | 노드 레지스트리 파일 경로 |

### 6. 벡터 데이터베이스 설정 (vectordb_config.py)
| 환경변수 | 기본값 | 타입 | 설명 |
|----------|--------|------|------|
| `QDRANT_HOST` | `localhost` | string | Qdrant 호스트 |
| `QDRANT_PORT` | `6333` | integer | Qdrant HTTP 포트 |
| `QDRANT_USE_GRPC` | `False` | boolean | Qdrant gRPC 사용 여부 |
| `QDRANT_GRPC_PORT` | `6334` | integer | Qdrant gRPC 포트 |

## 🚀 설정 사용법

### 1. 환경변수로 설정하기
```bash
# .env 파일 생성
cat > .env << EOF
# OpenAI 설정
OPENAI_API_KEY=sk-your-api-key
OPENAI_MODEL_DEFAULT=gpt-4

# 애플리케이션 설정
APP_ENVIRONMENT=production
APP_PORT=8080
DEBUG_MODE=false

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

# 설정 초기화 (자동 발견 포함)
composer = ConfigComposer()

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

## ➕ 새로운 설정 추가하기 (Step-by-Step)

### 🎯 Step 1: 설정 카테고리 결정

기존 카테고리에 추가할지, 새로운 카테고리를 만들지 결정합니다.

**기존 카테고리:**
- `openai_config.py` - OpenAI API 관련 설정
- `app_config.py` - 애플리케이션 기본 설정
- `database_config.py` - 데이터베이스 연결 설정
- `workflow_config.py` - 워크플로우 실행 설정
- `node_config.py` - 노드 시스템 설정
- `vectordb_config.py` - 벡터 데이터베이스 설정

### 🔧 Step 2: 기존 카테고리에 설정 추가

**예시: app_config.py에 파일 업로드 관련 설정 추가**

```python
# config/sub_config/app_config.py
from typing import Dict
from config.base_config import BaseConfig, PersistentConfig, convert_to_bool

class AppConfig(BaseConfig):
    """애플리케이션 기본 설정 관리"""
    
    def initialize(self) -> Dict[str, PersistentConfig]:
        """애플리케이션 기본 설정들을 초기화"""
        
        # ...기존 설정들...
        
        # 🆕 새로운 설정 추가
        self.MAX_FILE_SIZE = self.create_persistent_config(
            env_name="MAX_FILE_SIZE",           # 환경변수 이름
            config_path="app.max_file_size",    # DB 저장 경로
            default_value=10485760,             # 기본값 (10MB)
            type_converter=int                  # 타입 변환기
        )
        
        self.ALLOWED_FILE_EXTENSIONS = self.create_persistent_config(
            env_name="ALLOWED_FILE_EXTENSIONS",
            config_path="app.allowed_file_extensions",
            default_value=["pdf", "txt", "docx", "md"]  # JSON으로 자동 처리
        )
        
        self.UPLOAD_DIRECTORY = self.create_persistent_config(
            env_name="UPLOAD_DIRECTORY",
            config_path="app.upload_directory",
            default_value="uploads"
        )
        
        return self.configs
```

### 🏗️ Step 3: 새로운 카테고리 생성

**예시: 이메일 관련 설정을 위한 새 카테고리 생성**

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
        
        # 📧 이메일 템플릿 설정
        self.EMAIL_TEMPLATES = self.create_persistent_config(
            env_name="EMAIL_TEMPLATES",
            config_path="email.templates",
            default_value={
                "welcome": "Welcome to PlateERAG!",
                "notification": "You have a new notification"
            }
        )
        
        return self.configs
```

### 🎉 Step 4: 자동 발견 확인

**설정 파일을 저장하면 자동으로 발견됩니다!**

ConfigComposer가 자동으로 `sub_config/` 폴더의 `*_config.py` 파일들을 스캔하므로 별도의 등록 과정이 필요 없습니다.

```python
# 🚀 자동으로 로드됩니다!
composer = ConfigComposer()
# email_config.py가 자동으로 발견되어 composer.email로 접근 가능
```

### 🧪 Step 5: 타입 변환기 사용법

```python
# 🔢 정수 타입
self.MAX_CONNECTIONS = self.create_persistent_config(
    env_name="MAX_CONNECTIONS",
    config_path="category.max_connections",
    default_value=100,
    type_converter=int
)

# 🔢 부동소수점 타입
self.THRESHOLD_SCORE = self.create_persistent_config(
    env_name="THRESHOLD_SCORE",
    config_path="category.threshold_score",
    default_value=0.8,
    type_converter=float
)

# ✅ 불린 타입
from config.base_config import convert_to_bool
self.ENABLE_FEATURE = self.create_persistent_config(
    env_name="ENABLE_FEATURE",
    config_path="category.enable_feature",
    default_value=True,
    type_converter=convert_to_bool
)

# 📝 문자열 타입 (기본값, type_converter 생략)
self.SERVER_NAME = self.create_persistent_config(
    env_name="SERVER_NAME",
    config_path="category.server_name",
    default_value="PlateERAG Server"
)

# 📋 리스트 타입 (JSON 자동 처리)
self.SUPPORTED_FORMATS = self.create_persistent_config(
    env_name="SUPPORTED_FORMATS",
    config_path="category.supported_formats",
    default_value=["json", "yaml", "xml"]
)

# 🗂️ 딕셔너리 타입 (JSON 자동 처리)
self.API_ENDPOINTS = self.create_persistent_config(
    env_name="API_ENDPOINTS",
    config_path="category.api_endpoints",
    default_value={
        "v1": "/api/v1",
        "v2": "/api/v2"
    }
)
```

### 🔄 Step 6: 설정 사용하기

```python
# 설정 초기화
composer = ConfigComposer()

# 기존 설정 사용
api_key = composer.openai.API_KEY.value
port = composer.app.PORT.value

# 🆕 새로 추가된 설정 사용
max_file_size = composer.app.MAX_FILE_SIZE.value
allowed_extensions = composer.app.ALLOWED_FILE_EXTENSIONS.value
upload_dir = composer.app.UPLOAD_DIRECTORY.value

# 🆕 새로운 카테고리 설정 사용
if composer.email.EMAIL_ENABLED.value:
    smtp_host = composer.email.SMTP_HOST.value
    smtp_port = composer.email.SMTP_PORT.value
    print(f"이메일 서버: {smtp_host}:{smtp_port}")
```

### 🌍 Step 7: 환경변수로 설정하기

```bash
# .env 파일에 새로운 설정 추가
cat >> .env << EOF

# 🆕 파일 업로드 설정
MAX_FILE_SIZE=52428800  # 50MB
ALLOWED_FILE_EXTENSIONS='["pdf", "txt", "docx", "md", "jpg", "png"]'
UPLOAD_DIRECTORY=uploads

# 🆕 이메일 설정
EMAIL_ENABLED=true
SMTP_HOST=smtp.company.com
SMTP_PORT=587
SMTP_USER=noreply@company.com
SMTP_PASSWORD=your-email-password
EMAIL_TEMPLATES='{"welcome": "환영합니다!", "notification": "새로운 알림이 있습니다."}'
EOF
```

### 🎯 Step 8: 고급 설정 패턴

#### 📁 파일 경로 설정
```python
# 파일 경로 설정 (file_path 사용)
self.API_KEY = self.create_persistent_config(
    env_name="OPENAI_API_KEY",
    config_path="openai.api_key",
    default_value="",
    file_path="secrets/openai_key.txt"  # 파일에서 읽기
)
```

#### 🔗 의존성 있는 설정
```python
# 다른 설정에 의존하는 설정
def initialize(self) -> Dict[str, PersistentConfig]:
    # 기본 설정들 먼저 초기화
    self.ENABLE_CACHE = self.create_persistent_config(
        env_name="ENABLE_CACHE",
        config_path="category.enable_cache",
        default_value=True,
        type_converter=convert_to_bool
    )
    
    # 의존성 있는 설정
    cache_ttl = 3600 if self.ENABLE_CACHE.value else 0
    self.CACHE_TTL = self.create_persistent_config(
        env_name="CACHE_TTL",
        config_path="category.cache_ttl",
        default_value=cache_ttl,
        type_converter=int
    )
    
    return self.configs
```

#### 🎨 커스텀 타입 변환기
```python
def convert_to_url_list(value: str) -> List[str]:
    """콤마로 구분된 URL 리스트로 변환"""
    urls = [url.strip() for url in value.split(',')]
    # URL 유효성 검사
    import re
    url_pattern = re.compile(r'^https?://')
    return [url for url in urls if url_pattern.match(url)]

# 사용 예시
self.WEBHOOK_URLS = self.create_persistent_config(
    env_name="WEBHOOK_URLS",
    config_path="category.webhook_urls",
    default_value=["https://example.com/webhook"],
    type_converter=convert_to_url_list
)
```

### ✅ Step 9: 설정 검증 및 테스트

```python
# 설정 검증
def validate_email_config(self) -> bool:
    """이메일 설정 유효성 검사"""
    if not self.EMAIL_ENABLED.value:
        return True
    
    required_fields = [
        self.SMTP_HOST.value,
        self.SMTP_USER.value,
        self.SMTP_PASSWORD.value
    ]
    
    return all(field.strip() for field in required_fields)

# 테스트 스크립트
if __name__ == "__main__":
    composer = ConfigComposer()
    
    # 새로운 설정 테스트
    print("🧪 설정 테스트:")
    print(f"파일 크기 제한: {composer.app.MAX_FILE_SIZE.value / 1024 / 1024:.1f}MB")
    print(f"허용 확장자: {composer.app.ALLOWED_FILE_EXTENSIONS.value}")
    print(f"업로드 폴더: {composer.app.UPLOAD_DIRECTORY.value}")
    
    if hasattr(composer, 'email'):
        print(f"이메일 활성화: {composer.email.EMAIL_ENABLED.value}")
        if composer.email.EMAIL_ENABLED.value:
            print(f"SMTP 서버: {composer.email.SMTP_HOST.value}:{composer.email.SMTP_PORT.value}")
```

### � 자주 사용하는 패턴

#### 1. **환경별 설정**
```python
# config/sub_config/environment_config.py
class EnvironmentConfig(BaseConfig):
    def initialize(self) -> Dict[str, PersistentConfig]:
        self.ENVIRONMENT = self.create_persistent_config(
            env_name="APP_ENVIRONMENT",
            config_path="environment.current",
            default_value="development"
        )
        
        # 환경별 설정
        is_production = self.ENVIRONMENT.value == "production"
        self.RATE_LIMIT = self.create_persistent_config(
            env_name="RATE_LIMIT",
            config_path="environment.rate_limit",
            default_value=1000 if is_production else 10000,
            type_converter=int
        )
        
        return self.configs
```

#### 2. **기능 플래그**
```python
# config/sub_config/feature_config.py
class FeatureConfig(BaseConfig):
    def initialize(self) -> Dict[str, PersistentConfig]:
        self.FEATURE_FLAGS = self.create_persistent_config(
            env_name="FEATURE_FLAGS",
            config_path="features.flags",
            default_value={
                "new_ui": False,
                "experimental_ai": False,
                "beta_features": False
            }
        )
        
        return self.configs
```

#### 3. **성능 튜닝 설정**
```python
# config/sub_config/performance_config.py
class PerformanceConfig(BaseConfig):
    def initialize(self) -> Dict[str, PersistentConfig]:
        self.WORKER_THREADS = self.create_persistent_config(
            env_name="WORKER_THREADS",
            config_path="performance.worker_threads",
            default_value=4,
            type_converter=int
        )
        
        self.MEMORY_LIMIT = self.create_persistent_config(
            env_name="MEMORY_LIMIT",
            config_path="performance.memory_limit",
            default_value="1GB"
        )
        
        return self.configs
```

## 🎯 완성! 이제 새로운 설정을 자유롭게 추가할 수 있습니다.

1. **기존 카테고리 확장**: 기존 `*_config.py` 파일에 설정 추가
2. **새 카테고리 생성**: 새로운 `*_config.py` 파일 생성
3. **자동 발견**: ConfigComposer가 자동으로 로드
4. **타입 안전**: 적절한 타입 변환기 사용
5. **환경변수 지원**: 즉시 환경변수로 설정 가능
6. **데이터베이스 저장**: 자동으로 영속성 관리

**🎉 축하합니다! 이제 PlateERAG의 설정 시스템 마스터입니다!**

## � API로 설정 관리하기

### 설정 조회

```bash
# 전체 설정 요약 조회
curl http://localhost:8000/app/config

# 영속성 설정 상세 정보 조회
curl http://localhost:8000/app/config/persistent

# 특정 카테고리 설정 조회
curl http://localhost:8000/app/config/category/openai
curl http://localhost:8000/app/config/category/app
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

# 리스트 타입 설정 변경
curl -X PUT http://localhost:8000/app/config/persistent/ALLOWED_FILE_EXTENSIONS \
  -H "Content-Type: application/json" \
  -d '{"value": ["pdf", "txt", "docx", "md", "jpg", "png"]}'
```

### 설정 관리

```bash
# 모든 설정을 데이터베이스에 저장
curl -X POST http://localhost:8000/app/config/persistent/save

# 데이터베이스에서 설정 새로고침
curl -X POST http://localhost:8000/app/config/persistent/refresh

# 설정 백업 내보내기
curl http://localhost:8000/app/config/export > config_backup.json
```

## 🛠️ 개발 및 디버깅 팁

### 1. 설정 로딩 과정 디버깅

```python
import logging
from config.config_composer import ConfigComposer

# 디버그 로깅 활성화
logging.basicConfig(level=logging.DEBUG)

# 설정 초기화 (로그 확인)
composer = ConfigComposer()

# 특정 설정의 상태 확인
config = composer.app.PORT
print(f"환경변수 이름: {config.env_name}")
print(f"현재 값: {config.value}")
print(f"기본값: {config.env_value}")
print(f"DB 저장값: {config.config_value}")
```

### 2. 설정 검증 및 테스트

```python
# 설정 검증 함수
def validate_configuration(composer):
    """설정 유효성 검사"""
    errors = []
    
    # OpenAI API 키 확인
    if not composer.openai.API_KEY.value:
        errors.append("OpenAI API 키가 설정되지 않았습니다.")
    
    # 포트 번호 확인
    port = composer.app.PORT.value
    if not (1 <= port <= 65535):
        errors.append(f"유효하지 않은 포트 번호: {port}")
    
    # 데이터베이스 설정 확인
    if composer.database.DATABASE_TYPE.value == "postgresql":
        if not composer.database.POSTGRES_HOST.value:
            errors.append("PostgreSQL 호스트가 설정되지 않았습니다.")
    
    return errors

# 사용 예시
composer = ConfigComposer()
errors = validate_configuration(composer)
if errors:
    for error in errors:
        print(f"❌ {error}")
else:
    print("✅ 모든 설정이 유효합니다.")
```

### 3. 설정 백업 및 복원

```python
from config.persistent_config import export_config_summary
import json

# 설정 백업
def backup_configuration(composer, filename="config_backup.json"):
    """설정을 JSON 파일로 백업"""
    config_data = export_config_summary()
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    print(f"설정이 {filename}에 백업되었습니다.")

# 특정 카테고리만 백업
def backup_category(composer, category_name, filename=None):
    """특정 카테고리의 설정만 백업"""
    if not filename:
        filename = f"{category_name}_config_backup.json"
    
    category_config = getattr(composer, category_name, None)
    if not category_config:
        print(f"❌ 카테고리 '{category_name}'를 찾을 수 없습니다.")
        return
    
    config_data = {}
    for name, config in category_config.configs.items():
        config_data[name] = {
            "value": config.value,
            "env_name": config.env_name,
            "config_path": config.config_path
        }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    print(f"카테고리 '{category_name}' 설정이 {filename}에 백업되었습니다.")

# 사용 예시
composer = ConfigComposer()
backup_configuration(composer)
backup_category(composer, "openai")
```

### 4. 환경별 설정 관리

```python
# 환경별 설정 로더
def load_environment_config(env_name):
    """환경별 설정 파일 로드"""
    import os
    
    config_files = {
        "development": ".env.development",
        "staging": ".env.staging",
        "production": ".env.production"
    }
    
    config_file = config_files.get(env_name)
    if not config_file or not os.path.exists(config_file):
        print(f"❌ 환경 설정 파일 '{config_file}'을 찾을 수 없습니다.")
        return
    
    # .env 파일 로드
    with open(config_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                os.environ[key] = value
    
    print(f"✅ 환경 설정 '{env_name}'이 로드되었습니다.")

# 사용 예시
load_environment_config("development")
composer = ConfigComposer()
```

## �🔧 문제 해결 가이드

### 자주 발생하는 오류

#### 1. **`AttributeError: 'ConfigClass' object has no attribute 'NEW_SETTING'`**
```python
# ❌ 잘못된 예시
class MyConfig(BaseConfig):
    def initialize(self):
        # self.NEW_SETTING을 정의하지 않음
        return self.configs

# ✅ 올바른 예시
class MyConfig(BaseConfig):
    def initialize(self):
        self.NEW_SETTING = self.create_persistent_config(
            env_name="NEW_SETTING",
            config_path="my.new_setting",
            default_value="default_value"
        )
        return self.configs
```

#### 2. **타입 변환 오류**
```bash
# ❌ 잘못된 예시
export MY_NUMBER=abc  # 숫자가 아님
export MY_BOOLEAN=maybe  # 불린이 아님

# ✅ 올바른 예시
export MY_NUMBER=123
export MY_BOOLEAN=true
```

#### 3. **JSON 형식 오류**
```bash
# ❌ 잘못된 예시
export MY_LIST=[item1, item2]  # 따옴표 없음
export MY_DICT={key: value}    # 따옴표 없음

# ✅ 올바른 예시
export MY_LIST='["item1", "item2"]'
export MY_DICT='{"key": "value"}'
```

#### 4. **데이터베이스 연결 문제**
```python
# 데이터베이스 연결 상태 확인
from config.database_manager import DatabaseManager

def check_database_connection():
    """데이터베이스 연결 상태 확인"""
    try:
        db_manager = DatabaseManager()
        db_type = db_manager.determine_database_type()
        print(f"✅ 데이터베이스 타입: {db_type}")
        
        if db_type == "postgresql":
            # PostgreSQL 연결 테스트
            pass
        else:
            # SQLite 연결 테스트
            pass
            
    except Exception as e:
        print(f"❌ 데이터베이스 연결 실패: {e}")
        print("💡 JSON 백업 모드로 전환됩니다.")
```

### 디버깅 체크리스트

1. **환경변수 확인**
   ```bash
   # 환경변수 출력
   env | grep -E "(OPENAI|APP|DATABASE|WORKFLOW|NODE|QDRANT)"
   ```

2. **설정 파일 존재 확인**
   ```bash
   # 설정 파일들 확인
   ls -la config/sub_config/*_config.py
   ```

3. **로그 확인**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   # 설정 로딩 시 로그 확인
   ```

4. **데이터베이스 백업 파일 확인**
   ```bash
   # JSON 백업 파일 확인
   ls -la constants/config.json
   ```

---

## 📚 참고 자료

### 주요 파일들
- **[base_config.py](base_config.py)**: 기본 설정 클래스 및 타입 변환기
- **[persistent_config.py](persistent_config.py)**: 영속성 설정 관리
- **[config_composer.py](config_composer.py)**: 설정 통합 관리자
- **[database_manager.py](database_manager.py)**: 데이터베이스 연결 관리

### 설정 카테고리 파일들
- **[openai_config.py](sub_config/openai_config.py)**: OpenAI API 설정
- **[app_config.py](sub_config/app_config.py)**: 애플리케이션 기본 설정
- **[database_config.py](sub_config/database_config.py)**: 데이터베이스 연결 설정
- **[workflow_config.py](sub_config/workflow_config.py)**: 워크플로우 실행 설정
- **[node_config.py](sub_config/node_config.py)**: 노드 시스템 설정
- **[vectordb_config.py](sub_config/vectordb_config.py)**: 벡터 데이터베이스 설정

### 사용 예시
메인 애플리케이션에서의 설정 사용법은 `../main.py`를 참조하세요.

---

**PlateERAG Backend Configuration System**  
*🚀 자동 발견 • 🔧 타입 안전 • 🗄️ 영속성 • 🌍 환경변수 지원*

**이제 설정을 자유롭게 확장하고 관리할 수 있습니다!**

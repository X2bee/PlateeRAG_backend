# PlateERAG Backend

## 🎯 프로젝트 개요

**PlateERAG Backend**는 **Canvas 기반 NodeEditor**를 위한 고성능 백엔드 시스템입니다. 사용자가 시각적 인터페이스에서 노드를 연결하여 복잡한 RAG(Retrieval-Augmented Generation) 워크플로우를 구성하고 실행할 수 있도록 지원합니다.

### 🌟 핵심 특징

- **🎨 Canvas 기반 NodeEditor**: 시각적 드래그 앤 드롭으로 워크플로우 구성
- **🧠 RAG 시스템**: 문서 처리, 벡터 검색, LLM 통합 완전 지원
- **🔗 노드 기반 아키텍처**: 재사용 가능한 독립적 기능 단위
- **⚡ 실시간 실행**: 워크플로우 실시간 실행 및 결과 반환
- **📊 성능 모니터링**: 실시간 성능 추적 및 분석
- **🔧 확장 가능**: 새로운 노드 타입 쉽게 추가 가능

### 🏗️ 시스템 아키텍처 (예시)

```
Frontend (Canvas UI) ←→ Backend (PlateERAG) ←→ External Services
        │                       │                     │
   ┌────▼────┐           ┌─────▼─────┐         ┌─────▼─────┐
   │ Node    │           │ Workflow  │         │ OpenAI    │
   │ Editor  │           │ Executor  │         │ Qdrant    │
   │ Canvas  │           │ Engine    │         │ Database  │
   └─────────┘           └───────────┘         └───────────┘
```

## 🎯 주요 기능

### 1. 🎨 Canvas NodeEditor 백엔드

사용자가 Canvas에서 노드를 연결하여 만든 워크플로우를 실행하는 완전한 백엔드 시스템입니다.

#### 워크플로우 구조
```
사용자 입력 (Canvas) → Start Node → 처리 노드들 → End Node → 결과 반환
```

#### 지원 노드 타입
- **📥 입력 노드**: 사용자 입력, 파일 입력, 데이터 소스
- **🔄 처리 노드**: 텍스트 변환, 수학 연산, 조건 분기
- **🤖 AI 노드**: OpenAI Chat, 임베딩 생성, RAG 검색
- **📤 출력 노드**: 결과 반환, 파일 저장, 알림

### 2. 🧠 완전한 RAG 시스템

#### 문서 처리 파이프라인
```
문서 업로드 → 텍스트 추출 → 청크 분할 → 임베딩 생성 → 벡터 저장
```

#### 검색 및 생성 파이프라인
```
사용자 질문 → 벡터 검색 → 컨텍스트 구성 → LLM 생성 → 답변 반환
```

### 3. ⚡ 실시간 워크플로우 실행

- **위상 정렬**: 노드 의존성 자동 분석
- **병렬 처리**: 독립적 노드 동시 실행
- **스트리밍**: 실시간 결과 스트리밍
- **에러 처리**: 견고한 에러 복구

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론
git clone https://github.com/X2bee/PlateeRAG_backend.git
cd PlateeRAG_backend

# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
cp .env.example .env
# .env 파일 편집하여 API 키 설정
```

### 2. 서비스 실행

```bash
# 개발 서버 실행
python main.py

# 또는 Docker로 실행
docker-compose up
```

### 3. API 테스트

```bash
# 건강 상태 확인
curl http://localhost:8000/app/status

# 워크플로우 실행
curl -X POST http://localhost:8000/api/workflow/execute \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_name": "simple_chat",
    "workflow_id": "chat_001",
    "input_data": "안녕하세요!",
    "interaction_id": "user_001"
  }'
```

## 🏗️ 프로젝트 구조

```
plateerag_backend/
├── 📄 README.md                 # 이 문서
├── 🚀 main.py                   # FastAPI 애플리케이션 진입점
├── 📦 requirements.txt          # 의존성 목록
├── 🐳 docker-compose.yaml       # Docker 구성
├── 🔧 config/                   # 설정 관리 시스템
│   ├── 📖 README.md            # 설정 시스템 상세 가이드
│   ├── 🏗️ base_config.py       # 기본 설정 클래스
│   ├── 🎼 config_composer.py   # 설정 통합 관리
│   ├── 🗄️ database_manager.py  # 데이터베이스 관리
│   └── 📂 sub_config/          # 도메인별 설정
├── 🎮 controller/              # API 컨트롤러
│   ├── 📖 README.md            # 컨트롤러 시스템 가이드
│   ├── 🖥️ appController.py     # 앱 상태 관리
│   ├── 💬 chatController.py    # 채팅 API
│   ├── 🔄 workflowController.py # 워크플로우 실행
│   └── 🧠 ragController.py     # RAG 시스템 API
├── 🎨 editor/                  # NodeEditor 시스템
│   ├── 📖 README.md            # NodeEditor 상세 가이드
│   ├── 🎼 node_composer.py     # 노드 등록 및 관리
│   ├── ⚡ workflow_executor.py  # 워크플로우 실행 엔진
│   ├── 📋 model/               # 노드 데이터 모델
│   └── 🔗 nodes/               # 노드 구현체들
│       ├── 🤖 chat/            # AI 채팅 노드
│       ├── 🔢 math/            # 수학 연산 노드
│       └── 🔧 tool/            # 유틸리티 노드
└── 🛠️ service/                 # 비즈니스 로직 서비스
    ├── 📖 README.md            # 서비스 시스템 가이드
    ├── 🗄️ database/            # 데이터베이스 서비스
    ├── 🔤 embedding/           # 임베딩 서비스
    ├── 📊 monitoring/          # 성능 모니터링
    ├── 🔍 retrieval/           # 문서 검색 서비스
    └── 🗂️ vector_db/           # 벡터 데이터베이스
```

## 🎨 NodeEditor 작동 원리

### 1. 노드 정의 및 등록

```python
# 새로운 노드 정의
class CustomProcessNode(Node):
    categoryId = "utilities"
    functionId = "tools"
    nodeId = "custom/process"
    nodeName = "Custom Process"
    
    inputs = [{"id": "input", "name": "Input", "type": "STR"}]
    outputs = [{"id": "output", "name": "Output", "type": "STR"}]
    
    def execute(self, input: str) -> str:
        return f"처리됨: {input}"
```

### 2. Canvas에서 워크플로우 생성

Frontend Canvas에서 사용자가 노드를 연결하면 다음과 같은 JSON이 생성됩니다:

```json
{
  "workflow_id": "my_workflow",
  "workflow_name": "My Workflow",
  "nodes": [
    {
      "id": "start_node",
      "type": "tool/input_str",
      "data": {"parameters": {"input_str": "{{user_input}}"}}
    },
    {
      "id": "process_node",
      "type": "custom/process",
      "data": {"parameters": {}}
    },
    {
      "id": "end_node",
      "type": "tool/output_str",
      "data": {"parameters": {}}
    }
  ],
  "edges": [
    {
      "source": {"nodeId": "start_node", "portId": "result"},
      "target": {"nodeId": "process_node", "portId": "input"}
    },
    {
      "source": {"nodeId": "process_node", "portId": "output"},
      "target": {"nodeId": "end_node", "portId": "input"}
    }
  ]
}
```

### 3. 워크플로우 실행

```python
# 워크플로우 실행기가 자동으로:
# 1. 노드 의존성 분석 (위상 정렬)
# 2. 실행 순서 결정
# 3. 데이터 흐름 관리
# 4. 병렬 실행 최적화
# 5. 결과 수집 및 반환

executor = WorkflowExecutor(workflow_data)
result = executor.execute(user_input="안녕하세요!")
```

### 4. 실시간 결과 반환

```javascript
// Frontend에서 실시간 결과 수신
const result = await fetch('/api/workflow/execute', {
  method: 'POST',
  body: JSON.stringify(workflowData)
});

console.log(result.final_output); // "처리됨: 안녕하세요!"
```

## 🧠 RAG 시스템 활용

### 1. 문서 업로드 워크플로우

```
Canvas UI → 문서 업로드 노드 → 텍스트 추출 노드 → 청크 분할 노드 → 임베딩 노드 → 벡터 저장 노드
```

### 2. 지능형 검색 워크플로우

```
Canvas UI → 질문 입력 노드 → 벡터 검색 노드 → 컨텍스트 병합 노드 → LLM 생성 노드 → 답변 출력 노드
```

### 3. 실제 RAG 워크플로우 예시

```json
{
  "workflow_name": "RAG Chat System",
  "nodes": [
    {"id": "user_input", "type": "tool/input_str"},
    {"id": "vector_search", "type": "rag/vector_search"},
    {"id": "context_merge", "type": "tool/string_concat"},
    {"id": "llm_chat", "type": "chat/openai"},
    {"id": "final_output", "type": "tool/output_str"}
  ],
  "edges": [
    {"source": "user_input", "target": "vector_search"},
    {"source": "user_input", "target": "context_merge"},
    {"source": "vector_search", "target": "context_merge"},
    {"source": "context_merge", "target": "llm_chat"},
    {"source": "llm_chat", "target": "final_output"}
  ]
}
```

## 🔧 확장 가능성

### 1. 새로운 노드 타입 추가

```python
# 새로운 카테고리 추가
CATEGORIES_LABEL_MAP = {
    'custom': 'Custom Tools',  # 새 카테고리
    # ...기존 카테고리들
}

# 새로운 노드 구현
class WeatherAPINode(Node):
    categoryId = "custom"
    functionId = "api"
    # ...노드 구현
```

### 2. 외부 서비스 통합

```python
# 새로운 서비스 연동 노드
class SlackNotificationNode(Node):
    def execute(self, message: str, channel: str):
        # Slack API 호출
        return send_slack_message(message, channel)
```

### 3. 복잡한 워크플로우 패턴

- **조건부 분기**: 조건에 따른 다른 경로 실행
- **반복 처리**: 배치 데이터 처리
- **병렬 실행**: 독립적 작업 동시 처리
- **에러 처리**: 실패 시 대체 경로 실행

## 📊 성능 및 모니터링

### 실시간 성능 추적

```python
# 워크플로우 실행 시 자동 성능 측정
with PerformanceLogger(workflow_name, node_id):
    result = node.execute(input_data)
    
# 성능 데이터 API로 조회
GET /api/performance/workflow/{workflow_name}/{workflow_id}
```

### 모니터링 대시보드

- **실행 시간**: 각 노드별 실행 시간
- **메모리 사용량**: 실시간 메모리 모니터링
- **처리량**: 초당 처리 워크플로우 수
- **오류율**: 실패한 워크플로우 비율

## 🔒 보안 및 인증

### API 인증

```python
# JWT 토큰 기반 인증
@router.post("/workflow/execute")
async def execute_workflow(
    request: WorkflowRequest,
    current_user: User = Depends(get_current_user)
):
    # 워크플로우 실행
    pass
```

### 데이터 보안

- **데이터 암호화**: 민감한 데이터 암호화 저장
- **접근 제어**: 사용자별 워크플로우 권한 관리
- **감사 로그**: 모든 실행 기록 추적

## 🚀 배포 및 운영

### Docker 배포

```bash
# 프로덕션 배포
docker-compose -f docker-compose.prod.yml up -d

# 스케일링
docker-compose up --scale backend=3
```

### 환경 설정

```bash
# 환경별 설정
export NODE_ENV=production
export OPENAI_API_KEY=your_api_key
export QDRANT_HOST=your_qdrant_host
```

### 모니터링 및 알림

```python
# 시스템 상태 모니터링
GET /app/status
{
  "status": "healthy",
  "database": "connected",
  "vector_db": "connected",
  "services": {
    "embedding": "active",
    "rag": "active"
  }
}
```

## 🔧 개발 가이드

### 개발 환경 설정

```bash
# 개발 환경 설정
pip install -r requirements-dev.txt

# 테스트 실행
pytest tests/

# 코드 포맷팅
black .
```

### 새로운 기능 추가

1. **새로운 노드 추가**: `editor/nodes/` 폴더에 노드 구현
2. **새로운 서비스 추가**: `service/` 폴더에 서비스 구현
3. **새로운 API 추가**: `controller/` 폴더에 컨트롤러 구현
4. **테스트 작성**: `tests/` 폴더에 테스트 코드 작성

## 📚 상세 문서

각 시스템의 상세한 사용법과 구현 방법은 해당 폴더의 README.md를 참조하세요:

- **[⚙️ 설정 시스템](config/README.md)**: 유연한 설정 관리 시스템
- **[🎮 컨트롤러 시스템](controller/README.md)**: RESTful API 구현 가이드
- **[🎨 NodeEditor 시스템](editor/README.md)**: 노드 기반 워크플로우 엔진
- **[🛠️ 서비스 시스템](service/README.md)**: 비즈니스 로직 구현

### 📖 추가 리소스

- **[Plateer AI-LAB 기술 블로그](https://x2bee.tistory.com/)**: 최신 기술 동향 및 개발 인사이트
- **[GitHub 저장소](https://github.com/X2bee/PlateeRAG_backend)**: 소스 코드 및 이슈 관리
- **[프로젝트 Wiki](https://github.com/X2bee/PlateeRAG_backend/wiki)**: 상세 문서 및 FAQ

## 🤝 기여하기

### 개발 참여

1. **Fork** 저장소
2. **Feature branch** 생성 (`git checkout -b feature/amazing-feature`)
3. **Commit** 변경사항 (`git commit -m 'Add amazing feature'`)
4. **Push** 브랜치 (`git push origin feature/amazing-feature`)
5. **Pull Request** 생성

### 코딩 규칙

- **Python**: PEP 8 준수
- **문서화**: 모든 함수와 클래스에 독스트링 작성
- **테스트**: 새로운 기능에 대한 테스트 작성
- **타입 힌트**: 타입 힌트 사용 권장

## 📄 라이선스

이 프로젝트는 [GPL-3.0](LICENSE)를 따릅니다.

## 👥 개발팀

- **Plateer AI-LAB**
- **CocoRoF** - 장하렴
- **haesookimDev** - 김해수
- **SonAIengine** - 손성준 (AI-LAB Part Leader)

## 🆘 지원

### 문제 신고

- **GitHub Issues**: [Issues 페이지](https://github.com/X2bee/PlateeRAG_backend/issues)
- **버그 신고**: 버그 발견 시 이슈 생성
- **기능 요청**: 새로운 기능 제안

### 커뮤니티

- **기술 블로그**: [Plateer AI-LAB 기술 블로그](https://x2bee.tistory.com/)
- **문서**: [프로젝트 Wiki](https://github.com/X2bee/PlateeRAG_backend/wiki)

---

## 🎯 로드맵

### 현재 버전 (v1.0)
- ✅ 기본 NodeEditor 시스템
- ✅ RAG 시스템 구현
- ✅ 실시간 워크플로우 실행
- ✅ 성능 모니터링

### 다음 버전 (v1.1)
- 🔄 향상된 노드 타입
- 🔄 워크플로우 템플릿 시스템
- 🔄 사용자 인증 시스템
- 🔄 클러스터링 지원

### 장기 계획 (v2.0)
- 📋 GraphQL API 지원
- 📋 마이크로서비스 아키텍처
- 📋 실시간 협업 기능
- 📋 AI 기반 워크플로우 최적화

---

<div align="center">

**Made with ❤️ by Plateer AI-LAB**

[⭐ Star this repo](https://github.com/X2bee/PlateeRAG_backend) • [🐛 Report Bug](https://github.com/X2bee/PlateeRAG_backend/issues) • [� Request Feature](https://github.com/X2bee/PlateeRAG_backend/issues)

</div>

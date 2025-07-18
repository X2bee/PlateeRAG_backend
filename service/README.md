# PlateERAG Backend Service 시스템 가이드

## 📖 개요

PlateERAG Backend의 Service 시스템은 **비즈니스 로직과 데이터 처리**를 담당하는 핵심 컴포넌트들을 제공합니다. 각 서비스는 특정 도메인의 기능을 담당하며, 컨트롤러와 모델 사이의 중간 계층 역할을 합니다.

### 🎯 핵심 특징
- **모듈화된 설계**: 각 서비스는 독립적인 기능 영역을 담당
- **확장성**: 새로운 서비스를 쉽게 추가할 수 있는 구조
- **재사용성**: 서비스 간 공통 기능 공유
- **비즈니스 로직 분리**: 데이터 액세스와 비즈니스 로직의 명확한 분리
- **성능 모니터링**: 실시간 성능 추적 및 로깅

## 🏗️ Service 아키텍처

### 폴더 구조
```
service/
├── README.md                    # 📖 이 문서
├── database/                    # 🗄️ 데이터베이스 관리
│   ├── __init__.py
│   ├── connection.py            # 📡 AppDatabaseManager
│   ├── execution_meta_service.py # 🔄 실행 메타데이터 서비스
│   └── models/                  # 📋 데이터 모델
│       ├── __init__.py
│       ├── base_model.py        # 🏗️ 기본 모델 클래스
│       ├── chat.py              # 💬 채팅 모델
│       ├── executor.py          # 🔄 실행 모델
│       ├── performance.py       # 📊 성능 모델
│       ├── persistent_config_model.py # ⚙️ 설정 모델
│       └── user.py              # 👤 사용자 모델
├── embedding/                   # 🔤 임베딩 서비스
│   ├── __init__.py
│   ├── base_embedding.py        # 🏗️ 기본 임베딩 클래스
│   ├── custom_http_embedding.py # 🌐 HTTP 임베딩 클라이언트
│   ├── embedding_factory.py     # 🏭 임베딩 팩토리
│   ├── huggingface_embedding.py # 🤗 HuggingFace 임베딩
│   └── openai_embedding.py      # 🤖 OpenAI 임베딩
├── general_function/            # 🔧 일반 기능
│   ├── __init__.py
│   └── chat.py                  # 💬 채팅 기능
├── monitoring/                  # 📊 모니터링 서비스
│   ├── __init__.py
│   └── performance_logger.py    # 📈 성능 로깅
├── retrieval/                   # 🔍 검색 서비스
│   ├── __init__.py
│   ├── document_processor.py    # 📄 문서 처리
│   └── rag_service.py           # 🧠 RAG 서비스
└── vector_db/                   # 🗂️ 벡터 DB 서비스
    ├── __init__.py
    └── vector_manager.py        # 📊 벡터 DB 관리
```

## 🗄️ 데이터베이스 서비스 (Database)

### 📡 AppDatabaseManager

`AppDatabaseManager`는 애플리케이션 데이터베이스의 **핵심 관리 컴포넌트**입니다. ORM-like 인터페이스를 제공하여 모델 기반 데이터베이스 작업을 단순화합니다.

#### 🎯 주요 기능
- **모델 등록 시스템**: 데이터베이스 모델 클래스 자동 등록
- **테이블 자동 생성**: 등록된 모델 기반 테이블 생성
- **CRUD 작업**: Create, Read, Update, Delete 기본 작업
- **다중 DB 지원**: SQLite, PostgreSQL 지원
- **트랜잭션 관리**: 안전한 데이터베이스 작업

#### 🔧 초기화 및 설정
```python
from service.database.connection import AppDatabaseManager
from service.database.models.user import User
from service.database.models.chat import Chat

# 데이터베이스 매니저 초기화
db_manager = AppDatabaseManager(database_config)

# 모델 등록
db_manager.register_model(User)
db_manager.register_model(Chat)

# 또는 여러 모델 한 번에 등록
db_manager.register_models([User, Chat])

# 데이터베이스 초기화 (연결 + 테이블 생성)
if db_manager.initialize_database():
    print("데이터베이스 초기화 완료")
else:
    print("데이터베이스 초기화 실패")
```

#### 📋 CRUD 작업 예시

##### 1. **Create (생성)**
```python
from service.database.models.user import User

# 새 사용자 생성
user = User(
    username="john_doe",
    email="john@example.com",
    full_name="John Doe",
    created_at=datetime.now()
)

# 데이터베이스에 삽입
user_id = db_manager.insert(user)
if user_id:
    print(f"사용자 생성 완료: ID {user_id}")
else:
    print("사용자 생성 실패")
```

##### 2. **Read (조회)**
```python
from service.database.models.user import User

# ID로 사용자 조회
user = db_manager.find_by_id(User, user_id)
if user:
    print(f"사용자 조회: {user.username}")

# 모든 사용자 조회 (페이징)
users = db_manager.find_all(User, limit=10, offset=0)
for user in users:
    print(f"사용자: {user.username}")

# 조건으로 사용자 조회
active_users = db_manager.find_by_condition(
    User, 
    conditions={"is_active": True}, 
    limit=50
)
```

##### 3. **Update (수정)**
```python
# 사용자 정보 수정
user = db_manager.find_by_id(User, user_id)
if user:
    user.full_name = "John Smith"
    user.updated_at = datetime.now()
    
    if db_manager.update(user):
        print("사용자 정보 수정 완료")
    else:
        print("사용자 정보 수정 실패")
```

##### 4. **Delete (삭제)**
```python
# 사용자 삭제
if db_manager.delete(User, user_id):
    print("사용자 삭제 완료")
else:
    print("사용자 삭제 실패")
```

#### 🔍 고급 기능

##### 1. **복잡한 조건 검색**
```python
# 여러 조건으로 검색
users = db_manager.find_by_condition(
    User, 
    conditions={
        "is_active": True,
        "created_at": datetime(2024, 1, 1)  # 특정 날짜 이후
    },
    limit=100,
    offset=0
)
```

##### 2. **테이블 생성 확인**
```python
# 등록된 모든 모델의 테이블 생성
if db_manager.create_tables():
    print("모든 테이블 생성 완료")
else:
    print("테이블 생성 실패")
```

##### 3. **마이그레이션 실행**
```python
# 데이터베이스 스키마 마이그레이션
if db_manager.run_migrations():
    print("마이그레이션 완료")
else:
    print("마이그레이션 실패")
```

#### 🛠️ 설정 및 최적화

##### 1. **연결 설정**
```python
# 데이터베이스 설정 확인
db_type = db_manager.config_db_manager.db_type
print(f"데이터베이스 타입: {db_type}")

# 연결 상태 확인
if db_manager.config_db_manager.connection:
    print("데이터베이스 연결 활성")
else:
    print("데이터베이스 연결 비활성")
```

##### 2. **인덱스 최적화**
```python
# 특정 모델에 대한 인덱스 생성 (PersistentConfigModel 예시)
# 자동으로 처리되지만 수동으로도 가능
index_query = "CREATE INDEX IF NOT EXISTS idx_user_email ON users(email)"
db_manager.config_db_manager.execute_query(index_query)
```

##### 3. **연결 종료**
```python
# 애플리케이션 종료 시 연결 정리
db_manager.close()
```

## 🔤 임베딩 서비스 (Embedding)

### 🏭 EmbeddingFactory

임베딩 서비스는 **텍스트를 벡터로 변환**하는 기능을 제공합니다. 다양한 임베딩 제공자를 지원하는 팩토리 패턴을 사용합니다.

#### 🎯 지원 임베딩 제공자
- **OpenAI**: OpenAI의 임베딩 모델 (`text-embedding-ada-002`, `text-embedding-3-small`, `text-embedding-3-large`)
- **HuggingFace**: HuggingFace의 오픈소스 임베딩 모델
- **Custom HTTP**: 사용자 정의 HTTP API 임베딩 서비스

#### 🔧 사용 예시
```python
from service.embedding import EmbeddingFactory

# 임베딩 클라이언트 생성
embedding_client = EmbeddingFactory.create_embedding_client(vectordb_config)

# 텍스트 임베딩
text = "안녕하세요, 세계!"
embedding_vector = embedding_client.embed_text(text)
print(f"임베딩 차원: {len(embedding_vector)}")

# 배치 임베딩
texts = ["첫 번째 텍스트", "두 번째 텍스트", "세 번째 텍스트"]
embedding_vectors = embedding_client.embed_batch(texts)
print(f"배치 임베딩 완료: {len(embedding_vectors)}개")
```

#### 📊 제공자별 특징
| 제공자 | 모델 예시 | 차원 | 특징 |
|--------|-----------|------|------|
| OpenAI | text-embedding-ada-002 | 1536 | 고품질, 상용 서비스 |
| HuggingFace | sentence-transformers/all-MiniLM-L6-v2 | 384 | 오픈소스, 다양한 모델 |
| Custom HTTP | 사용자 정의 | 가변 | 사용자 맞춤형 API |

## 📊 모니터링 서비스 (Monitoring)

### 📈 PerformanceLogger

`PerformanceLogger`는 **워크플로우 노드의 성능을 실시간으로 측정**하고 로깅하는 컴포넌트입니다.

#### 🎯 측정 항목
- **처리 시간**: 노드 실행 시간
- **CPU 사용률**: 프로세스 기준 CPU 사용량
- **메모리 사용량**: RAM 사용량
- **GPU 사용률**: NVIDIA GPU 사용량 (선택사항)

#### 🔧 사용 예시
```python
from service.monitoring.performance_logger import PerformanceLogger

# 성능 로거 생성
logger = PerformanceLogger(
    workflow_name="chat_workflow",
    workflow_id="workflow_001",
    node_id="chat_node_1",
    node_name="OpenAI Chat",
    user_id="user_123",
    db_manager=db_manager
)

# 컨텍스트 매니저로 사용
with logger:
    # 모니터링할 작업 실행
    result = some_expensive_operation()
    
# 성능 데이터가 자동으로 데이터베이스에 저장됨
```

## 🔍 검색 서비스 (Retrieval)

### 📄 DocumentProcessor

`DocumentProcessor`는 **다양한 형식의 문서를 처리**하여 RAG 시스템에 적합한 형태로 변환하는 컴포넌트입니다.

#### 🎯 주요 기능
- **다양한 파일 형식 지원**: PDF, DOCX, DOC, TXT
- **텍스트 추출**: 문서에서 순수 텍스트 추출
- **텍스트 정리**: 불필요한 공백, 줄바꿈 정리
- **청크 분할**: 긴 문서를 검색에 적합한 크기로 분할

#### 📋 지원 파일 형식
```python
document_processor = DocumentProcessor()

# 지원 형식 확인
supported_types = document_processor.get_supported_types()
print(f"지원 형식: {supported_types}")
# 출력: ['pdf', 'docx', 'doc', 'txt']
```

#### 🔧 텍스트 추출 예시
```python
import asyncio
from service.retrieval.document_processor import DocumentProcessor

async def process_document():
    processor = DocumentProcessor()
    
    # PDF 파일 처리
    pdf_text = await processor.extract_text_from_file("document.pdf", "pdf")
    print(f"추출된 텍스트 길이: {len(pdf_text)}")
    
    # DOCX 파일 처리
    docx_text = await processor.extract_text_from_file("document.docx", "docx")
    print(f"추출된 텍스트 길이: {len(docx_text)}")
    
    # 텍스트 정리
    cleaned_text = processor.clean_text(pdf_text)
    print(f"정리된 텍스트 길이: {len(cleaned_text)}")

# 비동기 실행
asyncio.run(process_document())
```

#### 📊 텍스트 청크 분할
```python
from service.retrieval.document_processor import DocumentProcessor

processor = DocumentProcessor()

# 긴 텍스트를 청크로 분할
long_text = "매우 긴 문서 내용..." * 1000
chunks = processor.split_text_into_chunks(
    text=long_text,
    chunk_size=1000,      # 청크 크기
    chunk_overlap=100     # 청크 간 중복
)

print(f"총 청크 수: {len(chunks)}")
for i, chunk in enumerate(chunks[:3]):  # 처음 3개 청크만 출력
    print(f"청크 {i+1}: {len(chunk)} 문자")
```

#### 🔍 고급 기능

##### 1. **메타데이터 추출**
```python
# 파일 메타데이터와 함께 텍스트 추출
file_path = "document.pdf"
metadata = await processor.extract_metadata(file_path)
text = await processor.extract_text_from_file(file_path, "pdf")

print(f"파일 크기: {metadata.get('size', 'Unknown')}")
print(f"생성일: {metadata.get('created_at', 'Unknown')}")
print(f"페이지 수: {metadata.get('page_count', 'Unknown')}")
```

##### 2. **배치 처리**
```python
# 여러 파일 일괄 처리
file_paths = ["doc1.pdf", "doc2.docx", "doc3.txt"]
results = await processor.batch_process(file_paths)

for file_path, result in results.items():
    if result['success']:
        print(f"{file_path}: 성공 - {len(result['text'])} 문자")
    else:
        print(f"{file_path}: 실패 - {result['error']}")
```

## 🧠 RAG 서비스 (RAG Service)

### 🎯 RAGService

`RAGService`는 **RAG(Retrieval-Augmented Generation) 시스템의 핵심**으로, 문서 처리, 임베딩, 벡터 검색을 통합하여 완전한 RAG 워크플로우를 제공합니다.

#### 🏗️ 아키텍처 구성요소
- **DocumentProcessor**: 문서 처리 및 텍스트 추출
- **EmbeddingFactory**: 텍스트 임베딩 생성
- **VectorManager**: 벡터 데이터베이스 관리
- **비즈니스 로직**: RAG 워크플로우 조정

#### 🔧 초기화 및 설정
```python
from service.retrieval.rag_service import RAGService

# RAG 서비스 초기화
rag_service = RAGService(
    vectordb_config=vectordb_config,
    openai_config=openai_config  # 선택사항
)

# 연결 상태 확인
if rag_service.is_connected():
    print("RAG 서비스 연결 완료")
else:
    print("RAG 서비스 연결 실패")
```

#### 📚 컬렉션 관리

##### 1. **컬렉션 생성**
```python
# 새 컬렉션 생성
result = rag_service.create_collection(
    collection_name="knowledge_base",
    description="회사 지식 베이스",
    metadata={
        "category": "internal_docs",
        "department": "engineering",
        "version": "1.0"
    }
)

print(f"컬렉션 생성 결과: {result['status']}")
```

##### 2. **컬렉션 목록 조회**
```python
# 모든 컬렉션 조회
collections = rag_service.list_collections()
print(f"사용 가능한 컬렉션: {collections['collections']}")

# 컬렉션 상세 정보 조회
collection_info = rag_service.get_collection_info("knowledge_base")
print(f"벡터 수: {collection_info['vectors_count']}")
print(f"벡터 차원: {collection_info['config']['vector_size']}")
```

##### 3. **컬렉션 삭제**
```python
# 컬렉션 삭제
result = rag_service.delete_collection("old_collection")
print(f"삭제 결과: {result['message']}")
```

#### 📄 문서 업로드 및 처리

##### 1. **단일 문서 업로드**
```python
# 파일 업로드
with open("document.pdf", "rb") as file:
    result = await rag_service.upload_document(
        file=file,
        filename="document.pdf",
        collection_name="knowledge_base",
        metadata={
            "author": "John Doe",
            "department": "engineering",
            "tags": ["technical", "guide"]
        }
    )

print(f"문서 업로드 완료: {result['document_id']}")
print(f"생성된 청크 수: {result['chunks_count']}")
```

##### 2. **배치 문서 업로드**
```python
# 여러 문서 일괄 업로드
file_paths = ["doc1.pdf", "doc2.docx", "doc3.txt"]
results = await rag_service.batch_upload_documents(
    file_paths=file_paths,
    collection_name="knowledge_base",
    chunk_size=1000,
    chunk_overlap=100
)

for file_path, result in results.items():
    if result['success']:
        print(f"{file_path}: 성공 - {result['chunks_count']} 청크")
    else:
        print(f"{file_path}: 실패 - {result['error']}")
```

#### 🔍 벡터 검색 및 RAG 쿼리

##### 1. **기본 벡터 검색**
```python
# 벡터 검색
search_results = rag_service.search_vectors(
    collection_name="knowledge_base",
    query="Python 프로그래밍 가이드",
    top_k=5,
    score_threshold=0.7
)

print(f"검색 결과: {len(search_results['results'])}개")
for i, result in enumerate(search_results['results']):
    print(f"{i+1}. 점수: {result['score']:.3f}")
    print(f"   내용: {result['payload']['content'][:100]}...")
```

##### 2. **필터링된 검색**
```python
# 조건부 검색
filtered_results = rag_service.search_vectors(
    collection_name="knowledge_base",
    query="API 문서",
    top_k=10,
    filters={
        "department": "engineering",
        "tags": "technical"
    }
)

print(f"필터링된 검색 결과: {len(filtered_results['results'])}개")
```

##### 3. **RAG 쿼리 실행**
```python
# 완전한 RAG 쿼리
rag_result = await rag_service.query(
    query="Python에서 비동기 프로그래밍을 어떻게 하나요?",
    collection_name="knowledge_base",
    top_k=5,
    include_sources=True,
    temperature=0.7
)

print(f"RAG 응답: {rag_result['answer']}")
print(f"참조 문서: {len(rag_result['sources'])}개")
for source in rag_result['sources']:
    print(f"- {source['filename']} (점수: {source['score']:.3f})")
```

#### 🔧 고급 기능

##### 1. **사용자 정의 청크 설정**
```python
# 사용자 정의 청크 설정으로 문서 처리
custom_result = await rag_service.upload_document(
    file=file,
    filename="large_document.pdf",
    collection_name="knowledge_base",
    chunk_size=1500,      # 더 큰 청크 크기
    chunk_overlap=200,    # 더 큰 중복
    metadata={
        "processing_config": "large_chunks"
    }
)
```

##### 2. **실시간 업데이트**
```python
# 문서 업데이트 (기존 문서 덮어쓰기)
update_result = await rag_service.update_document(
    document_id="doc_12345",
    file=updated_file,
    filename="updated_document.pdf",
    collection_name="knowledge_base"
)

print(f"업데이트 완료: {update_result['message']}")
```

##### 3. **문서 삭제**
```python
# 특정 문서 삭제
delete_result = rag_service.delete_document(
    document_id="doc_12345",
    collection_name="knowledge_base"
)

print(f"삭제 완료: {delete_result['message']}")
```

##### 4. **통계 및 분석**
```python
# 컬렉션 통계
stats = rag_service.get_collection_stats("knowledge_base")
print(f"총 문서 수: {stats['document_count']}")
print(f"총 청크 수: {stats['chunk_count']}")
print(f"평균 청크 크기: {stats['avg_chunk_size']}")
print(f"최근 업로드: {stats['last_upload_date']}")
```

##### 5. **백업 및 복원**
```python
# 컬렉션 백업
backup_result = rag_service.backup_collection(
    collection_name="knowledge_base",
    backup_path="/backups/knowledge_base_backup.json"
)

# 컬렉션 복원
restore_result = rag_service.restore_collection(
    collection_name="knowledge_base_restored",
    backup_path="/backups/knowledge_base_backup.json"
)
```

## 🗂️ 벡터 DB 서비스 (Vector Database)

### 📊 VectorManager

`VectorManager`는 **Qdrant 벡터 데이터베이스의 저수준 관리**를 담당하는 핵심 컴포넌트입니다. 벡터 저장, 검색, 관리의 모든 기능을 제공합니다.

#### 🎯 주요 기능
- **컬렉션 관리**: 생성, 삭제, 조회
- **벡터 저장**: 대용량 벡터 데이터 저장
- **유사도 검색**: 고성능 벡터 검색
- **메타데이터 관리**: 벡터와 연결된 메타데이터 관리
- **필터링**: 복잡한 조건 기반 검색

#### 🔧 초기화 및 연결
```python
from service.vector_db.vector_manager import VectorManager

# 벡터 매니저 초기화
vector_manager = VectorManager(vectordb_config)

# 연결 상태 확인
if vector_manager.is_connected():
    print("Qdrant 연결 성공")
else:
    print("Qdrant 연결 실패")
    
# 연결 재시도
vector_manager.ensure_connected()
```

#### 📊 컬렉션 관리

##### 1. **컬렉션 생성**
```python
# 기본 컬렉션 생성
result = vector_manager.create_collection(
    collection_name="documents",
    vector_size=1536,
    distance="Cosine"
)
print(f"컬렉션 생성: {result['status']}")

# 메타데이터와 함께 컬렉션 생성
result = vector_manager.create_collection(
    collection_name="advanced_docs",
    vector_size=1536,
    distance="Cosine",
    description="고급 문서 컬렉션",
    metadata={
        "version": "2.0",
        "category": "premium",
        "max_documents": 10000
    }
)
```

##### 2. **컬렉션 목록 및 정보 조회**
```python
# 모든 컬렉션 목록
collections = vector_manager.list_collections()
print(f"컬렉션 목록: {collections['collections']}")

# 특정 컬렉션 정보
info = vector_manager.get_collection_info("documents")
print(f"벡터 수: {info['vectors_count']}")
print(f"벡터 차원: {info['config']['vector_size']}")
print(f"거리 메트릭: {info['config']['distance']}")
```

##### 3. **컬렉션 삭제**
```python
# 컬렉션 삭제
result = vector_manager.delete_collection("old_collection")
print(f"삭제 결과: {result['message']}")
```

#### 📝 벡터 포인트 작업

##### 1. **포인트 삽입**
```python
# 단일 포인트 삽입
points = [
    {
        "vector": [0.1, 0.2, 0.3, ...],  # 1536차원 벡터
        "payload": {
            "text": "이것은 샘플 텍스트입니다.",
            "category": "sample",
            "timestamp": "2024-01-01T00:00:00Z"
        }
    }
]

result = vector_manager.insert_points("documents", points)
print(f"삽입 완료: {result['message']}")

# 배치 포인트 삽입
batch_points = []
for i in range(100):
    batch_points.append({
        "vector": [random.random() for _ in range(1536)],
        "payload": {
            "text": f"문서 {i}",
            "category": "batch",
            "index": i
        }
    })

result = vector_manager.insert_points("documents", batch_points)
print(f"배치 삽입 완료: {result['message']}")
```

##### 2. **포인트 검색**
```python
# 기본 벡터 검색
query_vector = [0.1, 0.2, 0.3, ...]  # 검색 벡터
results = vector_manager.search_points(
    collection_name="documents",
    query_vector=query_vector,
    limit=10,
    score_threshold=0.5
)

print(f"검색 결과: {results['total']}개")
for result in results['results']:
    print(f"ID: {result['id']}, 점수: {result['score']:.3f}")
    print(f"내용: {result['payload']['text']}")
```

##### 3. **필터링된 검색**
```python
# 조건부 검색
filtered_results = vector_manager.search_points(
    collection_name="documents",
    query_vector=query_vector,
    limit=10,
    filter_criteria={
        "category": "technical",
        "timestamp": {
            "range": {
                "gte": "2024-01-01T00:00:00Z",
                "lte": "2024-12-31T23:59:59Z"
            }
        }
    }
)

print(f"필터링된 검색 결과: {len(filtered_results['results'])}개")
```

##### 4. **포인트 삭제**
```python
# 특정 포인트 삭제
point_ids = ["point_1", "point_2", "point_3"]
result = vector_manager.delete_points("documents", point_ids)
print(f"삭제 완료: {result['message']}")
```

#### 🔍 고급 검색 기능

##### 1. **스크롤 검색**
```python
# 대용량 데이터 스크롤 검색
points, next_offset = vector_manager.scroll_points(
    collection_name="documents",
    limit=1000,
    with_payload=True,
    with_vectors=False
)

print(f"스크롤 결과: {len(points)}개 포인트")
print(f"다음 오프셋: {next_offset}")

# 다음 배치 조회
if next_offset:
    next_points, final_offset = vector_manager.scroll_points(
        collection_name="documents",
        limit=1000,
        offset=next_offset,
        with_payload=True,
        with_vectors=False
    )
```

##### 2. **조건부 스크롤**
```python
# 필터 조건으로 스크롤
filtered_points, offset = vector_manager.scroll_points(
    collection_name="documents",
    filter_criteria={
        "category": "important",
        "score": {
            "range": {"gte": 0.8}
        }
    },
    limit=500
)
```

#### 🛠️ 관리 및 유지보수

##### 1. **컬렉션 문서 수 업데이트**
```python
# 문서 추가 시 카운트 증가
vector_manager.update_collection_document_count("documents", 1)

# 문서 삭제 시 카운트 감소
vector_manager.update_collection_document_count("documents", -1)
```

##### 2. **연결 관리**
```python
# 연결 상태 확인 및 재연결
try:
    vector_manager.ensure_connected()
    print("연결 상태 정상")
except Exception as e:
    print(f"연결 오류: {e}")

# 리소스 정리
vector_manager.cleanup()
```

##### 3. **성능 모니터링**
```python
# 컬렉션 통계 조회
info = vector_manager.get_collection_info("documents")
print(f"디스크 사용량: {info.get('disk_data_size', 'N/A')}")
print(f"메모리 사용량: {info.get('ram_data_size', 'N/A')}")
print(f"인덱스된 벡터 수: {info.get('indexed_vectors_count', 'N/A')}")
```

#### 🔧 고급 활용 패턴

##### 1. **배치 처리 최적화**
```python
def optimized_batch_insert(vector_manager, collection_name, large_dataset):
    """대용량 데이터 최적화된 배치 삽입"""
    batch_size = 1000
    
    for i in range(0, len(large_dataset), batch_size):
        batch = large_dataset[i:i + batch_size]
        
        try:
            result = vector_manager.insert_points(collection_name, batch)
            print(f"배치 {i//batch_size + 1} 완료: {len(batch)}개")
        except Exception as e:
            print(f"배치 {i//batch_size + 1} 실패: {e}")
```

##### 2. **유사도 검색 최적화**
```python
def semantic_search_with_fallback(vector_manager, collection_name, query_vector):
    """유사도 검색 + 폴백 로직"""
    
    # 1차 검색 (높은 임계값)
    results = vector_manager.search_points(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=5,
        score_threshold=0.8
    )
    
    # 결과가 부족하면 2차 검색 (낮은 임계값)
    if len(results['results']) < 3:
        results = vector_manager.search_points(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=10,
            score_threshold=0.5
        )
    
    return results
```

##### 3. **멀티 컬렉션 검색**
```python
def multi_collection_search(vector_manager, collections, query_vector):
    """여러 컬렉션에서 동시 검색"""
    all_results = []
    
    for collection_name in collections:
        try:
            results = vector_manager.search_points(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=5
            )
            
            # 컬렉션 정보 추가
            for result in results['results']:
                result['collection'] = collection_name
                all_results.append(result)
                
        except Exception as e:
            print(f"컬렉션 {collection_name} 검색 실패: {e}")
    
    # 점수 기준 정렬
    all_results.sort(key=lambda x: x['score'], reverse=True)
    return all_results[:10]  # 상위 10개 결과
```

## 🔧 서비스 통합 및 활용

### 🎯 통합 서비스 활용 예시

```python
# 완전한 RAG 시스템 구축
class IntegratedRAGSystem:
    def __init__(self, configs):
        self.db_manager = AppDatabaseManager(configs.database)
        self.vector_manager = VectorManager(configs.vectordb)
        self.rag_service = RAGService(configs.vectordb, configs.openai)
        self.document_processor = DocumentProcessor()
        
    async def process_and_store_document(self, file_path, collection_name):
        """문서 처리 및 저장 통합 워크플로우"""
        
        # 1. 문서 처리
        text = await self.document_processor.extract_text_from_file(file_path)
        chunks = self.document_processor.split_text_into_chunks(text)
        
        # 2. 임베딩 생성
        embedding_client = EmbeddingFactory.create_embedding_client(configs.vectordb)
        embeddings = embedding_client.embed_batch(chunks)
        
        # 3. 벡터 저장
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            points.append({
                "vector": embedding,
                "payload": {
                    "text": chunk,
                    "source": file_path,
                    "chunk_index": i
                }
            })
        
        self.vector_manager.insert_points(collection_name, points)
        
        # 4. 메타데이터 데이터베이스 저장
        # (필요에 따라 구현)
        
        return {"chunks": len(chunks), "embeddings": len(embeddings)}
    
    async def intelligent_search(self, query, collection_name):
        """지능형 검색 시스템"""
        
        # 1. 성능 모니터링 시작
        with PerformanceLogger("search", "search_001", "search_node", "Intelligent Search"):
            
            # 2. RAG 검색 실행
            results = await self.rag_service.query(
                query=query,
                collection_name=collection_name,
                top_k=5,
                include_sources=True
            )
            
            # 3. 검색 결과 데이터베이스 저장
            # (필요에 따라 구현)
            
            return results
```

---

## 📚 참고 자료

### 관련 파일들
- **[database/connection.py](database/connection.py)**: 데이터베이스 연결 관리
- **[embedding/embedding_factory.py](embedding/embedding_factory.py)**: 임베딩 팩토리
- **[monitoring/performance_logger.py](monitoring/performance_logger.py)**: 성능 모니터링
- **[retrieval/document_processor.py](retrieval/document_processor.py)**: 문서 처리
- **[retrieval/rag_service.py](retrieval/rag_service.py)**: RAG 서비스
- **[vector_db/vector_manager.py](vector_db/vector_manager.py)**: 벡터 DB 관리

### 외부 라이브러리 문서
- **[Qdrant](https://qdrant.tech/documentation/)**: 벡터 데이터베이스
- **[OpenAI](https://platform.openai.com/docs)**: OpenAI API
- **[LangChain](https://python.langchain.com/)**: LangChain 라이브러리
- **[HuggingFace](https://huggingface.co/docs)**: HuggingFace 생태계

### 모범 사례
- **[RAG 시스템 설계](https://docs.llamaindex.ai/en/stable/getting_started/concepts.html)**: RAG 아키텍처 가이드
- **[벡터 검색 최적화](https://qdrant.tech/documentation/guides/optimization/)**: 벡터 검색 성능 최적화

---

**PlateERAG Backend Service System**  
*🗄️ 데이터베이스 관리 • 🔤 임베딩 서비스 • 📊 성능 모니터링 • 🔍 문서 처리 • 🧠 RAG 시스템 • 🗂️ 벡터 DB*

**이제 서비스 계층을 완전히 이해하고 효과적으로 활용할 수 있습니다!**

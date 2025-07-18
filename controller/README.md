# PlateERAG Backend 컨트롤러 시스템 가이드

## 📖 개요

PlateERAG Backend의 컨트롤러 시스템은 **FastAPI의 APIRouter**를 기반으로 구축된 **RESTful API 엔드포인트**들을 제공합니다. 각 컨트롤러는 특정 도메인의 비즈니스 로직을 담당하며, 클린 아키텍처 원칙에 따라 서비스 계층과 분리되어 있습니다.

### 🎯 핵심 특징
- **도메인 분리**: 각 컨트롤러가 특정 기능 영역을 담당
- **의존성 주입**: FastAPI의 Request 객체를 통한 상태 관리
- **일관된 구조**: 모든 컨트롤러가 동일한 패턴을 따름
- **에러 핸들링**: HTTPException을 통한 표준화된 에러 응답
- **타입 안전**: Pydantic 모델을 통한 요청/응답 검증
- **로깅**: 구조화된 로깅 시스템

## 🏗️ 컨트롤러 아키텍처

### 폴더 구조
```
controller/
├── README.md                    # 📖 이 문서
├── __init__.py                  # 🔧 컨트롤러 패키지 초기화
├── appController.py             # 🖥️  애플리케이션 상태 및 설정 관리
├── chatController.py            # 💬 채팅 및 대화 관리
├── configController.py          # ⚙️  설정 관리 및 구성 API
├── embeddingController.py       # 🔤 임베딩 서비스 관리
├── interactionController.py     # 🔄 상호작용 기록 관리
├── nodeController.py            # 🔗 노드 탐색 및 관리
├── nodeStateController.py       # 📊 노드 상태 관리
├── performanceController.py     # 📈 성능 모니터링
├── ragController.py             # 🧠 RAG 시스템 관리
├── retrievalController.py       # 🔍 문서 검색 및 벡터 관리
└── workflowController.py        # 🔄 워크플로우 실행 관리
```

### 아키텍처 구성요소

#### 1. **공통 패턴**
모든 컨트롤러는 다음 패턴을 따릅니다:
```python
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import logging

logger = logging.getLogger("controller-name")
router = APIRouter(prefix="/api/prefix", tags=["tag"])

class RequestModel(BaseModel):
    # 요청 모델 정의
    pass

@router.get("/endpoint")
async def endpoint_function(request: Request):
    # 엔드포인트 구현
    pass
```

#### 2. **의존성 주입**
Request 객체를 통해 애플리케이션 상태에 접근:
```python
def get_service(request: Request):
    """의존성 주입 함수"""
    if hasattr(request.app.state, 'service') and request.app.state.service:
        return request.app.state.service
    else:
        raise HTTPException(status_code=500, detail="Service not available")
```

#### 3. **에러 핸들링**
표준화된 에러 응답:
```python
try:
    # 비즈니스 로직
    pass
except SpecificError as e:
    logger.error(f"Specific error: {e}")
    raise HTTPException(status_code=400, detail="Bad Request")
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise HTTPException(status_code=500, detail="Internal Server Error")
```

## 🗂️ 컨트롤러 상세 가이드

### 1. 애플리케이션 컨트롤러 (appController.py)
**경로**: `/app`  
**역할**: 애플리케이션 상태 및 설정 관리

#### 주요 엔드포인트
| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/app/status` | 애플리케이션 상태 정보 |
| `GET` | `/app/config` | 설정 요약 정보 |
| `GET` | `/app/config/persistent` | 영속성 설정 상세 정보 |
| `PUT` | `/app/config/persistent/{config_name}` | 설정 값 업데이트 |
| `POST` | `/app/config/persistent/save` | 모든 설정 저장 |
| `POST` | `/app/config/persistent/refresh` | 설정 새로고침 |

#### 사용 예시
```python
# 애플리케이션 상태 확인
curl http://localhost:8000/app/status

# 설정 업데이트
curl -X PUT http://localhost:8000/app/config/persistent/DEBUG_MODE \
  -H "Content-Type: application/json" \
  -d '{"value": true}'
```

### 2. 채팅 컨트롤러 (chatController.py)
**경로**: `/api/chat`  
**역할**: 채팅 및 대화 관리

#### 주요 엔드포인트
| 메서드 | 경로 | 설명 |
|--------|------|------|
| `POST` | `/api/chat/new` | 새로운 채팅 세션 시작 |
| `POST` | `/api/chat/execute` | 채팅 실행 |
| `GET` | `/api/chat/history/{interaction_id}` | 채팅 기록 조회 |

#### 요청 모델
```python
class ChatNewRequest(BaseModel):
    workflow_name: str = "default_mode"
    workflow_id: str = "default_mode"
    interaction_id: str
    input_data: Optional[str] = None

class ChatExecutionRequest(BaseModel):
    user_input: str
    interaction_id: str
    workflow_id: Optional[str] = None
    workflow_name: Optional[str] = None
```

#### 사용 예시
```python
# 새로운 채팅 시작
curl -X POST http://localhost:8000/api/chat/new \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_name": "default_mode",
    "workflow_id": "default_mode",
    "interaction_id": "chat_001"
  }'

# 채팅 실행
curl -X POST http://localhost:8000/api/chat/execute \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "안녕하세요",
    "interaction_id": "chat_001"
  }'
```

### 3. 설정 컨트롤러 (configController.py)
**경로**: `/api/config`  
**역할**: 설정 관리 및 구성 API

#### 주요 엔드포인트
| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/api/config/persistent/summary` | 설정 요약 정보 |
| `GET` | `/api/config/persistent/all` | 모든 설정 상세 정보 |
| `PUT` | `/api/config/persistent/{config_name}` | 특정 설정 업데이트 |
| `POST` | `/api/config/persistent/save` | 모든 설정 저장 |
| `POST` | `/api/config/persistent/refresh` | 설정 새로고침 |

#### 사용 예시
```python
# 설정 요약 조회
curl http://localhost:8000/api/config/persistent/summary

# 설정 업데이트
curl -X PUT http://localhost:8000/api/config/persistent/OPENAI_MODEL_DEFAULT \
  -H "Content-Type: application/json" \
  -d '{"value": "gpt-4", "save_to_db": true}'
```

### 4. 임베딩 컨트롤러 (embeddingController.py)
**경로**: `/api/embedding`  
**역할**: 임베딩 서비스 관리

#### 주요 엔드포인트
| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/api/embedding/providers` | 사용 가능한 임베딩 제공자 목록 |
| `GET` | `/api/embedding/status` | 현재 임베딩 제공자 상태 |
| `POST` | `/api/embedding/switch` | 임베딩 제공자 전환 |
| `POST` | `/api/embedding/test` | 임베딩 테스트 |

#### 사용 예시
```python
# 임베딩 제공자 목록 조회
curl http://localhost:8000/api/embedding/providers

# 임베딩 제공자 전환
curl -X POST http://localhost:8000/api/embedding/switch \
  -H "Content-Type: application/json" \
  -d '{"new_provider": "huggingface"}'

# 임베딩 테스트
curl -X POST http://localhost:8000/api/embedding/test \
  -H "Content-Type: application/json" \
  -d '{"query_text": "Hello, world!"}'
```

### 5. 노드 컨트롤러 (nodeController.py)
**경로**: `/api/node`  
**역할**: 노드 탐색 및 관리

#### 주요 엔드포인트
| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/api/node/get` | 모든 노드 목록 조회 |
| `POST` | `/api/node/discovery` | 노드 탐색 실행 |
| `GET` | `/api/node/registry` | 노드 레지스트리 조회 |
| `GET` | `/api/node/spec/{node_id}` | 특정 노드 스펙 조회 |

#### 사용 예시
```python
# 노드 목록 조회
curl http://localhost:8000/api/node/get

# 노드 탐색 실행
curl -X POST http://localhost:8000/api/node/discovery

# 특정 노드 스펙 조회
curl http://localhost:8000/api/node/spec/math_add
```

### 6. 노드 상태 컨트롤러 (nodeStateController.py)
**경로**: `/api/node-state`  
**역할**: 노드 상태 관리

#### 주요 엔드포인트
| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/api/node-state/registry` | 앱 상태의 노드 레지스트리 |
| `GET` | `/api/node-state/nodes` | 모든 노드 정보 |
| `GET` | `/api/node-state/node/{node_id}` | 특정 노드 정보 |

### 7. 워크플로우 컨트롤러 (workflowController.py)
**경로**: `/api/workflow`  
**역할**: 워크플로우 실행 관리

#### 주요 엔드포인트
| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/api/workflow/list` | 워크플로우 목록 조회 |
| `POST` | `/api/workflow/save` | 워크플로우 저장 |
| `POST` | `/api/workflow/execute` | 워크플로우 실행 |
| `GET` | `/api/workflow/load/{workflow_id}` | 워크플로우 로드 |
| `DELETE` | `/api/workflow/delete/{workflow_id}` | 워크플로우 삭제 |

#### 요청 모델
```python
class WorkflowRequest(BaseModel):
    workflow_name: str
    workflow_id: str
    input_data: str = ""
    interaction_id: str = "default"

class WorkflowData(BaseModel):
    workflow_name: str
    workflow_id: str
    view: Dict[str, Any]
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    interaction_id: str = "default"
```

#### 사용 예시
```python
# 워크플로우 목록 조회
curl http://localhost:8000/api/workflow/list

# 워크플로우 실행
curl -X POST http://localhost:8000/api/workflow/execute \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_name": "my_workflow",
    "workflow_id": "workflow_001",
    "input_data": "test input",
    "interaction_id": "interaction_001"
  }'
```

### 8. 상호작용 컨트롤러 (interactionController.py)
**경로**: `/api/interaction`  
**역할**: 상호작용 기록 관리

#### 주요 엔드포인트
| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/api/interaction/list` | 상호작용 목록 조회 |
| `GET` | `/api/interaction/detail/{interaction_id}` | 상호작용 상세 정보 |
| `DELETE` | `/api/interaction/delete/{interaction_id}` | 상호작용 삭제 |

#### 사용 예시
```python
# 상호작용 목록 조회
curl "http://localhost:8000/api/interaction/list?interaction_id=chat_001&limit=50"

# 상호작용 상세 정보 조회
curl http://localhost:8000/api/interaction/detail/chat_001
```

### 9. 성능 컨트롤러 (performanceController.py)
**경로**: `/api/performance`  
**역할**: 성능 모니터링

#### 주요 엔드포인트
| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/api/performance/workflow/{workflow_name}/{workflow_id}` | 워크플로우 성능 데이터 |
| `GET` | `/api/performance/summary` | 성능 요약 정보 |
| `GET` | `/api/performance/metrics` | 성능 메트릭 |

#### 사용 예시
```python
# 워크플로우 성능 데이터 조회
curl "http://localhost:8000/api/performance/workflow/my_workflow/workflow_001?limit=100"

# 성능 요약 정보 조회
curl http://localhost:8000/api/performance/summary
```

### 10. RAG 컨트롤러 (ragController.py)
**경로**: `/rag`  
**역할**: RAG 시스템 관리

#### 주요 엔드포인트
| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/rag/collections` | 컬렉션 목록 조회 |
| `POST` | `/rag/collections` | 컬렉션 생성 |
| `DELETE` | `/rag/collections/{collection_name}` | 컬렉션 삭제 |
| `POST` | `/rag/collections/{collection_name}/points` | 포인트 삽입 |
| `POST` | `/rag/collections/{collection_name}/search` | 벡터 검색 |
| `POST` | `/rag/documents/upload` | 문서 업로드 |
| `POST` | `/rag/query` | RAG 쿼리 실행 |

#### 사용 예시
```python
# 컬렉션 생성
curl -X POST http://localhost:8000/rag/collections \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "my_collection",
    "vector_size": 1536,
    "distance": "Cosine"
  }'

# 문서 업로드
curl -X POST http://localhost:8000/rag/documents/upload \
  -F "file=@document.pdf" \
  -F "collection_name=my_collection"

# RAG 쿼리 실행
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main topic?",
    "collection_name": "my_collection",
    "top_k": 5
  }'
```

### 11. 검색 컨트롤러 (retrievalController.py)
**경로**: `/api/retrieval`  
**역할**: 문서 검색 및 벡터 관리

#### 주요 엔드포인트
| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/api/retrieval/collections` | 컬렉션 목록 조회 |
| `POST` | `/api/retrieval/collections` | 컬렉션 생성 |
| `POST` | `/api/retrieval/collections/{collection_name}/points` | 포인트 삽입 |
| `POST` | `/api/retrieval/collections/{collection_name}/search` | 벡터 검색 |
| `POST` | `/api/retrieval/documents/upload` | 문서 업로드 |

## 🚀 새로운 컨트롤러 추가하기 (Step-by-Step)

### 🎯 Step 1: 컨트롤러 파일 생성

새로운 컨트롤러를 `controller/` 폴더에 생성합니다.

**예시: userController.py**
```python
"""
사용자 관리 컨트롤러

사용자 계정 관리, 인증, 권한 관리 등의 API 엔드포인트를 제공합니다.
"""

from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel, EmailStr
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger("user-controller")
router = APIRouter(prefix="/api/users", tags=["users"])

# Pydantic 모델 정의
class UserCreateRequest(BaseModel):
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    password: str

class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    full_name: Optional[str] = None
    created_at: datetime
    is_active: bool

class UserUpdateRequest(BaseModel):
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None
    is_active: Optional[bool] = None

# 의존성 주입 함수
def get_user_service(request: Request):
    """사용자 서비스 의존성 주입"""
    if hasattr(request.app.state, 'user_service') and request.app.state.user_service:
        return request.app.state.user_service
    else:
        raise HTTPException(status_code=500, detail="User service not available")

# 엔드포인트 정의
@router.post("/", response_model=UserResponse)
async def create_user(request: Request, user_data: UserCreateRequest):
    """새로운 사용자 생성"""
    try:
        user_service = get_user_service(request)
        
        # 사용자 생성 로직
        user = await user_service.create_user(
            username=user_data.username,
            email=user_data.email,
            full_name=user_data.full_name,
            password=user_data.password
        )
        
        logger.info(f"User created successfully: {user.username}")
        return user
        
    except ValueError as e:
        logger.error(f"Invalid user data: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.get("/", response_model=List[UserResponse])
async def list_users(request: Request, skip: int = 0, limit: int = 100):
    """사용자 목록 조회"""
    try:
        user_service = get_user_service(request)
        users = await user_service.list_users(skip=skip, limit=limit)
        return users
        
    except Exception as e:
        logger.error(f"Error listing users: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.get("/{user_id}", response_model=UserResponse)
async def get_user(request: Request, user_id: str):
    """특정 사용자 조회"""
    try:
        user_service = get_user_service(request)
        user = await user_service.get_user(user_id)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.put("/{user_id}", response_model=UserResponse)
async def update_user(request: Request, user_id: str, user_data: UserUpdateRequest):
    """사용자 정보 업데이트"""
    try:
        user_service = get_user_service(request)
        
        # 업데이트할 데이터만 추출
        update_data = user_data.dict(exclude_unset=True)
        
        user = await user_service.update_user(user_id, update_data)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        logger.info(f"User updated successfully: {user_id}")
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.delete("/{user_id}")
async def delete_user(request: Request, user_id: str):
    """사용자 삭제"""
    try:
        user_service = get_user_service(request)
        
        result = await user_service.delete_user(user_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="User not found")
        
        logger.info(f"User deleted successfully: {user_id}")
        return {"message": "User deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting user: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.post("/{user_id}/activate")
async def activate_user(request: Request, user_id: str):
    """사용자 활성화"""
    try:
        user_service = get_user_service(request)
        
        user = await user_service.activate_user(user_id)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        logger.info(f"User activated successfully: {user_id}")
        return {"message": "User activated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error activating user: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.post("/{user_id}/deactivate")
async def deactivate_user(request: Request, user_id: str):
    """사용자 비활성화"""
    try:
        user_service = get_user_service(request)
        
        user = await user_service.deactivate_user(user_id)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        logger.info(f"User deactivated successfully: {user_id}")
        return {"message": "User deactivated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deactivating user: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
```

### 🔧 Step 2: 메인 애플리케이션에 등록

새로운 컨트롤러를 `main.py`에 등록합니다.

```python
# main.py
from controller.userController import router as user_router

app = FastAPI(title="PlateERAG Backend")

# 기존 라우터들...
app.include_router(user_router)  # 새로운 라우터 추가
```

### 🧪 Step 3: 테스트

새로운 컨트롤러를 테스트합니다.

```python
# 사용자 생성
curl -X POST http://localhost:8000/api/users \
  -H "Content-Type: application/json" \
  -d '{
    "username": "john_doe",
    "email": "john@example.com",
    "full_name": "John Doe",
    "password": "secure_password"
  }'

# 사용자 목록 조회
curl http://localhost:8000/api/users

# 특정 사용자 조회
curl http://localhost:8000/api/users/user_123

# 사용자 정보 업데이트
curl -X PUT http://localhost:8000/api/users/user_123 \
  -H "Content-Type: application/json" \
  -d '{
    "full_name": "John Smith",
    "is_active": true
  }'

# 사용자 삭제
curl -X DELETE http://localhost:8000/api/users/user_123
```

### 🎨 Step 4: 고급 패턴

#### 1. **미들웨어 추가**
```python
from fastapi import Request
from fastapi.middleware.base import BaseHTTPMiddleware

class UserAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # 인증 로직
        if request.url.path.startswith("/api/users"):
            # 토큰 검증
            pass
        
        response = await call_next(request)
        return response
```

#### 2. **권한 검사**
```python
from fastapi import Depends, HTTPException
from typing import Optional

async def get_current_user(request: Request) -> Optional[dict]:
    """현재 사용자 정보 조회"""
    # JWT 토큰 검증 로직
    pass

async def require_admin(current_user: dict = Depends(get_current_user)):
    """관리자 권한 필요"""
    if not current_user or not current_user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user

@router.delete("/{user_id}")
async def delete_user(
    request: Request, 
    user_id: str, 
    current_user: dict = Depends(require_admin)
):
    """사용자 삭제 (관리자 권한 필요)"""
    # 삭제 로직
    pass
```

#### 3. **페이지네이션**
```python
from fastapi import Query

class PaginationParams:
    def __init__(
        self,
        page: int = Query(1, ge=1, description="페이지 번호"),
        size: int = Query(10, ge=1, le=100, description="페이지 크기")
    ):
        self.page = page
        self.size = size
        self.offset = (page - 1) * size

@router.get("/", response_model=Dict[str, Any])
async def list_users(
    request: Request,
    pagination: PaginationParams = Depends()
):
    """사용자 목록 조회 (페이지네이션)"""
    user_service = get_user_service(request)
    
    users = await user_service.list_users(
        offset=pagination.offset,
        limit=pagination.size
    )
    
    total_count = await user_service.count_users()
    
    return {
        "users": users,
        "pagination": {
            "page": pagination.page,
            "size": pagination.size,
            "total": total_count,
            "pages": (total_count + pagination.size - 1) // pagination.size
        }
    }
```

#### 4. **검색 및 필터링**
```python
from typing import Optional

@router.get("/search", response_model=List[UserResponse])
async def search_users(
    request: Request,
    q: Optional[str] = Query(None, description="검색 쿼리"),
    is_active: Optional[bool] = Query(None, description="활성 상태"),
    created_after: Optional[datetime] = Query(None, description="생성일 이후"),
    created_before: Optional[datetime] = Query(None, description="생성일 이전")
):
    """사용자 검색"""
    user_service = get_user_service(request)
    
    filters = {}
    if q:
        filters["search"] = q
    if is_active is not None:
        filters["is_active"] = is_active
    if created_after:
        filters["created_after"] = created_after
    if created_before:
        filters["created_before"] = created_before
    
    users = await user_service.search_users(filters)
    return users
```

### 🔄 Step 5: 배치 작업

```python
from fastapi import BackgroundTasks

@router.post("/batch/deactivate")
async def batch_deactivate_users(
    request: Request,
    background_tasks: BackgroundTasks,
    user_ids: List[str]
):
    """사용자 일괄 비활성화"""
    
    async def deactivate_users_task(user_ids: List[str]):
        user_service = get_user_service(request)
        for user_id in user_ids:
            try:
                await user_service.deactivate_user(user_id)
                logger.info(f"User {user_id} deactivated")
            except Exception as e:
                logger.error(f"Failed to deactivate user {user_id}: {e}")
    
    background_tasks.add_task(deactivate_users_task, user_ids)
    
    return {"message": f"Batch deactivation started for {len(user_ids)} users"}
```

## 🛠️ 개발 및 디버깅 팁

### 1. 로깅 설정

각 컨트롤러에 적절한 로깅을 추가합니다:

```python
import logging

logger = logging.getLogger("controller-name")

@router.post("/endpoint")
async def endpoint_function(request: Request):
    logger.info(f"Request received: {request.method} {request.url}")
    
    try:
        # 비즈니스 로직
        result = await service.do_something()
        logger.info(f"Operation completed successfully: {result}")
        return result
    except Exception as e:
        logger.error(f"Operation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")
```

### 2. 요청/응답 검증

Pydantic 모델을 사용하여 요청과 응답을 검증합니다:

```python
from pydantic import BaseModel, validator
from typing import Optional

class UserCreateRequest(BaseModel):
    username: str
    email: str
    password: str
    
    @validator('username')
    def validate_username(cls, v):
        if len(v) < 3:
            raise ValueError('Username must be at least 3 characters long')
        return v.lower()
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v

class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    created_at: datetime
    
    class Config:
        from_attributes = True  # Pydantic v2
```

### 3. 에러 처리

표준화된 에러 처리 패턴을 사용합니다:

```python
from fastapi import HTTPException
from typing import Dict, Any

class APIError(Exception):
    def __init__(self, status_code: int, detail: str, error_code: str = None):
        self.status_code = status_code
        self.detail = detail
        self.error_code = error_code

def handle_service_error(error: Exception) -> HTTPException:
    """서비스 에러를 HTTP 에러로 변환"""
    if isinstance(error, ValueError):
        return HTTPException(status_code=400, detail=str(error))
    elif isinstance(error, PermissionError):
        return HTTPException(status_code=403, detail="Permission denied")
    elif isinstance(error, FileNotFoundError):
        return HTTPException(status_code=404, detail="Resource not found")
    else:
        logger.error(f"Unexpected error: {error}", exc_info=True)
        return HTTPException(status_code=500, detail="Internal Server Error")

@router.post("/endpoint")
async def endpoint_function(request: Request):
    try:
        # 비즈니스 로직
        pass
    except Exception as e:
        raise handle_service_error(e)
```

### 4. API 문서화

OpenAPI 문서를 위한 메타데이터를 추가합니다:

```python
@router.post(
    "/",
    response_model=UserResponse,
    status_code=201,
    summary="사용자 생성",
    description="새로운 사용자 계정을 생성합니다.",
    responses={
        201: {"description": "사용자 생성 성공"},
        400: {"description": "잘못된 요청 데이터"},
        409: {"description": "이미 존재하는 사용자"},
        500: {"description": "서버 내부 오류"}
    },
    tags=["users"]
)
async def create_user(request: Request, user_data: UserCreateRequest):
    """새로운 사용자 생성"""
    pass
```

### 5. 성능 최적화

```python
from fastapi import BackgroundTasks
from asyncio import gather

@router.get("/dashboard")
async def get_dashboard_data(request: Request):
    """대시보드 데이터 조회 (병렬 처리)"""
    
    # 여러 서비스 호출을 병렬로 처리
    user_service = get_user_service(request)
    
    user_count_task = user_service.count_users()
    active_users_task = user_service.count_active_users()
    recent_users_task = user_service.get_recent_users(limit=10)
    
    user_count, active_users, recent_users = await gather(
        user_count_task,
        active_users_task,
        recent_users_task
    )
    
    return {
        "user_count": user_count,
        "active_users": active_users,
        "recent_users": recent_users
    }
```

## 🔧 문제 해결 가이드

### 자주 발생하는 오류

#### 1. **의존성 주입 오류**
```python
# ❌ 잘못된 예시
def get_service(request: Request):
    return request.app.state.service  # AttributeError 발생 가능

# ✅ 올바른 예시
def get_service(request: Request):
    if hasattr(request.app.state, 'service') and request.app.state.service:
        return request.app.state.service
    else:
        raise HTTPException(status_code=500, detail="Service not available")
```

#### 2. **Pydantic 모델 검증 오류**
```python
# ❌ 잘못된 예시
class UserRequest(BaseModel):
    email: str  # 이메일 형식 검증 없음

# ✅ 올바른 예시
from pydantic import EmailStr

class UserRequest(BaseModel):
    email: EmailStr  # 이메일 형식 자동 검증
```

#### 3. **에러 응답 일관성 부족**
```python
# ❌ 잘못된 예시
@router.get("/user/{user_id}")
async def get_user(user_id: str):
    user = await service.get_user(user_id)
    if not user:
        return {"error": "User not found"}  # 일관성 없는 응답

# ✅ 올바른 예시
@router.get("/user/{user_id}")
async def get_user(user_id: str):
    user = await service.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
```

#### 4. **비동기 처리 실수**
```python
# ❌ 잘못된 예시
@router.post("/process")
async def process_data(data: dict):
    result = service.process_data(data)  # await 누락
    return result

# ✅ 올바른 예시
@router.post("/process")
async def process_data(data: dict):
    result = await service.process_data(data)  # await 사용
    return result
```

### 디버깅 체크리스트

1. **라우터 등록 확인**
   ```python
   # main.py에서 라우터가 등록되었는지 확인
   app.include_router(your_router)
   ```

2. **의존성 주입 확인**
   ```python
   # app.state에 필요한 서비스가 있는지 확인
   print(hasattr(request.app.state, 'service'))
   ```

3. **로그 확인**
   ```python
   # 적절한 로그 레벨로 설정
   logging.basicConfig(level=logging.DEBUG)
   ```

4. **모델 검증 확인**
   ```python
   # Pydantic 모델이 올바르게 정의되었는지 확인
   model = UserRequest.parse_obj(data)
   ```

---

## 📚 참고 자료

### FastAPI 공식 문서
- **[FastAPI 공식 문서](https://fastapi.tiangolo.com/)**
- **[Pydantic 공식 문서](https://pydantic-docs.helpmanual.io/)**
- **[Starlette 공식 문서](https://www.starlette.io/)**

### 관련 파일들
- **[main.py](../main.py)**: 애플리케이션 메인 파일
- **[service/](../service/)**: 서비스 계층 파일들
- **[config/](../config/)**: 설정 관리 파일들

### 코딩 스타일 가이드
- **[PEP 8](https://peps.python.org/pep-0008/)**: Python 코딩 스타일 가이드
- **[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)**: 구글 파이썬 스타일 가이드

---

**PlateERAG Backend Controller System**  
*🚀 FastAPI 기반 • 🔧 타입 안전 • 🗄️ 의존성 주입 • 🌍 RESTful API*

**이제 컨트롤러를 자유롭게 확장하고 관리할 수 있습니다!**

# MCP Controller - API 문서

MCP Station과 통신하기 위한 게이트웨이 컨트롤러입니다.

## 개요

- **파일**: `/plateerag_backend/controller/mcpController.py`
- **라우터 Prefix**: `/api/mcp`
- **MCP Station URL**: `http://mcp_station:20100` (Docker Compose 네트워크 내부)

## API Endpoints

### 1. 헬스체크

#### `GET /api/mcp/health`
MCP Station 서비스의 상태를 확인합니다.

**응답 예시**:
```json
{
  "service": "MCP Station",
  "status": "running",
  "sessions_count": 3
}
```

#### `GET /api/mcp/health/detailed`
MCP Station의 상세한 상태 정보를 조회합니다.

**응답 예시**:
```json
{
  "status": "healthy",
  "total_sessions": 5,
  "running_sessions": 3,
  "error_sessions": 0
}
```

---

### 2. 세션 관리

#### `POST /api/mcp/sessions`
새로운 MCP 서버 세션을 생성합니다.

**요청 Body**:
```json
{
  "server_type": "python",
  "server_command": "/app/server.py",
  "server_args": ["--verbose"],
  "env_vars": {
    "API_KEY": "your-api-key"
  },
  "working_dir": "/app"
}
```

**응답 (201 Created)**:
```json
{
  "session_id": "abc123",
  "server_type": "python",
  "status": "running",
  "created_at": "2025-09-30T10:00:00",
  "pid": 12345,
  "error_message": null
}
```

#### `GET /api/mcp/sessions`
모든 활성 MCP 세션 목록을 조회합니다.

**응답**:
```json
[
  {
    "session_id": "abc123",
    "server_type": "python",
    "status": "running",
    "created_at": "2025-09-30T10:00:00",
    "pid": 12345,
    "error_message": null
  },
  {
    "session_id": "def456",
    "server_type": "node",
    "status": "running",
    "created_at": "2025-09-30T11:00:00",
    "pid": 12346,
    "error_message": null
  }
]
```

#### `GET /api/mcp/sessions/{session_id}`
특정 세션의 상세 정보를 조회합니다.

**파라미터**:
- `session_id`: 조회할 세션의 ID

**응답**:
```json
{
  "session_id": "abc123",
  "server_type": "python",
  "status": "running",
  "created_at": "2025-09-30T10:00:00",
  "pid": 12345,
  "error_message": null
}
```

#### `DELETE /api/mcp/sessions/{session_id}`
지정된 세션을 삭제하고 관련 프로세스를 종료합니다.

**파라미터**:
- `session_id`: 삭제할 세션의 ID

**응답 (204 No Content)**: 빈 응답

---

### 3. 도구 및 요청

#### `GET /api/mcp/sessions/{session_id}/tools`
특정 세션에서 사용 가능한 MCP 도구 목록을 조회합니다.

**파라미터**:
- `session_id`: 도구 목록을 조회할 세션의 ID

**응답 예시**:
```json
{
  "tools": [
    {
      "name": "read_file",
      "description": "파일을 읽습니다",
      "inputSchema": {
        "type": "object",
        "properties": {
          "path": {
            "type": "string",
            "description": "파일 경로"
          }
        },
        "required": ["path"]
      }
    }
  ]
}
```

#### `POST /api/mcp/request`
MCP 서버로 요청을 라우팅합니다.

**요청 Body**:
```json
{
  "session_id": "abc123",
  "method": "tools/call",
  "params": {
    "name": "read_file",
    "arguments": {
      "path": "/app/data.txt"
    }
  }
}
```

**지원 메서드**:
- `tools/list`: 사용 가능한 도구 목록 조회
- `tools/call`: 특정 도구 호출
- `prompts/list`: 사용 가능한 프롬프트 목록 조회
- `prompts/get`: 특정 프롬프트 가져오기

**응답**:
```json
{
  "success": true,
  "data": {
    "result": "파일 내용..."
  },
  "error": null
}
```

**실패 응답**:
```json
{
  "success": false,
  "data": null,
  "error": "Tool not found"
}
```

---

## 에러 처리

### HTTP 상태 코드

- `200 OK`: 요청 성공
- `201 Created`: 세션 생성 성공
- `204 No Content`: 세션 삭제 성공
- `404 Not Found`: 세션을 찾을 수 없음
- `500 Internal Server Error`: 서버 내부 오류
- `503 Service Unavailable`: MCP Station에 연결할 수 없음

### 에러 응답 형식

```json
{
  "detail": "에러 메시지"
}
```

---

## 사용 예시

### Python 예시

```python
import httpx

# 세션 생성
async def create_mcp_session():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/mcp/sessions",
            json={
                "server_type": "python",
                "server_command": "/app/mcp_server.py",
                "env_vars": {"DEBUG": "true"}
            }
        )
        session = response.json()
        return session["session_id"]

# 도구 목록 조회
async def get_tools(session_id: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"http://localhost:8000/api/mcp/sessions/{session_id}/tools"
        )
        return response.json()

# 도구 호출
async def call_tool(session_id: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/mcp/request",
            json={
                "session_id": session_id,
                "method": "tools/call",
                "params": {
                    "name": "read_file",
                    "arguments": {"path": "/data.txt"}
                }
            }
        )
        return response.json()
```

### JavaScript/TypeScript 예시

```typescript
// 세션 생성
async function createMCPSession() {
  const response = await fetch('http://localhost:8000/api/mcp/sessions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      server_type: 'node',
      server_command: '/app/server.js',
      env_vars: { DEBUG: 'true' }
    })
  });
  const session = await response.json();
  return session.session_id;
}

// 도구 호출
async function callTool(sessionId: string) {
  const response = await fetch('http://localhost:8000/api/mcp/request', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      method: 'tools/call',
      params: {
        name: 'read_file',
        arguments: { path: '/data.txt' }
      }
    })
  });
  return await response.json();
}
```

---

## 로깅

모든 API 호출은 로깅 시스템을 통해 기록됩니다:

- **성공 로그**: `backend_log.success()`
- **정보 로그**: `backend_log.info()`
- **경고 로그**: `backend_log.warning()`
- **에러 로그**: `backend_log.error()`

각 로그에는 요청한 사용자 ID, 세션 ID, 메서드 등의 메타데이터가 포함됩니다.

---

## 설정

### Docker Compose 네트워크

MCP Controller는 Docker Compose 네트워크 내부에서 `mcp_station` 서비스와 통신합니다:

```yaml
services:
  mcp_station:
    build: ./mcp_station/.
    container_name: mcp_station
    restart: unless-stopped
    ports:
      - 20100:20100

  polarag_backend:
    container_name: prg_backend
    # MCP Station과 같은 네트워크에 위치
```

백엔드는 `http://mcp_station:20100`으로 MCP Station에 접근합니다.

---

## 주의사항

1. **타임아웃**: 모든 HTTP 요청은 30초 타임아웃이 설정되어 있습니다.
2. **에러 처리**: MCP Station의 에러는 적절한 HTTP 상태 코드와 함께 프론트엔드로 전달됩니다.
3. **인증**: 현재 인증은 백엔드의 기존 인증 시스템을 사용합니다 (`extract_user_id_from_request`).
4. **로깅**: 모든 요청/응답은 데이터베이스 로거를 통해 기록됩니다.

---

## 향후 개선 사항

- [ ] 세션 타임아웃 자동 관리
- [ ] MCP Station 연결 풀링
- [ ] 재시도 로직 추가
- [ ] WebSocket 지원 (실시간 이벤트)
- [ ] 세션별 권한 관리

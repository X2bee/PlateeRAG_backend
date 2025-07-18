# PlateERAG Backend Node Editor 시스템 가이드

## 📖 개요

PlateERAG Backend의 Node Editor는 **시각적 워크플로우 편집기**를 위한 백엔드 시스템입니다. 각 노드는 특정 기능을 수행하는 독립적인 실행 단위이며, 이들을 연결하여 복잡한 데이터 처리 워크플로우를 구성할 수 있습니다.

### 🎯 핵심 특징
- **노드 기반 아키텍처**: 각 노드는 하나의 기능을 담당하는 독립적인 실행 단위
- **자동 노드 발견**: `nodes/` 폴더의 노드들을 자동으로 탐지하고 등록
- **타입 시스템**: 강력한 타입 검증 및 포트 시스템
- **워크플로우 실행**: 위상 정렬을 통한 효율적인 노드 실행
- **확장성**: 새로운 노드를 쉽게 추가할 수 있는 플러그인 아키텍처
- **카테고리 시스템**: 노드를 기능별로 분류하여 관리
- **매개변수 검증**: 런타임 전에 노드 매개변수 유효성 검증
- **JSON 스펙 생성**: 프론트엔드를 위한 노드 스펙 자동 생성

## 🏗️ Node Editor 아키텍처

### 폴더 구조
```
editor/
├── README.md                    # 📖 이 문서
├── __init__.py                  # 🔧 에디터 패키지 초기화
├── model/                       # 📋 데이터 모델
│   ├── __init__.py
│   └── node.py                  # 🔗 노드 타입 정의 및 검증
├── node_composer.py             # 🎼 노드 탐지 및 등록 시스템
├── workflow_executor.py         # 🔄 워크플로우 실행 엔진
└── nodes/                       # 📂 노드 구현 폴더
    ├── __init__.py
    ├── chat/                    # 💬 채팅 모델 노드
    │   ├── __init__.py
    │   └── chat_openai.py       # 🤖 OpenAI 채팅 노드
    ├── math/                    # 🔢 수학 연산 노드
    │   ├── __init__.py
    │   ├── math_add.py          # ➕ 덧셈 노드
    │   ├── math_multiply.py     # ✖️ 곱셈 노드
    │   └── math_subtract.py     # ➖ 뺄셈 노드
    └── tool/                    # 🔧 유틸리티 노드
        ├── __init__.py
        ├── input_int.py         # 📥 정수 입력 노드
        ├── input_str.py         # 📥 문자열 입력 노드
        ├── print_any.py         # 📤 출력 노드
        └── test_validation.py   # 🧪 검증 테스트 노드
```

### 아키텍처 구성요소


## 🗂️ 카테고리 및 기능 시스템

### 📋 카테고리 목록
현재 지원하는 카테고리들:

| 카테고리 ID | 카테고리 이름 | 아이콘 | 설명 |
|-------------|---------------|--------|------|
| `langchain` | LangChain | `SiLangchain` | LangChain 라이브러리 기반 노드 |
| `polar` | POLAR | `POLAR` | POLAR 시스템 전용 노드 |
| `utilities` | Utilities | `LuWrench` | 유틸리티 및 도구 노드 |
| `math` | Math | `LuWrench` | 수학 연산 노드 |

### 🔧 기능 목록
각 카테고리 내에서 사용할 수 있는 기능들:

| 기능 ID | 기능 이름 | 설명 |
|---------|-----------|------|
| `agents` | Agent | LangChain 에이전트 |
| `cache` | Cache | 캐싱 시스템 |
| `chain` | Chain | LangChain 체인 |
| `chat_models` | Chat Model | 채팅 모델 |
| `document_loaders` | Document Loader | 문서 로더 |
| `embeddings` | Embedding | 임베딩 모델 |
| `graph` | Graph | 그래프 처리 |
| `memory` | Memory | 메모리 시스템 |
| `moderation` | Moderation | 콘텐츠 검열 |
| `output_parsers` | Output Parser | 출력 파서 |
| `tools` | Tool | 도구 노드 |
| `arithmetic` | Arithmetic | 수학 연산 |
| `endnode` | End Node | 종료 노드 |
| `startnode` | Start Node | 시작 노드 |

## 🎯 노드 타입 시스템

### 🔌 포트 타입
노드 간 데이터 전송을 위한 포트 타입들:

| 타입 | 설명 | 예시 |
|------|------|------|
| `INT` | 정수 | `42`, `-10`, `0` |
| `STR` | 문자열 | `"Hello"`, `"World"` |
| `FLOAT` | 부동소수점 | `3.14`, `-0.5`, `1.0` |
| `BOOL` | 불린 | `true`, `false` |
| `ANY` | 모든 타입 | 모든 데이터 타입 허용 |

### 📊 매개변수 타입
노드 설정을 위한 매개변수 타입들:

| 타입 | 설명 | 추가 속성 |
|------|------|-----------|
| `STRING` | 문자열 매개변수 | - |
| `INTEGER` | 정수 매개변수 | `min`, `max`, `step` |
| `FLOAT` | 부동소수점 매개변수 | `min`, `max`, `step` |
| `BOOLEAN` | 불린 매개변수 | - |

### 🎚️ 매개변수 고급 설정
```python
parameters = [
    {
        "id": "temperature",
        "name": "Temperature",
        "type": "FLOAT",
        "value": 0.7,
        "required": False,
        "optional": True,      # 고급 모드에서만 표시
        "min": 0.0,
        "max": 2.0,
        "step": 0.1
    },
    {
        "id": "model",
        "name": "Model",
        "type": "STRING",
        "value": "gpt-3.5-turbo",
        "required": True,
        "options": [           # 드롭다운 옵션
            {"value": "gpt-3.5-turbo", "label": "GPT-3.5 Turbo"},
            {"value": "gpt-4", "label": "GPT-4"}
        ]
    }
]
```

## 🚀 새로운 노드 추가하기 (Step-by-Step)

### 🎯 Step 1: 노드 파일 생성

새로운 노드를 `nodes/` 폴더의 적절한 카테고리에 생성합니다.

#### 예시: 간단한 문자열 연결 노드

**파일**: `editor/nodes/tool/string_concat.py`

```python
"""
문자열 연결 노드

두 개의 문자열을 입력받아 연결한 결과를 반환합니다.
"""

from editor.node_composer import Node

class StringConcatNode(Node):
    # 카테고리 및 기능 정의
    categoryId = "utilities"        # 반드시 CATEGORIES_LABEL_MAP에 존재해야 함
    functionId = "tools"           # 반드시 FUNCTION_LABEL_MAP에 존재해야 함
    
    # 노드 기본 정보
    nodeId = "tool/string_concat"   # 고유 식별자 (카테고리/노드명 형식 권장)
    nodeName = "String Concat"      # 사용자에게 표시될 이름
    description = "두 개의 문자열을 입력받아 연결한 결과를 반환합니다. 구분자를 설정할 수 있습니다."
    tags = ["string", "concatenation", "text", "join", "utility"]  # 검색 태그
    
    # 입력 포트 정의
    inputs = [
        {
            "id": "str1",           # 포트 고유 식별자
            "name": "String 1",     # 포트 표시 이름
            "type": "STR",          # 데이터 타입
            "required": True,       # 필수 입력
            "multi": False          # 다중 연결 비허용
        },
        {
            "id": "str2",
            "name": "String 2",
            "type": "STR",
            "required": True,
            "multi": False
        }
    ]
    
    # 출력 포트 정의
    outputs = [
        {
            "id": "result",
            "name": "Result",
            "type": "STR"
        }
    ]
    
    # 매개변수 정의
    parameters = [
        {
            "id": "separator",
            "name": "Separator",
            "type": "STRING",
            "value": " ",          # 기본값: 공백
            "required": False,     # 필수 아님
            "optional": False      # 기본 모드에서 표시
        }
    ]
    
    def execute(self, str1: str, str2: str, separator: str = " ") -> str:
        """
        노드 실행 메서드
        
        Args:
            str1: 첫 번째 문자열
            str2: 두 번째 문자열
            separator: 구분자 (기본값: 공백)
            
        Returns:
            str: 연결된 문자열
        """
        return f"{str1}{separator}{str2}"
```

### 🔧 Step 2: 노드 등록 확인

노드가 정의되면 자동으로 등록됩니다. 등록 성공 시 다음과 같은 메시지가 출력됩니다:

```
-> 노드 'String Concat' 등록 완료.
```

등록 실패 시 다음과 같은 오류가 출력됩니다:

```
[Node Registration Failed] Node 'StringConcatNode': 'categoryId' is invalid.
-> Assigned value: 'invalid_category' (Allowed values: ['langchain', 'polar', 'utilities', 'math'])
```

### 🧪 Step 3: 노드 테스트

새로운 노드를 테스트하는 방법:

```python
# test_string_concat.py
from editor.nodes.tool.string_concat import StringConcatNode

def test_string_concat():
    node = StringConcatNode()
    
    # 기본 구분자 테스트
    result = node.execute("Hello", "World")
    assert result == "Hello World"
    
    # 사용자 정의 구분자 테스트
    result = node.execute("Hello", "World", separator="-")
    assert result == "Hello-World"
    
    print("✅ String Concat Node 테스트 통과")

if __name__ == "__main__":
    test_string_concat()
```

### 🎨 Step 4: 고급 노드 패턴

#### 1. **상태를 가진 노드**
```python
class CounterNode(Node):
    categoryId = "utilities"
    functionId = "tools"
    nodeId = "tool/counter"
    nodeName = "Counter"
    description = "호출될 때마다 카운트를 증가시키는 노드입니다."
    tags = ["counter", "state", "increment"]
    
    inputs = [
        {
            "id": "trigger",
            "name": "Trigger",
            "type": "ANY",
            "required": True,
            "multi": False
        }
    ]
    
    outputs = [
        {
            "id": "count",
            "name": "Count",
            "type": "INT"
        }
    ]
    
    parameters = [
        {
            "id": "start_value",
            "name": "Start Value",
            "type": "INTEGER",
            "value": 0,
            "required": False
        }
    ]
    
    def __init__(self):
        super().__init__()
        self.count = 0
    
    def execute(self, trigger: any, start_value: int = 0) -> int:
        if self.count == 0:
            self.count = start_value
        self.count += 1
        return self.count
```

#### 2. **조건부 실행 노드**
```python
class ConditionalNode(Node):
    categoryId = "utilities"
    functionId = "tools"
    nodeId = "tool/conditional"
    nodeName = "Conditional"
    description = "조건에 따라 다른 값을 반환하는 노드입니다."
    tags = ["conditional", "if", "logic", "branch"]
    
    inputs = [
        {
            "id": "condition",
            "name": "Condition",
            "type": "BOOL",
            "required": True,
            "multi": False
        },
        {
            "id": "true_value",
            "name": "True Value",
            "type": "ANY",
            "required": True,
            "multi": False
        },
        {
            "id": "false_value",
            "name": "False Value",
            "type": "ANY",
            "required": True,
            "multi": False
        }
    ]
    
    outputs = [
        {
            "id": "result",
            "name": "Result",
            "type": "ANY"
        }
    ]
    
    parameters = []
    
    def execute(self, condition: bool, true_value: any, false_value: any) -> any:
        return true_value if condition else false_value
```

#### 3. **배치 처리 노드**
```python
from typing import List

class BatchProcessNode(Node):
    categoryId = "utilities"
    functionId = "tools"
    nodeId = "tool/batch_process"
    nodeName = "Batch Process"
    description = "여러 입력을 배치로 처리하는 노드입니다."
    tags = ["batch", "process", "list", "multiple"]
    
    inputs = [
        {
            "id": "items",
            "name": "Items",
            "type": "ANY",
            "required": True,
            "multi": True  # 다중 입력 허용
        }
    ]
    
    outputs = [
        {
            "id": "results",
            "name": "Results",
            "type": "ANY"
        }
    ]
    
    parameters = [
        {
            "id": "batch_size",
            "name": "Batch Size",
            "type": "INTEGER",
            "value": 10,
            "required": False,
            "min": 1,
            "max": 100
        }
    ]
    
    def execute(self, items: List[any], batch_size: int = 10) -> List[any]:
        results = []
        
        # 배치 단위로 처리
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            # 배치 처리 로직
            processed_batch = [self.process_item(item) for item in batch]
            results.extend(processed_batch)
        
        return results
    
    def process_item(self, item: any) -> any:
        """개별 아이템 처리 로직"""
        return item  # 실제 처리 로직 구현
```

#### 4. **외부 API 호출 노드**
```python
import requests
from typing import Dict, Any

class HttpRequestNode(Node):
    categoryId = "utilities"
    functionId = "tools"
    nodeId = "tool/http_request"
    nodeName = "HTTP Request"
    description = "HTTP 요청을 보내고 응답을 받는 노드입니다."
    tags = ["http", "request", "api", "web", "rest"]
    
    inputs = [
        {
            "id": "url",
            "name": "URL",
            "type": "STR",
            "required": True,
            "multi": False
        },
        {
            "id": "data",
            "name": "Request Data",
            "type": "ANY",
            "required": False,
            "multi": False
        }
    ]
    
    outputs = [
        {
            "id": "response",
            "name": "Response",
            "type": "ANY"
        },
        {
            "id": "status_code",
            "name": "Status Code",
            "type": "INT"
        }
    ]
    
    parameters = [
        {
            "id": "method",
            "name": "Method",
            "type": "STRING",
            "value": "GET",
            "required": True,
            "options": [
                {"value": "GET", "label": "GET"},
                {"value": "POST", "label": "POST"},
                {"value": "PUT", "label": "PUT"},
                {"value": "DELETE", "label": "DELETE"}
            ]
        },
        {
            "id": "timeout",
            "name": "Timeout",
            "type": "INTEGER",
            "value": 30,
            "required": False,
            "optional": True,
            "min": 1,
            "max": 300
        }
    ]
    
    def execute(self, url: str, data: any = None, method: str = "GET", timeout: int = 30) -> Dict[str, Any]:
        try:
            response = requests.request(
                method=method,
                url=url,
                json=data if data else None,
                timeout=timeout
            )
            
            return {
                "response": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text,
                "status_code": response.status_code
            }
        except Exception as e:
            return {
                "response": {"error": str(e)},
                "status_code": -1
            }
```

#### 5. **데이터 변환 노드**
```python
import json
from typing import Dict, Any

class DataTransformNode(Node):
    categoryId = "utilities"
    functionId = "tools"
    nodeId = "tool/data_transform"
    nodeName = "Data Transform"
    description = "데이터를 다른 형식으로 변환하는 노드입니다."
    tags = ["transform", "convert", "data", "format"]
    
    inputs = [
        {
            "id": "data",
            "name": "Input Data",
            "type": "ANY",
            "required": True,
            "multi": False
        }
    ]
    
    outputs = [
        {
            "id": "transformed_data",
            "name": "Transformed Data",
            "type": "ANY"
        }
    ]
    
    parameters = [
        {
            "id": "transform_type",
            "name": "Transform Type",
            "type": "STRING",
            "value": "to_json",
            "required": True,
            "options": [
                {"value": "to_json", "label": "To JSON"},
                {"value": "to_string", "label": "To String"},
                {"value": "to_upper", "label": "To Uppercase"},
                {"value": "to_lower", "label": "To Lowercase"}
            ]
        }
    ]
    
    def execute(self, data: any, transform_type: str = "to_json") -> any:
        try:
            if transform_type == "to_json":
                return json.dumps(data, ensure_ascii=False, indent=2)
            elif transform_type == "to_string":
                return str(data)
            elif transform_type == "to_upper":
                return str(data).upper()
            elif transform_type == "to_lower":
                return str(data).lower()
            else:
                return data
        except Exception as e:
            return {"error": str(e)}
```

### 🔄 Step 5: 노드 테스트 및 검증

#### 1. **단위 테스트 작성**
```python
# test_nodes.py
import pytest
from editor.nodes.tool.string_concat import StringConcatNode
from editor.nodes.tool.conditional import ConditionalNode

class TestStringConcatNode:
    def setup_method(self):
        self.node = StringConcatNode()
    
    def test_basic_concat(self):
        result = self.node.execute("Hello", "World")
        assert result == "Hello World"
    
    def test_custom_separator(self):
        result = self.node.execute("Hello", "World", separator="-")
        assert result == "Hello-World"
    
    def test_empty_strings(self):
        result = self.node.execute("", "")
        assert result == " "

class TestConditionalNode:
    def setup_method(self):
        self.node = ConditionalNode()
    
    def test_true_condition(self):
        result = self.node.execute(True, "yes", "no")
        assert result == "yes"
    
    def test_false_condition(self):
        result = self.node.execute(False, "yes", "no")
        assert result == "no"
```

#### 2. **노드 검증 도구**
```python
# node_validator.py
from editor.node_composer import get_node_registry, run_discovery

def validate_all_nodes():
    """모든 노드의 유효성을 검증합니다."""
    run_discovery()
    registry = get_node_registry()
    
    print(f"총 {len(registry)}개의 노드가 등록되었습니다.")
    
    # 카테고리별 분류
    categories = {}
    for node in registry:
        cat_id = node['categoryId']
        if cat_id not in categories:
            categories[cat_id] = []
        categories[cat_id].append(node)
    
    # 검증 결과 출력
    for cat_id, nodes in categories.items():
        print(f"\n📂 {cat_id}: {len(nodes)}개 노드")
        for node in nodes:
            print(f"  - {node['nodeName']} ({node['id']})")
            
            # 입력/출력 검증
            if not node['inputs'] and not node['outputs']:
                print(f"    ⚠️  입력/출력이 모두 없습니다.")
            
            # 매개변수 검증
            for param in node['parameters']:
                if param.get('required') and param.get('optional'):
                    print(f"    ❌ 매개변수 '{param['id']}': required와 optional이 모두 True입니다.")

if __name__ == "__main__":
    validate_all_nodes()
```

## 🏗️ 새로운 카테고리 추가하기

### 🎯 Step 1: 카테고리 정의

`editor/model/node.py`에서 새로운 카테고리를 추가합니다:

```python
CATEGORIES_LABEL_MAP = {
    'langchain': 'LangChain',
    'polar': 'POLAR',
    'utilities': 'Utilities',
    'math': 'Math',
    'database': 'Database',    # 새로운 카테고리 추가
    'network': 'Network',      # 새로운 카테고리 추가
    # ...
}

ICON_LABEL_MAP = {
    'langchain': 'SiLangchain',
    'polar': 'POLAR',
    'utilities': 'LuWrench',
    'math': 'LuWrench',
    'database': 'BiDatabase',   # 새로운 아이콘 추가
    'network': 'BiNetwork',     # 새로운 아이콘 추가
    # ...
}
```

### 🔧 Step 2: 기능 정의

필요한 경우 새로운 기능도 추가합니다:

```python
FUNCTION_LABEL_MAP = {
    # ...기존 기능들...
    'sql': 'SQL',
    'nosql': 'NoSQL',
    'crud': 'CRUD',
    'http': 'HTTP',
    'websocket': 'WebSocket',
    'tcp': 'TCP',
    # ...
}
```

### 📁 Step 3: 폴더 구조 생성

새로운 카테고리를 위한 폴더를 생성합니다:

```
editor/nodes/
├── database/
│   ├── __init__.py
│   ├── mysql_query.py
│   ├── mongodb_find.py
│   └── redis_get.py
└── network/
    ├── __init__.py
    ├── http_get.py
    ├── websocket_send.py
    └── tcp_connect.py
```

### 🎨 Step 4: 카테고리별 노드 예시

#### 데이터베이스 노드 예시
```python
# editor/nodes/database/mysql_query.py
import mysql.connector
from editor.node_composer import Node

class MySQLQueryNode(Node):
    categoryId = "database"
    functionId = "sql"
    nodeId = "database/mysql_query"
    nodeName = "MySQL Query"
    description = "MySQL 데이터베이스에 쿼리를 실행하는 노드입니다."
    tags = ["mysql", "database", "sql", "query"]
    
    inputs = [
        {
            "id": "query",
            "name": "SQL Query",
            "type": "STR",
            "required": True,
            "multi": False
        }
    ]
    
    outputs = [
        {
            "id": "result",
            "name": "Query Result",
            "type": "ANY"
        }
    ]
    
    parameters = [
        {
            "id": "host",
            "name": "Host",
            "type": "STRING",
            "value": "localhost",
            "required": True
        },
        {
            "id": "port",
            "name": "Port",
            "type": "INTEGER",
            "value": 3306,
            "required": True
        },
        {
            "id": "database",
            "name": "Database",
            "type": "STRING",
            "value": "",
            "required": True
        },
        {
            "id": "username",
            "name": "Username",
            "type": "STRING",
            "value": "",
            "required": True
        },
        {
            "id": "password",
            "name": "Password",
            "type": "STRING",
            "value": "",
            "required": True
        }
    ]
    
    def execute(self, query: str, host: str, port: int, database: str, username: str, password: str):
        try:
            connection = mysql.connector.connect(
                host=host,
                port=port,
                database=database,
                user=username,
                password=password
            )
            
            cursor = connection.cursor()
            cursor.execute(query)
            
            if query.strip().upper().startswith('SELECT'):
                result = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                return {
                    "data": result,
                    "columns": columns
                }
            else:
                connection.commit()
                return {
                    "affected_rows": cursor.rowcount,
                    "message": "Query executed successfully"
                }
                
        except Exception as e:
            return {"error": str(e)}
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
```

#### 네트워크 노드 예시
```python
# editor/nodes/network/http_get.py
import requests
from editor.node_composer import Node

class HttpGetNode(Node):
    categoryId = "network"
    functionId = "http"
    nodeId = "network/http_get"
    nodeName = "HTTP GET"
    description = "HTTP GET 요청을 보내는 노드입니다."
    tags = ["http", "get", "request", "api", "web"]
    
    inputs = [
        {
            "id": "url",
            "name": "URL",
            "type": "STR",
            "required": True,
            "multi": False
        }
    ]
    
    outputs = [
        {
            "id": "response",
            "name": "Response",
            "type": "ANY"
        },
        {
            "id": "status_code",
            "name": "Status Code",
            "type": "INT"
        }
    ]
    
    parameters = [
        {
            "id": "timeout",
            "name": "Timeout",
            "type": "INTEGER",
            "value": 30,
            "required": False,
            "min": 1,
            "max": 300
        },
        {
            "id": "headers",
            "name": "Headers",
            "type": "STRING",
            "value": "{}",
            "required": False,
            "optional": True
        }
    ]
    
    def execute(self, url: str, timeout: int = 30, headers: str = "{}"):
        try:
            import json
            headers_dict = json.loads(headers) if headers else {}
            
            response = requests.get(url, timeout=timeout, headers=headers_dict)
            
            try:
                response_data = response.json()
            except:
                response_data = response.text
            
            return {
                "response": response_data,
                "status_code": response.status_code
            }
        except Exception as e:
            return {
                "response": {"error": str(e)},
                "status_code": -1
            }
```

## 🔄 워크플로우 구조 및 특수 노드

### 📋 워크플로우 필수 구조
모든 워크플로우는 **반드시** 다음과 같은 구조를 가져야 합니다:

```
사용자 입력 (Interaction)
         ↓
    [Start Node]
         ↓
   [중간 처리 노드들]
         ↓
     [End Node]
         ↓
    사용자 출력 (Result)
```

### 🎯 특수 노드: Start Node와 End Node

#### 🚀 Start Node (시작 노드)
Start Node는 워크플로우의 **진입점**이며, 사용자의 Interaction을 받아들이는 특수한 노드입니다.

##### **특징**
- **단일성**: 워크플로우당 반드시 하나만 존재해야 함
- **입력 없음**: 외부 포트로부터 입력을 받지 않음
- **Interaction 연결**: 사용자의 입력(Interaction)을 직접 받음
- **고정 기능**: `functionId = "startnode"`로 고정됨

##### **Start Node 예시**
```python
class InputStringNode(Node):
    categoryId = "utilities"
    functionId = "startnode"        # 고정값: 시작 노드
    nodeId = "tool/input_str"
    nodeName = "Input String"
    description = "사용자가 설정한 문자열 값을 출력하는 입력 노드입니다. 워크플로우에서 텍스트 데이터의 시작점으로 사용됩니다."
    tags = ["input", "string", "text", "parameter", "source", "start_node", "user_input"]
    
    inputs = []  # 입력 포트 없음
    outputs = [
        {
            "id": "result",
            "name": "Result",
            "type": "STR"
        }
    ]
    parameters = [
        {
            "id": "input_str",
            "name": "String",
            "type": "STRING",
            "value": "",
            "required": True
        }
    ]
    
    def execute(self, input_str: str) -> str:
        """사용자 입력을 그대로 출력"""
        return input_str
```

#### 🏁 End Node (종료 노드)
End Node는 워크플로우의 **출구점**이며, 최종 결과를 사용자에게 반환하는 특수한 노드입니다.

##### **특징**
- **단일성**: 워크플로우당 반드시 하나만 존재해야 함
- **출력 없음**: 다른 노드로 출력을 전달하지 않음
- **결과 반환**: 워크플로우의 최종 결과를 사용자에게 반환
- **고정 기능**: `functionId = "endnode"`로 고정됨

##### **End Node 예시**
```python
class OutputStringNode(Node):
    categoryId = "utilities"
    functionId = "endnode"          # 고정값: 종료 노드
    nodeId = "tool/output_str"
    nodeName = "Output String"
    description = "입력받은 값을 최종 결과로 출력하는 종료 노드입니다. 워크플로우의 최종 출력점으로 사용됩니다."
    tags = ["output", "string", "text", "result", "end_node", "final_output"]
    
    inputs = [
        {
            "id": "input",
            "name": "Input",
            "type": "ANY",
            "required": True,
            "multi": False
        }
    ]
    outputs = []  # 출력 포트 없음
    parameters = []
    
    def execute(self, input: any) -> any:
        """입력을 최종 결과로 반환"""
        return input
```

### 🔄 워크플로우 실행 흐름

#### 1. **Interaction 입력 단계**
```python
# 사용자 입력 예시
user_interaction = {
    "type": "text",
    "content": "안녕하세요, 날씨는 어떤가요?",
    "timestamp": "2025-07-18T10:30:00Z"
}
```

#### 2. **Start Node 실행**
```python
# Start Node가 사용자 입력을 처리
start_node_result = start_node.execute(user_interaction["content"])
# 결과: "안녕하세요, 날씨는 어떤가요?"
```

#### 3. **중간 노드 처리**
```python
# 예: 채팅 모델을 통한 응답 생성
chat_node_result = chat_node.execute(start_node_result)
# 결과: "안녕하세요! 오늘 날씨는 맑고 기온은 25도입니다."
```

#### 4. **End Node 실행**
```python
# End Node가 최종 결과를 사용자에게 반환
final_result = end_node.execute(chat_node_result)
# 결과: "안녕하세요! 오늘 날씨는 맑고 기온은 25도입니다."
```

### 🎨 워크플로우 구성 예시

#### 간단한 채팅 워크플로우
```json
{
  "workflow_name": "Simple Chat",
  "workflow_id": "simple_chat",
  "nodes": [
    {
      "id": "start_node",
      "type": "tool/input_str",
      "data": {
        "nodeId": "tool/input_str",
        "parameters": {
          "input_str": "{{user_input}}"  // Interaction에서 자동 주입
        }
      }
    },
    {
      "id": "chat_node",
      "type": "chat/openai",
      "data": {
        "nodeId": "chat/openai",
        "parameters": {
          "model": "gpt-3.5-turbo",
          "temperature": 0.7
        }
      }
    },
    {
      "id": "end_node",
      "type": "tool/output_str",
      "data": {
        "nodeId": "tool/output_str",
        "parameters": {}
      }
    }
  ],
  "edges": [
    {
      "id": "edge1",
      "source": {"nodeId": "start_node", "portId": "result"},
      "target": {"nodeId": "chat_node", "portId": "text"}
    },
    {
      "id": "edge2",
      "source": {"nodeId": "chat_node", "portId": "result"},
      "target": {"nodeId": "end_node", "portId": "input"}
    }
  ]
}
```

#### 복잡한 RAG 워크플로우
```json
{
  "workflow_name": "RAG Chat",
  "workflow_id": "rag_chat",
  "nodes": [
    {
      "id": "start_node",
      "type": "tool/input_str",
      "data": {
        "nodeId": "tool/input_str",
        "parameters": {
          "input_str": "{{user_input}}"
        }
      }
    },
    {
      "id": "retrieval_node",
      "type": "rag/vector_search",
      "data": {
        "nodeId": "rag/vector_search",
        "parameters": {
          "collection_name": "knowledge_base",
          "top_k": 5
        }
      }
    },
    {
      "id": "context_merge_node",
      "type": "tool/string_concat",
      "data": {
        "nodeId": "tool/string_concat",
        "parameters": {
          "separator": "\n\n"
        }
      }
    },
    {
      "id": "chat_node",
      "type": "chat/openai",
      "data": {
        "nodeId": "chat/openai",
        "parameters": {
          "model": "gpt-4",
          "temperature": 0.3
        }
      }
    },
    {
      "id": "end_node",
      "type": "tool/output_str",
      "data": {
        "nodeId": "tool/output_str",
        "parameters": {}
      }
    }
  ],
  "edges": [
    {
      "id": "edge1",
      "source": {"nodeId": "start_node", "portId": "result"},
      "target": {"nodeId": "retrieval_node", "portId": "query"}
    },
    {
      "id": "edge2",
      "source": {"nodeId": "start_node", "portId": "result"},
      "target": {"nodeId": "context_merge_node", "portId": "str1"}
    },
    {
      "id": "edge3",
      "source": {"nodeId": "retrieval_node", "portId": "result"},
      "target": {"nodeId": "context_merge_node", "portId": "str2"}
    },
    {
      "id": "edge4",
      "source": {"nodeId": "context_merge_node", "portId": "result"},
      "target": {"nodeId": "chat_node", "portId": "text"}
    },
    {
      "id": "edge5",
      "source": {"nodeId": "chat_node", "portId": "result"},
      "target": {"nodeId": "end_node", "portId": "input"}
    }
  ]
}
```

### 🎯 Best Practices

#### 1. **Start Node 설계**
- **단순성**: 복잡한 로직 지양, 입력 변환에 집중
- **유연성**: 다양한 입력 형식 지원
- **검증**: 입력 데이터 유효성 검증

#### 2. **End Node 설계**
- **포맷팅**: 사용자 친화적인 결과 형식
- **메타데이터**: 실행 정보, 타임스탬프 등 추가
- **에러 처리**: 실행 오류 시 적절한 에러 메시지

#### 3. **워크플로우 설계**
- **선형성**: Start → 중간 → End 선형 구조 유지
- **검증**: 연결성 및 구조 검증
- **테스트**: 다양한 입력에 대한 테스트

---

# Output Dependency 기반 라우팅 구현 완료

## 구현 개요

노드의 출력 포트에 `dependency`와 `dependencyValue` 속성을 추가하여, 파라미터 값에 따라 동적으로 활성화되는 출력 포트를 결정하는 기능을 구현했습니다.

## 구현 파일

### 1. 핵심 로직: `/plateerag_backend/editor/async_workflow_executor.py`

#### 추가된 메서드

##### `_get_active_output_ports(node_info, parameters)`
- 노드의 출력 포트 중 현재 파라미터 값에 따라 활성화된 포트만 반환
- dependency와 dependencyValue를 확인하여 조건 평가
- 상세한 디버그 로깅 포함

##### `_compare_dependency_values(actual_value, expected_value)`
- 다양한 타입(Boolean, String, Number, None)에 대한 강건한 값 비교
- 대소문자 무관 비교
- Boolean-like 문자열 자동 변환 ("true", "1", "yes" 등)
- 숫자 타입 자동 변환

#### 수정된 로직

##### 일반 노드 출력 처리 (라인 980-1025)
```python
# 기존: 첫 번째 출력 포트로 무조건 전달
output_port_id = node_info['data']['outputs'][0]['id']
node_outputs[node_id] = {output_port_id: result}

# 신규: dependency 기반 활성 포트 필터링
active_ports = self._get_active_output_ports(node_info, current_parameters)
# 활성 포트로만 데이터 전달
# 비활성 포트와 연결된 노드들은 excluded_nodes에 추가
```

주요 변경 사항:
1. 파라미터 값 수집
2. 활성 출력 포트 결정
3. 활성 포트가 하나면 해당 포트로만 전달
4. 활성 포트가 여러 개면 모든 활성 포트로 전달
5. 비활성 포트와 연결된 노드들을 재귀적으로 제외

### 2. 테스트 파일: `/plateerag_backend/test_dependency_output.py`

다음 시나리오를 테스트:
- ✓ 스트리밍 모드별 출력 포트 선택
- ✓ dependency 없는 출력 포트 (항상 활성화)
- ✓ 여러 dependency 조건
- ✓ Boolean 값 비교
- ✓ 혼합 dependency (일부만 dependency 있음)

### 3. 문서

- `/plateerag_backend/docs/OUTPUT_DEPENDENCY_ROUTING.md`: 상세 기능 설명
- `/plateerag_backend/examples/dependency_output_routing_example.py`: 통합 예시

## 사용 예시

### Agent Xgen 노드 (이미 적용됨)

```python
outputs = [
    {
        "id": "stream",
        "name": "Stream",
        "type": "STREAM STR",
        "stream": True,
        "dependency": "streaming",
        "dependencyValue": True
    },
    {
        "id": "result",
        "name": "Result",
        "type": "STR",
        "dependency": "streaming",
        "dependencyValue": False
    }
]

parameters = [
    {
        "id": "streaming",
        "name": "Streaming",
        "type": "BOOL",
        "value": True,
        # ...
    }
]
```

**동작**:
- `streaming=True` → `stream` 포트만 활성화
- `streaming=False` → `result` 포트만 활성화

## 실행 흐름

```
노드 실행 시작
    ↓
파라미터 값 수집
    ↓
_get_active_output_ports() 호출
    ↓
각 출력 포트의 dependency 확인
    ↓
_compare_dependency_values()로 조건 평가
    ↓
활성 포트 목록 반환
    ↓
노드 execute() 실행
    ↓
활성 포트로만 결과 전달
    ↓
비활성 포트와 연결된 노드들 제외
    ↓
다음 노드 실행 (활성 경로만)
```

## 로그 예시

```
-> 출력 포트 'stream': dependency='streaming' (실제=True, 기대=True), 일치, 활성화
-> 출력 포트 'result': dependency='streaming' (실제=True, 기대=False), 불일치, 비활성화
-> 활성화된 출력 포트: ['stream'] (총 1/2개)
-> 출력 (활성 포트: stream): {'stream': <generator>}
-> 비활성 포트 감지 (1개): ['result']
-> dependency 조건으로 인해 노드 'result_handler' 및 후속 노드들 제외 (비활성 포트: result)
```

## 주요 특징

### 1. 강건한 타입 비교
- Boolean, String, Number, None 모두 지원
- 유연한 Boolean 매칭 (true/True/1/yes/on 등)
- 대소문자 무관 비교

### 2. 재귀적 노드 제외
- 비활성 포트와 연결된 노드뿐만 아니라
- 그 후속 노드들도 모두 제외 (DFS)

### 3. 다중 활성 포트 지원
- 여러 포트가 동시에 활성화될 수 있음
- 모든 활성 포트로 동일한 결과 전달

### 4. 프론트엔드 동기화
- 프론트엔드에서 보이는 포트 = 백엔드에서 실행되는 포트
- 일관된 사용자 경험

## 호환성

### 하위 호환성 보장
- dependency가 없는 출력 포트는 기존과 동일하게 작동 (항상 활성화)
- 기존 노드들은 수정 없이 그대로 작동
- RouterNode의 라우팅 로직과 독립적으로 동작

### 적용 가능한 노드
- 모든 노드 타입에 적용 가능
- Agent, Chat, Tool, Util 등 모든 카테고리

## 테스트 결과

```bash
$ python test_dependency_output.py

============================================================
Output Dependency 라우팅 테스트 시작
============================================================
✓ 모든 테스트 통과!
```

## 다음 단계

이 기능을 활용하여 다양한 노드에 적용할 수 있습니다:

1. **Conditional Processing Node**
   - 처리 모드에 따라 다른 출력 경로

2. **Format Converter Node**
   - 출력 형식(JSON/XML/CSV)에 따른 라우팅

3. **Error Handling Node**
   - 성공/실패 경로 분리

4. **Multi-Model Node**
   - 모델 타입에 따른 다른 출력 처리

## 참고 자료

- 상세 문서: `/plateerag_backend/docs/OUTPUT_DEPENDENCY_ROUTING.md`
- 예시 코드: `/plateerag_backend/examples/dependency_output_routing_example.py`
- 테스트 코드: `/plateerag_backend/test_dependency_output.py`
- README 파라미터 가이드: `/plateerag_backend/editor/README.md`

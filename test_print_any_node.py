#!/usr/bin/env python3
"""
PrintAnyNode 테스트 스크립트 - 수정된 버전
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from editor.nodes.xgen.tool.print_any import PrintAnyNode

def test_print_any_node():
    """PrintAnyNode의 객체 자동 변환 기능을 테스트합니다."""
    print("=== 수정된 PrintAnyNode 테스트 시작 ===\n")

    # PrintAnyNode 인스턴스 생성
    print_node = PrintAnyNode()

    # 테스트 케이스들
    test_cases = [
        # 기본 타입들 - 그대로 반환되어야 함
        {"name": "문자열", "input": "Hello World", "expected_type": str},
        {"name": "정수", "input": 42, "expected_type": int},
        {"name": "실수", "input": 3.14, "expected_type": float},
        {"name": "불린(True)", "input": True, "expected_type": bool},
        {"name": "불린(False)", "input": False, "expected_type": bool},
        {"name": "None", "input": None, "expected_type": type(None)},

        # 객체들 - JSON 문자열로 변환되어야 함
        {"name": "딕셔너리", "input": {"sentiment": False, "score": 0.8}, "expected_type": str},
        {"name": "리스트", "input": [1, 2, 3, "test"], "expected_type": str},
        {"name": "중첩 딕셔너리", "input": {"data": {"sentiment": False}, "meta": {"version": 1}}, "expected_type": str},
        {"name": "빈 딕셔너리", "input": {}, "expected_type": str},
        {"name": "빈 리스트", "input": [], "expected_type": str},
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"테스트 {i}: {test_case['name']}")
        print(f"입력: {test_case['input']} (타입: {type(test_case['input']).__name__})")

        try:
            # execute 메서드 호출
            result = print_node.execute(input_print=test_case['input'])

            print(f"출력: {result}")
            print(f"출력 타입: {type(result).__name__}")
            print(f"예상 타입과 일치: {isinstance(result, test_case['expected_type'])}")

            # 딕셔너리나 리스트의 경우 JSON 형태로 변환되었는지 확인
            if isinstance(test_case['input'], (dict, list)):
                print(f"JSON 형태 변환: {'OK' if isinstance(result, str) and ('{' in result or '[' in result) else 'FAIL'}")

        except Exception as e:
            print(f"오류 발생: {e}")

        print("-" * 50)

    print("\n=== 실제 워크플로우 상황 시뮬레이션 ===")
    # 실제 워크플로우에서 발생했던 상황과 유사한 테스트
    workflow_output = {"sentiment": False}
    print(f"워크플로우 출력: {workflow_output}")

    result = print_node.execute(input_print=workflow_output)
    print(f"PrintAnyNode 결과: {result}")
    print(f"결과 타입: {type(result).__name__}")
    print(f"프론트엔드에서 표시 가능: {'OK' if isinstance(result, str) else 'POTENTIAL ISSUE'}")

    print("\n=== PrintAnyNode 테스트 완료 ===")

if __name__ == "__main__":
    test_print_any_node()

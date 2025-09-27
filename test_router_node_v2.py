#!/usr/bin/env python3
"""
RouterNode 테스트 스크립트 - 수정된 버전
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from editor.nodes.xgen.tool.router import RouterNode

def test_router_node():
    """RouterNode의 기본 동작을 테스트합니다."""
    print("=== 수정된 RouterNode 테스트 시작 ===\n")

    # RouterNode 인스턴스 생성
    router = RouterNode()

    # 테스트 케이스 1: 정상적인 Dict 입력과 라우팅
    print("테스트 1: 정상적인 라우팅")
    test_input_1 = {"is_human": False, "content": "AI가 생성한 콘텐츠입니다."}
    result_1 = router.execute(agent_output=test_input_1, routing_criteria="is_human")
    print(f"입력: {test_input_1}")
    print("라우팅 기준: is_human")
    print(f"결과 타입: {type(result_1)}")
    print(f"결과: {result_1}")
    print(f"원본과 동일: {result_1 == test_input_1}")
    print()

    # 테스트 케이스 2: 다른 값으로 라우팅
    print("테스트 2: True 값으로 라우팅")
    test_input_2 = {"is_human": True, "content": "사람이 작성한 콘텐츠입니다."}
    result_2 = router.execute(agent_output=test_input_2, routing_criteria="is_human")
    print(f"입력: {test_input_2}")
    print("라우팅 기준: is_human")
    print(f"결과 타입: {type(result_2)}")
    print(f"결과: {result_2}")
    print(f"원본과 동일: {result_2 == test_input_2}")
    print()

    # 테스트 케이스 3: print로 바로 사용 가능한지 확인
    print("테스트 3: print 직접 사용")
    print("print(result_1):")
    print(result_1)
    print("print(result_2):")
    print(result_2)
    print()

    print("=== RouterNode 테스트 완료 ===")

if __name__ == "__main__":
    test_router_node()

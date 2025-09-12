"""
피드백 루프 노드 테스트 스크립트
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from editor.nodes.xgen.agent.agent_feedback_loop import AgentFeedbackLoopNode
from editor.utils.helper.feedback_evaluator import FeedbackEvaluator, IterativeImprover

def test_feedback_evaluator():
    """피드백 평가기 테스트"""
    print("=== FeedbackEvaluator 테스트 ===")
    
    evaluator = FeedbackEvaluator()
    
    # 테스트 케이스 1: 좋은 결과
    user_request = "파이썬으로 피보나치 수열을 계산하는 함수를 만들어주세요"
    good_result = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# 사용 예시
print(fibonacci(10))  # 55 출력
"""
    
    score = evaluator.evaluate_result(user_request, good_result)
    print(f"좋은 결과 점수: {score.overall_score}/10")
    print(f"이유: {score.reasoning}")
    print(f"장점: {score.strengths}")
    print(f"개선사항: {score.improvements}")
    print()
    
    # 테스트 케이스 2: 부족한 결과
    bad_result = "오류가 발생했습니다"
    score2 = evaluator.evaluate_result(user_request, bad_result)
    print(f"부족한 결과 점수: {score2.overall_score}/10")
    print(f"이유: {score2.reasoning}")
    print(f"개선사항: {score2.improvements}")
    print()

def test_iterative_improver():
    """반복적 개선기 테스트"""
    print("=== IterativeImprover 테스트 ===")
    
    evaluator = FeedbackEvaluator()
    improver = IterativeImprover(evaluator)
    
    user_request = "간단한 계산기 함수를 만들어주세요"
    
    # 첫 번째 결과 (부족함)
    result1 = "def add(a, b): return a + b"
    score1 = evaluator.evaluate_result(user_request, result1)
    improver.track_improvement(1, result1, score1)
    
    # 두 번째 결과 (개선됨)
    result2 = """
def calculator(a, b, operation):
    if operation == 'add':
        return a + b
    elif operation == 'subtract':
        return a - b
    elif operation == 'multiply':
        return a * b
    elif operation == 'divide':
        return a / b if b != 0 else "Division by zero error"
    else:
        return "Unknown operation"
"""
    score2 = evaluator.evaluate_result(user_request, result2)
    improver.track_improvement(2, result2, score2)
    
    print(f"반복 1 점수: {score1.overall_score}/10")
    print(f"반복 2 점수: {score2.overall_score}/10")
    
    # 개선 분석
    analysis = improver.analyze_improvement_trend()
    print(f"개선 추세: {analysis['trend']}")
    print(f"개선 정도: {analysis['improvement']}")
    
    best_result = improver.get_best_result()
    print(f"최고 점수: {best_result['score']}/10")
    print()

def test_feedback_loop_node_basic():
    """피드백 루프 노드 기본 테스트"""
    print("=== AgentFeedbackLoopNode 기본 테스트 ===")
    
    # 노드 생성
    node = AgentFeedbackLoopNode()
    
    # 기본 속성 확인
    print(f"노드 ID: {node.nodeId}")
    print(f"노드 이름: {node.nodeName}")
    print(f"설명: {node.description}")
    print(f"태그: {node.tags}")
    print()
    
    # 입력/출력 확인
    print("입력:")
    for inp in node.inputs:
        required = inp.get('required', False)
        print(f"  - {inp['name']} ({inp['type']}): 필수={required}")
    
    print("\n출력:")
    for out in node.outputs:
        print(f"  - {out['name']} ({out['type']})")
    
    print("\n파라미터:")
    for param in node.parameters:
        print(f"  - {param['name']} ({param['type']}): 기본값={param.get('value', 'N/A')}")
    print()

def simulate_feedback_loop():
    """피드백 루프 시뮬레이션 (실제 LLM 없이)"""
    print("=== 피드백 루프 시뮬레이션 ===")
    
    # 모의 도구 클래스
    class MockTool:
        def __init__(self, name, responses):
            self.name = name
            self.responses = responses
            self.call_count = 0
        
        def invoke(self, input_text):
            response = self.responses[min(self.call_count, len(self.responses) - 1)]
            self.call_count += 1
            return response
    
    # 모의 LLM 클래스
    class MockLLM:
        def __init__(self):
            self.call_count = 0
            self.evaluation_responses = [
                '{"score": 4, "reasoning": "기본적인 기능만 구현됨", "improvements": ["더 많은 기능 필요"], "strengths": ["문법적으로 올바름"]}',
                '{"score": 7, "reasoning": "충분한 기능 구현", "improvements": ["문서화 필요"], "strengths": ["완성도 높음"]}',
                '{"score": 9, "reasoning": "매우 완성도 높음", "improvements": ["미세한 개선"], "strengths": ["모든 요구사항 충족"]}'
            ]
        
        def invoke(self, prompt):
            # 평가 프롬프트인지 확인
            if "점 척도로 평가" in str(prompt) or "JSON 형식으로 응답" in str(prompt):
                response = self.evaluation_responses[min(self.call_count, len(self.evaluation_responses) - 1)]
                self.call_count += 1
                return type('Response', (), {'content': response})()
            
            # 일반 응답
            return type('Response', (), {'content': f"Mock response {self.call_count}"})()
    
    # 시뮬레이션 실행
    try:
        # 피드백 평가기 생성 (Mock LLM 사용)
        mock_llm = MockLLM()
        evaluator = FeedbackEvaluator(llm=mock_llm)
        improver = IterativeImprover(evaluator)
        
        user_request = "파이썬으로 정렬 알고리즘을 구현해주세요"
        
        # 3번의 반복 시뮬레이션
        results = [
            "def sort(arr): return sorted(arr)",  # 간단한 구현
            "def bubble_sort(arr):\n    for i in range(len(arr)):\n        for j in range(len(arr)-1-i):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr",  # 개선된 구현
            "def bubble_sort(arr):\n    \"\"\"버블 정렬 구현\"\"\"\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr\n\n# 사용 예시\ntest_arr = [64, 34, 25, 12, 22, 11, 90]\nprint(f'정렬 전: {test_arr}')\nprint(f'정렬 후: {bubble_sort(test_arr.copy())}')"  # 완성된 구현
        ]
        
        for i, result in enumerate(results, 1):
            print(f"--- 반복 {i} ---")
            score = evaluator.evaluate_result(user_request, result)
            improver.track_improvement(i, result, score)
            
            print(f"결과: {result[:50]}...")
            print(f"점수: {score.overall_score}/10")
            print(f"개선사항: {score.improvements}")
            
            if score.overall_score >= 8:
                print("만족스러운 결과 달성!")
                break
            print()
        
        # 최종 분석
        analysis = improver.analyze_improvement_trend()
        print(f"\n최종 분석:")
        print(f"- 개선 추세: {analysis['trend']}")
        print(f"- 총 개선 정도: {analysis['improvement']} 점")
        print(f"- 최고 점수: {analysis['best_score']}/10")
        
    except Exception as e:
        print(f"시뮬레이션 오류: {str(e)}")
    print()

def run_all_tests():
    """모든 테스트 실행"""
    print("피드백 루프 노드 테스트 시작\n")
    
    try:
        test_feedback_evaluator()
        test_iterative_improver()
        test_feedback_loop_node_basic()
        simulate_feedback_loop()
        
        print("✅ 모든 테스트 완료")
        
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()
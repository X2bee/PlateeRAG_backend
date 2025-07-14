"""
Performance Logger 사용 예제
"""
import os
import json
from src.monitoring.performance_logger import PerformanceLogger
from database.connection import AppDatabaseManager
from controller.performanceController import PerformanceController

def extract_workflow_info_from_file(workflow_file_path: str) -> tuple[str, str]:
    """
    워크플로우 파일에서 workflow_name과 workflow_id를 추출합니다.
    
    Args:
        workflow_file_path: 워크플로우 JSON 파일 경로
        
    Returns:
        tuple: (workflow_name, workflow_id)
    """
    try:
        # workflow_name은 파일명에서 추출 (확장자 제외)
        workflow_name = os.path.splitext(os.path.basename(workflow_file_path))[0]
        
        # workflow_id는 JSON 파일 내부의 id 필드에서 추출
        with open(workflow_file_path, 'r', encoding='utf-8') as f:
            workflow_data = json.load(f)
            workflow_id = workflow_data.get('id', '')
        
        return workflow_name, workflow_id
        
    except Exception as e:
        print(f"Error extracting workflow info: {e}")
        return "", ""

def performance_logger_usage_example():
    """Performance Logger 사용 예제"""
    
    # 1. 워크플로우 정보 추출
    workflow_file = "/plateerag_backend/downloads/Workflow.json"
    workflow_name, workflow_id = extract_workflow_info_from_file(workflow_file)
    
    print(f"Workflow Name: {workflow_name}")
    print(f"Workflow ID: {workflow_id}")
    
    # 2. 데이터베이스 매니저 초기화 (실제 구현에서는 app.state에서 가져와야 함)
    # db_manager = app.state.app_db  # 실제 사용 시
    db_manager = None  # 예제용
    
    # 3. Performance Logger를 사용한 노드 성능 측정
    node_id = "chat/openai-1752219026167"
    node_name = "Chat OpenAI"
    
    with PerformanceLogger(workflow_name, workflow_id, node_id, node_name, db_manager) as perf_logger:
        # 여기서 실제 노드 작업을 수행
        input_data = {"text": "Hello, how are you?"}
        
        # 시뮬레이션: 실제 작업 (예: OpenAI API 호출)
        import time
        time.sleep(0.1)  # 100ms 작업 시뮬레이션
        
        output_data = {"result": "I'm doing well, thank you for asking!"}
        
        # 성능 데이터 로깅 (파일 + DB)
        perf_logger.log(input_data, output_data)
    
    print("Performance data logged successfully!")

def performance_analysis_example():
    """성능 분석 예제"""
    
    # 데이터베이스 매니저 초기화 (실제 구현에서는 app.state에서 가져와야 함)
    db_manager = None  # 예제용
    controller = PerformanceController(db_manager)
    
    workflow_name = "Workflow"
    workflow_id = "workflow_80b6de6259d615e8dc9063ca9fa3258ff1801947"
    
    # 1. 성능 평균 계산
    print("=== Performance Average ===")
    avg_data = controller.get_performance_average(workflow_name, workflow_id)
    print(json.dumps(avg_data, indent=2, ensure_ascii=False))
    
    # 2. 노드별 성능 요약
    print("\n=== Node Performance Summary ===")
    summary_data = controller.get_node_performance_summary(workflow_name, workflow_id)
    print(json.dumps(summary_data, indent=2, ensure_ascii=False))
    
    # 3. 최근 성능 데이터 조회
    print("\n=== Recent Performance Data ===")
    recent_data = controller.get_performance_data(workflow_name, workflow_id, limit=10)
    print(f"Found {len(recent_data)} recent performance records")

if __name__ == "__main__":
    print("Performance Logger Example")
    print("=" * 50)
    
    # 사용 예제 실행
    performance_logger_usage_example()
    
    print("\nPerformance Analysis Example")
    print("=" * 50)
    
    # 분석 예제 실행
    performance_analysis_example()

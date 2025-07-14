#!/usr/bin/env python3
"""
Performance Logger 간단 테스트 (API 호출 방식)
백엔드가 실행 중일 때 사용
"""
import requests
import json
import time

def test_performance_api():
    """Performance API 테스트"""
    base_url = "http://localhost:8000"  # 백엔드 서버 주소
    
    # 워크플로우 정보
    workflow_name = "Workflow"
    workflow_id = "workflow_80b6de6259d615e8dc9063ca9fa3258ff1801947"
    
    print("🧪 Performance API Test")
    print("=" * 50)
    
    # 1. 성능 데이터 조회 테스트
    print("\n1️⃣ Testing performance data retrieval...")
    try:
        response = requests.get(f"{base_url}/api/performance/data")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Found {data.get('count', 0)} performance records")
        else:
            print(f"❌ API Error: {response.status_code}")
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        print("💡 Make sure the backend server is running with: python main.py")
        return
    
    # 2. 성능 평균 조회 테스트
    print("\n2️⃣ Testing performance average...")
    try:
        response = requests.get(f"{base_url}/api/performance/average/{workflow_name}/{workflow_id}")
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                avg_data = data['data']
                print(f"✅ Execution Count: {avg_data.get('execution_count', 0)}")
                if avg_data.get('average_performance'):
                    avg_perf = avg_data['average_performance']
                    print(f"✅ Average Processing Time: {avg_perf.get('processing_time_ms', 0)}ms")
            else:
                print("⚠️ No performance data found")
        else:
            print(f"❌ API Error: {response.status_code}")
    except Exception as e:
        print(f"❌ Request Error: {e}")
    
    # 3. 노드별 성능 요약 테스트
    print("\n3️⃣ Testing node performance summary...")
    try:
        response = requests.get(f"{base_url}/api/performance/summary/{workflow_name}/{workflow_id}")
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                summary = data['data']
                print(f"✅ Total Nodes: {summary.get('total_nodes', 0)}")
                for node in summary.get('nodes_summary', []):
                    print(f"   📌 {node['node_name']}: {node['avg_processing_time_ms']}ms")
            else:
                print("⚠️ No summary data found")
        else:
            print(f"❌ API Error: {response.status_code}")
    except Exception as e:
        print(f"❌ Request Error: {e}")
    
    print("\n🎉 API Test completed!")

def create_sample_performance_data():
    """샘플 성능 데이터 생성 (데이터베이스 직접 입력용)"""
    print("\n📝 Sample performance data structure:")
    
    sample_data = {
        "workflow_name": "Workflow",
        "workflow_id": "workflow_80b6de6259d615e8dc9063ca9fa3258ff1801947",
        "node_id": "chat/openai-1752219026167",
        "node_name": "Chat OpenAI",
        "timestamp": "2025-07-13T10:30:00Z",
        "processing_time_ms": 150.25,
        "cpu_usage_percent": 15.8,
        "ram_usage_mb": 128.5,
        "gpu_usage_percent": None,
        "gpu_memory_mb": None,
        "input_data": '{"text": "Hello, how are you?"}',
        "output_data": '{"result": "I\'m doing well, thank you!"}'
    }
    
    print(json.dumps(sample_data, indent=2, ensure_ascii=False))
    
    # SQL 삽입 쿼리 예시
    print("\n📝 Sample SQL insert query:")
    sql = """
    INSERT INTO node_performance (
        workflow_name, workflow_id, node_id, node_name, timestamp,
        processing_time_ms, cpu_usage_percent, ram_usage_mb,
        gpu_usage_percent, gpu_memory_mb, input_data, output_data
    ) VALUES (
        'Workflow', 'workflow_80b6de6259d615e8dc9063ca9fa3258ff1801947',
        'chat/openai-1752219026167', 'Chat OpenAI', datetime('now'),
        150.25, 15.8, 128.5, NULL, NULL,
        '{"text": "Hello, how are you?"}',
        '{"result": "I''m doing well, thank you!"}'
    );
    """
    print(sql)

if __name__ == "__main__":
    # API 테스트 실행
    test_performance_api()
    
    # 샘플 데이터 구조 출력
    create_sample_performance_data()

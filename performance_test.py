#!/usr/bin/env python3
"""
Performance Logger 테스트 스크립트
프로젝트 루트 디렉토리에서 실행: python performance_test.py
"""
import sys
import os
import json
import time
import asyncio
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.config_composer import config_composer
from database.connection import AppDatabaseManager
from models.performance import NodePerformance
from src.monitoring.performance_logger import PerformanceLogger
from controller.performanceController import PerformanceController

def get_existing_database():
    """기존 데이터베이스 연결 재사용"""
    print("📁 Using existing database connection...")
    
    try:
        # 기존 데이터베이스 설정 가져오기
        database_config = config_composer.initialize_database_config_only()
        if not database_config:
            print("❌ Failed to get database configuration")
            return None
        
        # 기존 연결을 재사용하는 매니저 생성
        app_db = AppDatabaseManager(database_config)
        
        # NodePerformance 모델만 추가 등록 (이미 다른 모델들은 main.py에서 등록됨)
        app_db.register_model(NodePerformance)
        
        # 연결 테스트
        if app_db.config_db_manager.connect():
            print("✅ Database connection successful")
            
            # NodePerformance 테이블이 없다면 생성
            try:
                create_query = NodePerformance.get_create_table_query(app_db.config_db_manager.db_type)
                app_db.config_db_manager.execute_query(create_query)
                print("✅ NodePerformance table ready")
            except Exception as e:
                print(f"⚠️ Table creation info: {e}")
            
            return app_db
        else:
            print("❌ Failed to connect to existing database")
            return None
            
    except Exception as e:
        print(f"❌ Error accessing existing database: {e}")
        return None

def extract_workflow_info():
    """워크플로우 정보 추출"""
    print("\n📋 Extracting workflow information...")
    
    workflow_file = project_root / "downloads" / "Workflow.json"
    
    if not workflow_file.exists():
        print(f"❌ Workflow file not found: {workflow_file}")
        return None, None
    
    try:
        # workflow_name은 파일명에서 추출
        workflow_name = workflow_file.stem  # 확장자 제외한 파일명
        
        # workflow_id는 JSON 파일 내부에서 추출
        with open(workflow_file, 'r', encoding='utf-8') as f:
            workflow_data = json.load(f)
            workflow_id = workflow_data.get('id', '')
        
        print(f"✅ Workflow Name: {workflow_name}")
        print(f"✅ Workflow ID: {workflow_id}")
        
        return workflow_name, workflow_id
        
    except Exception as e:
        print(f"❌ Error extracting workflow info: {e}")
        return None, None

def simulate_node_execution(workflow_name, workflow_id, db_manager):
    """노드 실행 시뮬레이션"""
    print("\n🔄 Simulating node executions...")
    
    # 테스트용 노드들
    test_nodes = [
        {
            "node_id": "chat/openai-1752219026167",
            "node_name": "Chat OpenAI",
            "input_data": {"text": "Hello, how are you?"},
            "output_data": {"result": "I'm doing well, thank you!"},
            "simulation_time": 0.15  # 150ms
        },
        {
            "node_id": "math/add-123456789",
            "node_name": "Math Add",
            "input_data": {"a": 10, "b": 20},
            "output_data": {"result": 30},
            "simulation_time": 0.05  # 50ms
        },
        {
            "node_id": "tool/print-987654321",
            "node_name": "Print Tool",
            "input_data": {"value": "Test output"},
            "output_data": {"printed": True},
            "simulation_time": 0.02  # 20ms
        }
    ]
    
    execution_count = 0
    
    # 각 노드를 여러 번 실행하여 테스트 데이터 생성
    for round_num in range(1, 4):  # 3라운드
        print(f"\n📊 Round {round_num} execution...")
        
        for node in test_nodes:
            print(f"  🚀 Executing {node['node_name']}...")
            
            # PerformanceLogger 사용
            with PerformanceLogger(
                workflow_name=workflow_name,
                workflow_id=workflow_id,
                node_id=node["node_id"],
                node_name=node["node_name"],
                db_manager=db_manager
            ) as perf_logger:
                
                # 실제 작업 시뮬레이션 (시간은 라운드마다 약간 다르게)
                simulation_time = node["simulation_time"] * (0.8 + round_num * 0.1)
                time.sleep(simulation_time)
                
                # 성능 데이터 로깅
                perf_logger.log(node["input_data"], node["output_data"])
                execution_count += 1
    
    print(f"\n✅ Completed {execution_count} node executions")
    return execution_count

def test_performance_analysis(workflow_name, workflow_id, db_manager):
    """성능 분석 테스트"""
    print("\n📈 Testing performance analysis...")
    
    controller = PerformanceController(db_manager)
    
    # 1. 전체 성능 데이터 조회
    print("\n1️⃣ Getting all performance data...")
    all_data = controller.get_performance_data(workflow_name, workflow_id)
    print(f"   Found {len(all_data)} performance records")
    
    if all_data:
        latest = all_data[0]
        print(f"   Latest execution: {latest.get('node_name')} - {latest.get('processing_time_ms')}ms")
    
    # 2. 성능 평균 계산
    print("\n2️⃣ Calculating performance average...")
    avg_data = controller.get_performance_average(workflow_name, workflow_id)
    
    if avg_data.get('execution_count', 0) > 0:
        avg_perf = avg_data['average_performance']
        print(f"   Execution Count: {avg_data['execution_count']}")
        print(f"   Average Processing Time: {avg_perf['processing_time_ms']}ms")
        print(f"   Average CPU Usage: {avg_perf['cpu_usage_percent']}%")
        print(f"   Average RAM Usage: {avg_perf['ram_usage_mb']}MB")
    else:
        print("   No performance data found for averaging")
    
    # 3. 노드별 성능 요약
    print("\n3️⃣ Getting node performance summary...")
    summary_data = controller.get_node_performance_summary(workflow_name, workflow_id)
    
    if summary_data.get('nodes_summary'):
        print(f"   Total Nodes: {summary_data['total_nodes']}")
        for node_summary in summary_data['nodes_summary']:
            print(f"   📌 {node_summary['node_name']}: {node_summary['avg_processing_time_ms']}ms (avg)")
    else:
        print("   No node summary data found")

def test_database_direct_query(db_manager):
    """데이터베이스 직접 쿼리 테스트"""
    print("\n🗄️ Testing direct database query...")
    
    try:
        # 직접 SQL 쿼리로 데이터 확인
        query = "SELECT COUNT(*) as count FROM node_performance"
        result = db_manager.config_db_manager.execute_query(query)
        
        if result:
            count = result[0]['count']
            print(f"   Total records in node_performance table: {count}")
        else:
            print("   No results from database query")
            
    except Exception as e:
        print(f"   ❌ Database query error: {e}")

def cleanup_test_data(db_manager):
    """테스트 데이터 정리"""
    print("\n🧹 Cleaning up test data...")
    
    try:
        # 테스트 데이터 삭제
        query = "DELETE FROM node_performance WHERE workflow_name = 'Workflow'"
        db_manager.config_db_manager.execute_query(query)
        print("   ✅ Test data cleaned up")
        
    except Exception as e:
        print(f"   ❌ Cleanup error: {e}")

def main():
    """메인 테스트 실행"""
    print("🧪 Performance Logger Test")
    print("=" * 50)
    
    # 1. 기존 데이터베이스 연결 사용
    db_manager = get_existing_database()
    if not db_manager:
        print("❌ Database connection failed. Exiting.")
        return
    
    # 2. 워크플로우 정보 추출
    workflow_name, workflow_id = extract_workflow_info()
    if not workflow_name or not workflow_id:
        print("❌ Workflow info extraction failed. Exiting.")
        return
    
    try:
        # 3. 노드 실행 시뮬레이션
        execution_count = simulate_node_execution(workflow_name, workflow_id, db_manager)
        
        # 4. 데이터베이스 직접 확인
        test_database_direct_query(db_manager)
        
        # 5. 성능 분석 테스트
        test_performance_analysis(workflow_name, workflow_id, db_manager)
        
        print("\n🎉 All tests completed successfully!")
        
        # 6. 테스트 데이터 정리 (선택사항)
        cleanup_choice = input("\n🗑️ Clean up test data? (y/N): ").strip().lower()
        if cleanup_choice == 'y':
            cleanup_test_data(db_manager)
        
    except Exception as e:
        print(f"❌ Test execution error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n👋 Test completed.")

if __name__ == "__main__":
    main()

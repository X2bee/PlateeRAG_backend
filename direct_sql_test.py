#!/usr/bin/env python3
"""
직접 SQL을 이용한 성능 데이터 삽입 테스트
"""
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.config_composer import config_composer

def test_direct_sql_insert():
    """직접 SQL로 샘플 데이터 삽입"""
    print("🔧 Direct SQL Insert Test")
    print("=" * 40)
    
    # 데이터베이스 설정 가져오기
    database_config = config_composer.initialize_database_config_only()
    if not database_config:
        print("❌ Failed to get database configuration")
        return
    
    from config.database_manager import DatabaseManager
    
    db_manager = DatabaseManager(database_config)
    if not db_manager.connect():
        print("❌ Failed to connect to database")
        return
    
    print(f"✅ Connected to {db_manager.db_type} database")
    
    # 샘플 데이터 삽입
    try:
        if db_manager.db_type == "postgresql":
            # PostgreSQL에서는 %s를 사용
            query = """
            INSERT INTO node_performance (
                workflow_name, workflow_id, node_id, node_name, timestamp,
                processing_time_ms, cpu_usage_percent, ram_usage_mb,
                gpu_usage_percent, gpu_memory_mb, input_data, output_data
            ) VALUES (%s, %s, %s, %s, NOW(), %s, %s, %s, %s, %s, %s, %s)
            """
        else:
            query = """
            INSERT INTO node_performance (
                workflow_name, workflow_id, node_id, node_name, timestamp,
                processing_time_ms, cpu_usage_percent, ram_usage_mb,
                gpu_usage_percent, gpu_memory_mb, input_data, output_data
            ) VALUES (?, ?, ?, ?, datetime('now'), ?, ?, ?, ?, ?, ?, ?)
            """
        
        # 샘플 데이터
        sample_data = (
            "Workflow",  # workflow_name
            "workflow_80b6de6259d615e8dc9063ca9fa3258ff1801947",  # workflow_id
            "chat/openai-1752219026167",  # node_id
            "Chat OpenAI",  # node_name
            # timestamp는 SQL에서 자동 생성
            150.25,  # processing_time_ms
            15.8,    # cpu_usage_percent
            128.5,   # ram_usage_mb
            None,    # gpu_usage_percent
            None,    # gpu_memory_mb
            '{"text": "Hello, how are you?"}',  # input_data
            '{"result": "I\'m doing well, thank you!"}'  # output_data
        )
        
        print("📝 Inserting sample data...")
        result = db_manager.execute_query(query, sample_data)
        
        if result is not None:
            print("✅ Sample data inserted successfully")
            
            # 데이터 확인
            count_query = "SELECT COUNT(*) as count FROM node_performance"
            count_result = db_manager.execute_query(count_query)
            if count_result:
                count = count_result[0]['count']
                print(f"📊 Total records in database: {count}")
                
                # 최근 데이터 조회
                recent_query = "SELECT * FROM node_performance ORDER BY created_at DESC LIMIT 1"
                recent_result = db_manager.execute_query(recent_query)
                if recent_result:
                    recent_data = recent_result[0]
                    print(f"🔍 Latest record: {recent_data['node_name']} - {recent_data['processing_time_ms']}ms")
        else:
            print("❌ Failed to insert sample data")
            
    except Exception as e:
        print(f"❌ Error during direct SQL test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        db_manager.connection.close()

if __name__ == "__main__":
    test_direct_sql_insert()

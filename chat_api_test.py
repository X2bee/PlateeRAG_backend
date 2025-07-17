"""
Chat API 테스트
"""
import requests
import json
import time
from datetime import datetime

# API 기본 URL
BASE_URL = "http://localhost:8000" 

def test_chat_new_api():
    """
    /api/chat/new API 테스트
    """
    print("=" * 50)
    print("테스트 1: /api/chat/new API")
    print("=" * 50)
    
    # 새로운 채팅 세션 시작
    interaction_id = f"test_chat_session_{int(time.time())}"
    
    request_data = {
        "workflow_name": "default_mode",
        "workflow_id": "default_mode", 
        "interaction_id": interaction_id,
        "input_data": "안녕하세요! 새로운 채팅을 시작합니다."
    }
    
    try:
        print(f"📤 요청 데이터: {json.dumps(request_data, ensure_ascii=False, indent=2)}")
        
        response = requests.post(
            f"{BASE_URL}/api/chat/new",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"📥 응답 상태: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 성공! 응답: {json.dumps(result, ensure_ascii=False, indent=2)}")
            return result.get("interaction_id")
        else:
            print(f"❌ 실패! 응답: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("❌ 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
        return None
    except Exception as e:
        print(f"❌ 오류: {str(e)}")
        return None


def test_chat_execution_api(interaction_id):
    """
    /api/chat/execution API 테스트
    """
    print("\n" + "=" * 50)
    print("테스트 2: /api/chat/execution API")
    print("=" * 50)
    
    if not interaction_id:
        print("❌ interaction_id가 없어서 테스트를 건너뜁니다.")
        return False
    
    request_data = {
        "user_input": "제가 방금 뭐라고 인사했나요?",
        "interaction_id": interaction_id
    }
    
    try:
        print(f"📤 요청 데이터: {json.dumps(request_data, ensure_ascii=False, indent=2)}")
        
        response = requests.post(
            f"{BASE_URL}/api/chat/execution",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"📥 응답 상태: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 성공! 응답: {json.dumps(result, ensure_ascii=False, indent=2)}")
            return True
        else:
            print(f"❌ 실패! 응답: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
        return False
    except Exception as e:
        print(f"❌ 오류: {str(e)}")
        return False


def test_invalid_workflow_mode():
    """
    잘못된 workflow_mode로 테스트 (에러 케이스)
    """
    print("\n" + "=" * 50)
    print("테스트 3: 잘못된 workflow_mode 테스트")
    print("=" * 50)
    
    interaction_id = f"test_invalid_{int(time.time())}"
    
    request_data = {
        "workflow_name": "invalid_mode",  # 잘못된 값
        "workflow_id": "default_mode",
        "interaction_id": interaction_id,
        "input_data": "테스트 메시지"
    }
    
    try:
        print(f"📤 요청 데이터: {json.dumps(request_data, ensure_ascii=False, indent=2)}")
        
        response = requests.post(
            f"{BASE_URL}/api/chat/new",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"📥 응답 상태: {response.status_code}")
        
        if response.status_code == 400:
            print(f"✅ 예상대로 400 에러 발생: {response.text}")
            return True
        else:
            print(f"❌ 예상과 다른 응답: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 오류: {str(e)}")
        return False


def main():
    """메인 테스트 함수"""
    print("🚀 Chat API 테스트 시작")
    print(f"📅 테스트 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌐 서버 URL: {BASE_URL}")
    
    # 테스트 실행
    interaction_id = test_chat_new_api()
    test2_result = test_chat_execution_api(interaction_id)
    test3_result = test_invalid_workflow_mode()
    
    # 결과 요약
    print("\n" + "=" * 50)
    print("📊 테스트 결과 요약")
    print("=" * 50)
    print(f"테스트 1 (chat/new): {'✅ 통과' if interaction_id else '❌ 실패'}")
    print(f"테스트 2 (chat/execution): {'✅ 통과' if test2_result else '❌ 실패'}")
    print(f"테스트 3 (invalid mode): {'✅ 통과' if test3_result else '❌ 실패'}")
    
    if interaction_id and test2_result and test3_result:
        print("\n🎉 모든 테스트가 성공적으로 완료되었습니다!")
    else:
        print("\n⚠️  일부 테스트가 실패했습니다.")
        print("💡 서버가 실행 중인지 확인하세요: python main.py")


if __name__ == "__main__":
    main()

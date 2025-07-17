"""
ChatGPT 대화 기능 테스트
"""
import os
import sys
import asyncio
from datetime import datetime

# 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.general_function import create_conversation_function


class MockConfigComposer:
    """테스트용 설정 컴포저 모의 객체"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
    
    def get_config_by_name(self, config_name: str):
        """설정 가져오기 모의 메서드"""
        if config_name == "openai.api_key":
            return MockConfig(self.api_key)
        raise KeyError(f"Configuration '{config_name}' not found")


class MockConfig:
    """테스트용 설정 객체"""
    
    def __init__(self, value: str):
        self.value = value


def test_conversation_without_db():
    """DB 연결 없이 대화 기능 테스트"""
    print("=" * 50)
    print("테스트 1: DB 연결 없이 대화 기능 테스트")
    print("=" * 50)
    
    # 환경변수에서 API 키 확인
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("   export OPENAI_API_KEY='your-api-key-here' 명령으로 설정하세요.")
        return False
    
    print(f"✅ OpenAI API 키 확인됨: {api_key[:8]}...")
    
    try:
        # 모의 설정 컴포저 생성
        config_composer = MockConfigComposer(api_key)
        
        # 대화 함수 생성 (DB 매니저 없음)
        conversation = create_conversation_function(config_composer, db_manager=None)
        
        print("\n📝 대화 함수 생성 완료")
        
        # 테스트 대화 1
        print("\n🤖 테스트 대화 1:")
        result1 = conversation(
            user_input="안녕하세요! 반갑습니다.",
            workflow_id=None,
            workflow_name=None,
            interaction_id="default"
        )
        
        if result1["status"] == "success":
            print(f"✅ 사용자: {result1['user_input']}")
            print(f"✅ AI: {result1['ai_response']}")
            print(f"✅ 세션 ID: {result1['session_id']}")
        else:
            print(f"❌ 오류 발생: {result1['error_message']}")
            return False
        
        # 테스트 대화 2 (같은 세션이지만 DB가 없으므로 기억하지 못함)
        print("\n🤖 테스트 대화 2:")
        result2 = conversation(
            user_input="제가 방금 뭐라고 인사했나요?",
            workflow_id=None,
            workflow_name=None,
            interaction_id="default"
        )
        
        if result2["status"] == "success":
            print(f"✅ 사용자: {result2['user_input']}")
            print(f"✅ AI: {result2['ai_response']}")
            print(f"✅ 세션 ID: {result2['session_id']}")
            print("ℹ️  DB 연결이 없으므로 이전 대화를 기억하지 못합니다.")
        else:
            print(f"❌ 오류 발생: {result2['error_message']}")
            return False
        
        # 워크플로우 정보가 있는 테스트
        print("\n🤖 테스트 대화 3 (워크플로우 정보 포함):")
        result3 = conversation(
            user_input="이 워크플로우에서 무엇을 할 수 있나요?",
            workflow_id="test_workflow_123",
            workflow_name="테스트 워크플로우",
            interaction_id="session_456"
        )
        
        if result3["status"] == "success":
            print(f"✅ 사용자: {result3['user_input']}")
            print(f"✅ AI: {result3['ai_response']}")
            print(f"✅ 워크플로우 ID: {result3['workflow_id']}")
            print(f"✅ 워크플로우 이름: {result3['workflow_name']}")
            print(f"✅ 세션 ID: {result3['session_id']}")
        else:
            print(f"❌ 오류 발생: {result3['error_message']}")
            return False
        
        print("\n✅ 모든 테스트 통과!")
        return True
        
    except Exception as e:
        print(f"❌ 예외 발생: {str(e)}")
        return False


def main():
    """메인 테스트 함수"""
    print("🚀 ChatGPT 대화 기능 테스트 시작")
    print(f"📅 테스트 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 테스트 실행
    test1_result = test_conversation_without_db()
    
    # 결과 요약
    print("\n" + "=" * 50)
    print("📊 테스트 결과 요약")
    print("=" * 50)
    print(f"테스트 1 (DB 없는 대화): {'✅ 통과' if test1_result else '❌ 실패'}")
    
    if test1_result:
        print("\n🎉 테스트가 성공적으로 완료되었습니다!")
    else:
        print("\n⚠️  테스트가 실패했습니다.")

if __name__ == "__main__":
    main()

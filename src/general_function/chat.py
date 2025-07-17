"""
LangChain을 이용한 OpenAI 채팅 기능
"""
import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

logger = logging.getLogger(__name__)

class DatabaseChatHistory(BaseChatMessageHistory):
    """DB에서 대화 기록을 가져와서 메모리로 사용하는 클래스"""
    def __init__(self, db_manager, session_id: str):
        self.db_manager = db_manager
        self.session_id = session_id
        self._messages: List[BaseMessage] = []
        self._load_messages_from_db()
    
    def _load_messages_from_db(self):
        """DB에서 대화 기록을 로드"""
        if not self.db_manager or self.session_id == "default":
            return
            
        try:
            query = """
            SELECT input_data, output_data, created_at 
            FROM execution_io 
            WHERE interaction_id = %s 
            ORDER BY created_at ASC
            """
            
            if hasattr(self.db_manager, 'config_db_manager') and self.db_manager.config_db_manager.db_type == "sqlite":
                query = query.replace("%s", "?")
            
            result = self.db_manager.config_db_manager.execute_query(query, (self.session_id,))
            
            if result:
                for row in result:
                    try:
                        # 입력 데이터 파싱
                        input_data = json.loads(row['input_data']) if row['input_data'] else {}
                        output_data = json.loads(row['output_data']) if row['output_data'] else {}
                        
                        # 사용자 메시지 추가 (input_data에서 추출)
                        if input_data:
                            user_content = self._extract_content(input_data)
                            if user_content:
                                self._messages.append(HumanMessage(content=user_content))
                        
                        # AI 메시지 추가 (output_data에서 추출)
                        if output_data:
                            ai_content = self._extract_content(output_data)
                            if ai_content:
                                self._messages.append(AIMessage(content=ai_content))
                                
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Failed to parse message data: {e}")
                        
        except Exception as e:
            logger.error(f"Error loading chat history: {e}")
    
    def _extract_content(self, data) -> Optional[str]:
        """데이터에서 텍스트 내용 추출"""
        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            # result 필드가 있으면 그것을 사용 (직접적인 결과)
            if 'result' in data:
                result = data['result']
                if isinstance(result, str):
                    return result
                elif isinstance(result, dict):
                    return self._extract_content(result)
            # inputs 필드가 있으면 첫 번째 input 값 사용
            elif 'inputs' in data and data['inputs']:
                first_input = list(data['inputs'].values())[0]
                return str(first_input) if first_input else None
        return None
    
    @property
    def messages(self) -> List[BaseMessage]:
        return self._messages
    
    def add_message(self, message: BaseMessage) -> None:
        """메시지 추가"""
        self._messages.append(message)
    
    def clear(self) -> None:
        """히스토리 초기화"""
        self._messages.clear()

def create_conversation_function(config_composer, db_manager=None):
    """
    LangChain을 이용한 대화 함수 생성
    
    Args:
        config_composer: 설정 컴포저 객체
        db_manager: 데이터베이스 매니저 (선택적)
    
    Returns:
        대화 함수
    """
    
    def conversation(
        user_input: str,
        workflow_id: Optional[str] = None,
        workflow_name: Optional[str] = None,
        interaction_id: str = "default"
    ) -> Dict[str, Any]:
        """
        사용자 입력에 대한 AI 응답 생성
        
        Args:
            user_input: 사용자 입력
            workflow_id: 워크플로우 ID (선택적)
            workflow_name: 워크플로우 이름 (선택적)
            interaction_id: 상호작용 ID (기본값: "default")
        
        Returns:
            AI 응답과 메타데이터
        """
        try:
            # OpenAI API 키 가져오기
            openai_config = config_composer.get_config_by_name("OPENAI_API_KEY")
            api_key = openai_config.value
            
            if not api_key:
                raise ValueError("OpenAI API key not found")
            
            # ChatOpenAI 모델 초기화
            llm = ChatOpenAI(
                api_key=api_key,
                model="gpt-3.5-turbo",
                temperature=0.7
            )
            
            # 프롬프트 템플릿 설정
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI assistant. Maintain context from previous conversations."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}")
            ])
            
            # 체인 구성
            chain = prompt | llm
            
            # 세션 ID 결정 (interaction_id 우선)
            session_id = interaction_id if interaction_id != "default" else workflow_id or "default"
            
            # 대화 히스토리가 있는 경우 메모리 기능 사용
            if db_manager and session_id != "default":
                # 메모리 기반 대화 체인 생성
                def get_session_history(session_id: str) -> DatabaseChatHistory:
                    return DatabaseChatHistory(db_manager, session_id)
                
                conversation_chain = RunnableWithMessageHistory(
                    chain,
                    get_session_history,
                    input_messages_key="input",
                    history_messages_key="history",
                )
                
                # 대화 실행
                response = conversation_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}}
                )
            else:
                # 단순 응답 (히스토리 없음)
                response = chain.invoke({
                    "input": user_input,
                    "history": []
                })
            
            # 응답 반환
            return {
                "status": "success",
                "user_input": user_input,
                "ai_response": response.content,
                "session_id": session_id,
                "workflow_id": workflow_id,
                "workflow_name": workflow_name,
                "interaction_id": interaction_id,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in conversation: {e}")
            return {
                "status": "error",
                "user_input": user_input,
                "error_message": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    return conversation

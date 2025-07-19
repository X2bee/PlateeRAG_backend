from editor.node_composer import Node
import logging
import json
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

class DBMemoryNode(Node):
    categoryId = "utilities"
    functionId = "memory"
    nodeId = "memory/db_memory"
    nodeName = "DB Memory"
    description = "DB에서 대화 기록을 로드하여 ConversationBufferMemory로 반환하는 노드입니다."
    tags = ["memory", "database", "chat_history", "langchain"]

    inputs = []
    outputs = [
        {
            "id": "memory",
            "name": "Memory",
            "type": "OBJECT"
        },
    ]
    parameters = [
        {
            "id": "interaction_id",
            "name": "Interaction ID",
            "type": "STR",
            "value": "",
        },
    ]

    def _get_db_manager(self):
        """FastAPI 앱에서 DB 매니저를 가져오는 함수"""
        try:
            import sys
            if 'main' in sys.modules:
                main_module = sys.modules['main']
                if hasattr(main_module, 'app') and hasattr(main_module.app, 'state'):
                    if hasattr(main_module.app.state, 'rag_service'):
                        db_manager = main_module.app.state.app_db
                        self._cached_db_manager = db_manager
                        logger.info("main 모듈에서 DB 매니저를 찾았습니다.")
                        return db_manager

            for module_name, module in sys.modules.items():
                if hasattr(module, 'app'):
                    app = getattr(module, 'app')
                    if hasattr(app, 'state') and hasattr(app.state, 'app_db'):
                        db_manager = app.state.app_db
                        self._cached_db_manager = db_manager
                        logger.info(f"{module_name} 모듈에서 DB 매니저를 찾았습니다.")
                        return db_manager

            logger.warning("DB 매니저를 찾을 수 없습니다. 서버가 실행되지 않았을 수 있습니다.")
            return None

        except Exception as e:
            logger.error(f"DB 매니저 접근 중 오류: {e}")
            return None

    def _load_messages_from_db(self, interaction_id: str) -> List[Dict[str, str]]:
        """DB에서 대화 기록을 로드하여 메시지 리스트로 반환"""
        db_manager = self._get_db_manager()
        if not db_manager or interaction_id == "default":
            return []

        try:
            query = """
            SELECT input_data, output_data, created_at
            FROM execution_io
            WHERE interaction_id = %s
            ORDER BY created_at ASC
            """

            if hasattr(db_manager, 'config_db_manager') and db_manager.config_db_manager.db_type == "sqlite":
                query = query.replace("%s", "?")

            result = db_manager.config_db_manager.execute_query(query, (interaction_id,))

            messages = []
            if result:
                for row in result:
                    try:
                        input_data = json.loads(row['input_data']) if row['input_data'] else {}
                        output_data = json.loads(row['output_data']) if row['output_data'] else {}

                        if input_data:
                            user_content = self._extract_content(input_data)
                            if user_content:
                                messages.append({"role": "user", "content": user_content})

                        if output_data:
                            ai_content = self._extract_content(output_data)
                            if ai_content:
                                messages.append({"role": "ai", "content": ai_content})

                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Failed to parse message data: {e}")

            return messages

        except Exception as e:
            logger.error(f"Error loading chat history: {e}")
            return []

    def _extract_content(self, data) -> Optional[str]:
        """데이터에서 텍스트 내용 추출"""
        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            if 'result' in data:
                result = data['result']
                if isinstance(result, str):
                    return result
                elif isinstance(result, dict):
                    return self._extract_content(result)
            elif 'inputs' in data and data['inputs']:
                first_input = list(data['inputs'].values())[0]
                return str(first_input) if first_input else None
        return None

    def load_memory_from_db(self, db_messages: List[Dict[str, str]]):
        """
        DB에서 로드한 메시지 리스트를 기반으로 LangChain 메모리 객체를 생성합니다.

        Args:
            db_messages: {'role': 'user' | 'ai', 'content': '...'} 형태의 딕셔너리 리스트

        Returns:
            대화 기록이 채워진 ConversationBufferMemory 객체
        """
        try:
            from langchain.memory import ConversationBufferMemory
            from langchain_community.chat_message_histories import ChatMessageHistory

            chat_history = ChatMessageHistory()
            for msg in db_messages:
                if msg.get("role") == "user":
                    chat_history.add_user_message(msg.get("content"))
                elif msg.get("role") == "ai":
                    chat_history.add_ai_message(msg.get("content"))

            memory = ConversationBufferMemory(
                chat_memory=chat_history,
                memory_key="chat_history",
                return_messages=True
            )
            return memory

        except ImportError as e:
            logger.error(f"LangChain import error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error creating memory object: {e}")
            return None

    def execute(self, interaction_id: str):
        """
        DB에서 대화 기록을 로드하여 ConversationBufferMemory 객체를 반환합니다.

        Args:
            interaction_id: 상호작용 ID

        Returns:
            ConversationBufferMemory 객체 또는 오류 메시지
        """
        try:
            # DB에서 메시지 로드
            db_messages = self._load_messages_from_db(interaction_id)

            if not db_messages:
                logger.info(f"No chat history found for interaction_id: {interaction_id}")
                # 빈 메모리 객체 반환
                return self.load_memory_from_db([])

            # ConversationBufferMemory 객체 생성
            memory = self.load_memory_from_db(db_messages)

            if memory is None:
                return None

            logger.info(f"Successfully loaded {len(db_messages)} messages for interaction_id: {interaction_id}")
            return memory

        except Exception as e:
            logger.error(f"Error in execute: {e}")
            return None

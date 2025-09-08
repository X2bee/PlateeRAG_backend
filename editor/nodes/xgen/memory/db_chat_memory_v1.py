from editor.node_composer import Node
import logging
import json
from typing import List, Optional, Dict, Any
from editor.utils.helper.service_helper import AppServiceManager

logger = logging.getLogger(__name__)

class DBMemoryNode(Node):
    categoryId = "xgen"
    functionId = "memory"
    nodeId = "memory/db_memory"
    nodeName = "DB Memory v1"
    description = "DB에서 대화 기록을 로드하여 ConversationBufferMemory로 반환하는 노드입니다."
    tags = ["memory", "database", "chat_history", "xgen"]

    inputs = []
    outputs = [
        {"id": "memory", "name": "Memory", "type": "OBJECT"},
    ]
    parameters = [
        {"id": "interaction_id", "name": "Interaction ID", "type": "STR", "value": ""},
        {"id": "include_thinking", "name": "Include Thinking", "type": "BOOL", "value": False, "required": False, "optional": True},

    ]

    def _load_messages_from_db(self, interaction_id: str, include_thinking: bool = False) -> List[Dict[str, str]]:
        """DB에서 대화 기록을 로드하여 메시지 리스트로 반환"""
        db_manager = AppServiceManager.get_db_manager()
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
                            user_content = self._extract_content(input_data, include_thinking)
                            if user_content:
                                messages.append({"role": "user", "content": user_content})

                        if output_data:
                            ai_content = self._extract_content(output_data, include_thinking)
                            if ai_content:
                                messages.append({"role": "ai", "content": ai_content})

                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Failed to parse message data: {e}")

            return messages

        except Exception as e:
            logger.error(f"Error loading chat history: {e}")
            return []

    def _extract_content(self, data, include_thinking: bool = False) -> Optional[str]:
        """데이터에서 텍스트 내용 추출"""
        if isinstance(data, str):
            content = data
        elif isinstance(data, dict):
            if 'result' in data:
                result = data['result']
                if isinstance(result, str):
                    content = result
                elif isinstance(result, dict):
                    return self._extract_content(result, include_thinking)
                else:
                    return None
            elif 'inputs' in data and data['inputs']:
                first_input = list(data['inputs'].values())[0]
                content = str(first_input) if first_input else None
            else:
                return None
        else:
            return None

        if content:
            import re
            content = re.sub(r'\[Cite\.\s*\{[^}]*\}\]', '', content, flags=re.DOTALL | re.IGNORECASE)
            
            if not include_thinking:
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
            
            content = content.strip()

        return content if content else None

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

    def execute(self, interaction_id: str, include_thinking: bool = False):
        """
        DB에서 대화 기록을 로드하여 ConversationBufferMemory 객체를 반환합니다.

        Args:
            interaction_id: 상호작용 ID

        Returns:
            ConversationBufferMemory 객체 또는 오류 메시지
        """
        try:
            # DB에서 메시지 로드
            db_messages = self._load_messages_from_db(interaction_id, include_thinking)

            if not db_messages:
                logger.info(f"No chat history found for interaction_id: {interaction_id}")
                # 빈 메모리 객체 반환
                return self.load_memory_from_db([])

            memory = self.load_memory_from_db(db_messages)

            if memory is None:
                return None

            logger.info(f"Successfully loaded {len(db_messages)} messages for interaction_id: {interaction_id}")
            return memory

        except Exception as e:
            logger.error(f"Error in execute: {e}")
            return None

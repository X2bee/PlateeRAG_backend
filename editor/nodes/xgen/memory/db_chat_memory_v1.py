from editor.node_composer import Node
import logging
import json
from typing import List, Optional, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from editor.utils.helper.service_helper import AppServiceManager

logger = logging.getLogger(__name__)

class DBMemoryNode(Node):
    categoryId = "xgen"
    functionId = "memory"
    nodeId = "memory/db_memory_v1"
    nodeName = "DB Memory v1"
    description = "DB에서 대화 기록을 로드하여 메시지 리스트로 반환 (LangChain 1.0.0)"
    tags = ["memory", "database", "chat_history", "xgen", "langchain_1.0"]

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

    def convert_to_langchain_messages(self, db_messages: List[Dict[str, str]]) -> List[BaseMessage]:
        """
        DB에서 로드한 메시지 리스트를 LangChain 1.0.0 메시지 객체로 변환합니다.

        Args:
            db_messages: {'role': 'user' | 'ai', 'content': '...'} 형태의 딕셔너리 리스트

        Returns:
            LangChain BaseMessage 객체 리스트 (HumanMessage, AIMessage)
        """
        try:
            messages: List[BaseMessage] = []
            for msg in db_messages:
                role = msg.get("role")
                content = msg.get("content", "")

                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "ai":
                    messages.append(AIMessage(content=content))
                else:
                    logger.warning(f"Unknown message role: {role}. Skipping message.")

            return messages

        except Exception as e:
            logger.error(f"Error converting to LangChain messages: {e}")
            return []

    def execute(self, interaction_id: str, include_thinking: bool = False):
        """
        DB에서 대화 기록을 로드하여 LangChain 1.0.0 메시지 리스트를 반환합니다.

        Args:
            interaction_id: 상호작용 ID
            include_thinking: thinking 태그 포함 여부

        Returns:
            List[BaseMessage]: LangChain 1.0.0 메시지 리스트
        """
        try:
            # DB에서 메시지 로드
            db_messages = self._load_messages_from_db(interaction_id, include_thinking)

            if not db_messages:
                logger.info(f"No chat history found for interaction_id: {interaction_id}")
                return []

            # LangChain 1.0.0 메시지 객체로 변환
            messages = self.convert_to_langchain_messages(db_messages)

            logger.info(f"Successfully loaded {len(messages)} messages for interaction_id: {interaction_id}")

            return messages

        except Exception as e:
            logger.error(f"Error in execute: {e}")
            return []

from editor.node_composer import Node
import logging
import json
from typing import List, Optional, Dict, Any
from editor.utils.helper.service_helper import AppServiceManager
import re
from collections import Counter
import math

logger = logging.getLogger(__name__)

class DBMemoryNode(Node):
    categoryId = "xgen"
    functionId = "memory"
    nodeId = "memory/db_memory"
    nodeName = "DB Memory"
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

    def _preprocess_text(self, text: str) -> List[str]:
        """텍스트를 전처리하여 키워드 리스트로 반환"""
        if not text:
            return []
        
        # 소문자 변환 및 특수문자 제거
        text = re.sub(r'[^\w\s가-힣]', ' ', text.lower())
        # 단어 분리 (한글, 영문 모두 지원)
        words = text.split()
        # 길이가 2 이상인 단어만 유지
        words = [word for word in words if len(word) >= 2]
        
        return words

    def _calculate_tf_idf(self, query_words: List[str], document_words: List[str], all_documents: List[List[str]]) -> float:
        """TF-IDF 기반 유사도 계산"""
        if not query_words or not document_words:
            return 0.0
        
        # TF (Term Frequency) 계산
        doc_word_count = Counter(document_words)
        query_word_count = Counter(query_words)
        
        total_words = len(document_words)
        total_docs = len(all_documents)
        
        score = 0.0
        
        for word in set(query_words):
            if word in doc_word_count:
                # TF 계산
                tf = doc_word_count[word] / total_words
                
                # IDF 계산 (해당 단어가 포함된 문서 수)
                docs_containing_word = sum(1 for doc in all_documents if word in doc)
                idf = math.log(total_docs / (docs_containing_word + 1))
                
                # TF-IDF 스코어
                tfidf = tf * idf
                score += tfidf
                
        return score

    def _calculate_message_relevance(self, current_input: str, historical_messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """현재 입력과 과거 메시지들 간의 관련도 계산"""
        if not current_input or not historical_messages:
            return []
        
        query_words = self._preprocess_text(current_input)
        if not query_words:
            return []
        
        # 모든 문서(메시지)의 단어 리스트 생성
        all_documents = []
        for msg in historical_messages:
            content = msg.get('content', '')
            doc_words = self._preprocess_text(content)
            all_documents.append(doc_words)
        
        # 각 메시지에 대해 관련도 계산
        messages_with_relevance = []
        for i, msg in enumerate(historical_messages):
            content = msg.get('content', '')
            doc_words = all_documents[i]
            
            relevance_score = self._calculate_tf_idf(query_words, doc_words, all_documents)
            
            messages_with_relevance.append({
                'role': msg.get('role'),
                'content': content,
                'relevance_score': relevance_score
            })
        
        # 관련도 순으로 정렬
        messages_with_relevance.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return messages_with_relevance

    def _simple_message_summary(self, messages: List[Dict[str, Any]], max_length: int = 500) -> str:
        """LLM을 사용할 수 없을 때 간단한 요약 방식"""
        if not messages:
            return ""
        
        summary_parts = []
        current_length = 0
        
        for msg in messages:
            role = "사용자" if msg['role'] == "user" else "AI"
            content = msg['content']
            
            # 긴 내용은 줄임
            if len(content) > 100:
                content = content[:100] + "..."
            
            part = f"{role}: {content}"
            
            if current_length + len(part) > max_length:
                break
                
            summary_parts.append(part)
            current_length += len(part)
        
        return "\n".join(summary_parts)

    def _select_and_summarize_relevant_messages(self, current_input: str, historical_messages: List[Dict[str, str]], n_messages: int = 5, llm=None) -> str:
        """현재 입력과 관련된 메시지들을 선택하고 요약 (연결된 agent 모델 사용)"""
        if not historical_messages:
            return ""
        
        # 관련도 계산
        messages_with_relevance = self._calculate_message_relevance(current_input, historical_messages)
        
        # 상위 n개 메시지 선택
        top_relevant_messages = messages_with_relevance[:n_messages]
        
        # 관련도가 0인 메시지들은 제외
        meaningful_messages = [msg for msg in top_relevant_messages if msg['relevance_score'] > 0]
        
        if not meaningful_messages:
            logger.info("No relevant messages found for current input")
            return ""
        
        # 선택된 메시지들에 대한 상세 로그
        for i, msg in enumerate(meaningful_messages, 1):
            role = msg['role']
            content = msg['content']
            score = msg['relevance_score']
            # 긴 메시지는 앞부분만 로그에 출력
            content_preview = content[:100] + "..." if len(content) > 100 else content
            logger.info(f"  [{i}] {role} (score: {score:.4f}): {content_preview}")
        
        logger.info(f"Current input for comparison: {current_input}")
        
        # agent에서 전달받은 LLM을 사용하여 요약
        if llm:
            try:
                # 요약할 메시지들 구성 (user-ai 쌍으로 그룹화)
                conversation_pairs = []
                current_pair = {}
                
                for msg in meaningful_messages:
                    if msg['role'] == "user":
                        # 이전 쌍이 완성되었다면 저장
                        if current_pair:
                            conversation_pairs.append(current_pair)
                        current_pair = {"user": msg['content'], "ai": ""}
                    elif msg['role'] == "ai" and current_pair:
                        current_pair["ai"] = msg['content']
                
                # 마지막 쌍 저장
                if current_pair:
                    conversation_pairs.append(current_pair)
                
                # 대화 쌍을 텍스트로 변환
                conversation_text = ""
                for i, pair in enumerate(conversation_pairs, 1):
                    conversation_text += f"[대화 {i}]\n"
                    conversation_text += f"사용자: {pair.get('user', '')}\n"
                    if pair.get('ai'):
                        conversation_text += f"AI: {pair['ai']}\n"
                    conversation_text += "\n"
                
                logger.info(f"Grouped conversation pairs for summarization:\n{conversation_text}")
                
                summary_prompt = f"""다음은 이전 대화 내용입니다. 현재 사용자의 질문과 관련된 핵심 내용만 간결하게 요약해주세요.

{conversation_text}

요약 조건:
1. 현재 질문과 관련된 중요한 정보만 포함
2. 각 대화 쌍의 핵심 내용을 간결하게 정리
3. 3-4문장으로 전체를 요약
4. 질문-답변의 맥락을 유지하면서 요약

요약:"""
                
                response = llm.invoke(summary_prompt)
                summary = response.content.strip()
                
                logger.info(f"Generated summary: {summary}")
                return summary
                
            except Exception as e:
                logger.warning(f"LLM summarization failed: {e}. Using simple concatenation.")
                fallback_summary = self._simple_message_summary(meaningful_messages)
                return fallback_summary
        else:
            # LLM이 전달되지 않은 경우 간단한 요약 사용
            logger.info("No LLM provided, using simple message summary")
            simple_summary = self._simple_message_summary(meaningful_messages)
            return simple_summary

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

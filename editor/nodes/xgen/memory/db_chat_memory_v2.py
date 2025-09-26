from editor.node_composer import Node
import logging
import json
from typing import List, Optional, Dict, Any
from langchain.schema import SystemMessage, HumanMessage
from editor.utils.helper.service_helper import AppServiceManager
import re
from collections import Counter
import math
from functools import lru_cache

logger = logging.getLogger(__name__)

class DBMemoryNode(Node):
    categoryId = "xgen"
    functionId = "memory"
    nodeId = "memory/db_memory_v2"
    nodeName = "DB Memory V2"
    description = "DB에서 대화 기록을 로드하여 ConversationBufferMemory로 반환하는 노드입니다."
    tags = ["memory", "database", "chat_history", "xgen"]

    inputs = []
    outputs = [
        {"id": "memory", "name": "Memory", "type": "OBJECT"},
    ]
    parameters = [
        {"id": "interaction_id", "name": "Interaction ID", "type": "STR", "value": ""},
        {"id": "include_thinking", "name": "Include Thinking", "type": "BOOL", "value": False, "required": False, "optional": True},
        {"id": "top_n_messages", "name": "Top N Messages", "type": "INT", "value": 10, "required": False, "optional": True},
        {"id": "min_tokens_for_similarity", "name": "Min Tokens For Similarity", "type": "INT", "value": 0, "required": False, "optional": True},
        {"id": "enable_similarity_filter", "name": "Enable Similarity Filter", "type": "BOOL", "value": False, "required": False, "optional": True},
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
            content = self._sanitize_content(content, include_thinking)

        return content if content else None

    def _sanitize_content(self, content: str, include_thinking: bool = False) -> str:
        import re
        content = re.sub(r'\[Cite\.\s*\{[^}]*\}\]', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r"<FEEDBACK_(LOOP|RESULT|STATUS)>.*?</FEEDBACK_(LOOP|RESULT|STATUS)>", '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r"<TODO_DETAILS>.*?</TODO_DETAILS>", '', content, flags=re.DOTALL | re.IGNORECASE)

        if not include_thinking:
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)

        content = re.sub(r'\n{3,}', '\n\n', content)
        content = content.strip()
        if not include_thinking:
            content = re.sub(r'<TOOLUSELOG>.*?</TOOLUSELOG>', '', content, flags=re.DOTALL | re.IGNORECASE)
            content = re.sub(r'<TOOLOUTPUTLOG>.*?</TOOLOUTPUTLOG>', '', content, flags=re.DOTALL | re.IGNORECASE)

        return content

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

    def _calculate_bm25(
        self,
        query_words: List[str],
        document_words: List[str],
        avg_doc_len: float,
        doc_freq_cache: Dict[str, int],
        total_docs: int,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> float:
        """간단한 BM25 스코어 계산"""
        if not query_words or not document_words:
            return 0.0

        doc_len = len(document_words) or 1
        word_counts = Counter(document_words)
        score = 0.0

        for word in set(query_words):
            freq = word_counts.get(word)
            if not freq:
                continue

            doc_freq = max(doc_freq_cache.get(word, 0), 1)
            idf = math.log((total_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1)

            numerator = freq * (k1 + 1)
            denominator = freq + k1 * (1 - b + b * (doc_len / avg_doc_len))
            score += idf * (numerator / denominator)

        return score

    def _calculate_message_relevance(
        self,
        current_input: str,
        historical_messages: List[Dict[str, str]],
        embedding_fn=None,
    ) -> List[Dict[str, Any]]:
        """현재 입력과 과거 메시지들 간의 관련도 계산"""
        if not current_input or not historical_messages:
            return []

        if embedding_fn:
            try:
                target_embedding = embedding_fn(current_input)
                if target_embedding is None:
                    raise ValueError("embedding_fn returned None")

                scored_messages = []
                for msg in historical_messages:
                    content = msg.get("content", "")
                    msg_embedding = embedding_fn(content)
                    if msg_embedding is None:
                        similarity = 0.0
                    else:
                        similarity = self._cosine_similarity(target_embedding, msg_embedding)
                    scored_messages.append({
                        "role": msg.get("role"),
                        "content": content,
                        "relevance_score": similarity,
                    })

                scored_messages.sort(key=lambda x: x["relevance_score"], reverse=True)
                return scored_messages
            except Exception as embedding_error:
                logger.warning(
                    "Embedding relevance calculation failed: %s. Falling back to lexical similarity.",
                    embedding_error,
                )

        query_words = self._preprocess_text(current_input)
        if not query_words:
            return []

        document_word_lists = []
        doc_freq_cache: Dict[str, int] = {}

        for msg in historical_messages:
            content = msg.get("content", "")
            doc_words = self._preprocess_text(content)
            document_word_lists.append(doc_words)
            unique_terms = set(doc_words)
            for term in unique_terms:
                doc_freq_cache[term] = doc_freq_cache.get(term, 0) + 1

        total_docs = max(len(document_word_lists), 1)
        avg_doc_len = sum(len(doc) for doc in document_word_lists) / total_docs if total_docs else 1.0

        messages_with_relevance = []
        for doc_words, message in zip(document_word_lists, historical_messages):
            score = self._calculate_bm25(
                query_words,
                doc_words,
                avg_doc_len,
                doc_freq_cache,
                total_docs,
            )
            messages_with_relevance.append({
                "role": message.get("role"),
                "content": message.get("content", ""),
                "relevance_score": score,
            })

        messages_with_relevance.sort(key=lambda x: x["relevance_score"], reverse=True)
        return messages_with_relevance

    def _resolve_embedding_function(self):
        """AppServiceManager에서 임베딩 서비스를 가져와 callable 형태로 반환"""
        try:
            get_service = getattr(AppServiceManager, "get_embedding_service", None)
            if not callable(get_service):
                return None

            embedding_service = get_service()
            if not embedding_service:
                return None

            if hasattr(embedding_service, "get_embedding"):
                def _embed(text: str):
                    if not text:
                        return None
                    try:
                        vector = embedding_service.get_embedding(text)
                        if vector is None:
                            return None
                        return list(vector)
                    except Exception as embedding_error:
                        logger.warning(
                            "Embedding service get_embedding failed: %s", embedding_error
                        )
                        return None

                return _embed

            return None
        except Exception as service_error:
            logger.warning(
                "Unable to resolve embedding function: %s", service_error
            )
            return None

    @staticmethod
    def _cosine_similarity(vec1, vec2) -> float:
        if not vec1 or not vec2:
            return 0.0

        try:
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = math.sqrt(sum(a * a for a in vec1))
            norm2 = math.sqrt(sum(b * b for b in vec2))
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot_product / (norm1 * norm2)
        except Exception:
            return 0.0

    @staticmethod
    @lru_cache(maxsize=1)
    def _get_token_encoder():
        """tiktoken 인코더를 캐싱해서 재사용"""
        try:
            import tiktoken
        except ImportError:
            logger.warning("tiktoken library not available; falling back to regex token estimate")
            return None

        try:
            return tiktoken.get_encoding("o200k_base")
        except Exception:
            logger.warning("Unable to load tiktoken encoder; falling back to regex token estimate")
            return None

    def _estimate_token_count(self, text: str) -> int:
        """토큰 수를 tiktoken 기반으로 계산하고 실패 시 간단 추정 사용"""
        if not text:
            return 0

        encoder = self._get_token_encoder()
        if encoder:
            try:
                return len(encoder.encode(text))
            except Exception as exc:
                logger.warning(f"tiktoken encoding failed, falling back to regex estimate: {exc}")

        tokens = re.findall(r"[가-힣A-Za-z0-9']+|[^\w\s]", text)
        return len(tokens)

    def _count_tokens_in_pairs(self, pairs: List[Dict[str, str]], limit: Optional[int] = None) -> int:
        """대화 쌍 리스트에서 토큰 수 추정"""
        if not pairs:
            return 0

        relevant_pairs = pairs[-limit:] if limit and limit > 0 else pairs
        total_tokens = 0

        for pair in relevant_pairs:
            total_tokens += self._estimate_token_count(pair.get("user", ""))
            total_tokens += self._estimate_token_count(pair.get("ai", ""))

        return total_tokens

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

    def _select_and_summarize_relevant_messages(
        self,
        current_input: str,
        historical_messages: List[Dict[str, str]],
        n_messages: int = 5,
        llm=None,
        use_similarity_filter: bool = True,
    ) -> str:
        """현재 입력과 관련된 메시지들을 선택하고 요약 (연결된 agent 모델 사용)"""
        if not historical_messages:
            return ""
        
        # 유사도 필터링 사용 여부에 따라 메시지 선택
        if use_similarity_filter:
            embedding_fn = self._resolve_embedding_function()

            messages_with_relevance = self._calculate_message_relevance(
                current_input,
                historical_messages,
                embedding_fn=embedding_fn,
            )

            top_relevant_messages = messages_with_relevance[: n_messages * 2]

            meaningful_messages = [
                msg for msg in top_relevant_messages if msg["relevance_score"] > 0
            ]

            if not meaningful_messages:
                logger.info(
                    "All relevance scores zero; falling back to recent messages for context."
                )
                recent_slice = historical_messages[-n_messages:]
                meaningful_messages = [
                    {"role": msg["role"], "content": msg["content"], "relevance_score": 0.0}
                    for msg in recent_slice
                ]
        else:
            meaningful_messages = [
                {
                    "role": msg["role"],
                    "content": msg["content"],
                    "relevance_score": 1.0,
                }
                for msg in historical_messages
            ]
        
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
                logger.info(
                    "Processing %s meaningful messages for conversation pairing:",
                    len(meaningful_messages),
                )
                for i, msg in enumerate(meaningful_messages):
                    preview = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
                    logger.info(
                        "  Message %s: role=%s, content='%s'",
                        i + 1,
                        msg["role"],
                        preview,
                    )

                conversation_pairs = []
                current_pair = {}

                for msg in meaningful_messages:
                    role = msg["role"]
                    content = msg.get("content", "")

                    if role == "user":
                        if current_pair:
                            conversation_pairs.append(current_pair)
                        current_pair = {"user": content, "ai": ""}
                    elif role == "ai":
                        if current_pair:
                            current_pair["ai"] = content
                        else:
                            current_pair = {"user": "", "ai": content}

                if current_pair:
                    conversation_pairs.append(current_pair)

                logger.info("Created %s conversation pairs", len(conversation_pairs))

                conversation_text = ""
                char_limit = 4000
                for i, pair in enumerate(conversation_pairs, 1):
                    segment = f"[대화 {i}]\n"
                    user_content = pair.get("user", "").strip()
                    ai_content = pair.get("ai", "").strip()

                    if user_content:
                        segment += f"사용자: {user_content}\n"
                    if ai_content:
                        segment += f"AI: {ai_content}\n"
                    segment += "\n"

                    if len(conversation_text) + len(segment) > char_limit:
                        remaining = char_limit - len(conversation_text)
                        if remaining > 0:
                            conversation_text += segment[:remaining]
                        break

                    conversation_text += segment

                logger.info(
                    "Generated conversation text (%s characters)",
                    len(conversation_text),
                )

                if not conversation_text.strip():
                    logger.warning(
                        "Conversation text is empty! Trying simple message concatenation"
                    )
                    conversation_text = ""
                    for i, msg in enumerate(meaningful_messages, 1):
                        role = "사용자" if msg["role"] == "user" else "AI"
                        conversation_text += f"[메시지 {i}] {role}: {msg['content']}\n\n"

                if not conversation_text.strip():
                    logger.warning(
                        "All conversation text generation failed! Using simple summary"
                    )
                    return self._simple_message_summary(meaningful_messages)

                summary_prompt_messages = [
                    SystemMessage(
                        content="이전 대화에서 현재 사용자 질문 해결에 필요한 정보를 5-7문장으로 정리하세요."
                    ),
                    HumanMessage(content=conversation_text),
                ]

                response = llm.invoke(summary_prompt_messages)
                summary = getattr(response, "content", "").strip()

                if not summary:
                    raise ValueError("Empty summary response")

                logger.info(f"Generated summary: {summary}")
                return summary

            except Exception as e:
                logger.warning(
                    "LLM summarization failed: %s. Using simple concatenation.", e
                )
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

    def _filter_messages_by_similarity(
        self,
        messages: List[Dict[str, str]],
        current_input: str,
        top_n: int,
        min_tokens_for_similarity: int = 0,
    ) -> List[Dict[str, str]]:
        """현재 입력과 가장 유사한 상위 n개의 사용자-응답 쌍을 기반으로 메시지 필터링"""
        if not current_input or not messages or top_n <= 0:
            return messages

        conversation_pairs = self._group_messages_into_pairs(messages)
        if not conversation_pairs:
            return messages

        embedding_fn = self._resolve_embedding_function()
        query_embedding = None
        if embedding_fn:
            try:
                query_embedding = embedding_fn(current_input)
                if not query_embedding:
                    embedding_fn = None
            except Exception as embedding_error:
                logger.warning(
                    "Failed to create query embedding: %s. Falling back to lexical scoring.",
                    embedding_error,
                )
                embedding_fn = None

        query_words = self._preprocess_text(current_input) if not embedding_fn else []
        if not embedding_fn and not query_words:
            return messages

        if min_tokens_for_similarity and min_tokens_for_similarity > 0:
            token_count = self._count_tokens_in_pairs(conversation_pairs, limit=top_n)
            if token_count < min_tokens_for_similarity:
                logger.info(
                    "Skipping similarity filtering because token count %s is below threshold %s",
                    token_count,
                    min_tokens_for_similarity,
                )
                return messages

        processed_pairs = []
        all_documents = []

        for index, pair in enumerate(conversation_pairs):
            combined_text = " ".join(filter(None, [pair.get("user", ""), pair.get("ai", "")])).strip()
            if not combined_text:
                continue

            doc_words = self._preprocess_text(combined_text)
            if not doc_words:
                continue

            processed_pairs.append({
                "index": index,
                "pair": pair,
                "doc_words": doc_words,
                "combined_text": combined_text,
            })
            all_documents.append(doc_words)

        if not processed_pairs:
            return messages

        total_docs = max(len(all_documents), 1)
        avg_doc_len = (
            sum(len(doc) for doc in all_documents) / total_docs if total_docs else 1.0
        )
        doc_freq_cache: Dict[str, int] = {}
        for doc in all_documents:
            for term in set(doc):
                doc_freq_cache[term] = doc_freq_cache.get(term, 0) + 1

        for entry in processed_pairs:
            if embedding_fn and query_embedding:
                pair_embedding = embedding_fn(entry["combined_text"])
                relevance_score = (
                    self._cosine_similarity(query_embedding, pair_embedding)
                    if pair_embedding
                    else 0.0
                )
            else:
                relevance_score = self._calculate_bm25(
                    query_words,
                    entry["doc_words"],
                    avg_doc_len,
                    doc_freq_cache,
                    total_docs,
                )
            entry["relevance_score"] = relevance_score

        meaningful_pairs = [
            entry for entry in processed_pairs if entry["relevance_score"] > 0
        ]
        meaningful_pairs.sort(key=lambda x: x["relevance_score"], reverse=True)

        top_pairs = meaningful_pairs[:top_n]
        if not top_pairs:
            logger.info("No relevant pairs identified; reverting to most recent pairs")
            start_index = max(len(conversation_pairs) - top_n, 0)
            fallback_pairs = conversation_pairs[start_index:]
            top_pairs = [
                {"index": start_index + idx, "pair": pair, "relevance_score": 0.0}
                for idx, pair in enumerate(fallback_pairs)
            ]

        # 시간 순서를 유지하기 위해 원래 인덱스 순으로 재정렬
        top_pairs.sort(key=lambda x: x["index"])

        filtered_messages = []
        for entry in top_pairs:
            user_content = entry["pair"].get("user", "").strip()
            ai_content = entry["pair"].get("ai", "").strip()

            if user_content:
                filtered_messages.append({"role": "user", "content": user_content})
            if ai_content:
                filtered_messages.append({"role": "ai", "content": ai_content})

        logger.info(
            "Filtered %s conversation pairs to top %s most relevant pairs",
            len(conversation_pairs),
            len(top_pairs)
        )

        return filtered_messages
    
    def _group_messages_into_pairs(self, chat_history):
        """연속된 메시지들을 user-ai 대화 쌍으로 묶어서 반환"""
        if not chat_history:
            return []

        pairs = []
        current_pair = {"user": [], "ai": []}

        def finalize_pair():
            user_text = " ".join(current_pair["user"]).strip()
            ai_text = " ".join(current_pair["ai"]).strip()
            if user_text or ai_text:
                pairs.append({"user": user_text, "ai": ai_text})
            current_pair["user"].clear()
            current_pair["ai"].clear()

        for msg in chat_history:
            role = None
            content = ""

            if isinstance(msg, dict):
                role = msg.get("role")
                content = (msg.get("content") or "").strip()
            else:
                msg_type = getattr(msg, "type", None)
                raw_content = getattr(msg, "content", "")
                content = raw_content.strip() if isinstance(raw_content, str) else ""
                if msg_type == "human":
                    role = "user"
                elif msg_type == "ai":
                    role = "ai"
                else:
                    role = msg_type

            if not role or not content:
                continue

            if role == "user":
                if current_pair["user"] and current_pair["ai"]:
                    finalize_pair()
                elif current_pair["ai"] and not current_pair["user"]:
                    finalize_pair()
                current_pair["user"].append(content)
            elif role == "ai":
                if current_pair["user"] and current_pair["ai"]:
                    finalize_pair()
                current_pair["ai"].append(content)
            else:
                continue

        finalize_pair()
        return pairs


    def execute(
        self,
        interaction_id: str,
        include_thinking: bool = False,
        top_n_messages: int = 10,
        min_tokens_for_similarity: int = 0,
        enable_similarity_filter: bool = False,
        current_input: str = "",
    ):
        """
        DB에서 대화 기록을 로드하여 ConversationBufferMemory 객체를 반환합니다.

        Args:
            interaction_id: 상호작용 ID
            include_thinking: thinking 태그 포함 여부
            top_n_messages: 유사도 필터링 시 선택할 메시지 수
            min_tokens_for_similarity: 유사도 필터링을 적용하기 위한 최소 토큰 수
            enable_similarity_filter: 유사도 기반 필터링 활성화 여부
            current_input: 현재 사용자 입력 (유사도 계산 기준)

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

            # 유사도 필터링 적용 (활성화된 경우)
            if enable_similarity_filter and current_input:
                original_messages = db_messages
                filtered_messages = self._filter_messages_by_similarity(
                    db_messages,
                    current_input,
                    top_n_messages,
                    min_tokens_for_similarity,
                )
                if filtered_messages is original_messages:
                    logger.info(
                        "Similarity filtering skipped due to token threshold or insufficient relevance"
                    )
                else:
                    logger.info(f"Applied similarity filtering with top {top_n_messages} messages")
                db_messages = filtered_messages

            memory = self.load_memory_from_db(db_messages)

            if memory is None:
                return None

            logger.info(f"Successfully loaded {len(db_messages)} messages for interaction_id: {interaction_id}")
            return memory

        except Exception as e:
            logger.error(f"Error in execute: {e}")
            return None

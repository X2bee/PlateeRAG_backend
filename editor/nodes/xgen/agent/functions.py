import re
import logging
from editor.utils.helper.async_helper import sync_run_async
from editor.utils.citation_prompt import citation_prompt
logger = logging.getLogger(__name__)

def rag_context_builder(text, rag_context, strict_citation=True):
    additional_rag_context = ""
    search_params = rag_context.get('search_params', {})
    rerank_flag = search_params.get('rerank', False)
    rerank_top_k = search_params.get('rerank_top_k', search_params.get('top_k', 10))

    if search_params.get('use_model_prompt', False):
        query = search_params.get('embedding_model_prompt', '') + text
    else:
        query = text

    search_result = sync_run_async(rag_context['rag_service'].search_documents(
        collection_name=search_params.get('collection_name'),
        query_text=query,
        limit=search_params.get('top_k', 5),
        score_threshold=search_params.get('score_threshold', 0.3),
        rerank=rerank_flag,
        rerank_top_k=rerank_top_k
    ))
    results = search_result.get("results", [])
    if results:
        context_parts = []
        for i, item in enumerate(results, 1):
            if "chunk_text" in item and item["chunk_text"]:
                item_file_name = item.get("file_name", "Unknown")
                item_file_path = item.get("file_path", "Unknown")
                item_page_number = item.get("page_number", 0)
                item_line_start = item.get("line_start", 0)
                item_line_end = item.get("line_end", 0)

                score = item.get("score", 0.0)
                chunk_text = item["chunk_text"]
                context_parts.append(f"[문서 {i}](관련도: {score:.3f})\n[파일명] {item_file_name}\n[파일경로] {item_file_path}\n[페이지번호] {item_page_number}\n[문장시작줄] {item_line_start}\n[문장종료줄] {item_line_end}\n\n[내용]\n{chunk_text}")
        if context_parts:
            if strict_citation:
                context_parts.append(f"{citation_prompt}")
            context_text = "\n".join(context_parts)
            additional_rag_context = f"""{rag_context['search_params']['enhance_prompt']}
{context_text}"""

    return additional_rag_context

def _sanitize_tool_name(name):
    if not name:
        return "unnamed_tool"

    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    sanitized = re.sub(r'_+', '_', sanitized)
    sanitized = sanitized.strip('_')

    if not sanitized:
        sanitized = "unnamed_tool"

    return sanitized

def _validate_and_fix_tools(tools_list):
    if not tools_list:
        return []

    validated_tools = []
    for tool in tools_list:
        if hasattr(tool, 'name'):
            original_name = tool.name
            sanitized_name = _sanitize_tool_name(original_name)

            if original_name != sanitized_name:
                logger.warning(f"Tool name sanitized: '{original_name}' -> '{sanitized_name}'")
                tool.name = sanitized_name

        if hasattr(tool, 'func') and hasattr(tool.func, '__name__'):
            original_func_name = tool.func.__name__
            sanitized_func_name = _sanitize_tool_name(original_func_name)

            if original_func_name != sanitized_func_name:
                logger.warning(f"Tool func name sanitized: '{original_func_name}' -> '{sanitized_func_name}'")
                tool.func.__name__ = sanitized_func_name
        validated_tools.append(tool)
    return validated_tools

def _flatten_tools_list(tools_list):
    """중첩된 리스트 구조의 tools를 평탄화하여 단일 리스트로 변환"""
    if not tools_list:
        return []

    flattened = []
    for item in tools_list:
        if isinstance(item, list):
            # 재귀적으로 중첩된 리스트를 평탄화
            flattened.extend(_flatten_tools_list(item))
        else:
            flattened.append(item)

    logger.info(f"Tools flattened: {len(tools_list)} items -> {len(flattened)} items")
    return flattened

def prepare_llm_components(text, tools, memory, model, temperature, max_tokens, base_url, n_messages, streaming=True):
    from langchain_openai import ChatOpenAI
    from editor.utils.helper.service_helper import AppServiceManager

    config_composer = AppServiceManager.get_config_composer()
    if not config_composer:
        raise ValueError("No Config Composer set.")

    llm_provider = config_composer.get_config_by_name("DEFAULT_LLM_PROVIDER").value
    if llm_provider == "openai":
        api_key = config_composer.get_config_by_name("OPENAI_API_KEY").value
        print(api_key)
        if not api_key:
            logger.error(f"OpenAI API Key is not set")
            raise ValueError("OpenAI API Key is not set.")

    elif llm_provider == "vllm":
        api_key = None
        logger.info(f"vLLM API Key is set to None [currently not used]")

        # TODO: vLLM API 키 설정 로직 추가
        # api_key = config_composer.get_config_by_name("VLLM_API_KEY").value
        # if not api_key:
        #     return "vLLM API Key is not set."
    else:
        logger.error(f"Unsupported LLM Provider: {llm_provider}")
        raise ValueError(f"Unsupported LLM Provider: {llm_provider}")

    llm = ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens, base_url=base_url, streaming=streaming)

    tools_list = []
    if tools:
        tools_list = tools if isinstance(tools, list) else [tools]
        tools_list = _flatten_tools_list(tools_list)
        # OpenAI API 요구사항에 맞게 tool name 검증 및 수정
        tools_list = _validate_and_fix_tools(tools_list)

    chat_history = []
    if memory:
        try:
            optimized_chat_history = prepare_optimized_chat_history(memory, text, n_messages, llm)
            chat_history = optimized_chat_history
            logger.info(f"Using optimized multiturn memory with {len(chat_history)} messages")
        except Exception as e:
            logger.warning(f"Failed to use optimized memory, using standard: {e}")
            chat_history = []

    return llm, tools_list, chat_history

def _group_messages_into_pairs(chat_history):
    """연속된 메시지들을 user-ai 쌍으로 그룹핑"""
    if not chat_history:
        return []

    grouped_messages = []
    current_user_messages = []
    current_ai_messages = []

    for msg in chat_history:
        if not hasattr(msg, 'type') or not hasattr(msg, 'content'):
            continue

        msg_type = msg.type
        content = msg.content.strip()

        if not content:
            continue

        if msg_type == "human":  # 사용자 메시지
            # AI 메시지가 쌓여있다면 이전 쌍을 완성
            if current_ai_messages:
                if current_user_messages:  # 이전 사용자 메시지가 있었다면
                    user_content = " ".join(current_user_messages)
                    ai_content = " ".join(current_ai_messages)
                    grouped_messages.append({
                        'role': 'user',
                        'content': user_content
                    })
                    grouped_messages.append({
                        'role': 'ai',
                        'content': ai_content
                    })
                    current_user_messages = []
                    current_ai_messages = []

            current_user_messages.append(content)

        elif msg_type == "ai":  # AI 메시지
            current_ai_messages.append(content)

    # 마지막 메시지들 처리
    if current_user_messages and current_ai_messages:
        user_content = " ".join(current_user_messages)
        ai_content = " ".join(current_ai_messages)
        grouped_messages.append({
            'role': 'user',
            'content': user_content
        })
        grouped_messages.append({
            'role': 'ai',
            'content': ai_content
        })
    elif current_user_messages:  # AI 답변이 없는 사용자 메시지만 있는 경우
        user_content = " ".join(current_user_messages)
        grouped_messages.append({
            'role': 'user',
            'content': user_content
        })

    return grouped_messages

def prepare_optimized_chat_history(memory, current_input, n_messages, llm):
    """최적화된 chat_history 생성 - 기존 inputs 구조와 호환"""
    if not memory:
        return []

    try:
        from editor.nodes.xgen.memory.db_chat_memory_v2 import DBMemoryNode

        # 메모리에서 모든 대화 기록 추출
        memory_vars = memory.load_memory_variables({})
        full_chat_history = memory_vars.get("chat_history", [])

        if not full_chat_history:
            return []

        # 메시지를 user-ai 쌍으로 그룹핑
        historical_messages = _group_messages_into_pairs(full_chat_history)

        if not historical_messages:
            logger.info("No grouped messages found, using original chat history")
            return full_chat_history[-n_messages:] if n_messages > 0 else []

        logger.info(f"Grouped {len(full_chat_history)} raw messages into {len(historical_messages)} processed messages")

        # DBMemoryNode 인스턴스 생성하여 최적화된 요약 생성
        db_memory_node = DBMemoryNode()
        optimized_summary = db_memory_node._select_and_summarize_relevant_messages(
            current_input=current_input,
            historical_messages=historical_messages,
            n_messages=n_messages,
            llm=llm
        )

        # 요약이 있으면 시스템 메시지로 변환하여 반환
        if optimized_summary and optimized_summary.strip():
            from langchain_core.messages import SystemMessage
            return [SystemMessage(content=f"이전 대화 요약: {optimized_summary}")]
        else:
            # 요약이 없거나 빈 문자열이면 최근 n_messages만 반환 (기존 방식)
            return full_chat_history[-n_messages:] if n_messages > 0 else []

    except Exception as e:
        logger.error(f"Error in prepare_optimized_chat_history: {e}")
        # 오류 발생 시 기존 방식으로 fallback
        memory_vars = memory.load_memory_variables({})
        chat_history = memory_vars.get("chat_history", [])
        return chat_history[-n_messages:] if n_messages > 0 else []

def create_json_output_prompt(args_schema, original_prompt):
    from langchain_core.output_parsers import JsonOutputParser
    parser = JsonOutputParser(pydantic_object=args_schema)
    format_instructions = parser.get_format_instructions()
    escaped_instructions = format_instructions.replace("{", "{{").replace("}", "}}")
    return f"{original_prompt}\n\n{escaped_instructions}"

def create_tool_context_prompt(additional_rag_context, default_prompt, n_messages, memory=None, current_input=None, llm=None, use_optimization=True):
    """
    Tool context prompt 생성 - 메모리가 있을 때만 최적화 기능 사용

    Args:
        use_optimization: 최적화 기능 사용 여부 (기본값: True)
        memory: 메모리 객체 (최적화 사용 시 필요)
        current_input: 현재 입력 텍스트 (최적화 사용 시 필요)
        llm: LLM 객체 (최적화 사용 시 필요)
    """
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

    # 메모리가 있고 최적화 기능을 사용하는 경우
    if use_optimization and memory and current_input and llm:
        try:
            return create_optimized_tool_context_prompt(additional_rag_context, default_prompt, memory, current_input, n_messages, llm)
        except Exception as e:
            logger.warning(f"Optimized context creation failed, falling back to standard: {e}")

    # 기존 방식 (메모리가 없거나 최적화 비활성화)
    if additional_rag_context and additional_rag_context.strip():
        final_prompt = ChatPromptTemplate.from_messages([
            ("system", default_prompt),
            MessagesPlaceholder(variable_name="chat_history", n_messages=n_messages),
            ("user", "{input}"),
            ("user", "{additional_rag_context}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
    else:
        final_prompt = ChatPromptTemplate.from_messages([
            ("system", default_prompt),
            MessagesPlaceholder(variable_name="chat_history", n_messages=n_messages),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
    return final_prompt

def create_optimized_tool_context_prompt(additional_rag_context, default_prompt, memory, current_input, n_messages, llm):
    """최적화된 멀티턴 컨텍스트를 사용한 tool 프롬프트 생성 (기존 구조 유지)"""
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

    # 기존 구조를 유지하되, chat_history만 최적화된 것으로 교체
    if additional_rag_context and additional_rag_context.strip():
        final_prompt = ChatPromptTemplate.from_messages([
            ("system", default_prompt),
            MessagesPlaceholder(variable_name="chat_history", n_messages=n_messages),
            ("user", "{input}"),
            ("user", "{additional_rag_context}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
    else:
        final_prompt = ChatPromptTemplate.from_messages([
            ("system", default_prompt),
            MessagesPlaceholder(variable_name="chat_history", n_messages=n_messages),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

    return final_prompt

def create_context_prompt(additional_rag_context, default_prompt, n_messages, strict_citation, memory=None, current_input=None, llm=None, use_optimization=True):
    """
    Context prompt 생성 - 메모리가 있을 때만 최적화 기능 사용

    Args:
        use_optimization: 최적화 기능 사용 여부 (기본값: True)
        memory: 메모리 객체 (최적화 사용 시 필요)
        current_input: 현재 입력 텍스트 (최적화 사용 시 필요)
        llm: LLM 객체 (최적화 사용 시 필요)
    """
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

    # 메모리가 있고 최적화 기능을 사용하는 경우
    if use_optimization and memory and current_input and llm:
        try:
            return create_optimized_context_prompt(additional_rag_context, default_prompt, memory, current_input, n_messages, llm, strict_citation)
        except Exception as e:
            logger.warning(f"Optimized context creation failed, falling back to standard: {e}")

    # 기존 방식 (메모리가 없거나 최적화 비활성화)
    if additional_rag_context and additional_rag_context.strip():
        if strict_citation:
            default_prompt = default_prompt + citation_prompt
        final_prompt = ChatPromptTemplate.from_messages([
            ("system", default_prompt),
            MessagesPlaceholder(variable_name="chat_history", n_messages=n_messages),
            ("user", "{input}"),
            ("user", "{additional_rag_context}"),
        ])
    else:
        final_prompt = ChatPromptTemplate.from_messages([
            ("system", default_prompt),
            MessagesPlaceholder(variable_name="chat_history", n_messages=n_messages),
            ("user", "{input}")
        ])

    return final_prompt

def create_optimized_context_prompt(additional_rag_context, default_prompt, memory, current_input, n_messages, llm, strict_citation):
    """최적화된 멀티턴 컨텍스트를 사용한 일반 프롬프트 생성 (기존 구조 유지)"""
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

    # 기존 구조를 유지하되, chat_history만 최적화된 것으로 교체
    if additional_rag_context and additional_rag_context.strip():
        if strict_citation:
            default_prompt = default_prompt + citation_prompt
        final_prompt = ChatPromptTemplate.from_messages([
            ("system", default_prompt),
            MessagesPlaceholder(variable_name="chat_history", n_messages=n_messages),
            ("user", "{input}"),
            ("user", "{additional_rag_context}"),
        ])
    else:
        final_prompt = ChatPromptTemplate.from_messages([
            ("system", default_prompt),
            MessagesPlaceholder(variable_name="chat_history", n_messages=n_messages),
            ("user", "{input}")
        ])

    return final_prompt

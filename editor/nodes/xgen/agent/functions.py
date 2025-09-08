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


def prepare_llm_components(tools, memory, model, temperature, max_tokens, base_url, streaming=True):
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

    chat_history = []
    if memory:
        try:
            memory_vars = memory.load_memory_variables({})
            chat_history = memory_vars.get("chat_history", [])
        except Exception:
            chat_history = []

    return llm, tools_list, chat_history

def prepare_optimized_memory(memory, current_input, n_messages, llm):
    """최적화된 멀티턴 메모리 생성 - 관련도 기반 메시지 선택 및 요약"""
    if not memory:
        return []
    
    try:
        from editor.nodes.xgen.memory.db_chat_memory import DBMemoryNode
        
        # 메모리에서 모든 대화 기록 추출
        memory_vars = memory.load_memory_variables({})
        full_chat_history = memory_vars.get("chat_history", [])
        
        if not full_chat_history:
            return []
        
        # 메시지를 dict 형태로 변환
        historical_messages = []
        for msg in full_chat_history:
            if hasattr(msg, 'type') and hasattr(msg, 'content'):
                role = "user" if msg.type == "human" else "ai"
                historical_messages.append({
                    'role': role,
                    'content': msg.content
                })
        
        if not historical_messages:
            return []
        
        # DBMemoryNode 인스턴스 생성하여 최적화된 요약 생성
        db_memory_node = DBMemoryNode()
        optimized_summary = db_memory_node._select_and_summarize_relevant_messages(
            current_input=current_input,
            historical_messages=historical_messages,
            n_messages=n_messages,
            llm=llm
        )
        
        # 요약이 있으면 시스템 메시지로 변환하여 반환
        if optimized_summary:
            from langchain_core.messages import SystemMessage
            return [SystemMessage(content=f"이전 대화 요약: {optimized_summary}")]
        else:
            # 요약이 없으면 최근 n_messages만 반환 (기존 방식)
            return full_chat_history[-n_messages:] if n_messages > 0 else []
            
    except Exception as e:
        logger.error(f"Error in prepare_optimized_memory: {e}")
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

def create_tool_context_prompt(additional_rag_context, default_prompt, n_messages):
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
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
    """최적화된 멀티턴 컨텍스트를 사용한 tool 프롬프트 생성"""
    from langchain.prompts import ChatPromptTemplate
    
    # 최적화된 메모리 생성
    optimized_history = prepare_optimized_memory(memory, current_input, n_messages, llm)
    
    messages = [("system", default_prompt)]
    
    # 최적화된 히스토리 추가
    if optimized_history:
        messages.extend([(msg.type if hasattr(msg, 'type') else "system", msg.content) for msg in optimized_history])
    
    messages.append(("user", "{input}"))
    
    if additional_rag_context and additional_rag_context.strip():
        messages.append(("user", "{additional_rag_context}"))
    
    messages.append(("placeholder", "{agent_scratchpad}"))
    
    return ChatPromptTemplate.from_messages(messages)

def create_context_prompt(additional_rag_context, default_prompt, n_messages, strict_citation):
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
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
    """최적화된 멀티턴 컨텍스트를 사용한 일반 프롬프트 생성"""
    from langchain.prompts import ChatPromptTemplate
    
    if additional_rag_context and additional_rag_context.strip() and strict_citation:
        default_prompt = default_prompt + citation_prompt
    
    # 최적화된 메모리 생성
    optimized_history = prepare_optimized_memory(memory, current_input, n_messages, llm)
    
    messages = [("system", default_prompt)]
    
    # 최적화된 히스토리 추가
    if optimized_history:
        for msg in optimized_history:
            msg_type = getattr(msg, 'type', 'system')
            if msg_type == 'human':
                msg_type = 'user'
            elif msg_type == 'ai':
                msg_type = 'assistant'
            messages.append((msg_type, msg.content))
    
    messages.append(("user", "{input}"))
    
    if additional_rag_context and additional_rag_context.strip():
        messages.append(("user", "{additional_rag_context}"))
    
    return ChatPromptTemplate.from_messages(messages)

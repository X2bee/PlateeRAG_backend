import re
import logging
from typing import Any, List, Optional

from langchain_core.messages import SystemMessage

from editor.utils.helper.async_helper import sync_run_async
from editor.utils.citation_prompt import citation_prompt
logger = logging.getLogger(__name__)

def rag_context_builder(text, rag_context, strict_citation=True):
    additional_rag_context = ""
    search_params = rag_context.get('search_params', {})
    rerank_flag = search_params.get('rerank', False)
    rerank_top_k = search_params.get('rerank_top_k', search_params.get('top_k', 10))

    # text가 dict인 경우 텍스트 추출 시도
    if isinstance(text, dict):
        # 우선순위: user_input > input > text > query
        extracted_text = None
        for key in ['user_input', 'input', 'text', 'query']:
            if key in text and text[key]:
                extracted_text = text[key]
                logger.info(f"[RAG_CONTEXT_BUILDER] Dict에서 텍스트 추출 성공: 키='{key}', 값='{extracted_text}'")
                break

        # 추출 실패 시 dict 전체를 JSON 문자열로 변환
        if extracted_text is None:
            try:
                import json
                extracted_text = json.dumps(text, ensure_ascii=False)
                logger.warning(f"[RAG_CONTEXT_BUILDER] Dict에서 특정 키를 찾지 못함, JSON 문자열로 변환: {extracted_text[:100]}...")
            except Exception as e:
                # JSON 변환도 실패하면 str() 사용
                extracted_text = str(text)
                logger.warning(f"[RAG_CONTEXT_BUILDER] JSON 변환 실패, str() 사용: {extracted_text[:100]}...")

        text = extracted_text

    # text가 문자열이 아닌 다른 타입인 경우 문자열로 변환
    if not isinstance(text, str):
        text = str(text)
        logger.warning(f"[RAG_CONTEXT_BUILDER] 텍스트가 문자열이 아님, str()로 변환: {type(text)} -> str")

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

        # OpenAI API의 tool calling을 위한 추가 검증
        if hasattr(tool, 'args_schema'):
            # args_schema가 None인 경우 기본 스키마로 대체
            if tool.args_schema is None:
                from pydantic import BaseModel, Field
                class DefaultToolSchema(BaseModel):
                    """Default schema for tools without explicit arguments"""
                    pass
                tool.args_schema = DefaultToolSchema
                logger.info(f"Tool '{tool.name}' had no args_schema, assigned default schema")

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

def prepare_llm_components(text, tools, memory, model, temperature, max_tokens, n_messages=None, streaming=True, plan=None):
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    from langchain_google_genai import ChatGoogleGenerativeAI
    from editor.utils.helper.service_helper import AppServiceManager

    config_composer = AppServiceManager.get_config_composer()
    if not config_composer:
        raise ValueError("No Config Composer set.")

    # Anthropic 모델 감지 - 모든 Claude 모델은 "claude-"로 시작
    is_anthropic_model = model.startswith('claude-')
    is_google_model = model.startswith('gemini-')

    if is_anthropic_model:
        # Anthropic 모델 사용
        api_key = config_composer.get_config_by_name("ANTHROPIC_API_KEY").value
        if not api_key:
            logger.error(f"Anthropic API Key is not set")
            raise ValueError("Anthropic API Key is not set.")

        llm = ChatAnthropic(model=model, temperature=temperature, max_tokens=max_tokens, streaming=streaming, anthropic_api_key=api_key)

    elif is_google_model:
        # Google Gemini 모델 사용
        api_key = config_composer.get_config_by_name("GEMINI_API_KEY").value
        if not api_key:
            logger.error(f"Google Generative AI API Key is not set")
            raise ValueError("Google Generative AI API Key is not set.")

        llm = ChatGoogleGenerativeAI(model=model, temperature=temperature, max_tokens=max_tokens, streaming=streaming, google_api_key=api_key)
    else:
        # OpenAI 또는 다른 모델 사용
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

        llm = ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens, streaming=streaming)

    tools_list = []
    if tools:
        tools_list = tools if isinstance(tools, list) else [tools]
    if plan and "tools" in plan and isinstance(plan["tools"], list) and len(plan["tools"]) > 0:
        tools_list.extend(plan["tools"])

    tools_list = _flatten_tools_list(tools_list)
    tools_list = _validate_and_fix_tools(tools_list)

    chat_history = []
    if memory:
        try:
            chat_history = prepare_chat_history(
                memory,
                current_input=text,
                llm=llm
            )
            logger.info(
                "Using multiturn memory with %s messages",
                len(chat_history),
            )
        except Exception as e:
            logger.warning(f"Failed to use optimized memory, using standard: {e}")
            chat_history = []

    return llm, tools_list, chat_history

def prepare_chat_history(
    memory,
    current_input: Optional[str] = None,
    llm: Optional[Any] = None,
):
    """
    LangChain 1.0.0 메모리 형식 처리 - List[BaseMessage] 직접 반환

    Args:
        memory: List[BaseMessage] 또는 None
        current_input: 현재 사용자 입력
        llm: LLM 객체 (요약 생성용, 선택사항)

    Returns:
        List[BaseMessage]: 대화 기록 메시지 리스트
    """
    if not memory:
        return []

    try:
        # LangChain 1.0.0: memory는 이미 List[BaseMessage] 형태
        if isinstance(memory, list):
            chat_history = memory
        else:
            # 혹시 다른 형태가 올 경우 빈 리스트 반환
            logger.warning(f"Unexpected memory type: {type(memory)}. Expected List[BaseMessage]")
            return []

        if not chat_history:
            return []

        # LLM이 제공되고 현재 입력이 있으면 요약 메시지 추가
        if llm and current_input:
            try:
                summary_message = _summarize_chat_history_with_llm(
                    chat_history,
                    current_input,
                    llm,
                )
                if summary_message:
                    return list(chat_history) + [summary_message]
            except Exception as summarize_error:
                logger.warning(f"Failed to summarize chat history: {summarize_error}")

        return chat_history

    except Exception as e:
        logger.error(f"Error in prepare_chat_history: {e}")
        return []

def _message_role_and_content(message) -> Optional[tuple]:
    role = None
    content = None

    if hasattr(message, "type") and hasattr(message, "content"):
        role_type = message.type
        if role_type in ("human", "user"):
            role = "사용자"
        elif role_type in ("ai", "assistant"):
            role = "AI"
        else:
            role = role_type or "기타"
        content = message.content if isinstance(message.content, str) else None
    elif isinstance(message, dict):
        role_key = message.get("role")
        if role_key in ("user", "human"):
            role = "사용자"
        elif role_key in ("ai", "assistant"):
            role = "AI"
        else:
            role = role_key or "기타"
        content = message.get("content")

    if content:
        normalized_content = content.strip()
        if normalized_content:
            return role, normalized_content
    return None


def _build_conversation_text(messages: List[Any]) -> str:
    lines = []
    for idx, msg in enumerate(messages, 1):
        normalized = _message_role_and_content(msg)
        if not normalized:
            continue
        role, content = normalized
        lines.append(f"[{idx}] {role}: {content}")

    return "\n".join(lines)


def _summarize_chat_history_with_llm(
    chat_history,
    current_input: str,
    llm: Any
) -> Optional[SystemMessage]:
    if not chat_history or not current_input:
        return None

    conversation_text = _build_conversation_text(chat_history)

    if not conversation_text.strip():
        return None

    summary_prompt = f"""/no_think 당신은 대화 요약 전문가입니다. 아래 이전 대화 기록을 읽고, 현재 사용자의 질문에 답하는 데 필요한 핵심 정보만 요약하세요.

현재 사용자 입력: {current_input}

이전 대화 기록:
{conversation_text}

요약 지침:
1. 현재 질문과 직접적으로 관련된 사실과 맥락만 포함
2. 사용자와 AI의 의도, 결론, 미해결 항목을 명확히 정리
3. 3-6문장으로 간결하지만 충분히 정보를 제공
4. 새로운 정보를 추가하지 말고 원문의 사실만 사용

요약:"""

    result = llm.invoke(summary_prompt)
    summary_text = getattr(result, "content", None)
    if not summary_text:
        return None

    cleaned_summary = summary_text.strip()
    if not cleaned_summary:
        return None

    logger.info("Generated chat history summary for current input")
    return SystemMessage(content=f"이전 대화 요약: {cleaned_summary}")

def create_json_output_prompt(args_schema, original_prompt):
    from langchain_core.output_parsers import JsonOutputParser
    parser = JsonOutputParser(pydantic_object=args_schema)
    format_instructions = parser.get_format_instructions()
    escaped_instructions = format_instructions.replace("{", "{{").replace("}", "}}")
    return f"{original_prompt}\n\n{escaped_instructions}"

def create_tool_context_prompt(additional_rag_context, default_prompt, n_messages=None, memory=None, current_input=None, llm=None, use_optimization=True, plan=None):
    """
    Tool context prompt 생성 - 메모리가 있을 때만 최적화 기능 사용

    Args:
        use_optimization: 최적화 기능 사용 여부 (기본값: True)
        memory: 메모리 객체 (최적화 사용 시 필요)
        current_input: 현재 입력 텍스트 (최적화 사용 시 필요)
        llm: LLM 객체 (최적화 사용 시 필요)
    """
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    if plan and "steps" in plan and isinstance(plan["steps"], list) and len(plan["steps"]) > 0:
        plan_description = "\n".join(plan["steps"])
        default_prompt = f"사용자 요청이 적절한 경우, 다음의 계획에 따라 문제를 해결하십시오: {plan_description}\n\n{default_prompt}"

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

def create_context_prompt(additional_rag_context, default_prompt, strict_citation, n_messages=None, memory=None, current_input=None, llm=None, use_optimization=True, plan=None):
    """
    Context prompt 생성 - 메모리가 있을 때만 최적화 기능 사용

    Args:
        use_optimization: 최적화 기능 사용 여부 (기본값: True)
        memory: 메모리 객체 (최적화 사용 시 필요)
        current_input: 현재 입력 텍스트 (최적화 사용 시 필요)
        llm: LLM 객체 (최적화 사용 시 필요)
    """
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    if plan and "steps" in plan and isinstance(plan["steps"], list) and len(plan["steps"]) > 0:
        plan_description = "\n".join(plan["steps"])
        default_prompt = f"사용자 요청이 적절한 경우, 다음의 계획에 따라 문제를 해결하십시오: {plan_description}\n\n{default_prompt}"

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

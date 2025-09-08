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

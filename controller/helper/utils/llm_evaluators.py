"""
LLM 평가 관련 유틸리티 함수들
"""
import re
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from controller.workflow.models.requests import ScoreModelParser
from controller.workflow.utils.data_parsers import clean_llm_output
from service.database.models.executor import ExecutionIO

logger = logging.getLogger("llm-evaluators")

async def evaluate_with_llm(
    unique_interaction_id: str,
    input_data: str,
    expected_output: str,
    actual_output: str,
    llm_eval_type: str,
    llm_eval_model: str,
    app_db,
    config_composer
) -> float:
    """
    LLM을 사용하여 실제 출력과 예상 출력을 비교하고 점수를 반환합니다.
    """
    logger.info(f"LLM 평가 시작: unique_interaction_id={unique_interaction_id}")

    # 출력 정리
    actual_output = clean_llm_output(actual_output)

    try:
        if llm_eval_type == "OpenAI":
            api_key = config_composer.get_config_by_name("OPENAI_API_KEY").value
            base_url = "https://api.openai.com/v1"
            model_name = llm_eval_model
            if not api_key:
                logger.error(f"[LLM_EVAL] OpenAI API 키가 설정되지 않았습니다")
                raise ValueError("OpenAI API 키가 설정되지 않았습니다.")

        elif llm_eval_type == "vLLM":
            api_key = None
            base_url = config_composer.get_config_by_name("VLLM_API_BASE_URL").value
            model_name = config_composer.get_config_by_name("VLLM_MODEL_NAME").value

        else:
            raise ValueError(f"지원되지 않는 LLM 평가 타입입니다: {llm_eval_type}")

        temperature = 0.1
        if llm_eval_model == "gpt-5" or llm_eval_model == "gpt-5-nano" or llm_eval_model == "gpt-5-mini":
            temperature = 1

        llm_client = ChatOpenAI(
            api_key=api_key,
            model=model_name,
            temperature=temperature,
            max_tokens=1000,
            base_url=base_url
        )

        parser = JsonOutputParser(pydantic_object=ScoreModelParser)

        system_msg = SystemMessage(
            content="""당신은 정확한 답변 평가 전문가입니다.
주어진 입력에 대해 실제 생성된 답변이 레퍼런스 정답과 얼마나 일치하는지 평가해주세요.

평가 기준:
1. 레퍼런스 정답에서 요구하는 핵심 정보나 값이 정확히 포함되어 있는가?
2. 답변이 적절해 보여도 레퍼런스가 지정하는 정확한 값과 다르면 낮은 점수를 주어야 합니다.
3. 부분적으로 맞더라도 핵심 내용이 틀리면 낮은 점수를 주어야 합니다.
4. 완전히 정확한 경우에만 높은 점수(0.9-1.0)를 주세요.

점수 기준:
- 1.0: 레퍼런스와 완전히 일치하거나 동등한 정확성
- 0.7-0.9: 대부분 정확하지만 일부 세부사항이 다름
- 0.4-0.6: 부분적으로 맞지만 중요한 부분이 틀림
- 0.1-0.3: 대부분 틀렸지만 일부 관련성 있음
- 0.0: 완전히 틀렸거나 관련성 없음

응답은 반드시 JSON 형식으로 소수점 2자리까지 정확하게 제공해주세요."""
        )

        evaluation_prompt = f"""다음 내용을 평가해주세요:

**입력 질문/요청:**
{input_data}

**레퍼런스 정답 (기준):**
{expected_output or "없음"}

**실제 생성된 답변:**
{actual_output}

위 실제 답변이 레퍼런스 정답과 얼마나 정확히 일치하는지 0.00~1.00 사이의 점수로 평가해주세요.
**답변 형식**
{parser.get_format_instructions()}"""
        
        human_msg = HumanMessage(content=evaluation_prompt)

        # LLM 호출
        response = await llm_client.ainvoke([system_msg, human_msg])
        content = response.content.strip()

        # JSON 파싱
        parsed_result = parser.parse(content)
        try:
            score = parsed_result.llm_eval_score
        except:
            score = parsed_result.get('llm_eval_score', 0.0)

        score = max(0.0, min(1.0, round(score, 2)))

        # DB 업데이트
        existing_data = app_db.find_by_condition(
            ExecutionIO,
            {
                "interaction_id": unique_interaction_id,
            },
            limit=1
        )

        if existing_data:
            updated_data = existing_data[0]
            updated_data.llm_eval_score = score
            app_db.update(updated_data)
        else:
            logger.warning(f"해당 interaction_id로 레코드를 찾을 수 없습니다: {unique_interaction_id}")

        return score

    except Exception as e:
        logger.error(f"LLM 평가 중 오류 발생: {str(e)}", exc_info=True)
        return 0.0
import logging

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.schema.output_parser import StrOutputParser

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from editor.type_model.feedback_state import FeedbackState

from editor.utils.helper.agent_helper import NonStreamingAgentHandlerWithToolOutput, NonStreamingAgentHandler

logger = logging.getLogger(__name__)

memory = MemorySaver()

def create_feedback_graph(llm, tools_list, prompt_template, additional_rag_context, feedback_criteria, return_intermediate_steps=True, feedback_threshold=8, enable_auto_feedback=True):
        """LangGraph 피드백 루프 그래프 생성"""
        
        def execute_task(state: FeedbackState) -> FeedbackState:
            """도구를 사용하여 작업 실행"""
            try:
                user_input = state["user_input"]
                iteration = state["iteration_count"]
                
                # 이전 시도들의 컨텍스트 생성
                previous_attempts = ""
                if state["tool_results"]:
                    previous_attempts = f"\n\nPrevious attempts:\n"
                    for i, result in enumerate(state["tool_results"][-3:]):  # 최근 3개만
                        previous_attempts += f"Attempt {i+1}: {result.get('result', '')}\nScore: {result.get('feedback_score', 'N/A')}\n\n"
                
                enhanced_input = f"{user_input}{previous_attempts}"
                inputs = {
                    "input": enhanced_input,
                    "chat_history": state["messages"][-5:] if state["messages"] else [],
                    "additional_rag_context": additional_rag_context
                }

                if tools_list and len(tools_list) > 0:
                    agent = create_tool_calling_agent(llm, tools_list, prompt_template)
                    agent_executor = AgentExecutor(
                        agent=agent,
                        tools=tools_list,
                        verbose=True,
                        handle_parsing_errors=True,
                        max_iterations=3,
                        max_execution_time=300,
                    )
                    if return_intermediate_steps:
                        handler = NonStreamingAgentHandlerWithToolOutput()
                    else:
                        handler = NonStreamingAgentHandler()
                    response = agent_executor.invoke(inputs, {"callbacks": [handler]})
                    result = handler.get_formatted_output(response["output"])
                else:
                    chain = prompt_template | llm | StrOutputParser()
                    result = chain.invoke(inputs)

                # 도구 실행 결과 저장
                import time
                tool_result = {
                    "iteration": iteration,
                    "result": result,
                    "timestamp": time.time()
                }
                
                new_tool_results = state["tool_results"] + [tool_result]
                
                return {
                    **state,
                    "tool_results": new_tool_results,
                    "messages": state["messages"] + [{"role": "assistant", "content": f"Iteration {iteration}: {result}"}]
                }
                
            except Exception as e:
                logger.error(f"Task execution failed: {str(e)}")
                import time
                error_result = {
                    "iteration": state["iteration_count"],
                    "result": f"Error: {str(e)}",
                    "timestamp": time.time(),
                    "error": True
                }
                return {
                    **state,
                    "tool_results": state["tool_results"] + [error_result],
                    "messages": state["messages"] + [{"role": "assistant", "content": f"Error in iteration {state['iteration_count']}: {str(e)}"}]
                }

        def evaluate_feedback(state: FeedbackState) -> FeedbackState:
            """결과에 대한 피드백 평가"""
            try:
                # enable_auto_feedback가 False인 경우 피드백 평가를 건너뛰고 기본 점수 설정
                if not enable_auto_feedback:
                    if not state["tool_results"]:
                        return {**state, "feedback_score": 0}

                    latest_result = state["tool_results"][-1]
                    updated_tool_results = state["tool_results"][:-1] + [{
                        **latest_result,
                        "feedback_score": 5,  # 중간 점수로 설정
                        "evaluation": {"score": 5, "reasoning": "자동 피드백이 비활성화됨", "improvements": "", "strengths": ""}
                    }]

                    return {
                        **state,
                        "feedback_score": 5,
                        "tool_results": updated_tool_results
                    }

                if not state["tool_results"]:
                    return {**state, "feedback_score": 0}

                latest_result = state["tool_results"][-1]
                if latest_result.get("error"):
                    return {**state, "feedback_score": 1}
                
                # 피드백 평가 프롬프트 생성
                evaluation_prompt = f"""
원본 사용자 요청: {state["user_input"]}

피드백 기준: {feedback_criteria if feedback_criteria else "일반적인 품질, 정확성, 완성도"}

현재 결과: {latest_result["result"]}

위 결과를 1-10점 척도로 평가해주세요. 다음 기준을 고려하세요:
1-3: 매우 부족함, 요구사항을 전혀 충족하지 못함
4-5: 부족함, 일부 요구사항만 충족
6-7: 보통, 기본 요구사항 충족하나 개선 필요
8-9: 좋음, 대부분의 요구사항 충족
10: 완벽함, 모든 요구사항을 완벽하게 충족

다음 JSON 형식으로 응답해주세요:
{{
    "score": <1-10 사이의 점수>,
    "reasoning": "<평가 이유>",
    "improvements": "<개선이 필요한 부분들>",
    "strengths": "<잘된 부분들>"
}}
"""
                
                # 평가용 간단한 프롬프트 직접 처리
                formatted_evaluation_prompt = f"추가 컨텍스트: {additional_rag_context}\n\n{evaluation_prompt}"
                
                # 직접 LLM 호출
                from langchain.schema import HumanMessage
                messages = [HumanMessage(content=formatted_evaluation_prompt)]
                llm_response = llm.invoke(messages)
                
                def parse_evaluation_result(content):
                    """평가 결과를 파싱하는 함수 - 여러 fallback 방법 포함"""
                    import json
                    import re

                    # 1차 시도: 정상적인 JSON 파싱
                    try:
                        result = json.loads(content)
                        return result
                    except Exception as e:
                        print(f"[FEEDBACK DEBUG] 1차 JSON 파싱 실패: {e}")

                    # 2차 시도: 잘린 JSON 수정 (마지막 }가 없는 경우)
                    try:
                        if content.strip().endswith('"') or content.strip().endswith(']'):
                            fixed_content = content.strip() + "}"
                            result = json.loads(fixed_content)
                            return result
                    except Exception as e:
                        print(f"[FEEDBACK DEBUG] 2차 JSON 파싱 실패: {e}")

                    # 3차 시도: 정규식으로 필드 추출
                    try:
                        score_match = re.search(r'"score":\s*(\d+)', content)
                        reasoning_match = re.search(r'"reasoning":\s*"([^"]*)"', content)
                        improvements_match = re.search(r'"improvements":\s*\[(.*?)\]', content, re.DOTALL)
                        strengths_match = re.search(r'"strengths":\s*\[(.*?)\]', content, re.DOTALL)

                        result = {
                            "score": int(score_match.group(1)) if score_match else 5,
                            "reasoning": reasoning_match.group(1) if reasoning_match else "정규식 파싱으로 추출",
                            "improvements": [],
                            "strengths": []
                        }

                        # improvements와 strengths 배열 파싱
                        if improvements_match:
                            improvements_text = improvements_match.group(1)
                            result["improvements"] = re.findall(r'"([^"]*)"', improvements_text)

                        if strengths_match:
                            strengths_text = strengths_match.group(1)
                            result["strengths"] = re.findall(r'"([^"]*)"', strengths_text)

                        return result

                    except Exception as e:
                        print(f"[FEEDBACK DEBUG] 3차 정규식 파싱 실패: {e}")

                    # 최종 fallback: 기본값
                    return {
                        "score": 5,
                        "reasoning": f"파싱 실패 - 원본: {content[:200]}...",
                        "improvements": ["파싱 실패로 인한 기본값"],
                        "strengths": ["파싱 실패로 인한 기본값"]
                    }

                evaluation_result = parse_evaluation_result(llm_response.content)
                
                score = evaluation_result.get("score", 0)
                
                # 결과 업데이트
                updated_tool_results = state["tool_results"][:-1] + [{
                    **latest_result,
                    "feedback_score": score,
                    "evaluation": evaluation_result
                }]
                
                return {
                    **state,
                    "feedback_score": score,
                    "tool_results": updated_tool_results,
                    "messages": state["messages"] + [{"role": "system", "content": f"Feedback evaluation: Score {score}/10 - {evaluation_result.get('reasoning', '')}"}]
                }
                
            except Exception as e:
                logger.error(f"Feedback evaluation failed: {str(e)}")
                return {**state, "feedback_score": 0}

        def check_completion(state: FeedbackState) -> str:
            """완료 조건 확인"""
            # 최대 반복 횟수 확인
            if state["iteration_count"] >= state["max_iterations"]:
                return "finalize"

            # 피드백 점수 확인 (enable_auto_feedback가 True일 때만)
            if enable_auto_feedback and state["feedback_score"] >= feedback_threshold:
                return "finalize"

            # 에러가 발생한 경우
            if state["tool_results"] and state["tool_results"][-1].get("error"):
                return "finalize"

            # enable_auto_feedback가 False인 경우, 최대 반복 횟수에만 의존
            if not enable_auto_feedback and state["iteration_count"] >= state["max_iterations"]:
                return "finalize"

            return "continue"

        def increment_iteration(state: FeedbackState) -> FeedbackState:
            """반복 횟수 증가"""
            return {
                **state,
                "iteration_count": state["iteration_count"] + 1
            }

        def finalize_result(state: FeedbackState) -> FeedbackState:
            """최종 결과 생성"""
            try:
                if not state["tool_results"]:
                    final_result = "No results generated"
                    requirements_met = False
                else:
                    # 가장 높은 점수의 결과 선택
                    best_result = max(
                        [r for r in state["tool_results"] if not r.get("error", False)],
                        key=lambda x: x.get("feedback_score", 0),
                        default=state["tool_results"][-1]
                    )
                    final_result = best_result["result"]
                    requirements_met = best_result.get("feedback_score", 0) >= feedback_threshold
                
                return {
                    **state,
                    "final_result": final_result,
                    "requirements_met": requirements_met
                }
                
            except Exception as e:
                logger.error(f"Result finalization failed: {str(e)}")
                return {
                    **state,
                    "final_result": f"Error during finalization: {str(e)}",
                    "requirements_met": False
                }

        # 그래프 구성
        workflow = StateGraph(FeedbackState)
        
        # 노드 추가
        workflow.add_node("execute_task", execute_task)
        workflow.add_node("evaluate_feedback", evaluate_feedback)
        workflow.add_node("increment_iteration", increment_iteration)
        workflow.add_node("finalize_result", finalize_result)
        
        # 엣지 설정
        workflow.set_entry_point("execute_task")
        workflow.add_edge("execute_task", "evaluate_feedback")
        workflow.add_conditional_edges(
            "evaluate_feedback",
            check_completion,
            {
                "continue": "increment_iteration",
                "finalize": "finalize_result"
            }
        )
        workflow.add_edge("increment_iteration", "execute_task")
        workflow.add_edge("finalize_result", END)
        
        return workflow.compile(checkpointer=memory)

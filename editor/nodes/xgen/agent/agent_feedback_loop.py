import logging
import asyncio
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from editor.node_composer import Node
from langchain.schema.output_parser import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from editor.nodes.xgen.agent.functions import (
    prepare_llm_components, rag_context_builder, 
    create_json_output_prompt, create_tool_context_prompt, create_context_prompt
)
from editor.utils.helper.agent_helper import NonStreamingAgentHandler, NonStreamingAgentHandlerWithToolOutput
from editor.utils.prefix_prompt import prefix_prompt
from langchain.agents import create_tool_calling_agent, AgentExecutor

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict, Annotated

logger = logging.getLogger(__name__)

default_prompt = """You are a helpful AI assistant with feedback loop capabilities. 
You will execute tasks, evaluate results against user requirements, and iterate until satisfactory results are achieved."""

# State definitions for LangGraph
class FeedbackState(TypedDict):
    messages: Annotated[List[Dict], add_messages]
    user_input: str
    tool_results: List[Dict]
    feedback_score: int  # 1-10 scale
    iteration_count: int
    final_result: Optional[str]
    requirements_met: bool
    max_iterations: int

class AgentFeedbackLoopNode(Node):
    categoryId = "xgen"
    functionId = "agents"
    nodeId = "agents/feedback_loop"
    nodeName = "Agent Feedback Loop"
    description = "LangGraph 기반 피드백 루프를 통해 사용자 요구사항에 맞는 결과를 반복적으로 생성하는 Agent 노드"
    tags = ["agent", "feedback", "loop", "langgraph", "iterative", "evaluation"]

    inputs = [
        {"id": "text", "name": "Text", "type": "STR", "multi": False, "required": True},
        {"id": "tools", "name": "Tools", "type": "TOOL", "multi": True, "required": False, "value": []},
        {"id": "memory", "name": "Memory", "type": "OBJECT", "multi": False, "required": False},
        {"id": "rag_context", "name": "RAG Context", "type": "DocsContext", "multi": False, "required": False},
        {"id": "args_schema", "name": "ArgsSchema", "type": "OutputSchema", "required": False},
        {"id": "feedback_criteria", "name": "Feedback Criteria", "type": "FeedbackCrit", "multi": False, "required": False, "value": ""},
    ]
    outputs = [
        {"id": "feedback_result", "name": "Feedback Loop Result", "type": "FeedbackDICT", "required": True, "multi": False, "stream": False},
    ]
    parameters = [
        {
            "id": "model", "name": "Model", "type": "STR", "value": "gpt-4.1-mini", "required": True, "description": "사용할 LLM 모델 이름 (예: gpt-4, gpt-3.5-turbo 등)"
        },
        {"id": "temperature", "name": "Temperature", "type": "FLOAT", "value": 0.7, "required": False, "optional": True, "min": 0.0, "max": 2.0, "step": 0.1},
        {"id": "max_tokens", "name": "Max Tokens", "type": "INT", "value": 8192, "required": False, "optional": True, "min": 1, "max": 65536, "step": 1},
        {"id": "n_messages", "name": "Max Memory", "type": "INT", "value": 3, "min": 1, "max": 10, "step": 1, "optional": True},
        {"id": "base_url", "name": "Base URL", "type": "STR", "value": "https://api.openai.com/v1", "required": False, "optional": True},
        {"id": "strict_citation", "name": "Strict Citation", "type": "BOOL", "value": True, "required": False, "optional": True},
        {"id": "return_intermediate_steps", "name": "Return Intermediate Steps", "type": "BOOL", "value": True, "required": False, "optional": True, "description": "중간 단계를 반환할지 여부입니다."},
        {"id": "default_prompt", "name": "Default Prompt", "type": "STR", "value": default_prompt, "required": False, "optional": True, "expandable": True, "description": "기본 프롬프트로 AI가 따르는 System 지침을 의미합니다."},
        {"id": "max_iterations", "name": "Max Iterations", "type": "INT", "value": 5, "min": 1, "max": 20, "step": 1, "optional": True, "description": "최대 반복 횟수"},
        {"id": "feedback_threshold", "name": "Feedback Threshold", "type": "INT", "value": 8, "min": 1, "max": 10, "step": 1, "optional": True, "description": "만족스러운 결과로 간주할 점수 임계값 (1-10)"},
        {"id": "enable_auto_feedback", "name": "Enable Auto Feedback", "type": "BOOL", "value": True, "required": False, "optional": True, "description": "자동 피드백 평가 활성화"},
    ]

    def __init__(self):
        super().__init__()
        self.memory = MemorySaver()
        
    def create_feedback_graph(self, llm, tools_list, prompt_template, additional_rag_context, feedback_criteria):
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
                    handler = NonStreamingAgentHandlerWithToolOutput()
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
                
                evaluation_inputs = {
                    "input": evaluation_prompt,
                    "chat_history": [],
                    "additional_rag_context": additional_rag_context,
                    "agent_scratchpad": ""
                }
                
                # 평가용 간단한 프롬프트 직접 처리
                formatted_evaluation_prompt = f"{additional_rag_context}\n\n{evaluation_prompt}"
                
                # 직접 LLM 호출
                from langchain.schema import HumanMessage
                messages = [HumanMessage(content=formatted_evaluation_prompt)]
                llm_response = llm.invoke(messages)
                
                try:
                    # JSON 파싱 시도
                    import json
                    evaluation_result = json.loads(llm_response.content)
                except:
                    # JSON 파싱 실패시 기본값
                    evaluation_result = {"score": 5, "reasoning": "평가 파싱 실패", "improvements": "", "strengths": ""}
                
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
            feedback_threshold = 8  # 기본값, 실제로는 파라미터로 전달받아야 함
            
            # 최대 반복 횟수 확인
            if state["iteration_count"] >= state["max_iterations"]:
                return "finalize"
            
            # 피드백 점수 확인
            if state["feedback_score"] >= feedback_threshold:
                return "finalize"
            
            # 에러가 발생한 경우
            if state["tool_results"] and state["tool_results"][-1].get("error"):
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
                    requirements_met = best_result.get("feedback_score", 0) >= 8
                
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
        
        return workflow.compile(checkpointer=self.memory)

    def execute(
        self,
        text: str,
        tools: Optional[Any] = None,
        memory: Optional[Any] = None,
        rag_context: Optional[Dict[str, Any]] = None,
        args_schema: Optional[BaseModel] = None,
        feedback_criteria: str = "",
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 8192,
        n_messages: int = 3,
        base_url: str = "https://api.openai.com/v1",
        strict_citation: bool = True,
        return_intermediate_steps: bool = True,
        default_prompt: str = default_prompt,
        max_iterations: int = 5,
        feedback_threshold: int = 8,
        enable_auto_feedback: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        try:
            # LLM 컴포넌트 준비
            enhanced_prompt = prefix_prompt + default_prompt
            llm, tools_list, chat_history = prepare_llm_components(
                text, tools, memory, model, temperature, max_tokens, base_url, n_messages, streaming=False
            )

            # RAG 컨텍스트 구성
            additional_rag_context = ""
            if rag_context:
                additional_rag_context = rag_context_builder(text, rag_context, strict_citation)

            # JSON 출력 스키마 처리
            if args_schema:
                enhanced_prompt = create_json_output_prompt(args_schema, enhanced_prompt)

            # 프롬프트 템플릿 생성
            if tools_list and len(tools_list) > 0:
                prompt_template = create_tool_context_prompt(additional_rag_context, enhanced_prompt, n_messages)
            else:
                prompt_template = create_context_prompt(additional_rag_context, enhanced_prompt, n_messages, strict_citation)

            # 피드백 기준 설정
            if not feedback_criteria:
                feedback_criteria = f"""
사용자 요청을 얼마나 잘 충족했는지 평가해주세요:
- 정확성: 요청한 내용과 일치하는가?
- 완성도: 필요한 모든 정보가 포함되었는가?
- 품질: 결과의 품질이 높은가?
- 유용성: 사용자에게 도움이 되는가?
"""

            # LangGraph 워크플로우 생성
            workflow = self.create_feedback_graph(
                llm, tools_list, prompt_template, additional_rag_context, feedback_criteria
            )

            # 초기 상태 설정
            initial_state = FeedbackState(
                messages=[{"role": "user", "content": text}],
                user_input=text,
                tool_results=[],
                feedback_score=0,
                iteration_count=1,
                final_result=None,
                requirements_met=False,
                max_iterations=max_iterations
            )

            # 워크플로우 실행
            import time
            thread_id = f"feedback_loop_{hash(text)}_{int(time.time())}"
            final_state = workflow.invoke(
                initial_state,
                config={"configurable": {"thread_id": thread_id}}
            )

            # 결과 구성
            iteration_log = []
            feedback_scores = []
            
            for result in final_state["tool_results"]:
                iteration_log.append({
                    "iteration": result["iteration"],
                    "result": result["result"],
                    "score": result.get("feedback_score", 0),
                    "evaluation": result.get("evaluation", {}),
                    "timestamp": result["timestamp"]
                })
                feedback_scores.append(result.get("feedback_score", 0))

            return {
                    "result": final_state["final_result"] or "No final result generated",
                    "iteration_log": iteration_log,
                    "feedback_scores": feedback_scores,
                    "total_iterations": len(iteration_log),
                    "final_score": feedback_scores[-1] if feedback_scores else 0,
                    "average_score": sum(feedback_scores) / len(feedback_scores) if feedback_scores else 0 
            }

        except Exception as e:
            logger.error(f"[FEEDBACK_LOOP_EXECUTE] Feedback Loop Agent 실행 중 오류 발생: {str(e)}")
            logger.exception(f"[FEEDBACK_LOOP_EXECUTE] 상세 스택 트레이스:")
            return {
                "result": f"죄송합니다. 피드백 루프 실행 중 오류가 발생했습니다: {str(e)}",
                "iteration_log": [],
                "feedback_scores": [],
                "total_iterations": 0,
                "final_score": 0,
                "average_score": 0,
                "error": True
                
            }
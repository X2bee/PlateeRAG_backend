import logging
import asyncio
from typing import Dict, Any, Optional, List
from controller.helper.singletonHelper import get_config_composer
from editor.type_model.feedback_state import FeedbackState
from editor.utils.feedback.create_feedback_graph import create_feedback_graph
from editor.utils.feedback.create_todos import create_todos
from editor.utils.feedback.todo_executor import todo_executor
from pydantic import BaseModel

from fastapi import Request
from editor.node_composer import Node
from editor.nodes.xgen.agent.functions import (
    prepare_llm_components, rag_context_builder, 
    create_json_output_prompt, create_tool_context_prompt, create_context_prompt
)
from editor.utils.prefix_prompt import prefix_prompt

logger = logging.getLogger(__name__)

default_prompt = """You are a helpful AI assistant with feedback loop capabilities. 
You will execute tasks, evaluate results against user requirements, and iterate until satisfactory results are achieved."""

# TODO tool 결과에 대한 정보를 피드백 결과에 추가하기 현재는 안보이는것 같음

class AgentFeedbackLoopNode(Node):
    categoryId = "xgen"
    functionId = "agents"
    nodeId = "agents/feedback_loop_vllm"
    nodeName = "Agent VLLM Feedback Loop"
    description = "사용자 요청을 TODO로 분해하고 각 TODO를 피드백 루프를 통해 순차적으로 실행하는 Agent 노드"
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
        {"id": "model", "name": "Model", "type": "STR", "value": "", "is_api": True, "api_name": "api_vllm_model_name", "required": True},
        {"id": "temperature", "name": "Temperature", "type": "FLOAT", "value": 0.7, "required": False, "optional": True, "min": 0.0, "max": 2.0, "step": 0.1},
        {"id": "max_tokens", "name": "Max Tokens", "type": "INT", "value": 8192, "required": False, "optional": True, "min": 1, "max": 65536, "step": 1},
        {"id": "n_messages", "name": "Max Memory", "type": "INT", "value": 3, "min": 1, "max": 10, "step": 1, "optional": True},
        {"id": "base_url", "name": "Base URL", "type": "STR", "value": "", "is_api": True, "api_name": "api_vllm_api_base_url", "required": True},
        {"id": "strict_citation", "name": "Strict Citation", "type": "BOOL", "value": True, "required": False, "optional": True},
        {"id": "return_intermediate_steps", "name": "Return Intermediate Steps", "type": "BOOL", "value": True, "required": False, "optional": True, "description": "중간 단계를 반환할지 여부입니다."},
        {"id": "default_prompt", "name": "Default Prompt", "type": "STR", "value": default_prompt, "required": False, "optional": True, "expandable": True, "description": "기본 프롬프트로 AI가 따르는 System 지침을 의미합니다."},
        {"id": "max_iterations", "name": "Max Sub-task Iterations", "type": "INT", "value": 3, "min": 1, "max": 10, "step": 1, "optional": True, "description": "개별 TODO 실행 시 최대 반복 횟수"},
        {"id": "feedback_threshold", "name": "Feedback Threshold", "type": "INT", "value": 8, "min": 1, "max": 10, "step": 1, "optional": True, "description": "만족스러운 결과로 간주할 점수 임계값 (1-10)"},
        {"id": "enable_auto_feedback", "name": "Enable Auto Feedback", "type": "BOOL", "value": True, "required": False, "optional": True, "description": "자동 피드백 평가 활성화"},
    ]

    def __init__(self):
        super().__init__()

    def api_vllm_model_name(self, request: Request) -> Dict[str, Any]:
        config_composer = get_config_composer(request)
        return config_composer.get_config_by_name("VLLM_MODEL_NAME").value

    def api_vllm_api_base_url(self, request: Request) -> Dict[str, Any]:
        config_composer = get_config_composer(request)
        return config_composer.get_config_by_name("VLLM_API_BASE_URL").value

        
    def execute(
        self,
        text: str,
        tools: Optional[Any] = None,
        memory: Optional[Any] = None,
        rag_context: Optional[Dict[str, Any]] = None,
        args_schema: Optional[BaseModel] = None,
        feedback_criteria: str = "",
        model: str = "x2bee/Polar-14B",
        temperature: float = 0.7,
        max_tokens: int = 8192,
        n_messages: int = 3,
        base_url: str = "",
        strict_citation: bool = True,
        return_intermediate_steps: bool = True,
        default_prompt: str = default_prompt,
        max_iterations: int = 3,
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
            prompt_template_with_tool = create_tool_context_prompt(additional_rag_context, enhanced_prompt, n_messages)
            prompt_template_without_tool = create_context_prompt(additional_rag_context, enhanced_prompt, n_messages, strict_citation)
            

            # 사용자 요청을 TODO로 분해
            todos = create_todos(llm, text)

            # 피드백 기준 설정
            if not feedback_criteria:
                feedback_criteria = f"""
각 TODO 작업을 얼마나 잘 완료했는지 평가해주세요:
- 정확성: 요청한 내용과 일치하는가?
- 완성도: 필요한 모든 정보가 포함되었는가?
- 품질: 결과의 품질이 높은가?
- 유용성: 사용자에게 도움이 되는가?
"""

            # TODO별 워크플로우 생성
            workflow = create_feedback_graph(
                llm, tools_list, chat_history, prompt_template_with_tool, prompt_template_without_tool, additional_rag_context,
                feedback_criteria, return_intermediate_steps, feedback_threshold, enable_auto_feedback
            )

            # 2단계: 각 TODO를 순차적으로 실행
            all_results = []
            todo_execution_log = []

            all_results, todo_execution_log = todo_executor(todos, text, max_iterations, workflow, return_intermediate_steps)

            # 3단계: 전체 결과 종합
            all_scores = [result["score"] for result in all_results if "score" in result]
            completed_todos = [todo for todo in todo_execution_log if todo.get("requirements_met", False)]

            # 최종 결과 구성
            # final_summary = ""

            # for todo_log in todo_execution_log:
            #     final_summary += f"\n- {todo_log['todo_title']}: {todo_log['result']}"

            final_summary=todo_execution_log[-1]['result'] if todo_execution_log else "No TODOs executed."

            result_dict = {
                "result": final_summary,
                "todos_generated": len(todos),
                "todos_completed": len(completed_todos),
                "total_iterations": len(all_results),
                "final_score": all_scores[-1] if all_scores else 0,
                "average_score": sum(all_scores) / len(all_scores) if all_scores else 0,
                "completion_rate": len(completed_todos) / len(todos) if todos else 0
            }

            # return_intermediate_steps가 True인 경우에만 상세 정보 포함
            if return_intermediate_steps:
                result_dict.update({
                    "todos_list": todos,
                    "todo_execution_log": todo_execution_log,
                    "iteration_log": all_results,
                    "feedback_scores": all_scores
                })

            return result_dict

        except Exception as e:
            logger.error(f"[FEEDBACK_LOOP_EXECUTE] TODO 기반 피드백 루프 실행 중 오류 발생: {str(e)}")
            logger.exception(f"[FEEDBACK_LOOP_EXECUTE] 상세 스택 트레이스:")
            # 에러 시에도 return_intermediate_steps에 따라 결과 구조 결정
            error_result = {
                "result": f"죄송합니다. TODO 기반 피드백 루프 실행 중 오류가 발생했습니다: {str(e)}",
                "todos_generated": 0,
                "todos_completed": 0,
                "total_iterations": 0,
                "final_score": 0,
                "average_score": 0,
                "completion_rate": 0,
                "error": True
            }

            if return_intermediate_steps:
                error_result.update({
                    "todos_list": [],
                    "todo_execution_log": [],
                    "iteration_log": [],
                    "feedback_scores": []
                })

            return error_result
import logging
import re
from typing import Dict, List

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.schema.output_parser import StrOutputParser

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from editor.type_model.feedback_state import FeedbackState

from editor.utils.helper.agent_helper import NonStreamingAgentHandlerWithToolOutput, NonStreamingAgentHandler
from editor.utils.helper.stream_helper import (
    EnhancedAgentStreamingHandler,
    EnhancedAgentStreamingHandlerWithToolOutput,
    execute_agent_streaming,
)

logger = logging.getLogger(__name__)

memory = MemorySaver()

def create_feedback_graph(
    llm,
    tools_list,
    chat_history,
    prompt_template_with_tool,
    prompt_template_without_tool,
    additional_rag_context,
    feedback_criteria,
    return_intermediate_steps=True,
    feedback_threshold=8,
    enable_auto_feedback=True,
    tool_agent_max_iterations=6,
    stream_emitter=None,
):
        """LangGraph 피드백 루프 그래프 생성"""
        
        use_tools = bool(tools_list)
        tool_agent_executor = None
        if use_tools:
            tool_calling_agent = create_tool_calling_agent(llm, tools_list, prompt_template_with_tool)
            tool_agent_executor = AgentExecutor(
                agent=tool_calling_agent,
                tools=tools_list,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=tool_agent_max_iterations,
                max_execution_time=300,
            )

        streaming_chain = None
        sync_chain = None
        if prompt_template_without_tool is not None:
            if stream_emitter:
                streaming_chain = prompt_template_without_tool | llm
            sync_chain = prompt_template_without_tool | llm | StrOutputParser()

        def _sanitize_result_text(text):
            if not text:
                return ""
            if not isinstance(text, str):
                text = str(text)

            def _compact_tool_block(label: str, block: str) -> str:
                snippet = block.strip()
                if not snippet:
                    return ""
                snippet = re.sub(r"\s+", " ", snippet)
                if len(snippet) > 600:
                    snippet = snippet[:600].rstrip() + "..."
                return f"{label}: {snippet}"

            cleaned = re.sub(
                r"<TOOLUSELOG>(.*?)</TOOLUSELOG>",
                lambda m: _compact_tool_block("도구 실행", m.group(1)),
                text,
                flags=re.DOTALL,
            )
            cleaned = re.sub(
                r"<TOOLOUTPUTLOG>(.*?)</TOOLOUTPUTLOG>",
                lambda m: _compact_tool_block("도구 결과", m.group(1)),
                cleaned,
                flags=re.DOTALL,
            )
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

            if "작업 세부 요약:" in cleaned:
                cleaned = cleaned.split("작업 세부 요약:", 1)[0].rstrip()

            paragraphs = []
            seen = set()
            for block in [p.strip() for p in cleaned.split("\n\n") if p.strip()]:
                if block not in seen:
                    seen.add(block)
                    paragraphs.append(block)

            return "\n\n".join(paragraphs)

        def _has_explicit_output(text: str) -> bool:
            if not text:
                return False

            lowered = text.lower()
            explicit_keywords = [
                "도구 결과",
                "결과 없음",
                "데이터 없음",
                "no result",
                "no data",
                "0 row",
                "0개 행",
                "행 없음",
                "returned",
                "rows",
            ]
            if any(keyword in lowered for keyword in explicit_keywords):
                return True

            if re.search(r"\|\s*[^|]+\|", text):
                return True

            json_blocks = re.findall(r"\{[^}]+\}", text)
            for block in json_blocks:
                lower_block = block.lower()
                if "query" in lower_block and not any(token in lower_block for token in ("rows", "result", "data", "output", "value", "records")):
                    continue
                return True

            list_blocks = re.findall(r"\[[^\]]+\]", text)
            for block in list_blocks:
                lower_block = block.lower()
                if "query" in lower_block and not any(token in lower_block for token in ("rows", "result", "data", "output", "value", "records")):
                    continue
                return True

            lines = [line.strip() for line in text.splitlines() if line.strip()]
            data_lines = []
            for line in lines:
                lower_line = line.lower()
                if lower_line.startswith("select "):
                    continue
                if "query" in lower_line or "postgresql" in lower_line or "tool" in lower_line:
                    continue
                data_lines.append(line)

            for line in data_lines:
                if any(char.isdigit() for char in line):
                    return True
                if len(line.split()) >= 3:
                    return True

            return False

        def execute_task(state: FeedbackState) -> FeedbackState:
            nonlocal sync_chain
            """도구를 사용하여 작업 실행"""
            try:
                user_input = state["user_input"]
                iteration = state["iteration_count"]

                # TODO의 tool_required 정보 추출 (state에서 가져오기)
                todo_requires_tools = state.get("todo_requires_tools", True)  # 기본값은 True (하위 호환성)
                todo_title = state.get("current_todo_title", "")
                todo_index = state.get("current_todo_index")
                total_todos = state.get("total_todos")
                todo_directive = state.get("todo_directive") or user_input
                todo_description = state.get("current_todo_description", "")
                previous_context = state.get("previous_results_context", "")

                if stream_emitter:
                    stream_emitter.emit_iteration_start(todo_index, total_todos, iteration, todo_title)

                # 이전 시도들의 컨텍스트 생성
                previous_attempts_lines = []
                if state["tool_results"]:
                    previous_attempts_lines.append("이전 반복 결과 요약:")
                    recent_results = state["tool_results"][-3:]
                    for idx, result_entry in enumerate(recent_results, start=1):
                        raw = result_entry.get('clean_result') or result_entry.get('result') or ""
                        summary = _sanitize_result_text(raw)
                        if len(summary) > 350:
                            summary = summary[:350].rstrip() + "..."
                        score = result_entry.get('feedback_score', 'N/A')
                        previous_attempts_lines.append(f"- #{idx} (점수 {score}): {summary}")
                previous_attempts = "\n".join(previous_attempts_lines)

                remediation_notes = [note for note in (state.get("remediation_notes") or []) if note]
                improvement_section = ""
                if remediation_notes:
                    truncated_notes = remediation_notes[:5]
                    formatted_notes = "\n".join(f"- {note}" for note in truncated_notes)
                    improvement_section = f"\n\n반영해야 할 개선 사항:\n{formatted_notes}"

                iteration_brief = [todo_directive]

                iteration_brief.append(
                    "결과 보고 시 이번 반복에서 확인한 실제 데이터(또는 비어 있음을 명시)를 직접 서술하고, '이전에 기록됨' 같은 표현으로 생략하지 마세요."
                )

                if todo_title or todo_description:
                    todo_meta = f"\n현재 TODO 정보:\n- 제목: {todo_title or 'Untitled'}"
                    if todo_description:
                        todo_meta += f"\n- 설명: {todo_description.strip()}"
                    if previous_context:
                        todo_meta += f"\n\n{previous_context}"
                    iteration_brief.append(todo_meta)

                iteration_brief.append(f"\n현재 반복 번호: {iteration}")

                if previous_attempts:
                    iteration_brief.append("\n" + previous_attempts)

                if improvement_section:
                    iteration_brief.append(improvement_section)

                if state.get("last_result_duplicate"):
                    iteration_brief.append(
                        "이전 반복 출력이 거의 동일했습니다. 완전히 다른 접근과 쿼리를 시도하세요."
                    )

                enhanced_input = "\n\n".join(section.strip() for section in iteration_brief if section and section.strip())
                inputs = {
                    "input": enhanced_input,
                    "chat_history": chat_history,
                    "additional_rag_context": additional_rag_context
                }

                result = ""

                def _invoke_tool_sync() -> str:
                    fallback_handler = (
                        NonStreamingAgentHandlerWithToolOutput()
                        if return_intermediate_steps
                        else NonStreamingAgentHandler()
                    )
                    response = tool_agent_executor.invoke(inputs, {"callbacks": [fallback_handler]})
                    return fallback_handler.get_formatted_output(response["output"])

                # 도구 필요성에 따른 처리 분기
                use_agent = bool(todo_requires_tools and tool_agent_executor)
                if use_agent:
                    # 복잡한 작업: 도구 사용
                    if stream_emitter:
                        if return_intermediate_steps:
                            handler = EnhancedAgentStreamingHandlerWithToolOutput()
                        else:
                            handler = EnhancedAgentStreamingHandler()

                        async_executor = lambda: tool_agent_executor.ainvoke(inputs, {"callbacks": [handler]})
                        try:
                            for chunk in execute_agent_streaming(async_executor, handler):
                                if chunk:
                                    stream_emitter.emit_llm_chunk(todo_index, iteration, chunk)
                        except Exception as streaming_exc:  # pragma: no cover - runtime safety
                            logger.error(
                                f"Streaming agent execution failed (iteration {iteration}): {streaming_exc}",
                                exc_info=True,
                            )
                            stream_emitter.emit_iteration_error(todo_index, iteration, streaming_exc)

                        raw_output = "".join(handler.streamed_tokens).strip()
                        if return_intermediate_steps and handler.tool_logs:
                            tool_log_text = "\n".join(handler.tool_logs)
                            if tool_log_text and raw_output:
                                result = f"{raw_output}\n\n{tool_log_text}"
                            else:
                                result = raw_output or tool_log_text
                        else:
                            result = raw_output

                        if streaming_error:
                            raise streaming_error

                        # 스트리밍 중 수집된 결과가 비어있는 경우 안전한 fallback 수행
                        if not result:
                            fallback_handler = (
                                NonStreamingAgentHandlerWithToolOutput()
                                if return_intermediate_steps
                                else NonStreamingAgentHandler()
                            )
                        else:
                            raw_output = "".join(handler.streamed_tokens).strip()
                            if return_intermediate_steps and handler.tool_logs:
                                tool_log_text = "\n".join(handler.tool_logs)
                                if tool_log_text and raw_output:
                                    result = f"{tool_log_text}\n\n{raw_output}"
                                else:
                                    result = tool_log_text or raw_output
                            else:
                                result = raw_output

                            # 스트리밍 중 수집된 결과가 비어있는 경우 안전한 fallback 수행
                            if not result:
                                result = _invoke_tool_sync()
                    else:
                        result = _invoke_tool_sync()
                else:
                    if stream_emitter:
                        chain = streaming_chain or (prompt_template_without_tool | llm)
                        collected_text = []
                        try:
                            for chunk in chain.stream(inputs):
                                content = getattr(chunk, "content", "") or ""
                                if content:
                                    stream_emitter.emit_llm_chunk(todo_index, iteration, content)
                                    collected_text.append(content)
                        except Exception as streaming_exc:  # pragma: no cover - runtime safety
                            streaming_error = streaming_exc
                            logger.error(
                                f"Streaming simple response failed (iteration {iteration}): {streaming_exc}",
                                exc_info=True,
                            )
                            stream_emitter.emit_iteration_error(todo_index, iteration, streaming_exc)

                        result = "".join(collected_text).strip()

                        if streaming_error:
                            raise streaming_error

                        if not result:
                            if not sync_chain:
                                sync_chain = prompt_template_without_tool | llm | StrOutputParser()
                            result = sync_chain.invoke(inputs)
                    else:
                        if not sync_chain:
                            sync_chain = prompt_template_without_tool | llm | StrOutputParser()
                        result = sync_chain.invoke(inputs)

                # 도구 실행 결과 저장
                import time
                clean_result = _sanitize_result_text(result)

                if stream_emitter:
                    stream_emitter.emit_iteration_complete(todo_index, iteration, clean_result)

                result_signature = clean_result.strip()

                seen_results = state.get("seen_results") or []
                seen_results = [entry for entry in seen_results if entry]
                duplicate_result = False
                if result_signature:
                    if result_signature in seen_results:
                        duplicate_result = True
                    else:
                        if len(seen_results) >= 10:
                            seen_results = seen_results[-9:]
                        seen_results.append(result_signature)

                result_frequencies = state.get("result_frequencies") or {}
                if result_signature:
                    result_frequencies[result_signature] = (
                        result_frequencies.get(result_signature, 0) + 1
                    )
                    if len(result_frequencies) > 25:
                        # 오래된 항목 제거
                        sorted_items = sorted(
                            result_frequencies.items(),
                            key=lambda item: item[1],
                            reverse=True,
                        )[:20]
                        result_frequencies = dict(sorted_items)

                duplicate_run_length = state.get("duplicate_run_length", 0)
                if duplicate_result:
                    duplicate_run_length += 1
                else:
                    duplicate_run_length = 0

                tool_result = {
                    "iteration": iteration,
                    "result": result,
                    "clean_result": clean_result,
                    "timestamp": time.time(),
                    "duplicate": duplicate_result,
                }
                
                new_tool_results = state["tool_results"] + [tool_result]
                
                if duplicate_result and stream_emitter:
                    stream_emitter.emit_status(
                        f"<FEEDBACK_STATUS>경고: 동일한 출력이 연속 {duplicate_run_length}회 감지되었습니다. 접근 방식을 변경하세요.</FEEDBACK_STATUS>\n"
                    )

                return {
                    **state,
                    "tool_results": new_tool_results,
                    "messages": state["messages"] + [{"role": "assistant", "content": f"Iteration {iteration}: {clean_result}"}],
                    "seen_results": seen_results,
                    "last_result_signature": result_signature,
                    "last_result_duplicate": duplicate_result,
                    "result_frequencies": result_frequencies,
                    "duplicate_run_length": duplicate_run_length,
                }
                
            except Exception as e:
                logger.error(f"Task execution failed: {str(e)}")
                if stream_emitter:
                    stream_emitter.emit_iteration_error(
                        state.get("current_todo_index"),
                        state.get("iteration_count"),
                        e,
                    )
                import time
                clean_error = f"Error: {str(e)}"
                error_result = {
                    "iteration": state["iteration_count"],
                    "result": clean_error,
                    "clean_result": clean_error,
                    "timestamp": time.time(),
                    "error": True
                }
                error_signature = clean_error.strip()
                seen_results = state.get("seen_results") or []
                seen_results = [entry for entry in seen_results if entry]
                if error_signature and error_signature not in seen_results:
                    if len(seen_results) >= 10:
                        seen_results = seen_results[-9:]
                    seen_results.append(error_signature)
                return {
                    **state,
                    "tool_results": state["tool_results"] + [error_result],
                    "messages": state["messages"] + [{"role": "assistant", "content": f"Error in iteration {state['iteration_count']}: {clean_error}"}],
                    "seen_results": seen_results,
                    "last_result_signature": error_signature,
                    "last_result_duplicate": False,
                    "result_frequencies": state.get("result_frequencies") or {},
                    "duplicate_run_length": 0,
                }

        def execute_direct(state: FeedbackState) -> FeedbackState:
            """단일 실행 경로 (TODO 생략)"""
            direct_state = execute_task(state)
            return {**direct_state, "execution_mode": "direct"}

        def execution_router(state: FeedbackState) -> FeedbackState:
            """실행 모드 분기 준비"""
            return state

        def route_execution_mode(state: FeedbackState) -> str:
            """실행 경로 결정"""
            return "direct" if state.get("execution_mode") == "direct" else "todo"

        def evaluate_feedback(state: FeedbackState) -> FeedbackState:
            """결과에 대한 피드백 평가"""
            try:
                todo_index = state.get("current_todo_index")
                iteration = state.get("iteration_count")
                todo_title = state.get("current_todo_title", "")

                if state.get("skip_feedback_eval"):
                    remediation_notes = state.get("remediation_notes") or []
                    if not state["tool_results"]:
                        if stream_emitter:
                            stream_emitter.emit_feedback_score(
                                todo_index,
                                iteration,
                                {
                                    "score": 10,
                                    "reasoning": "직접 실행 경로에서 평가할 결과가 없습니다.",
                                    "improvements": [],
                                    "strengths": [],
                                },
                            )
                        return {
                            **state,
                            "feedback_score": 10,
                            "remediation_notes": remediation_notes,
                            "stagnation_count": 0,
                        }

                    latest_result = state["tool_results"][-1]
                    score = latest_result.get("feedback_score") or 10
                    evaluation_payload = latest_result.get("evaluation") or {
                        "score": score,
                        "reasoning": "직접 실행 경로에서 자동 평가를 건너뛰고 승인되었습니다.",
                        "improvements": [],
                        "strengths": ["빠른 직접 실행"],
                    }

                    updated_tool_results = state["tool_results"][:-1] + [
                        {
                            **latest_result,
                            "feedback_score": score,
                            "evaluation": evaluation_payload,
                        }
                    ]

                    if stream_emitter:
                        stream_emitter.emit_feedback_score(todo_index, iteration, evaluation_payload)

                    return {
                        **state,
                        "feedback_score": score,
                        "tool_results": updated_tool_results,
                        "remediation_notes": remediation_notes,
                        "stagnation_count": 0,
                    }

                # enable_auto_feedback가 False인 경우 피드백 평가를 건너뛰고 기본 점수 설정
                if not enable_auto_feedback:
                    if not state["tool_results"]:
                        if stream_emitter:
                            stream_emitter.emit_feedback_score(todo_index, iteration, {
                                "score": 0,
                                "reasoning": "자동 피드백이 비활성화됨",
                                "improvements": [],
                                "strengths": []
                            })
                        return {
                            **state,
                            "feedback_score": 0,
                            "remediation_notes": state.get("remediation_notes") or [],
                            "stagnation_count": state.get("stagnation_count") or 0,
                        }

                    latest_result = state["tool_results"][-1]
                    updated_tool_results = state["tool_results"][:-1] + [{
                        **latest_result,
                        "feedback_score": 5,  # 중간 점수로 설정
                        "evaluation": {"score": 5, "reasoning": "자동 피드백이 비활성화됨", "improvements": "", "strengths": ""}
                    }]

                    if stream_emitter:
                        stream_emitter.emit_feedback_score(todo_index, iteration, {
                            "score": 5,
                            "reasoning": "자동 피드백이 비활성화됨",
                            "improvements": "",
                            "strengths": ""
                        })

                    return {
                        **state,
                        "feedback_score": 5,
                        "tool_results": updated_tool_results,
                        "remediation_notes": state.get("remediation_notes") or [],
                        "stagnation_count": state.get("stagnation_count") or 0,
                    }

                if not state["tool_results"]:
                    if stream_emitter:
                        stream_emitter.emit_feedback_score(todo_index, iteration, {
                            "score": 0,
                            "reasoning": "평가할 결과가 없습니다.",
                            "improvements": [],
                            "strengths": []
                        })
                    return {
                        **state,
                        "feedback_score": 0,
                        "remediation_notes": state.get("remediation_notes") or [],
                        "stagnation_count": state.get("stagnation_count") or 0,
                    }

                latest_result = state["tool_results"][-1]
                latest_clean = latest_result.get("clean_result", latest_result.get("result", ""))
                duplicate_result = latest_result.get("duplicate") or state.get("last_result_duplicate", False)
                remediation_notes = state.get("remediation_notes") or []
                if latest_result.get("error"):
                    if stream_emitter:
                        stream_emitter.emit_feedback_score(todo_index, iteration, {
                            "score": 1,
                            "reasoning": "도중에 오류가 발생했습니다.",
                            "improvements": [str(latest_clean)],
                            "strengths": []
                        })
                    error_note = f"오류 재발 방지: {latest_clean or '원인을 재검토하세요'}"
                    updated_notes = [error_note] + remediation_notes
                    deduped_notes = []
                    for note in updated_notes:
                        if note and note not in deduped_notes:
                            deduped_notes.append(note)
                    updated_tool_results = state["tool_results"][:-1] + [{
                        **latest_result,
                        "feedback_score": 1,
                        "evaluation": {
                            "score": 1,
                            "reasoning": "도구 실행 중 오류가 발생했습니다.",
                            "improvements": [str(latest_clean)],
                            "strengths": [],
                        },
                    }]
                    return {
                        **state,
                        "feedback_score": 1,
                        "remediation_notes": deduped_notes[:8],
                        "tool_results": updated_tool_results,
                        "stagnation_count": state.get("stagnation_count", 0) + 1,
                    }
                
                # 피드백 평가 프롬프트 생성
                original_request = state.get("original_user_request") or state.get("user_input")
                todo_description = state.get("current_todo_description", "")
                todo_index = state.get("current_todo_index")
                total_todos = state.get("total_todos")
                previous_context = state.get("previous_results_context") or "이전 TODO 결과 없음"

                todo_pos = ""
                if todo_index and total_todos:
                    todo_pos = f" (순번: {todo_index}/{total_todos})"

                evaluation_prompt = f"""
원본 사용자 요청 요약: {original_request}

현재 TODO 정보{todo_pos}:
- 제목: {state.get("current_todo_title", 'Untitled')}
- 설명: {todo_description or '설명 없음'}

참고할 이전 TODO 결과:
{previous_context}

피드백 기준: {feedback_criteria if feedback_criteria else "일반적인 품질, 정확성, 완성도"}

현재 TODO 실행 결과: {latest_clean}

중요: 오직 현재 TODO 목표 달성 여부만 평가하세요. 이후 TODO에서 처리될 내용을 현재 결과에 없다는 이유로 감점하지 마세요. 반복된 작업이나 이전 단계와의 불일치, 향후 단계 준비 미흡 여부는 정확성/안정성 평가에 반영하세요.

추가 고려 사항:
- 속도: 불필요한 반복 없이 목표를 달성했는가?
- 안정성: 이전 개선 사항과 컨텍스트를 반영했는가?
- 정확도: 현재 TODO 목표를 정확히 충족했는가?
- 결과 적합성: 실행 결과가 실제 출력 데이터를 포함하거나, 데이터가 비어있을 경우 그 사실을 명시했는가? 단순히 쿼리만 제시하고 결과를 보고하지 않았다면 감점하세요.

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

                try:
                    score = int(evaluation_result.get("score", 0))
                except (TypeError, ValueError):
                    score = 0

                import re as _re

                def _normalize_notes(value):
                    notes: List[str] = []
                    if not value:
                        return notes
                    if isinstance(value, list):
                        notes = [str(item).strip() for item in value if str(item).strip()]
                    elif isinstance(value, str):
                        notes = [segment.strip() for segment in _re.split(r"[\n;,]+", value) if segment.strip()]
                    else:
                        notes = [str(value).strip()]
                    return notes

                improvements_list = _normalize_notes(evaluation_result.get("improvements"))
                if not improvements_list:
                    reasoning_text = evaluation_result.get("reasoning")
                    if reasoning_text:
                        reasoning_notes = _normalize_notes(reasoning_text)
                        improvements_list.extend(reasoning_notes)
                if not improvements_list and evaluation_result.get("reasoning"):
                    reasoning_note = str(evaluation_result.get("reasoning")).strip()
                    if reasoning_note:
                        improvements_list.append(reasoning_note)
                duplicate_result = latest_result.get("duplicate") or state.get("last_result_duplicate", False)
                duplicate_run_length = state.get("duplicate_run_length", 0)
                if duplicate_result:
                    improvements_list.append("이전 출력과 동일합니다. 접근 방식을 변경하세요.")
                    if duplicate_run_length >= 2:
                        improvements_list.append(
                            f"연속 {duplicate_run_length}회 동일 출력이 감지되었습니다. 다른 도구를 사용하거나 프롬프트 구조를 재구성하세요."
                        )

                if not _has_explicit_output(latest_clean):
                    reminder = "쿼리 실행 결과(데이터 또는 비어 있음을 명시)를 함께 제공하세요."
                    if reminder not in improvements_list:
                        improvements_list.append(reminder)
                    score = min(score, feedback_threshold - 1, 6)

                evaluation_result["score"] = score
                evaluation_result["improvements"] = improvements_list

                remediation_notes = state.get("remediation_notes") or []
                combined_notes = improvements_list + remediation_notes
                deduped_notes: List[str] = []
                for note in combined_notes:
                    if note and note not in deduped_notes:
                        deduped_notes.append(note)
                remediation_notes_updated = deduped_notes[:8]

                previous_scores = [
                    entry.get("feedback_score", 0)
                    for entry in state["tool_results"][:-1]
                    if entry.get("feedback_score") is not None
                ]
                best_previous_score = max(previous_scores) if previous_scores else None
                stagnation_count = state.get("stagnation_count") or 0
                if best_previous_score is None:
                    stagnation_count = 0
                else:
                    if (score <= best_previous_score and not improvements_list) or duplicate_result:
                        stagnation_count += 1
                    else:
                        stagnation_count = 0

                if duplicate_run_length >= 2:
                    stagnation_count = max(stagnation_count, duplicate_run_length)

                max_iterations = state.get("max_iterations", 5)
                if max_iterations:
                    stagnation_count = min(stagnation_count, max_iterations)

                if stream_emitter:
                    stream_emitter.emit_feedback_score(todo_index, iteration, evaluation_result)
                
                # 결과 업데이트
                updated_tool_results = state["tool_results"][:-1] + [{
                    **latest_result,
                    "feedback_score": score,
                    "evaluation": {
                        **evaluation_result,
                        "score": score,
                        "improvements": improvements_list,
                    }
                }]
                
                return {
                    **state,
                    "feedback_score": score,
                    "tool_results": updated_tool_results,
                    "remediation_notes": remediation_notes_updated,
                    "stagnation_count": stagnation_count,
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

            if state.get("stagnation_count", 0) >= 2:
                return "finalize"

            if state.get("duplicate_run_length", 0) >= 3:
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
                    final_result = best_result.get("clean_result", best_result.get("result", ""))
                    requirements_met = best_result.get("feedback_score", 0) >= feedback_threshold
                if stream_emitter:
                    stream_emitter.emit_todo_finalization(
                        state.get("current_todo_index"),
                        state.get("current_todo_title", ""),
                        final_result,
                        requirements_met,
                    )
                
                return {
                    **state,
                    "final_result": final_result,
                    "final_result_clean": final_result,
                    "requirements_met": requirements_met
                }
                
            except Exception as e:
                logger.error(f"Result finalization failed: {str(e)}")
                if stream_emitter:
                    stream_emitter.emit_todo_finalization(
                        state.get("current_todo_index"),
                        state.get("current_todo_title", ""),
                        f"Error during finalization: {str(e)}",
                        False,
                    )
                return {
                    **state,
                    "final_result": f"Error during finalization: {str(e)}",
                    "requirements_met": False
                }

        # 그래프 구성
        workflow = StateGraph(FeedbackState)
        
        # 노드 추가
        workflow.add_node("execution_router", execution_router)
        workflow.add_node("execute_direct", execute_direct)
        workflow.add_node("execute_task", execute_task)
        workflow.add_node("evaluate_feedback", evaluate_feedback)
        workflow.add_node("increment_iteration", increment_iteration)
        workflow.add_node("finalize_result", finalize_result)
        
        # 엣지 설정
        workflow.set_entry_point("execution_router")
        workflow.add_conditional_edges(
            "execution_router",
            route_execution_mode,
            {
                "direct": "execute_direct",
                "todo": "execute_task"
            }
        )
        workflow.add_edge("execute_task", "evaluate_feedback")
        workflow.add_conditional_edges(
            "evaluate_feedback",
            check_completion,
            {
                "continue": "increment_iteration",
                "finalize": "finalize_result"
            }
        )
        workflow.add_edge("execute_direct", "evaluate_feedback")
        workflow.add_edge("increment_iteration", "execute_task")
        workflow.add_edge("finalize_result", END)
        
        return workflow.compile(checkpointer=memory)

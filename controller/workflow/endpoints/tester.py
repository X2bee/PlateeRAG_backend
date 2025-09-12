"""
워크플로우 테스터 관련 엔드포인트들
"""
import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from controller.helper.controllerHelper import extract_user_id_from_request
from controller.helper.singletonHelper import get_db_manager, get_config_composer
from controller.workflow.models.requests import TesterExecuteRequest, TesterTestCase, TesterTestResult, WorkflowRequest
from controller.helper.utils.data_parsers import parse_input_data
from controller.helper.utils.workflow_helpers import workflow_parameter_helper
from controller.helper.utils.llm_evaluators import evaluate_with_llm
from service.database.models.executor import ExecutionIO
from editor.async_workflow_executor import execution_manager
from service.database.logger_helper import create_logger

logger = logging.getLogger("tester-endpoints")
router = APIRouter()

# 테스터 작업 상태 저장용 (메모리 기반)
tester_status_storage = {}

async def execute_single_workflow_for_tester_with_callback(
    user_id: str,
    workflow_name: str,
    workflow_id: str,
    input_data: str,
    interaction_id: str,
    selected_collections: List[str],
    app_db,
    test_case: TesterTestCase,
    callback=None,
    expected_output=None
) -> Dict[str, Any]:
    """
    개별 워크플로우 실행 후 콜백 호출
    """
    start_time = time.time()

    try:
        downloads_path = os.path.join(os.getcwd(), "downloads")
        download_path_id = os.path.join(downloads_path, user_id)

        filename = f"{workflow_name}.json" if not workflow_name.endswith('.json') else workflow_name
        file_path = os.path.join(download_path_id, filename)

        if not os.path.exists(file_path):
            raise ValueError(f"워크플로우 파일을 찾을 수 없습니다: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            workflow_data = json.load(f)

        temp_request = WorkflowRequest(
            workflow_name=workflow_name,
            workflow_id=workflow_id,
            input_data=input_data,
            interaction_id=interaction_id,
            selected_collections=selected_collections
        )
        workflow_data = await workflow_parameter_helper(temp_request, workflow_data)

        if not workflow_data or 'nodes' not in workflow_data or 'edges' not in workflow_data:
            raise ValueError(f"워크플로우 데이터가 유효하지 않습니다")

        if input_data is not None:
            for node in workflow_data.get('nodes', []):
                if node.get('data', {}).get('functionId') == 'startnode':
                    parameters = node.get('data', {}).get('parameters', [])
                    if parameters and isinstance(parameters, list):
                        parameters[0]['value'] = input_data
                        break

        executor = execution_manager.create_executor(
            workflow_data=workflow_data,
            db_manager=app_db,
            interaction_id=interaction_id,
            user_id=user_id,
            expected_output=expected_output,
            test_mode=True
        )

        final_outputs = []
        async for chunk in executor.execute_workflow_async():
            final_outputs.append(chunk)

        if len(final_outputs) == 1:
            processed_output = final_outputs[0]
        elif len(final_outputs) > 1:
            if all(isinstance(item, str) for item in final_outputs):
                processed_output = ''.join(final_outputs)
            else:
                processed_output = final_outputs[-1]
        else:
            processed_output = "결과 없음"

        execution_time = int((time.time() - start_time) * 1000)

        result = {
            "success": True,
            "outputs": processed_output,
            "execution_time": execution_time
        }

        # 콜백 호출 - 개별 테스트 케이스 완료 시
        if callback:
            tester_result = TesterTestResult(
                id=test_case.id,
                input=test_case.input,
                expected_output=test_case.expected_output,
                actual_output=str(processed_output),
                status="success",
                execution_time=execution_time,
                error=None,
                llm_eval_score=None
            )
            await callback(tester_result)

        return result

    except Exception as e:
        execution_time = int((time.time() - start_time) * 1000)
        logger.error(f"테스터 워크플로우 실행 실패: {str(e)}")

        result = {
            "success": False,
            "error": str(e),
            "execution_time": execution_time
        }

        # 콜백 호출 - 에러 발생 시
        if callback:
            tester_result = TesterTestResult(
                id=test_case.id,
                input=test_case.input,
                expected_output=test_case.expected_output,
                actual_output=None,
                status="error",
                execution_time=execution_time,
                error=str(e),
                llm_eval_score=None
            )
            await callback(tester_result)

        return result

async def process_batch_group(
    user_id: str,
    workflow_name: str,
    workflow_id: str,
    test_cases: List[TesterTestCase],
    interaction_id: str,
    selected_collections: List[str],
    batch_id: str,
    app_db,
    individual_result_callback=None,
) -> List[TesterTestResult]:
    """
    배치 그룹을 병렬로 처리하며 개별 완료 시마다 콜백 호출
    """
    results = []

    # asyncio.gather를 사용해서 병렬 실행
    tasks = []
    for test_case in test_cases:
        unique_interaction_id = f"{interaction_id}____{workflow_name}____{batch_id}____{test_case.id}"
        task = execute_single_workflow_for_tester_with_callback(
            user_id=user_id,
            workflow_name=workflow_name,
            workflow_id=workflow_id,
            input_data=test_case.input,
            interaction_id=unique_interaction_id,
            selected_collections=selected_collections,
            app_db=app_db,
            test_case=test_case,
            callback=individual_result_callback,
            expected_output=test_case.expected_output,
        )
        tasks.append(task)

    # 모든 태스크를 병렬로 실행
    execution_results = await asyncio.gather(*tasks, return_exceptions=True)

    # 결과 처리
    for test_case, exec_result in zip(test_cases, execution_results):
        if isinstance(exec_result, Exception):
            result = TesterTestResult(
                id=test_case.id,
                input=test_case.input,
                expected_output=test_case.expected_output,
                actual_output=None,
                status="error",
                execution_time=0,
                error=str(exec_result),
                llm_eval_score=None
            )
        elif exec_result.get("success"):
            outputs = exec_result.get("outputs", "결과 없음")
            if isinstance(outputs, list):
                actual_output = outputs[0] if outputs else "결과 없음"
            else:
                actual_output = str(outputs)

            result = TesterTestResult(
                id=test_case.id,
                input=test_case.input,
                expected_output=test_case.expected_output,
                actual_output=actual_output,
                status="success",
                execution_time=exec_result.get("execution_time", 0),
                error=None,
                llm_eval_score=None
            )
        else:
            result = TesterTestResult(
                id=test_case.id,
                input=test_case.input,
                expected_output=test_case.expected_output,
                actual_output=None,
                status="error",
                execution_time=exec_result.get("execution_time", 0),
                error=exec_result.get("error", "알 수 없는 오류"),
                llm_eval_score=None
            )

        results.append(result)

    # 진행 상황 업데이트
    if batch_id in tester_status_storage:
        tester_status_storage[batch_id]["completed_count"] += len(test_cases)
        progress = (tester_status_storage[batch_id]["completed_count"] /
                   tester_status_storage[batch_id]["total_count"]) * 100
        tester_status_storage[batch_id]["progress"] = progress

    return results

@router.get("/io_logs")
async def get_workflow_io_logs_for_tester(request: Request, workflow_name: str):
    """
    특정 워크플로우의 ExecutionIO 로그를 interaction_batch_id별로 그룹화하여 반환합니다.
    """
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        backend_log.info("Retrieving workflow tester IO logs",
                        metadata={"workflow_name": workflow_name})

        result = app_db.find_by_condition(
            ExecutionIO,
            {
                "user_id": user_id,
                "workflow_name": workflow_name,
                "test_mode": True
            },
            limit=1000000,
            orderby="updated_at",
            orderby_asc=True,
            return_list=True
        )

        if not result:
            backend_log.info("No tester IO logs found",
                           metadata={"workflow_name": workflow_name})
            logger.info(f"No performance data found for workflow: {workflow_name}")
            return JSONResponse(content={
                "workflow_name": workflow_name,
                "response_data_list": [],
                "message": "No in_out_logs data found for this workflow"
            })

        # interaction_batch_id별로 그룹화
        tester_groups = {}

        for idx, row in enumerate(result):
            interaction_id = row['interaction_id']

            # interaction_id에서 마지막 숫자를 제외한 배치 ID 추출
            parts = interaction_id.split('____')
            if len(parts) >= 4:
                interaction_batch_id = '____'.join(parts[:-1])
            else:
                interaction_batch_id = interaction_id

            if interaction_batch_id not in tester_groups:
                tester_groups[interaction_batch_id] = []

            # input_data 파싱
            raw_input_data = json.loads(row['input_data']).get('result', None) if row['input_data'] else None
            parsed_input_data = parse_input_data(raw_input_data) if raw_input_data else None

            # interaction_id에서 마지막 번호를 추출하여 log_id로 사용
            parts = interaction_id.split('____')
            if len(parts) >= 4 and parts[-1].isdigit():
                log_id = int(parts[-1])
            else:
                log_id = len(tester_groups[interaction_batch_id]) + 1

            log_entry = {
                "log_id": log_id,
                "interaction_id": row['interaction_id'],
                "workflow_name": row['workflow_name'],
                "workflow_id": row['workflow_id'],
                "input_data": parsed_input_data,
                "output_data": json.loads(row['output_data']).get('result', None) if row['output_data'] else None,
                "expected_output": row['expected_output'],
                "llm_eval_score": row['llm_eval_score'],
                "updated_at": row['updated_at'].isoformat() if isinstance(row['updated_at'], datetime) else row['updated_at']
            }
            tester_groups[interaction_batch_id].append(log_entry)

        # 각 테스터 그룹을 response_data 형태로 변환
        response_data_list = []
        for interaction_batch_id, performance_stats in tester_groups.items():
            response_data = {
                "workflow_name": workflow_name,
                "interaction_batch_id": interaction_batch_id,
                "in_out_logs": performance_stats,
                "message": "In/Out logs retrieved successfully"
            }
            response_data_list.append(response_data)

        final_response = {
            "workflow_name": workflow_name,
            "response_data_list": response_data_list,
            "message": f"In/Out logs retrieved successfully for {len(response_data_list)} tester groups"
        }

        backend_log.success("Successfully retrieved workflow tester IO logs",
                          metadata={"workflow_name": workflow_name,
                                  "tester_groups": len(response_data_list),
                                  "total_logs": len(result)})

        logger.info(f"Performance stats retrieved for workflow: {workflow_name}, {len(response_data_list)} tester groups")
        return JSONResponse(content=final_response)

    except Exception as e:
        backend_log.error("Failed to retrieve workflow tester IO logs", exception=e,
                         metadata={"workflow_name": workflow_name})
        logger.error(f"Error retrieving workflow performance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve performance data: {str(e)}")

@router.delete("/io_logs")
async def delete_workflow_io_logs_for_tester(request: Request, workflow_name: str, interaction_batch_id: str):
    """
    특정 워크플로우의 ExecutionIO 로그를 삭제합니다.
    """
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        backend_log.info("Starting workflow tester IO logs deletion",
                        metadata={"workflow_name": workflow_name, "interaction_batch_id": interaction_batch_id})

        existing_data = app_db.find_by_condition(
            ExecutionIO,
            {
                "user_id": user_id,
                "workflow_name": workflow_name,
                "interaction_id__like__": interaction_batch_id,
                "test_mode": True
            },
            limit=1000000
        )

        delete_count = len(existing_data) if existing_data else 0

        if delete_count == 0:
            backend_log.info("No tester logs found to delete",
                           metadata={"workflow_name": workflow_name, "interaction_batch_id": interaction_batch_id})
            logger.info(f"No logs found to delete for workflow: {workflow_name}")
            return JSONResponse(content={
                "workflow_name": workflow_name,
                "deleted_count": 0,
                "message": "No logs found to delete"
            })

        app_db.delete_by_condition(
            ExecutionIO,
            {
                "user_id": user_id,
                "workflow_name": workflow_name,
                "interaction_id__like__": interaction_batch_id,
                "test_mode": True,
            }
        )

        backend_log.success("Successfully deleted workflow tester IO logs",
                          metadata={"workflow_name": workflow_name,
                                  "interaction_batch_id": interaction_batch_id,
                                  "deleted_count": delete_count})

        logger.info(f"Successfully deleted {delete_count} logs for workflow: {workflow_name}")

        return JSONResponse(content={
            "workflow_name": workflow_name,
            "deleted_count": delete_count,
            "interaction_batch_id": interaction_batch_id,
            "message": f"Successfully deleted {delete_count} execution logs"
        })

    except HTTPException:
        raise
    except Exception as e:
        backend_log.error("Failed to delete workflow tester IO logs", exception=e,
                         metadata={"workflow_name": workflow_name,
                                 "interaction_batch_id": interaction_batch_id,
                                 "expected_delete_count": delete_count if 'delete_count' in locals() else 0})
        logger.error(f"Error deleting workflow logs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete workflow logs: {str(e)}")

@router.post("/stream")
async def execute_workflow_tester_stream(request: Request, tester_request: TesterExecuteRequest):
    """
    워크플로우 테스터 실행 스트리밍 엔드포인트
    여러 테스트 케이스를 배치로 처리하며 개별 완료 시마다 실시간 진행 상황을 SSE로 스트리밍합니다.
    """
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    app_db = get_db_manager(request)
    config_composer = get_config_composer(request=request)
    backend_log = create_logger(app_db, user_id, request)

    async def tester_stream_generator():
        batch_id = str(uuid.uuid4())
        start_time = time.time()
        completed_count = 0
        result_queue = asyncio.Queue()

        try:
            backend_log.info("Starting workflow tester streaming execution",
                            metadata={"workflow_name": tester_request.workflow_name,
                                    "workflow_id": tester_request.workflow_id,
                                    "interaction_id": tester_request.interaction_id,
                                    "batch_id": batch_id,
                                    "total_test_cases": len(tester_request.test_cases),
                                    "batch_size": tester_request.batch_size,
                                    "llm_eval_enabled": tester_request.llm_eval_enabled,
                                    "llm_eval_type": tester_request.llm_eval_type,
                                    "llm_eval_model": tester_request.llm_eval_model})

            tester_status_storage[batch_id] = {
                "status": "running",
                "total_count": len(tester_request.test_cases),
                "completed_count": 0,
                "progress": 0.0,
                "start_time": start_time
            }

            initial_message = {
                "type": "tester_start",
                "batch_id": batch_id,
                "total_count": len(tester_request.test_cases),
                "batch_size": tester_request.batch_size,
                "workflow_name": tester_request.workflow_name
            }
            yield f"data: {json.dumps(initial_message, ensure_ascii=False)}\n\n"

            logger.info(f"테스터 스트림 {batch_id} 시작: 워크플로우={tester_request.workflow_name}, "
                       f"테스트 케이스={len(tester_request.test_cases)}개, 배치 크기={tester_request.batch_size}")

            all_results = []

            async def individual_completion_callback(result: TesterTestResult):
                await result_queue.put(result)

            async def batch_processor():
                nonlocal all_results
                try:
                    for i in range(0, len(tester_request.test_cases), tester_request.batch_size):
                        batch_group = tester_request.test_cases[i:i + tester_request.batch_size]
                        group_number = i // tester_request.batch_size + 1

                        logger.info(f"배치 그룹 {group_number} 처리 중: {len(batch_group)}개 병렬 실행")

                        group_results = await process_batch_group(
                            user_id=user_id,
                            workflow_name=tester_request.workflow_name,
                            workflow_id=tester_request.workflow_id,
                            test_cases=batch_group,
                            interaction_id=tester_request.interaction_id,
                            selected_collections=tester_request.selected_collections,
                            batch_id=batch_id,
                            app_db=app_db,
                            individual_result_callback=individual_completion_callback,
                        )

                        all_results.extend(group_results)

                        if i + tester_request.batch_size < len(tester_request.test_cases):
                            await asyncio.sleep(0.5)

                    await result_queue.put("TESTER_COMPLETE")
                except Exception as e:
                    await result_queue.put(f"ERROR:{str(e)}")

            batch_task = asyncio.create_task(batch_processor())

            while True:
                try:
                    result = await asyncio.wait_for(result_queue.get(), timeout=1.0)

                    if result == "TESTER_COMPLETE":
                        break
                    elif isinstance(result, str) and result.startswith("ERROR:"):
                        backend_log.error("Workflow tester execution failed",
                                        metadata={"batch_id": batch_id, "error": result[6:]})
                        error_message = {
                            "type": "error",
                            "batch_id": batch_id,
                            "error": result[6:],
                            "message": "테스터 실행 중 오류가 발생했습니다"
                        }
                        yield f"data: {json.dumps(error_message, ensure_ascii=False)}\n\n"
                        break
                    elif isinstance(result, TesterTestResult):
                        completed_count += 1
                        result_message = {
                            "type": "test_result",
                            "batch_id": batch_id,
                            "result": result.dict()
                        }
                        yield f"data: {json.dumps(result_message, ensure_ascii=False)}\n\n"

                        progress = (completed_count / len(tester_request.test_cases)) * 100

                        progress_message = {
                            "type": "progress",
                            "batch_id": batch_id,
                            "completed_count": completed_count,
                            "total_count": len(tester_request.test_cases),
                            "progress": round(progress, 2),
                            "elapsed_time": int((time.time() - start_time) * 1000)
                        }
                        yield f"data: {json.dumps(progress_message, ensure_ascii=False)}\n\n"

                        if batch_id in tester_status_storage:
                            tester_status_storage[batch_id]["completed_count"] = completed_count
                            tester_status_storage[batch_id]["progress"] = progress

                except asyncio.TimeoutError:
                    continue

            # 배치 태스크 완료 대기
            await batch_task

            # LLM 평가 처리
            if tester_request.llm_eval_enabled and all_results:
                backend_log.info("Starting LLM evaluation for tester results",
                               metadata={"batch_id": batch_id, "results_count": len(all_results),
                                       "llm_eval_type": tester_request.llm_eval_type,
                                       "llm_eval_model": tester_request.llm_eval_model})

                logger.info(f"LLM 평가 시작: {len(all_results)}개 결과 평가")

                eval_progress_message = {
                    "type": "eval_start",
                    "batch_id": batch_id,
                    "message": "LLM 평가를 시작합니다..."
                }
                yield f"data: {json.dumps(eval_progress_message, ensure_ascii=False)}\n\n"

                # 각 결과에 대해 LLM 평가 수행
                for idx, result in enumerate(all_results):
                    if result.status == "success" and result.actual_output:
                        try:
                            unique_interaction_id = f"{tester_request.interaction_id}____{tester_request.workflow_name}____{batch_id}____{result.id}"

                            llm_score = await evaluate_with_llm(
                                unique_interaction_id=unique_interaction_id,
                                input_data=result.input,
                                expected_output=result.expected_output,
                                actual_output=result.actual_output,
                                llm_eval_type=tester_request.llm_eval_type,
                                llm_eval_model=tester_request.llm_eval_model,
                                app_db=app_db,
                                config_composer=config_composer
                            )

                            # LLM 평가 결과 SSE 전송
                            eval_result_message = {
                                "type": "eval_result",
                                "batch_id": batch_id,
                                "test_id": result.id,
                                "llm_eval_score": llm_score,
                                "progress": f"{idx + 1}/{len(all_results)}"
                            }
                            yield f"data: {json.dumps(eval_result_message, ensure_ascii=False)}\n\n"

                            logger.info(f"테스트 케이스 {result.id} LLM 평가 완료: 점수={llm_score}, interaction_id={unique_interaction_id}")

                        except Exception as eval_error:
                            backend_log.error("LLM evaluation failed for test case", exception=eval_error,
                                            metadata={"batch_id": batch_id, "test_id": result.id})
                            logger.error(f"테스트 케이스 {result.id} LLM 평가 실패: {str(eval_error)}")

                            eval_error_message = {
                                "type": "eval_error",
                                "batch_id": batch_id,
                                "test_id": result.id,
                                "error": str(eval_error)
                            }
                            yield f"data: {json.dumps(eval_error_message, ensure_ascii=False)}\n\n"

                # LLM 평가 완료 메시지
                eval_complete_message = {
                    "type": "eval_complete",
                    "batch_id": batch_id,
                    "message": "LLM 평가가 완료되었습니다"
                }
                yield f"data: {json.dumps(eval_complete_message, ensure_ascii=False)}\n\n"

                backend_log.info("LLM evaluation completed for tester results",
                               metadata={"batch_id": batch_id})

            # 최종 결과 계산
            total_execution_time = int((time.time() - start_time) * 1000)
            success_count = sum(1 for r in all_results if r.status == "success")
            error_count = len(all_results) - success_count

            # 테스터 상태 완료로 업데이트
            tester_status_storage[batch_id]["status"] = "completed"
            tester_status_storage[batch_id]["progress"] = 100.0

            # 최종 완료 메시지
            final_message = {
                "type": "tester_complete",
                "batch_id": batch_id,
                "total_count": len(all_results),
                "success_count": success_count,
                "error_count": error_count,
                "total_execution_time": total_execution_time,
                "message": f"테스터 처리 완료: 성공={success_count}개, 실패={error_count}개"
            }
            yield f"data: {json.dumps(final_message, ensure_ascii=False)}\n\n"

            backend_log.success("Workflow tester streaming execution completed",
                              metadata={"workflow_name": tester_request.workflow_name,
                                      "batch_id": batch_id,
                                      "total_count": len(all_results),
                                      "success_count": success_count,
                                      "error_count": error_count,
                                      "total_execution_time": total_execution_time,
                                      "llm_eval_enabled": tester_request.llm_eval_enabled})

            logger.info(f"테스터 스트림 {batch_id} 완료: 성공={success_count}개, 실패={error_count}개, "
                       f"총 소요시간={total_execution_time}ms")

        except Exception as e:
            backend_log.error("Workflow tester streaming execution failed", exception=e,
                            metadata={"batch_id": batch_id if 'batch_id' in locals() else "unknown",
                                    "workflow_name": tester_request.workflow_name,
                                    "workflow_id": tester_request.workflow_id})
            logger.error(f"테스터 스트림 실행 중 오류: {str(e)}", exc_info=True)

            if 'batch_id' in locals() and batch_id in tester_status_storage:
                tester_status_storage[batch_id]["status"] = "error"
                tester_status_storage[batch_id]["error"] = str(e)

            error_message = {
                "type": "error",
                "batch_id": batch_id if 'batch_id' in locals() else "unknown",
                "error": str(e),
                "message": "테스터 실행 중 오류가 발생했습니다"
            }
            yield f"data: {json.dumps(error_message, ensure_ascii=False)}\n\n"

    return StreamingResponse(tester_stream_generator(), media_type="text/event-stream")

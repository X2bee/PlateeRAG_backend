"""
워크플로우 실행 관련 엔드포인트들
"""
import asyncio
import os
import json
import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from controller.helper.controllerHelper import extract_user_id_from_request
from controller.helper.singletonHelper import get_db_manager
from controller.workflow.models.requests import WorkflowRequest
from controller.helper.utils.auth_helpers import workflow_user_id_extractor
from controller.helper.utils.workflow_helpers import workflow_parameter_helper, default_workflow_parameter_helper
from service.database.execution_meta_service import get_or_create_execution_meta, update_execution_meta_count
from service.database.models.executor import ExecutionIO
from editor.async_workflow_executor import execution_manager
from service.database.logger_helper import create_logger
from service.database.models.workflow import WorkflowMeta

logger = logging.getLogger("execution-endpoints")
router = APIRouter()

@router.post("/based_id", response_model=Dict[str, Any])
async def execute_workflow_with_id(request: Request, request_body: WorkflowRequest):
    """
    주어진 노드와 엣지 정보로 워크플로우를 실행합니다.
    """
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        backend_log.info("Starting workflow execution",
                        metadata={"workflow_name": request_body.workflow_name,
                                "workflow_id": request_body.workflow_id,
                                "interaction_id": request_body.interaction_id,
                                "input_data_length": len(str(request_body.input_data)) if request_body.input_data else 0})

        extracted_user_id = workflow_user_id_extractor(app_db, user_id, request_body.user_id, request_body.workflow_name)

        ## 일반채팅인 경우 미리 정의된 워크플로우를 이용하여 일반 채팅에 사용.
        if request_body.workflow_name == 'default_mode':
            default_mode_workflow_folder = os.path.join(os.getcwd(), "constants")
            file_path = os.path.join(default_mode_workflow_folder, "base_chat_workflow.json")
            with open(file_path, 'r', encoding='utf-8') as f:
                workflow_data = json.load(f)
            workflow_data = await default_workflow_parameter_helper(request, request_body, workflow_data)
            backend_log.info("Using default mode workflow",
                           metadata={"file_path": file_path})

        ## 워크플로우 실행인 경우, 해당하는 워크플로우 파일을 찾아서 사용.
        else:
            # downloads_path = os.path.join(os.getcwd(), "downloads")
            # download_path_id = os.path.join(downloads_path, extracted_user_id)

            # if not request_body.workflow_name.endswith('.json'):
            #     filename = f"{request_body.workflow_name}.json"
            # else:
            #     filename = request_body.workflow_name
            # file_path = os.path.join(download_path_id, filename)
            # with open(file_path, 'r', encoding='utf-8') as f:
            #     workflow_data = json.load(f)

            #### DB방식으로 변경중
            workflow_meta = app_db.find_by_condition(WorkflowMeta, {"user_id": extracted_user_id, "workflow_name": request_body.workflow_name}, limit=1)
            workflow_data = workflow_meta[0].workflow_data if workflow_meta else None
            if isinstance(workflow_data, str):
                workflow_data = json.loads(workflow_data)

            workflow_data = await workflow_parameter_helper(request_body, workflow_data)
            backend_log.info("Loaded custom workflow",
                           metadata={"extracted_user_id": extracted_user_id})

        if not workflow_data or 'nodes' not in workflow_data or 'edges' not in workflow_data:
            backend_log.error("Invalid workflow data structure",
                            metadata={"has_nodes": 'nodes' in workflow_data if workflow_data else False,
                                    "has_edges": 'edges' in workflow_data if workflow_data else False})
            raise ValueError(f"워크플로우 데이터가 유효하지 않습니다: {file_path}")

        # 워크플로우 메타데이터 추출
        nodes = workflow_data.get('nodes', [])
        edges = workflow_data.get('edges', [])

        print("--- 워크플로우 실행 시작 ---")
        print(f"실행 워크플로우: {request_body.workflow_name} ({request_body.workflow_id})")
        print(f"입력 데이터: {request_body.input_data}")

        ## 모든 워크플로우는 startnode가 있어야 하며, 입력 데이터는 startnode의 첫 번째 파라미터로 설정되어야 함.
        if request_body.input_data is not None and request_body.input_data != "" and len(request_body.input_data) > 0:
            for node in workflow_data.get('nodes', []):
                if node.get('data', {}).get('functionId') == 'startnode':
                    parameters = node.get('data', {}).get('parameters', [])
                    if parameters and isinstance(parameters, list):
                        parameters[0]['value'] = request_body.input_data
                        break

        ## 대화형 실행인 경우 execution_meta 생성
        execution_meta = None
        if request_body.interaction_id != "default" and app_db:
            execution_meta = await get_or_create_execution_meta(
                app_db,
                user_id,
                request_body.interaction_id,
                request_body.workflow_id,
                request_body.workflow_name,
                request_body.input_data
            )

        ## 워크플로우 실행
        executor = execution_manager.create_executor(
            workflow_data=workflow_data,
            db_manager=app_db,
            interaction_id=request_body.interaction_id,
            user_id=user_id
        )

        # 비동기 실행 및 결과 수집
        final_outputs = []
        async for output in executor.execute_workflow_async():
            final_outputs.append(output)

        ## 대화형 실행인 경우 count 증가
        if execution_meta:
            await update_execution_meta_count(app_db, execution_meta)

        response_data = {"status": "success", "message": "워크플로우 실행 완료", "outputs": final_outputs}

        if execution_meta:
            response_data["execution_meta"] = {
                "interaction_id": execution_meta.interaction_id,
                "interaction_count": execution_meta.interaction_count + 1,
                "workflow_id": execution_meta.workflow_id,
                "workflow_name": execution_meta.workflow_name
            }

        backend_log.success("Workflow execution completed successfully",
                          metadata={"workflow_name": request_body.workflow_name,
                                  "workflow_id": request_body.workflow_id,
                                  "interaction_id": request_body.interaction_id,
                                  "node_count": len(nodes),
                                  "edge_count": len(edges),
                                  "output_count": len(final_outputs),
                                  "is_interactive": execution_meta is not None,
                                  "interaction_count": execution_meta.interaction_count + 1 if execution_meta else 0})

        return response_data

    except ValueError as e:
        backend_log.error("Workflow execution validation error", exception=e,
                         metadata={"workflow_name": request_body.workflow_name,
                                 "workflow_id": request_body.workflow_id})
        logging.error(f"Workflow execution error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        backend_log.error("Workflow execution failed", exception=e,
                         metadata={"workflow_name": request_body.workflow_name,
                                 "workflow_id": request_body.workflow_id,
                                 "interaction_id": request_body.interaction_id})
        logging.error(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    finally:
        # 완료된 실행들 정리
        pass
        execution_manager.cleanup_completed_executions()

@router.post("/based_id/stream")
async def execute_workflow_with_id_stream(request: Request, request_body: WorkflowRequest):
    """
    주어진 ID를 기반으로 워크플로우를 스트리밍 방식으로 실행합니다.
    """
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    backend_log.info("Starting streaming workflow execution with ID",
                    metadata={"workflow_name": request_body.workflow_name,
                            "workflow_id": request_body.workflow_id,
                            "interaction_id": request_body.interaction_id,
                            "input_data_length": len(str(request_body.input_data)) if request_body.input_data else 0})

    extracted_user_id = workflow_user_id_extractor(app_db, user_id, request_body.user_id, request_body.workflow_name)

    async def stream_generator(async_result_generator, db_manager, user_id, workflow_req):
        """결과 제너레이터를 SSE 형식으로 변환하는 비동기 제너레이터"""
        full_response_chunks = []
        chunk_count = 0
        try:
            async for chunk in async_result_generator:
                chunk_count += 1
                full_response_chunks.append(str(chunk))
                response_chunk = {"type": "data", "content": chunk}
                yield f"data: {json.dumps(response_chunk, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.01)

            end_message = {"type": "end", "message": "Stream finished"}
            yield f"data: {json.dumps(end_message)}\n\n"

            backend_log.success("Streaming workflow execution completed",
                              metadata={"workflow_name": workflow_req.workflow_name,
                                      "workflow_id": workflow_req.workflow_id,
                                      "interaction_id": workflow_req.interaction_id,
                                      "chunk_count": chunk_count,
                                      "total_output_length": len("".join(full_response_chunks))})

        except Exception as e:
            backend_log.error("Streaming workflow execution failed", exception=e,
                            metadata={"workflow_name": workflow_req.workflow_name,
                                    "workflow_id": workflow_req.workflow_id,
                                    "interaction_id": workflow_req.interaction_id,
                                    "chunk_count": chunk_count})
            logger.error(f"스트리밍 중 오류 발생: {e}", exc_info=True)
            error_message = {"type": "error", "detail": f"스트리밍 중 오류가 발생했습니다: {str(e)}"}
            yield f"data: {json.dumps(error_message)}\n\n"
        finally:
            # 스트림 완료 후 DB 로그 업데이트
            final_text = "".join(full_response_chunks)
            if not final_text:
                backend_log.warn("Empty stream results, skipping log update",
                               metadata={"workflow_name": workflow_req.workflow_name,
                                       "interaction_id": workflow_req.interaction_id})
                logger.info("스트림 결과가 비어있어 로그를 업데이트하지 않습니다.")
                return

            try:
                logger.info(f"스트림 완료. Interaction ID [{workflow_req.interaction_id}]의 로그 업데이트 시작.")

                log_to_update_list = db_manager.find_by_condition(
                    ExecutionIO,
                    {
                        "user_id": user_id,
                        "interaction_id": workflow_req.interaction_id,
                    },
                    limit=1,
                    orderby="created_at",
                    orderby_asc=False
                )

                if not log_to_update_list:
                    backend_log.warn("ExecutionIO log not found for update",
                                   metadata={"workflow_name": workflow_req.workflow_name,
                                           "interaction_id": workflow_req.interaction_id})
                    logger.warning(f"업데이트할 ExecutionIO 로그를 찾지 못했습니다. Interaction ID: {workflow_req.interaction_id}")
                    return

                log_to_update = log_to_update_list[0]

                # output_data 필드의 JSON을 실제 결과로 수정
                output_data_dict = json.loads(log_to_update.output_data)
                output_data_dict['result'] = final_text

                # inputs 필드 업데이트
                if 'inputs' in output_data_dict and isinstance(output_data_dict['inputs'], dict):
                    for key, value in output_data_dict['inputs'].items():
                        if value == "<generator_output>":
                            output_data_dict['inputs'][key] = final_text

                log_to_update.output_data = json.dumps(output_data_dict, ensure_ascii=False)
                db_manager.update(log_to_update)

                backend_log.info("ExecutionIO log updated with streaming results",
                               metadata={"workflow_name": workflow_req.workflow_name,
                                       "interaction_id": workflow_req.interaction_id,
                                       "final_text_length": len(final_text)})
                logger.info(f"Interaction ID [{workflow_req.interaction_id}]의 로그가 최종 스트림 결과로 업데이트되었습니다.")

            except Exception as db_error:
                backend_log.error("ExecutionIO log update failed", exception=db_error,
                                metadata={"workflow_name": workflow_req.workflow_name,
                                        "interaction_id": workflow_req.interaction_id})
                logger.error(f"ExecutionIO 로그 업데이트 중 DB 오류 발생: {db_error}", exc_info=True)
            finally:
                execution_manager.cleanup_completed_executions()

    try:
        if request_body.workflow_name == 'default_mode':
            default_mode_workflow_folder = os.path.join(os.getcwd(), "constants")
            file_path = os.path.join(default_mode_workflow_folder, "base_chat_workflow.json")
            with open(file_path, 'r', encoding='utf-8') as f:
                workflow_data = json.load(f)
            workflow_data = await default_workflow_parameter_helper(request, request_body, workflow_data)
            backend_log.info("Using default mode workflow for streaming",
                           metadata={"file_path": file_path})
        else:
            # downloads_path = os.path.join(os.getcwd(), "downloads")
            # download_path_id = os.path.join(downloads_path, extracted_user_id)
            # filename = f"{request_body.workflow_name}.json" if not request_body.workflow_name.endswith('.json') else request_body.workflow_name
            # file_path = os.path.join(download_path_id, filename)
            # with open(file_path, 'r', encoding='utf-8') as f:
            #     workflow_data = json.load(f)

            #### DB방식으로 변경중
            workflow_meta = app_db.find_by_condition(WorkflowMeta, {"user_id": extracted_user_id, "workflow_name": request_body.workflow_name}, limit=1)
            workflow_data = workflow_meta[0].workflow_data if workflow_meta else None
            if isinstance(workflow_data, str):
                workflow_data = json.loads(workflow_data)

            workflow_data = await workflow_parameter_helper(request_body, workflow_data)
            backend_log.info("Loaded custom workflow for streaming",
                           metadata={"extracted_user_id": extracted_user_id})

        if not workflow_data or 'nodes' not in workflow_data or 'edges' not in workflow_data:
            backend_log.error("Invalid workflow data structure for streaming",
                            metadata={"has_nodes": 'nodes' in workflow_data if workflow_data else False,
                                    "has_edges": 'edges' in workflow_data if workflow_data else False})
            raise ValueError(f"워크플로우 데이터가 유효하지 않습니다: {file_path}")

        # 워크플로우 메타데이터 추출
        nodes = workflow_data.get('nodes', [])
        edges = workflow_data.get('edges', [])

        if request_body.input_data is not None and request_body.input_data != "" and len(request_body.input_data) > 0:
            for node in workflow_data.get('nodes', []):
                if node.get('data', {}).get('functionId') == 'startnode':
                    parameters = node.get('data', {}).get('parameters', [])
                    if parameters:
                        parameters[0]['value'] = request_body.input_data
                        break

        execution_meta = None
        if request_body.interaction_id != "default" and app_db:
            execution_meta = await get_or_create_execution_meta(
                app_db, user_id, request_body.interaction_id,
                request_body.workflow_id, request_body.workflow_name, request_body.input_data
            )

        if execution_meta:
            await update_execution_meta_count(app_db, execution_meta)

        # 비동기 실행기 생성
        executor = execution_manager.create_executor(
            workflow_data=workflow_data,
            db_manager=app_db,
            interaction_id=request_body.interaction_id,
            user_id=user_id
        )

        backend_log.info("Starting workflow streaming execution",
                        metadata={"workflow_name": request_body.workflow_name,
                                "workflow_id": request_body.workflow_id,
                                "interaction_id": request_body.interaction_id,
                                "node_count": len(nodes),
                                "edge_count": len(edges),
                                "is_interactive": execution_meta is not None})

        # 비동기 제너레이터 시작 (스트리밍용)
        result_generator = executor.execute_workflow_async_streaming()

        # StreamingResponse를 사용하여 SSE 스트림 반환
        return StreamingResponse(
            stream_generator(result_generator, app_db, user_id, request_body),
            media_type="text/event-stream"
        )

    except ValueError as e:
        backend_log.error("Streaming workflow execution validation error", exception=e,
                         metadata={"workflow_name": request_body.workflow_name,
                                 "workflow_id": request_body.workflow_id})
        logger.error(f"Workflow execution error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        backend_log.error("Streaming workflow execution setup failed", exception=e,
                         metadata={"workflow_name": request_body.workflow_name,
                                 "workflow_id": request_body.workflow_id,
                                 "interaction_id": request_body.interaction_id})
        logger.error(f"An unexpected error occurred during workflow setup: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.get("/status")
async def get_all_execution_status(request: Request):
    """
    현재 실행 중인 모든 워크플로우의 상태를 반환합니다.
    """
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        backend_log.info("Retrieving all workflow execution status")

        status_data = execution_manager.get_all_execution_status()

        backend_log.success("Successfully retrieved execution status",
                          metadata={"active_executions": len(status_data)})

        return {
            "active_executions": len(status_data),
            "executions": status_data
        }
    except Exception as e:
        backend_log.error("Failed to retrieve execution status", exception=e)
        logger.error(f"실행 상태 조회 중 오류: {e}")
        raise HTTPException(status_code=500, detail="실행 상태 조회 실패")

@router.get("/status/{execution_id}")
async def get_execution_status(request: Request, execution_id: str):
    """
    특정 워크플로우 실행의 상태를 반환합니다.
    """
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        backend_log.info("Retrieving workflow execution status",
                        metadata={"execution_id": execution_id})

        status = execution_manager.get_execution_status(execution_id)
        if status is None:
            backend_log.warn("Execution not found",
                           metadata={"execution_id": execution_id})
            raise HTTPException(status_code=404, detail="실행을 찾을 수 없습니다")

        backend_log.success("Successfully retrieved execution status",
                          metadata={"execution_id": execution_id,
                                  "status": status.get("status", "unknown")})

        return status
    except HTTPException:
        raise
    except Exception as e:
        backend_log.error("Failed to retrieve execution status", exception=e,
                         metadata={"execution_id": execution_id})
        logger.error(f"실행 상태 조회 중 오류: {e}")
        raise HTTPException(status_code=500, detail="실행 상태 조회 실패")

@router.post("/cleanup")
async def cleanup_completed_executions():
    """
    완료된 워크플로우 실행들을 정리합니다.
    """
    try:
        before_count = len(execution_manager.get_all_execution_status())
        execution_manager.cleanup_completed_executions()
        after_count = len(execution_manager.get_all_execution_status())
        cleaned_count = before_count - after_count

        return {
            "message": f"{cleaned_count}개의 완료된 실행이 정리되었습니다",
            "before_count": before_count,
            "after_count": after_count,
            "cleaned_count": cleaned_count
        }
    except Exception as e:
        logger.error(f"실행 정리 중 오류: {e}")
        raise HTTPException(status_code=500, detail="실행 정리 실패")

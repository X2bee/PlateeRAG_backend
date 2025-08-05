import os
import json
import asyncio
import logging

from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from editor.workflow_executor import WorkflowExecutor

from service.database.models.executor import ExecutionIO
from service.database.execution_meta_service import get_or_create_execution_meta, update_execution_meta_count

from controller.workflow.helper import _workflow_parameter_helper, _default_workflow_parameter_helper
from controller.workflow.utils import get_db_manager
from controller.workflow.model import WorkflowRequest

logger = logging.getLogger("workflow-controller")
router = APIRouter(prefix="/api/workflow/deploy", tags=["workflow"])

@router.get("/load/{user_id}/{workflow_id}")
async def load_workflow(request: Request, user_id: str, workflow_id: str):
    """
    특정 workflow를 로드합니다.
    """
    try:
        user_id = user_id

        downloads_path = os.path.join(os.getcwd(), "downloads")
        download_path_id = os.path.join(downloads_path, user_id)

        filename = f"{workflow_id}.json"
        file_path = os.path.join(download_path_id, filename)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")

        with open(file_path, 'r', encoding='utf-8') as f:
            workflow_data = json.load(f)

        logger.info(f"Workflow loaded successfully: {filename}")
        return JSONResponse(content=workflow_data)

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")
    except Exception as e:
        logger.error(f"Error loading workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load workflow: {str(e)}")

@router.post("/execute/based_id", response_model=Dict[str, Any])
async def execute_workflow_with_id(request: Request, request_body: WorkflowRequest):
    """
    주어진 노드와 엣지 정보로 워크플로우를 실행합니다.
    """
    try:
        user_id = request_body.user_id

        ## 일반채팅인 경우 미리 정의된 워크플로우를 이용하여 일반 채팅에 사용.
        if request_body.workflow_name == 'default_mode':
            default_mode_workflow_folder = os.path.join(os.getcwd(), "constants")
            file_path = os.path.join(default_mode_workflow_folder, "base_chat_workflow.json")
            with open(file_path, 'r', encoding='utf-8') as f:
                workflow_data = json.load(f)
            workflow_data = await _default_workflow_parameter_helper(request, request_body, workflow_data)

        ## 워크플로우 실행인 경우, 해당하는 워크플로우 파일을 찾아서 사용.
        else:
            downloads_path = os.path.join(os.getcwd(), "downloads")
            download_path_id = os.path.join(downloads_path, str(user_id))

            if not request_body.workflow_name.endswith('.json'):
                filename = f"{request_body.workflow_name}.json"
            else:
                filename = request_body.workflow_name
            file_path = os.path.join(download_path_id, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                workflow_data = json.load(f)
            workflow_data = await _workflow_parameter_helper(request_body, workflow_data)

        ## ========== 워크플로우 데이터 검증 ==========
        ## TODO 워크플로우 아이디 정합성 관련 로직 생각해볼 것
        # if workflow_data.get('workflow_id') != request_body.workflow_id:
        #     raise ValueError(f"워크플로우 ID가 일치하지 않습니다: {workflow_data.get('workflow_id')} != {request_body.workflow_id}")

        if not workflow_data or 'nodes' not in workflow_data or 'edges' not in workflow_data:
            raise ValueError(f"워크플로우 데이터가 유효하지 않습니다: {file_path}")

        print("--- 워크플로우 실행 시작 ---")
        print(f"실행 워크플로우: {request_body.workflow_name} ({request_body.workflow_id})")
        print(f"입력 데이터: {request_body.input_data}")

        ## 모든 워크플로우는 startnode가 있어야 하며, 입력 데이터는 startnode의 첫 번째 파라미터로 설정되어야 함.
        ## 사용자의 인풋은 여기에 입력되고, 워크플로우가 실행됨.
        if request_body.input_data is not None:
            for node in workflow_data.get('nodes', []):
                if node.get('data', {}).get('functionId') == 'startnode':
                    parameters = node.get('data', {}).get('parameters', [])
                    if parameters and isinstance(parameters, list):
                        parameters[0]['value'] = request_body.input_data
                        break

        ## app에 저장된 db_manager를 가져옴. 이걸 통해 DB에 접근할 수 있음.
        ## DB에 접근하여 execution 데이터를 활용하여, 기록된 대화를 가져올지 말지 결정.
        app_db = get_db_manager(request)

        ## 일반적인 실행(execution)이 아닌 경우, 즉 대화형 실행(conversation execution)인 경우
        ## interaction_id가 "default"가 아닌 경우, 대화형 실행을 위한 메타데이터를 가져오거나 생성
        ## interaction_id가 "default"인 경우, execution_meta는 None으로 설정
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

        ## 워크플로우를 실질적으로 실행 (가장중요)
        ## 워크플로우 실행 관련 로직은 WorkflowExecutor 클래스에 정의되어 있음.
        ## WorkflowExecutor 클래스는 워크플로우의 노드와 엣지를 기반으로 워크플로우를 실행하는 역할을 함.
        ## 워크플로우 실행 시, interaction_id를 전달하여 대화형 실행을 지원함. (이렇게 되는 경우, interaction_id는 대화형 실행의 ID로 사용되어 DB에 저장됨)
        ## 워크플로우 실행 결과는 final_outputs에 저장됨.
        executor = WorkflowExecutor(workflow_data, app_db, request_body.interaction_id, user_id)
        final_outputs = executor.execute_workflow()

        ## 대화형 실행인 경우 execution_meta의 값을 가지고, 이 경우에는 대화 count를 증가.
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

        return response_data

    except ValueError as e:
        logging.error(f"Workflow execution error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.post("/execute/based_id/stream")
async def execute_workflow_with_id_stream(request: Request, request_body: WorkflowRequest):
    """
    주어진 ID를 기반으로 워크플로우를 스트리밍 방식으로 실행합니다.
    """
    user_id = request_body.user_id

    user_id=str(user_id)
    app_db = get_db_manager(request)

    async def stream_generator(result_generator, db_manager, user_id, workflow_req):
        """결과 제너레이터를 SSE 형식으로 변환하는 비동기 제너레이터"""
        full_response_chunks = []
        try:
            for chunk in result_generator:
                # 클라이언트에 보낼 데이터 형식 정의 (JSON)
                full_response_chunks.append(str(chunk))
                response_chunk = {"type": "data", "content": chunk}
                yield f"data: {json.dumps(response_chunk, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.01) # 짧은 딜레이로 이벤트 스트림 안정화

            end_message = {"type": "end", "message": "Stream finished"}
            yield f"data: {json.dumps(end_message)}\n\n"

        except Exception as e:
            logger.error(f"스트리밍 중 오류 발생: {e}", exc_info=True)
            error_message = {"type": "error", "detail": f"스트리밍 중 오류가 발생했습니다: {str(e)}"}
            yield f"data: {json.dumps(error_message)}\n\n"
        finally:
            # ✨ 2. 스트림이 모두 끝난 후, 수집된 내용으로 DB 로그 업데이트
            final_text = "".join(full_response_chunks)
            if not final_text:
                logger.info("스트림 결과가 비어있어 로그를 업데이트하지 않습니다.")
                return

            try:
                logger.info(f"스트림 완료. Interaction ID [{workflow_req.interaction_id}]의 로그 업데이트 시작.")

                # 가장 최근에 생성된 로그 레코드를 찾습니다.
                log_to_update_list = db_manager.find_by_condition(
                    ExecutionIO,
                    {
                        "user_id": user_id,
                        "interaction_id": workflow_req.interaction_id,
                        # "workflow_id": workflow_req.workflow_id, # 워크플로우 ID 로직 삭제
                    },
                    limit=1,
                    orderby="created_at",
                    orderby_asc=False # 최신순 정렬
                )

                if not log_to_update_list:
                    logger.warning(f"업데이트할 ExecutionIO 로그를 찾지 못했습니다. Interaction ID: {workflow_req.interaction_id}")
                    return

                log_to_update = log_to_update_list[0]

                # output_data 필드의 JSON을 실제 결과로 수정
                output_data_dict = json.loads(log_to_update.output_data)
                output_data_dict['result'] = final_text # placeholder를 최종 텍스트로 교체

                # inputs 필드에 있던 generator placeholder도 업데이트 (선택적)
                if 'inputs' in output_data_dict and isinstance(output_data_dict['inputs'], dict):
                    for key, value in output_data_dict['inputs'].items():
                        if value == "<generator_output>":
                             output_data_dict['inputs'][key] = final_text

                # 수정된 JSON으로 레코드를 업데이트
                log_to_update.output_data = json.dumps(output_data_dict, ensure_ascii=False)
                db_manager.update(log_to_update)

                logger.info(f"Interaction ID [{workflow_req.interaction_id}]의 로그가 최종 스트림 결과로 업데이트되었습니다.")

            except Exception as db_error:
                logger.error(f"ExecutionIO 로그 업데이트 중 DB 오류 발생: {db_error}", exc_info=True)


    try:
        user_id = request_body.user_id

        user_id = int(user_id)

        if request_body.workflow_name == 'default_mode':
            default_mode_workflow_folder = os.path.join(os.getcwd(), "constants")
            file_path = os.path.join(default_mode_workflow_folder, "base_chat_workflow.json")
            with open(file_path, 'r', encoding='utf-8') as f:
                workflow_data = json.load(f)
            workflow_data = await _default_workflow_parameter_helper(request, request_body, workflow_data)
        else:
            downloads_path = os.path.join(os.getcwd(), "downloads")
            download_path_id = os.path.join(downloads_path, str(user_id))
            filename = f"{request_body.workflow_name}.json" if not request_body.workflow_name.endswith('.json') else request_body.workflow_name
            file_path = os.path.join(download_path_id, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                workflow_data = json.load(f)
            workflow_data = await _workflow_parameter_helper(request_body, workflow_data)

        ## TODO 워크플로우 아이디 정합성 관련 로직 생각해볼 것
        # if workflow_data.get('workflow_id') != request_body.workflow_id:
        #     raise ValueError(f"워크플로우 ID가 일치하지 않습니다.")
        if not workflow_data or 'nodes' not in workflow_data or 'edges' not in workflow_data:
            raise ValueError(f"워크플로우 데이터가 유효하지 않습니다: {file_path}")

        if request_body.input_data is not None:
            for node in workflow_data.get('nodes', []):
                if node.get('data', {}).get('functionId') == 'startnode':
                    parameters = node.get('data', {}).get('parameters', [])
                    if parameters:
                        parameters[0]['value'] = request_body.input_data
                        break

        app_db = get_db_manager(request)
        execution_meta = None
        if request_body.interaction_id != "default" and app_db:
            execution_meta = await get_or_create_execution_meta(
                app_db, user_id, request_body.interaction_id,
                request_body.workflow_id, request_body.workflow_name, request_body.input_data
            )

        if execution_meta:
            await update_execution_meta_count(app_db, execution_meta)

        executor = WorkflowExecutor(workflow_data, app_db, request_body.interaction_id, user_id)
        result_generator = executor.execute_workflow()

        # StreamingResponse를 사용하여 SSE 스트림 반환
        return StreamingResponse(
            stream_generator(result_generator, app_db, user_id, request_body),
            media_type="text/event-stream"
        )

    except ValueError as e:
        logger.error(f"Workflow execution error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"An unexpected error occurred during workflow setup: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")
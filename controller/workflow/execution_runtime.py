import os
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from controller.workflow.models.requests import WorkflowRequest
from controller.helper.utils.auth_helpers import workflow_user_id_extractor
from controller.workflow.helper import (
    _workflow_parameter_helper,
    _default_workflow_parameter_helper,
)
from service.database.execution_meta_service import get_or_create_execution_meta
from service.database.models.workflow import WorkflowMeta
from service.database.models.executor import ExecutionIO
from editor.async_workflow_executor import AsyncWorkflowExecutor, execution_manager

logger = logging.getLogger("workflow-execution-runtime")


@dataclass
class WorkflowExecutionContext:
    """Container with prepared workflow execution artifacts."""

    workflow_data: Dict[str, Any]
    executor: AsyncWorkflowExecutor
    execution_meta: Optional[Any]
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    extracted_user_id: str
    source: str


def _ensure_request_like(request_like: Optional[Any], app: Any) -> Any:
    """
    Provide an object exposing `.app` so helpers expecting FastAPI Request keep working.
    """
    if request_like is not None:
        return request_like

    if app is None:
        raise ValueError("Application instance is required to prepare default workflow data")

    class _RequestShim:
        def __init__(self, app_obj: Any):
            self.app = app_obj

    return _RequestShim(app)


async def prepare_workflow_execution(
    *,
    app: Any,
    app_db: Any,
    login_user_id: str,
    request_body: WorkflowRequest,
    request_like: Optional[Any] = None,
    backend_log: Optional[Any] = None,
    eager_meta_creation: bool = True,
) -> WorkflowExecutionContext:
    """
    Build workflow execution context shared by SSE and WebSocket pathways.
    """
    extracted_user_id = workflow_user_id_extractor(
        app_db,
        login_user_id,
        request_body.user_id,
        request_body.workflow_name,
    )

    workflow_data: Optional[Dict[str, Any]] = None
    source_hint = ""

    if request_body.workflow_name == "default_mode":
        default_folder = os.path.join(os.getcwd(), "constants")
        file_path = os.path.join(default_folder, "base_chat_workflow.json")

        with open(file_path, "r", encoding="utf-8") as file_obj:
            workflow_data = json.load(file_obj)

        request_for_helper = _ensure_request_like(request_like, app)
        workflow_data = await _default_workflow_parameter_helper(
            request_for_helper,
            request_body,
            workflow_data,
        )
        source_hint = file_path

        if backend_log:
            backend_log.info(
                "Loaded default workflow",
                metadata={"file_path": file_path},
            )
    else:
        workflow_meta = app_db.find_by_condition(
            WorkflowMeta,
            {"user_id": extracted_user_id, "workflow_name": request_body.workflow_name},
            limit=1,
        )

        workflow_data = workflow_meta[0].workflow_data if workflow_meta else None
        if isinstance(workflow_data, str):
            workflow_data = json.loads(workflow_data)

        workflow_data = await _workflow_parameter_helper(request_body, workflow_data)
        source_hint = f"db:{extracted_user_id}:{request_body.workflow_name}"

        if backend_log:
            backend_log.info(
                "Loaded workflow from database",
                metadata={
                    "extracted_user_id": extracted_user_id,
                    "workflow_name": request_body.workflow_name,
                },
            )

    if not workflow_data or "nodes" not in workflow_data or "edges" not in workflow_data:
        raise ValueError(
            "워크플로우 데이터가 유효하지 않습니다. source=" + (source_hint or "unknown"),
        )

    nodes = workflow_data.get("nodes", [])
    edges = workflow_data.get("edges", [])

    if request_body.input_data:
        for node in nodes:
            data = node.get("data", {})
            if data.get("functionId") == "startnode":
                parameters = data.get("parameters", [])
                if parameters and isinstance(parameters, list):
                    parameters[0]["value"] = request_body.input_data
                break

    execution_meta = None
    if eager_meta_creation and request_body.interaction_id != "default" and app_db:
        execution_meta = await get_or_create_execution_meta(
            app_db,
            login_user_id,
            request_body.interaction_id,
            request_body.workflow_id,
            request_body.workflow_name,
            request_body.input_data,
        )

    executor = execution_manager.create_executor(
        workflow_data=workflow_data,
        db_manager=app_db,
        interaction_id=request_body.interaction_id,
        user_id=login_user_id,
    )

    return WorkflowExecutionContext(
        workflow_data=workflow_data,
        executor=executor,
        execution_meta=execution_meta,
        nodes=nodes,
        edges=edges,
        extracted_user_id=extracted_user_id,
        source=source_hint,
    )


async def persist_stream_results(db_manager, user_id, workflow_req, final_text, backend_log):
    """
    Update ExecutionIO logs once streaming completes and clean up executor state.
    """
    if not final_text:
        if backend_log:
            backend_log.warn(
                "Empty stream results, skipping log update",
                metadata={
                    "workflow_name": getattr(workflow_req, "workflow_name", None),
                    "interaction_id": getattr(workflow_req, "interaction_id", None),
                },
            )
        logger.info("스트림 결과가 비어있어 로그를 업데이트하지 않습니다.")
        execution_manager.cleanup_completed_executions()
        return

    try:
        log_to_update_list = db_manager.find_by_condition(
            ExecutionIO,
            {
                "user_id": user_id,
                "interaction_id": workflow_req.interaction_id,
            },
            limit=1,
            orderby="created_at",
            orderby_asc=False,
        )

        if not log_to_update_list:
            if backend_log:
                backend_log.warn(
                    "ExecutionIO log not found for update",
                    metadata={
                        "workflow_name": getattr(workflow_req, "workflow_name", None),
                        "interaction_id": getattr(workflow_req, "interaction_id", None),
                    },
                )
            logger.warning(
                "업데이트할 ExecutionIO 로그를 찾지 못했습니다. Interaction ID: %s",
                workflow_req.interaction_id,
            )
            execution_manager.cleanup_completed_executions()
            return

        log_to_update = log_to_update_list[0]
        output_data_dict = json.loads(log_to_update.output_data)
        output_data_dict["result"] = final_text

        if "inputs" in output_data_dict and isinstance(output_data_dict["inputs"], dict):
            for key, value in output_data_dict["inputs"].items():
                if value == "<generator_output>":
                    output_data_dict["inputs"][key] = final_text

        log_to_update.output_data = json.dumps(output_data_dict, ensure_ascii=False)
        db_manager.update(log_to_update)

        if backend_log:
            backend_log.info(
                "ExecutionIO log updated with streaming results",
                metadata={
                    "workflow_name": getattr(workflow_req, "workflow_name", None),
                    "interaction_id": getattr(workflow_req, "interaction_id", None),
                    "final_text_length": len(final_text),
                },
            )
        logger.info(
            "Interaction ID [%s]의 로그가 최종 스트림 결과로 업데이트되었습니다.",
            workflow_req.interaction_id,
        )
    except Exception as db_error:
        if backend_log:
            backend_log.error(
                "ExecutionIO log update failed",
                exception=db_error,
                metadata={
                    "workflow_name": getattr(workflow_req, "workflow_name", None),
                    "interaction_id": getattr(workflow_req, "interaction_id", None),
                },
            )
        logger.error(
            "ExecutionIO 로그 업데이트 중 DB 오류 발생: %s",
            db_error,
            exc_info=True,
        )
    finally:
        execution_manager.cleanup_completed_executions()

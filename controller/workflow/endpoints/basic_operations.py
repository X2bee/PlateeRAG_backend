"""
기본 워크플로우 CRUD 작업 관련 엔드포인트들
"""
import os
import json
import copy
from datetime import datetime
import logging
import secrets
import string
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from controller.helper.controllerHelper import extract_user_id_from_request
from controller.helper.singletonHelper import get_db_manager, get_config_composer
from controller.workflow.models.requests import SaveWorkflowRequest
from service.database.models.executor import ExecutionMeta, ExecutionIO
from controller.helper.utils.auth_helpers import workflow_user_id_extractor
from controller.helper.utils.data_parsers import parse_input_data
from service.database.models.user import User
from service.database.models.workflow import WorkflowMeta, WorkflowVersion
from service.database.models.deploy import DeployMeta
from service.database.logger_helper import create_logger

logger = logging.getLogger("basic-operations")
router = APIRouter()

@router.get("/list")
async def list_workflows(request: Request):
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        # downloads_path = os.path.join(os.getcwd(), "downloads")
        # download_path_id = os.path.join(downloads_path, user_id)

        # # downloads 폴더가 존재하지 않으면 생성
        # if not os.path.exists(download_path_id):
        #     os.makedirs(download_path_id)
        #     backend_log.info("Created downloads directory for user",
        #                    metadata={"path": download_path_id})
        #     return JSONResponse(content={"workflows": []})

        # # .json 확장자를 가진 파일들만 필터링
        # workflow_files = []
        # for file in os.listdir(download_path_id):
        #     if file.endswith('.json'):
        #         workflow_files.append(file)

        #### DB방식으로 변경중
        workflow_data = app_db.find_by_condition(
                    WorkflowMeta,
                    {
                        "user_id": user_id,
                    },
                    limit=1000,
                    select_columns=["workflow_name", "updated_at"],
                    orderby="updated_at"
                )

        workflow_files = []
        for wf in workflow_data:
            workflow_files.append(f"{wf.workflow_name}")

        backend_log.success("Workflow list retrieved successfully",
                          metadata={"workflow_count": len(workflow_files),
                                  "workflow_files": workflow_files[:10]})  # 처음 10개만 로깅

        logger.info(f"Found {len(workflow_files)} workflow files")
        return JSONResponse(content={"workflows": workflow_files})

    except Exception as e:
        backend_log.error("Failed to list workflows", exception=e)
        logger.error(f"Error listing workflows: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list workflows: {str(e)}")

@router.get("/version/list")
async def list_workflow_versions(request: Request, workflow_name: str, user_id):
    login_user_id = extract_user_id_from_request(request)
    if not login_user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, login_user_id, request)

    try:
        backend_log.info("Starting workflow version list operation")
        using_id = workflow_user_id_extractor(app_db, login_user_id, user_id, workflow_name)
        workflow_data = app_db.find_by_condition(WorkflowMeta, {"user_id": using_id, "workflow_name": workflow_name}, limit=1)

        if not workflow_data or len(workflow_data) == 0:
            backend_log.warn("Workflow not found for version listing",
                           metadata={"workflow_name": workflow_name, "user_id": using_id})
            raise HTTPException(status_code=404, detail=f"Workflow '{workflow_name}' not found")

        workflow_meta_id = workflow_data[0].id

        workflow_versions = app_db.find_by_condition(
            WorkflowVersion,
            {
                "user_id": using_id,
                "workflow_meta_id": workflow_meta_id
            },
            orderby="version",
            ignore_columns=["workflow_data"],
            return_list=True
        )

        return workflow_versions
    except Exception as e:
        backend_log.error("Failed to list workflows", exception=e)
        logger.error(f"Error listing workflows: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list workflows: {str(e)}")

@router.post("/version/change")
async def update_workflow_version(request: Request, workflow_name: str, version: float):
    user_id = extract_user_id_from_request(request)
    app_db = get_db_manager(request)
    config_composer = get_config_composer(request)

    try:
        existing_data = app_db.find_by_condition(
            WorkflowMeta,
            {
                "user_id": user_id,
                "workflow_name": workflow_name
            },
            limit=1
        )

        if not existing_data:
            raise HTTPException(status_code=404, detail="Workflow not found")
        existing_data = existing_data[0]

        workflow_version = app_db.find_by_condition(WorkflowVersion, {"user_id": user_id, "workflow_meta_id": existing_data.id})
        version_list = [wv.version for wv in workflow_version]

        if version not in version_list:
            raise HTTPException(status_code=400, detail=f"Version {version} not found for workflow '{workflow_name}'")

        deploy_data = app_db.find_by_condition(
            DeployMeta,
            {
                "user_id": user_id,
                "workflow_name": workflow_name,
            },
            limit=1
        )
        if not deploy_data:
            raise HTTPException(status_code=404, detail="배포 메타데이터를 찾을 수 없습니다")
        if not deploy_data[0].is_accepted:
            raise HTTPException(status_code=400, detail="해당 워크플로우에 대한 권한이 박탈되었습니다. 편집할 수 없습니다.")

        version_data = app_db.find_by_condition(WorkflowVersion, {"user_id": user_id, "workflow_meta_id": existing_data.id, "version": version}, limit=1)[0]
        if not version_data:
            raise HTTPException(status_code=404, detail="Version not found")

        existing_data.workflow_data = version_data.workflow_data
        existing_data.current_version = version_data.version

        db_type = app_db.config_db_manager.db_type
        if db_type == "postgresql":
            update_query = "UPDATE workflow_version SET current_use = %s WHERE user_id = %s AND workflow_meta_id = %s"
        else:
            update_query = "UPDATE workflow_version SET current_use = ? WHERE user_id = ? AND workflow_meta_id = ?"
        app_db.config_db_manager.execute_update_delete(update_query, (False, user_id, existing_data.id))
        version_data.current_use = True
        app_db.update(version_data)
        app_db.update(existing_data)

        return {
            "message": "Workflow updated successfully",
            "workflow_name": existing_data.workflow_name,
            "current_version": existing_data.current_version
        }

    except Exception as e:
        logger.error(f"Failed to update workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update workflow: {str(e)}")

@router.post("/version/label-name")
async def version_label_name(request: Request, workflow_name: str, version: float, new_version_label: str):
    user_id = extract_user_id_from_request(request)
    app_db = get_db_manager(request)

    try:
        existing_data = app_db.find_by_condition(
            WorkflowMeta,
            {
                "user_id": user_id,
                "workflow_name": workflow_name
            },
            limit=1
        )

        if not existing_data:
            raise HTTPException(status_code=404, detail="Workflow not found")
        existing_data = existing_data[0]

        workflow_version = app_db.find_by_condition(WorkflowVersion, {"user_id": user_id, "workflow_meta_id": existing_data.id})
        version_list = [wv.version for wv in workflow_version]

        if version not in version_list:
            raise HTTPException(status_code=400, detail=f"Version {version} not found for workflow '{workflow_name}'")

        deploy_data = app_db.find_by_condition(
            DeployMeta,
            {
                "user_id": user_id,
                "workflow_name": workflow_name,
            },
            limit=1
        )
        if not deploy_data:
            raise HTTPException(status_code=404, detail="배포 메타데이터를 찾을 수 없습니다")
        if not deploy_data[0].is_accepted:
            raise HTTPException(status_code=400, detail="해당 워크플로우에 대한 권한이 박탈되었습니다. 편집할 수 없습니다.")

        version_data = app_db.find_by_condition(WorkflowVersion, {"user_id": user_id, "workflow_meta_id": existing_data.id, "version": version}, limit=1)[0]
        if not version_data:
            raise HTTPException(status_code=404, detail="Version not found")

        version_data.version_label = new_version_label if new_version_label and new_version_label.strip() != "" else f"v{version_data.version}"
        app_db.update(version_data)

        return {
            "message": "Workflow updated successfully",
            "workflow_name": existing_data.workflow_name,
            "version_label": version_data.version_label
        }

    except Exception as e:
        logger.error(f"Failed to update workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update workflow: {str(e)}")


@router.delete("/delete/version")
async def delete_version(request: Request, workflow_name: str, version: float):
    user_id = extract_user_id_from_request(request)
    app_db = get_db_manager(request)

    try:
        existing_data = app_db.find_by_condition(
            WorkflowMeta,
            {
                "user_id": user_id,
                "workflow_name": workflow_name
            },
            limit=1
        )

        if not existing_data:
            raise HTTPException(status_code=404, detail="Workflow not found")
        existing_data = existing_data[0]

        workflow_version = app_db.find_by_condition(WorkflowVersion, {"user_id": user_id, "workflow_meta_id": existing_data.id})
        version_list = [wv.version for wv in workflow_version]

        if version not in version_list:
            raise HTTPException(status_code=400, detail=f"Version {version} not found for workflow '{workflow_name}'")

        deploy_data = app_db.find_by_condition(
            DeployMeta,
            {
                "user_id": user_id,
                "workflow_name": workflow_name,
            },
            limit=1
        )
        if not deploy_data:
            raise HTTPException(status_code=404, detail="배포 메타데이터를 찾을 수 없습니다")
        if not deploy_data[0].is_accepted:
            raise HTTPException(status_code=400, detail="해당 워크플로우에 대한 권한이 박탈되었습니다. 편집할 수 없습니다.")

        existing_data.latest_version = max((wv.version for wv in workflow_version if wv.version != version), default=1.0)
        app_db.update(existing_data)
        app_db.delete_by_condition(WorkflowVersion, {"user_id": user_id, "workflow_meta_id": existing_data.id, "version": version})

        return {
            "message": "Workflow updated successfully",
            "workflow_name": existing_data.workflow_name,
            "latest_version": existing_data.latest_version
        }

    except Exception as e:
        logger.error(f"Failed to update workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update workflow: {str(e)}")

@router.post("/save")
async def save_workflow(request: Request, workflow_request: SaveWorkflowRequest):
    """
    Frontend에서 받은 workflow 정보를 파일로 저장합니다.
    파일명: {workflow_name}.json
    """
    login_user_id = extract_user_id_from_request(request)
    if not login_user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    if workflow_request.user_id and str(workflow_request.user_id) != str(login_user_id):
        user_id = str(workflow_request.user_id)
    else:
        user_id = login_user_id

    logger.info(f"Saving workflow for user: {user_id}, workflow name: {workflow_request.workflow_name}")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        workflow_data = workflow_request.content.model_dump()

        backend_log.info("Starting workflow save operation",
                        metadata={"workflow_name": workflow_request.workflow_name,
                                "workflow_id": workflow_request.content.workflow_id})

        existing_data = app_db.find_by_condition(
            WorkflowMeta,
            {
                "user_id": user_id,
                "workflow_name": workflow_request.workflow_name,
            },
            limit=1,
        )

        if existing_data:
            deploy_data = app_db.find_by_condition(
                DeployMeta,
                {
                    "user_id": user_id,
                    "workflow_id": existing_data[0].workflow_id if existing_data else workflow_request.content.workflow_id,
                    "workflow_name": workflow_request.workflow_name,
                },
                limit=1
            )
            if deploy_data and len(deploy_data) > 0 and not deploy_data[0].is_accepted:
                backend_log.warn("Workflow access denied - permissions revoked",
                               metadata={"workflow_name": workflow_request.workflow_name})
                raise HTTPException(status_code=400, detail="해당 이름의 워크플로우에 대한 권한이 박탈되었습니다. 해당 이름 사용이 불가능합니다.")

        # downloads_path = os.path.join(os.getcwd(), "downloads")
        # download_path_id = os.path.join(downloads_path, user_id)

        # if not os.path.exists(download_path_id):
        #     os.makedirs(download_path_id)

        # if not workflow_request.workflow_name.endswith('.json'):
        #     filename = f"{workflow_request.workflow_name}.json"
        # else:
        #     filename = workflow_request.workflow_name
        # file_path = os.path.join(download_path_id, filename)

        # nodes 수 계산
        nodes = workflow_data.get('nodes', [])
        node_count = len(nodes) if isinstance(nodes, list) else 0
        has_startnode = any(
            node.get('data', {}).get('functionId') == 'startnode' for node in nodes
        )
        has_endnode = any(
            node.get('data', {}).get('functionId') == 'endnode' for node in nodes
        )

        # edges 수 계산
        edges = workflow_data.get('edges', [])
        edge_count = len(edges) if isinstance(edges, list) else 0

        workflow_meta = WorkflowMeta(
            user_id=user_id,
            workflow_id=workflow_request.content.workflow_id,
            workflow_name=workflow_request.workflow_name,
            node_count=node_count,
            edge_count=edge_count,
            has_startnode=has_startnode,
            has_endnode=has_endnode,
            is_completed=(has_startnode and has_endnode),
            is_shared=existing_data[0].is_shared if existing_data and len(existing_data) > 0 else False,
            share_group=existing_data[0].share_group if existing_data and len(existing_data) > 0 else None,
            share_permissions=existing_data[0].share_permissions if existing_data and len(existing_data) > 0 else 'read',
            workflow_data=workflow_data,
            current_version=existing_data[0].current_version if existing_data and len(existing_data) > 0 else 1.0,
            latest_version=existing_data[0].latest_version if existing_data and len(existing_data) > 0 else 1.0,
        )

        if existing_data and len(existing_data) > 0:
            existing_data_id = existing_data[0].id
            workflow_meta.id = existing_data_id

            # 버전 업데이트 (소수점 3자리까지 제한)
            version_new = round(existing_data[0].latest_version + 0.1, 3)
            workflow_meta.current_version = version_new
            workflow_meta.latest_version = version_new
            insert_result = app_db.update(workflow_meta)
        else:
            version_new = round(1.0, 3)
            insert_result = app_db.insert(workflow_meta)

        if insert_result and insert_result.get("result") == "success":
            # Deploy metadata 생성
            deploy_meta = DeployMeta(
                user_id=user_id,
                workflow_id=workflow_request.content.workflow_id,
                workflow_name=workflow_request.workflow_name,
                is_deployed=False,
                is_accepted=True,
                inquire_deploy=False,
                deploy_key=""
            )

            # 기존 Deploy metadata 확인 및 생성/업데이트
            existing_deploy_data = app_db.find_by_condition(
                DeployMeta,
                {
                    "user_id": user_id,
                    "workflow_name": workflow_request.workflow_name,
                },
                limit=1
            )
            if existing_deploy_data and len(existing_deploy_data) > 0:
                deploy_meta.id = existing_deploy_data[0].id
                deploy_meta.is_deployed = existing_deploy_data[0].is_deployed
                deploy_meta.is_accepted = existing_deploy_data[0].is_accepted
                deploy_meta.inquire_deploy = existing_deploy_data[0].inquire_deploy
                deploy_meta.deploy_key = existing_deploy_data[0].deploy_key
                app_db.update(deploy_meta)
            else:
                app_db.insert(deploy_meta)

            # 기존 버전들의 current_use를 False로 변경
            db_type = app_db.config_db_manager.db_type
            if db_type == "postgresql":
                update_query = "UPDATE workflow_version SET current_use = %s WHERE user_id = %s AND workflow_meta_id = %s"
            else:
                update_query = "UPDATE workflow_version SET current_use = ? WHERE user_id = ? AND workflow_meta_id = ?"

            app_db.config_db_manager.execute_update_delete(update_query, (False, user_id, workflow_meta.id))

            workflow_version_data = WorkflowVersion(
                user_id=user_id,
                workflow_meta_id=workflow_meta.id,
                workflow_id=workflow_meta.workflow_id,
                workflow_name=workflow_meta.workflow_name,
                version=version_new,
                current_use=True,
                workflow_data=workflow_data,
                version_label=f"v{version_new}"
            )

            app_db.insert(workflow_version_data)

            # with open(file_path, 'w', encoding='utf-8') as f:
            #     json.dump(workflow_data, f, indent=2, ensure_ascii=False)

            backend_log.success("Workflow saved successfully",
                              metadata={"workflow_name": workflow_request.workflow_name,
                                      "filename": workflow_request.workflow_name + ".json",
                                      "node_count": node_count,
                                      "edge_count": edge_count,
                                      "has_startnode": has_startnode,
                                      "has_endnode": has_endnode,
                                      "is_completed": has_startnode and has_endnode,
                                      "operation": "update" if existing_data else "create"})

            logger.info(f"Workflow metadata and deploy metadata saved successfully: {workflow_request.workflow_name}")
        else:
            backend_log.error("Failed to save workflow metadata",
                            metadata={"workflow_name": workflow_request.workflow_name,
                                    "error": insert_result.get('error', 'Unknown error')})
            logger.error(f"Failed to save workflow metadata: {insert_result.get('error', 'Unknown error')}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save workflow metadata: {insert_result.get('error', 'Unknown error')}"
            )

        logger.info(f"Workflow saved successfully: {workflow_request.workflow_name}.json")
        return JSONResponse(content={
            "success": True,
            "message": f"Workflow '{workflow_request.workflow_name}' saved successfully",
            # "filename": filename
        })

    except HTTPException:
        raise
    except Exception as e:
        backend_log.error("Workflow save operation failed", exception=e,
                         metadata={"workflow_name": workflow_request.workflow_name,
                                 "workflow_id": workflow_request.content.workflow_id if hasattr(workflow_request, 'content') else None})
        logger.error(f"Error saving workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save workflow: {str(e)}")

@router.get("/load/{workflow_name}")
async def load_workflow(request: Request, workflow_name: str, user_id):
    """
    특정 workflow를 로드합니다.
    """
    login_user_id = extract_user_id_from_request(request)
    if not login_user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, login_user_id, request)

    try:
        backend_log.info("Starting workflow load operation",
                        metadata={"workflow_name": workflow_name,
                                "requested_user_id": user_id,
                                "login_user_id": login_user_id})

        # downloads_path = os.path.join(os.getcwd(), "downloads")
        using_id = workflow_user_id_extractor(app_db, login_user_id, user_id, workflow_name)
        # download_path_id = os.path.join(downloads_path, using_id)

        # filename = f"{workflow_name}.json"
        # file_path = os.path.join(download_path_id, filename)

        # if not os.path.exists(file_path):
        #     backend_log.warn("Workflow file not found",
        #                    metadata={"workflow_name": workflow_name, "file_path": file_path})
        #     raise HTTPException(status_code=404, detail=f"Workflow '{workflow_name}' not found")

        # with open(file_path, 'r', encoding='utf-8') as f:
        #     workflow_data = json.load(f)


        #### DB방식으로 변경중
        workflow_meta = app_db.find_by_condition(WorkflowMeta, {"user_id": using_id, "workflow_name": workflow_name}, limit=1)
        workflow_data = workflow_meta[0].workflow_data if workflow_meta else None
        if isinstance(workflow_data, str):
            workflow_data = json.loads(workflow_data)

        # 워크플로우 메타데이터 계산
        nodes = workflow_data.get('nodes', [])
        edges = workflow_data.get('edges', [])
        workflow_id = workflow_data.get('workflow_id', 'unknown')

        backend_log.success("Workflow loaded successfully",
                          metadata={"workflow_name": workflow_name,
                                  "workflow_id": workflow_id,
                                  "filename": workflow_name + ".json",
                                  "node_count": len(nodes),
                                  "edge_count": len(edges),
                                  "using_user_id": using_id,
                                #   "file_size": os.path.getsize(file_path)
                                })

        logger.info(f"Workflow loaded successfully: {workflow_name}.json")
        return JSONResponse(content=workflow_data)

    except FileNotFoundError:
        backend_log.error("Workflow file not found",
                         metadata={"workflow_name": workflow_name})
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_name}' not found")
    except Exception as e:
        backend_log.error("Workflow load operation failed", exception=e,
                         metadata={"workflow_name": workflow_name, "requested_user_id": user_id})
        logger.error(f"Error loading workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load workflow: {str(e)}")

@router.get("/duplicate/{workflow_name}")
async def duplicate_workflow(request: Request, workflow_name: str, user_id):
    """
    특정 workflow를 복제합니다.
    """
    try:
        login_user_id = extract_user_id_from_request(request)
        # downloads_path = os.path.join(os.getcwd(), "downloads")
        app_db = get_db_manager(request)
        using_id = workflow_user_id_extractor(app_db, login_user_id, user_id, workflow_name)

        # origin_path_id = os.path.join(downloads_path, using_id)
        # target_path_id = os.path.join(downloads_path, login_user_id)

        # filename = f"{workflow_name}.json"
        # origin_path = os.path.join(origin_path_id, filename)

        # if not os.path.exists(origin_path):
        #     logger.info(f"Reading workflow data from: {origin_path}")
        #     raise HTTPException(status_code=404, detail=f"Workflow '{workflow_name}' not found")

        # copy_workflow_name = f"{workflow_name}_copy"
        # copy_file_name = f"{copy_workflow_name}.json"
        # target_path = os.path.join(copy_workflow_name, filename)

        # if os.path.exists(target_path):
        #     logger.warning(f"Workflow already exists for user '{login_user_id}': {filename}. Change target file name.")
        #     counter = 1
        #     while os.path.exists(target_path):
        #         copy_workflow_name = f"{workflow_name}_copy_{counter}"
        #         copy_file_name = f"{copy_workflow_name}.json"
        #         target_path = os.path.join(target_path_id, copy_file_name)
        #         counter += 1

        # with open(origin_path, 'r', encoding='utf-8') as f:
        #     logger.info(f"Reading workflow data from: {origin_path}")
        #     workflow_data = json.load(f)

        workflow_meta = app_db.find_by_condition(WorkflowMeta, {"user_id": using_id, "workflow_name": workflow_name}, limit=1)
        workflow_data = workflow_meta[0].workflow_data if workflow_meta else None
        if isinstance(workflow_data, str):
            workflow_data = json.loads(workflow_data)

        nodes = workflow_data.get('nodes', [])
        node_count = len(nodes) if isinstance(nodes, list) else 0
        has_startnode = any(node.get('data', {}).get('functionId') == 'startnode' for node in nodes)
        has_endnode = any(node.get('data', {}).get('functionId') == 'endnode' for node in nodes)

        edges = workflow_data.get('edges', [])
        edge_count = len(edges) if isinstance(edges, list) else 0

        copy_workflow_name = f"{workflow_name}_copy"
        existing_copy_data = app_db.find_by_condition(WorkflowMeta, {"user_id": login_user_id, "workflow_name": copy_workflow_name}, limit=1)
        if existing_copy_data:
            logger.warning(f"Workflow already exists for user '{login_user_id}': {copy_workflow_name}. Change target file name.")
            counter = 1
            while existing_copy_data:
                copy_workflow_name = f"{workflow_name}_copy_{counter}"
                existing_copy_data = app_db.find_by_condition(WorkflowMeta, {"user_id": login_user_id, "workflow_name": copy_workflow_name}, limit=1)
                counter += 1
        version_new = round(1.0, 3)

        workflow_meta = WorkflowMeta(
            user_id=login_user_id,
            workflow_id=workflow_data.get('workflow_id'),
            workflow_name=copy_workflow_name,
            node_count=node_count,
            edge_count=edge_count,
            has_startnode=has_startnode,
            has_endnode=has_endnode,
            is_completed=(has_startnode and has_endnode),
            workflow_data=workflow_data,
            current_version=version_new,
            latest_version=version_new,
        )

        insert_result = app_db.insert(workflow_meta)
        # with open(target_path, 'w', encoding='utf-8') as wf:
        #     json.dump(workflow_data, wf, ensure_ascii=False, indent=2)

        if insert_result and insert_result.get("result") == "success":
            # copy 데이터의 Deploy metadata 생성
            deploy_meta = DeployMeta(
                user_id=user_id,
                workflow_id=workflow_data.get('workflow_id'),
                workflow_name=copy_workflow_name,
                is_deployed=False,
                is_accepted=True,
                inquire_deploy=False,
                deploy_key=""
            )
            app_db.insert(deploy_meta)

            workflow_version_data = WorkflowVersion(
                user_id=user_id,
                workflow_meta_id=workflow_meta.id,
                workflow_id=workflow_meta.workflow_id,
                workflow_name=workflow_meta.workflow_name,
                version=version_new,
                current_use=True,
                workflow_data=workflow_data,
                version_label=f"v{version_new}"
            )
            app_db.insert(workflow_version_data)

        logger.info(f"Workflow duplicated successfully: {workflow_name}")
        return {"success": True, "message": f"Workflow '{workflow_name}' duplicated successfully", "filename": copy_workflow_name}

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_name}' not found")
    except Exception as e:
        logger.error(f"Error loading workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load workflow: {str(e)}")

@router.post("/update/{workflow_name}")
async def update_workflow(request: Request, workflow_name: str, update_dict: dict):
    user_id = extract_user_id_from_request(request)
    app_db = get_db_manager(request)
    config_composer = get_config_composer(request)

    try:
        existing_data = app_db.find_by_condition(
            WorkflowMeta,
            {
                "user_id": user_id,
                "workflow_name": workflow_name
            },
            limit=1
        )

        if not existing_data:
            raise HTTPException(status_code=404, detail="Workflow not found")
        existing_data = existing_data[0]

        deploy_data = app_db.find_by_condition(
            DeployMeta,
            {
                "user_id": user_id,
                "workflow_name": workflow_name,
            },
            limit=1
        )
        if not deploy_data:
            raise HTTPException(status_code=404, detail="배포 메타데이터를 찾을 수 없습니다")
        if not deploy_data[0].is_accepted:
            raise HTTPException(status_code=400, detail="해당 워크플로우에 대한 권한이 박탈되었습니다. 편집할 수 없습니다.")

        deploy_meta = deploy_data[0]

        existing_data.is_shared = update_dict.get("is_shared", existing_data.is_shared)
        existing_data.share_group = update_dict.get("share_group", existing_data.share_group)
        existing_data.share_permissions = update_dict.get("share_permissions", existing_data.share_permissions)

        deploy_enabled = update_dict.get("enable_deploy", deploy_meta.is_deployed)
        deploy_meta.is_deployed = deploy_enabled
        free_deploy_mode = config_composer.get_config_by_name("FREE_CHAT_DEPLOYMENT_MODE").value

        if deploy_enabled:
            alphabet = string.ascii_letters + string.digits
            deploy_key = ''.join(secrets.choice(alphabet) for _ in range(32))
            deploy_meta.deploy_key = deploy_key

            if not free_deploy_mode:
                from controller.admin.adminBaseController import is_superuser
                val_superuser = await is_superuser(request, user_id)
                if not val_superuser.get("superuser", False):
                    deploy_meta.is_deployed = False
                    deploy_meta.inquire_deploy = True

            logger.info(f"Generated new deploy key for workflow: {workflow_name}")
        else:
            deploy_meta.deploy_key = ""
            deploy_meta.inquire_deploy = False
            logger.info(f"Cleared deploy key for workflow: {workflow_name}")

        app_db.update(existing_data)
        app_db.update(deploy_meta)

        return {
            "message": "Workflow updated successfully",
            "workflow_name": existing_data.workflow_name,
            "deploy_key": deploy_meta.deploy_key if deploy_meta.is_deployed else None,
        }

    except Exception as e:
        logger.error(f"Failed to update workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update workflow: {str(e)}")

@router.delete("/delete/{workflow_name}")
async def delete_workflow(request: Request, workflow_name: str):
    """
    특정 workflow를 삭제합니다.
    """
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        backend_log.info("Starting workflow deletion",
                        metadata={"workflow_name": workflow_name})

        existing_data = app_db.find_by_condition(
            WorkflowMeta,
            {
                "user_id": user_id,
                "workflow_name": workflow_name,
            },
            ignore_columns=['workflow_data'],
            limit=1
        )

        if not existing_data:
            backend_log.warn("Workflow metadata not found for deletion",
                           metadata={"workflow_name": workflow_name})
            raise HTTPException(status_code=404, detail=f"Workflow '{workflow_name}' not found")

        deploy_meta = app_db.find_by_condition(
            DeployMeta,
            {
                "user_id": user_id,
                "workflow_id": existing_data[0].workflow_id,
                "workflow_name": workflow_name,
            },
            limit=1
        )

        if deploy_meta and not deploy_meta[0].is_accepted:
            backend_log.warn("Workflow deletion denied - permissions revoked",
                           metadata={"workflow_name": workflow_name})
            raise HTTPException(status_code=400, detail="해당 워크플로우에 대한 권한이 박탈되었습니다. 편집할 수 없습니다.")

        if deploy_meta and deploy_meta[0].is_deployed:
            backend_log.warn("Cannot delete deployed workflow",
                           metadata={"workflow_name": workflow_name, "is_deployed": True})
            raise HTTPException(status_code=400, detail="배포된 워크플로우는 삭제할 수 없습니다. 배포를 해제한 후 다시 시도하세요.")

        # 데이터베이스에서 메타데이터 삭제
        app_db.delete(WorkflowMeta, existing_data[0].id if existing_data else None)
        app_db.delete_by_condition(DeployMeta, {
            "user_id": user_id,
            "workflow_id": existing_data[0].workflow_id,
            "workflow_name": workflow_name,
        })
        app_db.delete_by_condition(WorkflowVersion, {
            "user_id": user_id,
            "workflow_meta_id": existing_data[0].id if existing_data else None,
        })

        # # 파일 시스템에서 파일 삭제
        # downloads_path = os.path.join(os.getcwd(), "downloads")
        # download_path_id = os.path.join(downloads_path, user_id)
        # filename = f"{workflow_name}.json"
        # file_path = os.path.join(download_path_id, filename)

        # if not os.path.exists(file_path):
        #     backend_log.warn("Workflow file not found on filesystem",
        #                    metadata={"workflow_name": workflow_name, "file_path": file_path})
        #     raise HTTPException(status_code=404, detail=f"Workflow '{workflow_name}' not found")

        # file_size = os.path.getsize(file_path)
        # os.remove(file_path)

        backend_log.success("Workflow deleted successfully",
                          metadata={"workflow_name": workflow_name,
                                  "workflow_id": existing_data[0].workflow_id,
                                #   "filename": filename,
                                #   "file_size": file_size,
                                  "was_deployed": deploy_meta[0].is_deployed if deploy_meta else False})

        logger.info(f"Workflow deleted successfully: {workflow_name}")
        return JSONResponse(content={
            "success": True,
            "message": f"Workflow '{workflow_name}' deleted successfully"
        })

    except FileNotFoundError:
        backend_log.error("Workflow file not found for deletion",
                         metadata={"workflow_name": workflow_name})
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_name}' not found")
    except HTTPException:
        raise
    except Exception as e:
        backend_log.error("Workflow deletion failed", exception=e,
                         metadata={"workflow_name": workflow_name})
        logger.error(f"Error deleting workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete workflow: {str(e)}")

@router.get("/list/detail")
async def list_workflows_detail(request: Request):
    """
    downloads 폴더에 있는 모든 workflow 파일들의 상세 정보를 반환합니다.
    각 워크플로우에 대해 파일명, workflow_id, 노드 수, 마지막 수정일자를 포함합니다.
    """
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        user = app_db.find_by_id(User, user_id)
        groups = user.groups
        user_name = user.username if user else "Unknown User"

        backend_log.info("Starting detailed workflow list retrieval",
                        metadata={"user_name": user_name, "groups": groups, "groups_count": len(groups) if groups else 0})

        # 직접 SQL 쿼리로 워크플로우와 사용자 정보 조인
        # 자신의 워크플로우
        own_workflows_query = """
            SELECT
                wm.id, wm.created_at, wm.updated_at,
                wm.user_id, wm.workflow_id, wm.workflow_name,
                wm.node_count, wm.edge_count, wm.has_startnode, wm.has_endnode,
                wm.is_completed, wm.metadata, wm.is_shared, wm.share_group, wm.share_permissions,
                u.username, u.full_name,
                dm.is_deployed, dm.deploy_key, dm.is_accepted, dm.inquire_deploy
            FROM workflow_meta wm
            LEFT JOIN users u ON wm.user_id = u.id
            LEFT JOIN deploy_meta dm ON wm.workflow_id = dm.workflow_id
            WHERE wm.user_id = %s
            ORDER BY wm.updated_at DESC
            LIMIT 10000
        """
        existing_data = app_db.config_db_manager.execute_query(own_workflows_query, (user_id,))
        own_workflow_count = len(existing_data) if existing_data else 0

        # 그룹 공유 워크플로우 추가
        shared_workflow_count = 0
        if groups and groups != None and groups != [] and len(groups) > 0:
            for group_name in groups:
                shared_workflows_query = """
                    SELECT
                        wm.id, wm.created_at, wm.updated_at,
                        wm.user_id, wm.workflow_id, wm.workflow_name,
                        wm.node_count, wm.edge_count, wm.has_startnode, wm.has_endnode,
                        wm.is_completed, wm.metadata, wm.is_shared, wm.share_group, wm.share_permissions,
                        u.username, u.full_name,
                        dm.is_deployed, dm.deploy_key, dm.is_accepted, dm.inquire_deploy
                    FROM workflow_meta wm
                    LEFT JOIN users u ON wm.user_id = u.id
                    LEFT JOIN deploy_meta dm ON wm.workflow_id = dm.workflow_id
                    WHERE wm.share_group = %s AND wm.is_shared = true
                    ORDER BY wm.updated_at DESC
                    LIMIT 10000
                """
                shared_data = app_db.config_db_manager.execute_query(shared_workflows_query, (group_name,))
                if shared_data:
                    existing_data.extend(shared_data)
                    shared_workflow_count += len(shared_data)

        seen_ids = set()
        unique_data = []
        for item in existing_data:
            if item.get("id") not in seen_ids:
                seen_ids.add(item.get("id"))
                item_dict = dict(item)
                if 'created_at' in item_dict and item_dict['created_at']:
                    item_dict['created_at'] = item_dict['created_at'].isoformat() if hasattr(item_dict['created_at'], 'isoformat') else str(item_dict['created_at'])
                if 'updated_at' in item_dict and item_dict['updated_at']:
                    item_dict['updated_at'] = item_dict['updated_at'].isoformat() if hasattr(item_dict['updated_at'], 'isoformat') else str(item_dict['updated_at'])
                unique_data.append(item_dict)

        backend_log.success("Detailed workflow list retrieved successfully",
                          metadata={"own_workflows": own_workflow_count,
                                  "shared_workflows": shared_workflow_count,
                                  "total_unique_workflows": len(unique_data),
                                  "user_name": user_name,
                                  "groups_count": len(groups) if groups else 0})

        logger.info(f"Found {len(unique_data)} workflow files with detailed information")

        return JSONResponse(content={"workflows": unique_data})

    except Exception as e:
        backend_log.error("Failed to retrieve detailed workflow list", exception=e,
                         metadata={"user_name": user_name if 'user_name' in locals() else None})
        logger.error(f"Error listing workflow details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list workflow details: {str(e)}")

@router.get("/io_logs")
async def get_workflow_io_logs(request: Request, workflow_name: str, workflow_id: str, interaction_id: str = 'default'):
    """
    특정 워크플로우의 ExecutionIO 로그를 반환합니다.
    """
    try:
        user_id = extract_user_id_from_request(request)
        app_db = get_db_manager(request)
        result = app_db.find_by_condition(
            ExecutionIO,
            {
                "user_id": user_id,
                "workflow_name": workflow_name,
                "interaction_id": interaction_id
            },
            limit=1000000,
            orderby="updated_at",
            orderby_asc=True,
            return_list=True
        )

        if not result:
            logger.info(f"No performance data found for workflow: {workflow_name} ({workflow_id})")
            return JSONResponse(content={
                "workflow_name": workflow_name,
                "workflow_id": workflow_id,
                "in_out_logs": [],
                "message": "No in_out_logs data found for this workflow"
            })

        performance_stats = []
        for idx, row in enumerate(result):
            # input_data 파싱
            raw_input_data = json.loads(row['input_data']).get('result', None) if row['input_data'] else None
            parsed_input_data = parse_input_data(raw_input_data) if raw_input_data else None

            log_entry = {
                "log_id": idx + 1,
                "io_id": row['id'],
                "interaction_id": row['interaction_id'],
                "workflow_name": row['workflow_name'],
                "workflow_id": row['workflow_id'],
                "input_data": parsed_input_data,
                "output_data": json.loads(row['output_data']).get('result', None) if row['output_data'] else None,
                "updated_at": row['updated_at'].isoformat() if isinstance(row['updated_at'], datetime) else row['updated_at']
            }
            performance_stats.append(log_entry)

        response_data = {
            "workflow_name": workflow_name,
            "workflow_id": workflow_id,
            "in_out_logs": performance_stats,
            "message": "In/Out logs retrieved successfully"
        }

        logger.info(f"Performance stats retrieved for workflow: {workflow_name} ({workflow_id})")
        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"Error retrieving workflow performance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve performance data: {str(e)}")

@router.post("/io_log/rating")
async def rate_workflow_io_log(request: Request, io_id: int, workflow_name: str, workflow_id: str, interaction_id: str = 'default', rating: int = 3):
    """
    특정 워크플로우의 ExecutionIO 로그에 대한 평가를 저장합니다.
    """
    try:
        user_id = extract_user_id_from_request(request)
        app_db = get_db_manager(request)
        io_log = app_db.find_by_condition(
            ExecutionIO,
            {
                "id": io_id,
                "user_id": user_id,
                "workflow_name": workflow_name,
                "interaction_id": interaction_id
            },
            limit=1,
        )

        if not io_log:
            logger.info(f"No performance data found for workflow: {workflow_name} ({workflow_id})")
            raise HTTPException(status_code=404, detail="ExecutionIO log not found")

        io_log = io_log[0]
        io_log.user_score = rating

        app_db.update(io_log)
        logger.info(f"Rating updated for ExecutionIO log ID {io_id} in workflow: {workflow_name} ({workflow_id})")
        return JSONResponse(content={
            "workflow_name": workflow_name,
            "workflow_id": workflow_id,
            "io_id": io_id,
            "new_rating": rating,
            "message": "Rating updated successfully"
        })
    except Exception as e:
        logger.error(f"Error updating rating for workflow log: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update rating: {str(e)}")

@router.delete("/io_logs")
async def delete_workflow_io_logs(request: Request, workflow_name: str, workflow_id: str, interaction_id: str = "default"):
    """
    특정 워크플로우의 ExecutionIO 로그를 삭제합니다.
    """
    try:
        user_id = extract_user_id_from_request(request)
        app_db = get_db_manager(request)

        existing_data = app_db.find_by_condition(
            ExecutionIO,
            {
                "user_id": user_id,
                "workflow_name": workflow_name,
                "interaction_id": interaction_id
            },
            limit=1000000
        )

        delete_count = len(existing_data) if existing_data else 0

        if delete_count == 0:
            logger.info(f"No logs found to delete for workflow: {workflow_name} ({workflow_id}), interaction_id: {interaction_id}")
            return JSONResponse(content={
                "workflow_name": workflow_name,
                "interaction_id": interaction_id,
                "deleted_count": 0,
                "message": "No logs found to delete"
            })

        app_db.delete_by_condition(
            ExecutionIO,
            {
                "user_id": user_id,
                "workflow_name": workflow_name,
                "interaction_id": interaction_id
            }
        )
        app_db.delete_by_condition(
            ExecutionMeta,
            {
                "user_id": user_id,
                "workflow_name": workflow_name,
                "interaction_id": interaction_id
            }
        )

        logger.info(f"Successfully deleted {delete_count} logs for workflow: {workflow_name} ({workflow_id}), interaction_id: {interaction_id}")

        return JSONResponse(content={
            "workflow_name": workflow_name,
            "interaction_id": interaction_id,
            "deleted_count": delete_count,
            "message": f"Successfully deleted {delete_count} execution logs"
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting workflow logs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete workflow logs: {str(e)}")

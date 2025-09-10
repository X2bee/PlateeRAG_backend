"""
워크플로우 파라미터 및 실행 관련 헬퍼 함수들
"""
import os
import json
import logging
from typing import Dict, Any

from service.database.models.workflow import WorkflowMeta
from controller.workflow.helper import _workflow_parameter_helper, _default_workflow_parameter_helper

logger = logging.getLogger("workflow-helpers")

# 기존 helper 함수들을 재 export
async def workflow_parameter_helper(request_body, workflow_data):
    """워크플로우 파라미터를 설정합니다."""
    return await _workflow_parameter_helper(request_body, workflow_data)

async def default_workflow_parameter_helper(request, request_body, workflow_data):
    """기본 워크플로우 파라미터를 설정합니다."""
    return await _default_workflow_parameter_helper(request, request_body, workflow_data)

async def workflow_data_synchronizer(app_db) -> Dict[str, Any]:
    """
    모든 사용자의 파일시스템과 WorkflowMeta DB 간의 데이터 동기화를 수행합니다.

    Args:
        app_db: 데이터베이스 매니저

    Returns:
        Dict: 동기화 결과 정보
    """
    downloads_path = os.path.join(os.getcwd(), "downloads")

    sync_results = {
        "files_added_to_db": 0,
        "files_created_from_db": 0,
        "orphaned_db_entries_removed": 0,
        "users_processed": 0,
        "errors": [],
        "success": True
    }

    try:
        # 1. downloads 폴더의 사용자 폴더 스캔 (숫자로 된 폴더만)
        user_folders = []
        if os.path.exists(downloads_path):
            for item in os.listdir(downloads_path):
                item_path = os.path.join(downloads_path, item)
                # 숫자로 된 폴더만 사용자 워크플로우 폴더로 인식
                if os.path.isdir(item_path) and item.isdigit():
                    user_folders.append(item)

        # 2. 각 사용자별로 처리
        for user_id in user_folders:
            user_downloads_path = os.path.join(downloads_path, user_id)

            # 해당 사용자의 파일시스템 워크플로우들 스캔
            filesystem_workflows = {}
            for filename in os.listdir(user_downloads_path):
                if filename.endswith('.json'):
                    workflow_name = filename[:-5]  # .json 제거
                    file_path = os.path.join(user_downloads_path, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            workflow_data = json.load(f)
                        filesystem_workflows[workflow_name] = workflow_data
                    except (json.JSONDecodeError, FileNotFoundError, IOError) as e:
                        sync_results["errors"].append(f"User {user_id}: Failed to read file {filename}: {str(e)}")

            # 해당 사용자의 DB 워크플로우 데이터 조회
            db_workflows = app_db.find_by_condition(
                WorkflowMeta,
                {"user_id": user_id},
                limit=10000,
                return_list=True
            )

            db_workflow_dict = {workflow['workflow_name']: workflow for workflow in db_workflows}

            # 3. 파일시스템에 존재하지만 DB에 없는 경우 처리
            for workflow_name, workflow_data in filesystem_workflows.items():
                if workflow_name not in db_workflow_dict:
                    try:
                        # 워크플로우 데이터에서 메타데이터 추출
                        nodes = workflow_data.get('nodes', [])
                        node_count = len(nodes) if isinstance(nodes, list) else 0
                        has_startnode = any(
                            node.get('data', {}).get('functionId') == 'startnode' for node in nodes
                        )
                        has_endnode = any(
                            node.get('data', {}).get('functionId') == 'endnode' for node in nodes
                        )

                        edges = workflow_data.get('edges', [])
                        edge_count = len(edges) if isinstance(edges, list) else 0

                        # WorkflowMeta 객체 생성 및 DB에 삽입
                        workflow_meta = WorkflowMeta(
                            user_id=user_id,
                            workflow_id=workflow_data.get('workflow_id', ''),
                            workflow_name=workflow_name,
                            node_count=node_count,
                            edge_count=edge_count,
                            has_startnode=has_startnode,
                            has_endnode=has_endnode,
                            is_completed=(has_startnode and has_endnode),
                            workflow_data=workflow_data,
                        )

                        app_db.insert(workflow_meta)
                        sync_results["files_added_to_db"] += 1
                        logger.info("Added workflow '%s' to database from filesystem for user %s", workflow_name, user_id)

                    except (IOError, ValueError) as e:
                        sync_results["errors"].append(f"User {user_id}: Failed to add workflow '{workflow_name}' to DB: {str(e)}")

            # 4. 파일시스템과 DB에 모두 존재하지만 DB에 workflow_data가 없는 경우 처리
            for workflow_name, workflow_data in filesystem_workflows.items():
                if workflow_name in db_workflow_dict:
                    db_workflow = db_workflow_dict[workflow_name]
                    if not db_workflow.get('workflow_data'):
                        try:
                            # DB에 workflow_data 업데이트 - JSON 문자열로 변환해서 저장
                            workflow_data_json = json.dumps(workflow_data, ensure_ascii=False)
                            app_db.update_list_columns(
                                WorkflowMeta,
                                {"workflow_data": workflow_data_json},
                                {"id": db_workflow['id']}
                            )
                            sync_results["files_added_to_db"] += 1
                            logger.info("Updated workflow_data for workflow '%s' in database for user %s", workflow_name, user_id)

                        except (ValueError, RuntimeError) as e:
                            sync_results["errors"].append(f"User {user_id}: Failed to update workflow_data for workflow '{workflow_name}': {str(e)}")

            # 5. DB에 존재하지만 파일시스템에 없는 경우 처리
            for workflow_name, db_workflow in db_workflow_dict.items():
                if workflow_name not in filesystem_workflows:
                    # workflow_data가 존재하는지 확인
                    if db_workflow.get('workflow_data'):
                        try:
                            # 파일시스템에 파일 생성
                            if not os.path.exists(user_downloads_path):
                                os.makedirs(user_downloads_path)

                            file_path = os.path.join(user_downloads_path, f"{workflow_name}.json")
                            workflow_data = db_workflow['workflow_data']

                            # workflow_data가 문자열인 경우 JSON 파싱
                            if isinstance(workflow_data, str):
                                workflow_data = json.loads(workflow_data)

                            with open(file_path, 'w', encoding='utf-8') as f:
                                json.dump(workflow_data, f, indent=2, ensure_ascii=False)

                            sync_results["files_created_from_db"] += 1
                            logger.info("Created file for workflow '%s' from database for user %s", workflow_name, user_id)

                        except (IOError, json.JSONDecodeError, OSError) as e:
                            sync_results["errors"].append(f"User {user_id}: Failed to create file for workflow '{workflow_name}': {str(e)}")
                    else:
                        # workflow_data가 없는 경우 DB에서 삭제
                        try:
                            app_db.delete(WorkflowMeta, db_workflow['id'])
                            sync_results["orphaned_db_entries_removed"] += 1
                            logger.info("Removed orphaned database entry for workflow '%s' for user %s", workflow_name, user_id)

                        except (ValueError, RuntimeError) as e:
                            sync_results["errors"].append(f"User {user_id}: Failed to remove orphaned DB entry for workflow '{workflow_name}': {str(e)}")

            sync_results["users_processed"] += 1

        # 3. DB에만 존재하는 사용자들의 워크플로우도 처리
        all_db_workflows = app_db.find_by_condition(
            WorkflowMeta,
            {},
            limit=100000,
            return_list=True
        )

        # 사용자별로 그룹핑
        db_users = {}
        for workflow in all_db_workflows:
            user_id = str(workflow['user_id'])
            if user_id not in db_users:
                db_users[user_id] = []
            db_users[user_id].append(workflow)

        # 파일시스템에 없는 사용자들 처리
        for user_id, workflows in db_users.items():
            if user_id not in user_folders:
                user_downloads_path = os.path.join(downloads_path, user_id)

                for workflow in workflows:
                    workflow_name = workflow['workflow_name']
                    if workflow.get('workflow_data'):
                        try:
                            # 사용자 폴더 생성
                            if not os.path.exists(user_downloads_path):
                                os.makedirs(user_downloads_path)

                            file_path = os.path.join(user_downloads_path, f"{workflow_name}.json")
                            workflow_data = workflow['workflow_data']

                            # workflow_data가 문자열인 경우 JSON 파싱
                            if isinstance(workflow_data, str):
                                workflow_data = json.loads(workflow_data)

                            with open(file_path, 'w', encoding='utf-8') as f:
                                json.dump(workflow_data, f, indent=2, ensure_ascii=False)

                            sync_results["files_created_from_db"] += 1
                            logger.info("Created file for workflow '%s' from database for user %s", workflow_name, user_id)

                        except (IOError, json.JSONDecodeError, OSError) as e:
                            sync_results["errors"].append(f"User {user_id}: Failed to create file for workflow '{workflow_name}': {str(e)}")
                    else:
                        # workflow_data가 없는 경우 DB에서 삭제
                        try:
                            app_db.delete(WorkflowMeta, workflow['id'])
                            sync_results["orphaned_db_entries_removed"] += 1
                            logger.info("Removed orphaned database entry for workflow '%s' for user %s", workflow_name, user_id)

                        except (ValueError, RuntimeError) as e:
                            sync_results["errors"].append(f"User {user_id}: Failed to remove orphaned DB entry for workflow '{workflow_name}': {str(e)}")

                if user_id not in user_folders:
                    sync_results["users_processed"] += 1

        # 4. 결과 검증
        if sync_results["errors"]:
            sync_results["success"] = False

        logger.info("Workflow synchronization completed: Added %d to DB, Created %d files, Removed %d orphaned entries, Processed %d users, Errors: %d",
                   sync_results['files_added_to_db'],
                   sync_results['files_created_from_db'],
                   sync_results['orphaned_db_entries_removed'],
                   sync_results['users_processed'],
                   len(sync_results['errors']))

        return sync_results

    except (OSError, RuntimeError, ValueError) as e:
        sync_results["success"] = False
        sync_results["errors"].append(f"Synchronization failed: {str(e)}")
        logger.error("Workflow synchronization failed: %s", str(e))
        return sync_results

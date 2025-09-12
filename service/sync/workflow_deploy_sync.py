"""
워크플로우 메타데이터와 배포 메타데이터 동기화 서비스
서버 시작 시 workflow_meta에는 있지만 deploy_meta에는 없는 워크플로우를 자동으로 동기화
"""
import logging
from typing import List, Dict, Any
from service.database.models.deploy import DeployMeta

logger = logging.getLogger(__name__)

class WorkflowDeploySync:
    """워크플로우와 배포 메타데이터 동기화 클래스"""

    def __init__(self, app_db_manager):
        """
        Args:
            app_db_manager: 애플리케이션 데이터베이스 매니저
        """
        self.app_db_manager = app_db_manager

    def get_workflow_metas(self) -> List[Dict[str, Any]]:
        """workflow_meta 테이블에서 모든 워크플로우 조회"""
        try:
            query = """
            SELECT id, user_id, workflow_id, workflow_name, created_at, updated_at
            FROM workflow_meta
            WHERE is_completed = TRUE
            ORDER BY created_at DESC
            """

            result = self.app_db_manager.config_db_manager.execute_query(query)
            return result if result else []

        except Exception as e:
            logger.error(f"Error fetching workflow metas: {e}")
            return []

    def get_existing_deploy_metas(self) -> List[Dict[str, Any]]:
        """deploy_meta 테이블에서 기존 배포 메타데이터 조회"""
        try:
            query = """
            SELECT id, user_id, workflow_id, workflow_name, created_at, updated_at
            FROM deploy_meta
            ORDER BY created_at DESC
            """

            result = self.app_db_manager.config_db_manager.execute_query(query)
            return result if result else []

        except Exception as e:
            logger.error(f"Error fetching deploy metas: {e}")
            return []

    def find_missing_deploy_metas(self, workflow_metas: List[Dict[str, Any]],
                                  deploy_metas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """workflow_meta에는 있지만 deploy_meta에는 없는 워크플로우 찾기 (set 기반 최적화)"""

        # deploy_meta에 있는 workflow_id들을 set으로 변환 (O(1) 검색)
        existing_deploy_workflow_ids = {
            deploy['workflow_id'] for deploy in deploy_metas
        }

        # workflow_meta의 workflow_id들을 set으로 변환
        workflow_ids_set = {
            workflow['workflow_id'] for workflow in workflow_metas
        }

        # set 연산으로 차집합 구하기 (O(n) 최적화)
        missing_workflow_ids = workflow_ids_set - existing_deploy_workflow_ids

        # 누락된 workflow_id에 해당하는 전체 데이터 필터링
        if not missing_workflow_ids:
            return []

        missing_workflows = [
            workflow for workflow in workflow_metas
            if workflow['workflow_id'] in missing_workflow_ids
        ]

        return missing_workflows

    def create_deploy_meta(self, workflow_meta: Dict[str, Any]) -> bool:
        """워크플로우 메타데이터를 기반으로 배포 메타데이터 생성"""
        try:

            # DeployMeta 모델 인스턴스 생성
            deploy_meta = DeployMeta(
                user_id=workflow_meta['user_id'],
                workflow_id=workflow_meta['workflow_id'],
                workflow_name=workflow_meta['workflow_name'],
                is_deployed=False,
                deploy_key=''
            )

            # 데이터베이스에 삽입
            success = self.app_db_manager.insert(deploy_meta)

            if success:
                logger.info(f"Created deploy meta for workflow: {workflow_meta['workflow_id']} "
                           f"(name: {workflow_meta['workflow_name']})")
                return True
            else:
                logger.error(f"Failed to insert deploy meta for workflow: {workflow_meta['workflow_id']}")
                return False

        except Exception as e:
            logger.error(f"Error creating deploy meta for workflow {workflow_meta['workflow_id']}: {e}")
            return False

    def sync_workflow_to_deploy(self) -> Dict[str, Any]:
        """
        워크플로우 메타데이터와 배포 메타데이터 동기화

        Returns:
            Dict: 동기화 결과 정보
        """
        logger.info("🔄 Starting workflow to deploy meta synchronization...")

        sync_result = {
            "total_workflows": 0,
            "existing_deploys": 0,
            "missing_deploys": 0,
            "created_deploys": 0,
            "failed_creates": 0,
            "success": False,
            "errors": []
        }

        try:
            # 1. workflow_meta에서 모든 완료된 워크플로우 조회
            workflow_metas = self.get_workflow_metas()
            sync_result["total_workflows"] = len(workflow_metas)

            if not workflow_metas:
                logger.info("📝 No completed workflows found in workflow_meta")
                sync_result["success"] = True
                return sync_result

            # 2. deploy_meta에서 기존 배포 메타데이터 조회
            deploy_metas = self.get_existing_deploy_metas()
            sync_result["existing_deploys"] = len(deploy_metas)

            # 3. 누락된 배포 메타데이터 찾기
            missing_workflows = self.find_missing_deploy_metas(workflow_metas, deploy_metas)
            sync_result["missing_deploys"] = len(missing_workflows)

            if not missing_workflows:
                logger.info("✅ All workflows are already synchronized with deploy meta")
                sync_result["success"] = True
                return sync_result

            logger.info(f"📋 Found {len(missing_workflows)} workflows missing from deploy_meta")

            # 4. 누락된 워크플로우들에 대해 배포 메타데이터 생성
            created_count = 0
            failed_count = 0

            for workflow in missing_workflows:
                if self.create_deploy_meta(workflow):
                    created_count += 1
                else:
                    failed_count += 1
                    sync_result["errors"].append(f"Failed to create deploy meta for {workflow['workflow_id']}")

            sync_result["created_deploys"] = created_count
            sync_result["failed_creates"] = failed_count

            # 5. 결과 로깅
            if failed_count == 0:
                logger.info(f"✅ Successfully synchronized {created_count} workflows to deploy meta")
                sync_result["success"] = True
            else:
                logger.warning(f"⚠️  Synchronization completed with {failed_count} failures. "
                             f"Created: {created_count}, Failed: {failed_count}")
                sync_result["success"] = created_count > 0

            return sync_result

        except Exception as e:
            error_msg = f"Error during workflow to deploy sync: {e}"
            logger.error(f"❌ {error_msg}")
            sync_result["errors"].append(error_msg)
            sync_result["success"] = False
            return sync_result


def sync_workflow_deploy_meta(app_db_manager) -> Dict[str, Any]:
    """
    워크플로우와 배포 메타데이터 동기화 실행 함수
    서버 시작 시 호출되는 메인 함수

    Args:
        app_db_manager: 애플리케이션 데이터베이스 매니저

    Returns:
        Dict: 동기화 결과
    """
    if not app_db_manager:
        logger.error("❌ App database manager not available for workflow-deploy sync")
        return {
            "success": False,
            "errors": ["Database manager not available"],
            "total_workflows": 0,
            "created_deploys": 0
        }

    try:
        sync_service = WorkflowDeploySync(app_db_manager)
        return sync_service.sync_workflow_to_deploy()

    except Exception as e:
        logger.error(f"❌ Failed to initialize workflow-deploy sync service: {e}")
        return {
            "success": False,
            "errors": [f"Sync service initialization failed: {e}"],
            "total_workflows": 0,
            "created_deploys": 0
        }

"""
인증 및 사용자 관리 관련 유틸리티 함수들
"""
import logging
from service.database.models.user import User
from service.database.models.workflow import WorkflowMeta

logger = logging.getLogger("auth-helpers")

def workflow_user_id_extractor(app_db, login_user_id, requested_user_id, workflow_id):
    """
    로그인된 사용자와 요청된 사용자 ID를 비교하여 적절한 사용자 ID를 반환합니다.
    그룹 공유 권한도 확인합니다.
    
    Args:
        app_db: 데이터베이스 매니저
        login_user_id: 로그인된 사용자 ID
        requested_user_id: 요청된 사용자 ID
        workflow_id: 워크플로우 ID
    
    Returns:
        str: 사용할 사용자 ID
    """
    if login_user_id is not None:
        login_user_id = str(login_user_id).strip()

    if requested_user_id is not None:
        requested_user_id = str(requested_user_id).strip()
    else:
        requested_user_id = None

    if (login_user_id == requested_user_id) or requested_user_id == None or len(requested_user_id) == 0:
        return login_user_id
    else:
        user = app_db.find_by_id(User, login_user_id)
        if not user:
            logger.error(f"Login user not found in database: {login_user_id}")
            return login_user_id

        groups = user.groups
        requested_workflow_meta = app_db.find_by_condition(
            WorkflowMeta, 
            {'user_id': requested_user_id, 'workflow_name': workflow_id}, 
            limit=1
        )

        if requested_workflow_meta:
            requested_workflow_meta = requested_workflow_meta[0]
            if requested_workflow_meta.is_shared:
                if requested_workflow_meta.share_group in groups:
                    logger.info(f"✓ Access granted! Login user belongs to share group '{requested_workflow_meta.share_group}'. Using requested_user_id: {requested_user_id}")
                    return requested_user_id
                else:
                    return login_user_id
            else:
                logger.warning(f"✗ Access denied! Workflow is not shared (is_shared=False). Using login_user_id: {login_user_id}")
                return login_user_id
        else:
            logger.warning(f"✗ No workflow metadata found for user_id: {requested_user_id}, workflow_name: {workflow_id}. Using login_user_id: {login_user_id}")
            return login_user_id
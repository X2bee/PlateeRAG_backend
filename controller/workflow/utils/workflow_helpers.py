"""
워크플로우 파라미터 및 실행 관련 헬퍼 함수들
"""
import logging
from controller.workflow.helper import _workflow_parameter_helper, _default_workflow_parameter_helper

logger = logging.getLogger("workflow-helpers")

# 기존 helper 함수들을 재 export
async def workflow_parameter_helper(request_body, workflow_data):
    """워크플로우 파라미터를 설정합니다."""
    return await _workflow_parameter_helper(request_body, workflow_data)

async def default_workflow_parameter_helper(request, request_body, workflow_data):
    """기본 워크플로우 파라미터를 설정합니다."""
    return await _default_workflow_parameter_helper(request, request_body, workflow_data)
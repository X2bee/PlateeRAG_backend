"""
워크플로우 관련 API 라우터 - 리팩토링된 버전
각 기능별로 분리된 엔드포인트들을 통합 관리
"""
from fastapi import APIRouter
from controller.workflow.endpoints import basic_operations, deploy

# 메인 워크플로우 라우터 생성
router = APIRouter(prefix="", tags=["workflow"])

# 기본 CRUD 작업 라우터 포함
router.include_router(basic_operations.router, prefix="")

# 배포 관련 라우터 포함
router.include_router(deploy.router, prefix="/deploy")

# TODO: 다른 기능별 라우터들도 점진적으로 추가
# router.include_router(execution.router, prefix="/execute")
# router.include_router(performance.router, prefix="/performance") 
# router.include_router(tester.router, prefix="/tester")
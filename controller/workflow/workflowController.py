"""
워크플로우 관련 API 라우터 - 리팩토링된 버전
각 기능별로 분리된 엔드포인트들을 통합 관리
"""
from fastapi import APIRouter
from controller.workflow.endpoints import basic_operations, deploy, execution, performance, tester

# 메인 워크플로우 라우터 생성
router = APIRouter(prefix="", tags=["workflow"])

# 기본 CRUD 작업 라우터 포함
router.include_router(basic_operations.router, prefix="")

# 배포 관련 라우터 포함
router.include_router(deploy.router, prefix="/deploy")

# 실행 관련 라우터 포함
router.include_router(execution.router, prefix="/execute")

# 성능 및 로그 관련 라우터 포함
router.include_router(performance.router, prefix="/performance")

# 테스터 관련 라우터 포함  
router.include_router(tester.router, prefix="/execute/tester")
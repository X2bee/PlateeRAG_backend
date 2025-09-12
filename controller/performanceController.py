"""
성능 데이터 관련 컨트롤러 및 라우터
"""
import json
import logging
from typing import List, Dict, Optional, Any
from fastapi import APIRouter, HTTPException, Request, Query
from controller.helper.singletonHelper import get_config_composer, get_vector_manager, get_rag_service, get_document_processor, get_db_manager, get_performance_controller
from controller.helper.controllerHelper import extract_user_id_from_request
from service.database.logger_helper import create_logger

logger = logging.getLogger("workflow-controller")
router = APIRouter(prefix="/api/performance", tags=["performance"])

def safe_float(value: Any) -> float:
    """Decimal, None 등을 안전하게 float으로 변환합니다."""
    if value is None:
        return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

@router.get("/data")
async def get_performance_data(
    request: Request,
    workflow_name: Optional[str] = None,
    workflow_id: Optional[str] = None,
    node_id: Optional[str] = None,
    limit: int = 100
) -> Dict[str, Any]:
    """성능 데이터를 조회합니다."""
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        backend_log.info("Retrieving performance data",
                       metadata={"workflow_name": workflow_name,
                               "workflow_id": workflow_id,
                               "node_id": node_id,
                               "limit": limit})

        controller = get_performance_controller(request)
        data = controller.get_performance_data(workflow_name, workflow_id, node_id, limit)

        response_data = {
            "success": True,
            "data": data,
            "count": len(data)
        }

        backend_log.success("Successfully retrieved performance data",
                          metadata={"workflow_name": workflow_name,
                                  "workflow_id": workflow_id,
                                  "node_id": node_id,
                                  "data_count": len(data)})

        return response_data

    except Exception as e:
        backend_log.error("Failed to retrieve performance data", exception=e,
                        metadata={"workflow_name": workflow_name,
                                "workflow_id": workflow_id,
                                "node_id": node_id})
        raise HTTPException(status_code=500, detail=f"Failed to fetch performance data: {str(e)}") from e

@router.get("/average/{workflow_name}/{workflow_id}")
async def get_performance_average(
    request: Request,
    workflow_name: str,
    workflow_id: str
) -> Dict[str, Any]:
    """동일한 workflow_name과 workflow_id를 가진 성능 데이터들의 평균을 계산합니다."""
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        backend_log.info("Calculating performance average",
                       metadata={"workflow_name": workflow_name,
                               "workflow_id": workflow_id})

        controller = get_performance_controller(request)
        average_data = controller.get_performance_average(workflow_name, workflow_id)

        response_data = {
            "success": True,
            "data": average_data
        }

        backend_log.success("Successfully calculated performance average",
                          metadata={"workflow_name": workflow_name,
                                  "workflow_id": workflow_id,
                                  "has_data": average_data is not None})

        return response_data

    except Exception as e:
        backend_log.error("Failed to calculate performance average", exception=e,
                        metadata={"workflow_name": workflow_name,
                                "workflow_id": workflow_id})
        raise HTTPException(status_code=500, detail=f"Failed to calculate performance average: {str(e)}") from e

@router.get("/summary/{workflow_name}/{workflow_id}")
async def get_node_performance_summary(
    request: Request,
    workflow_name: str,
    workflow_id: str
) -> Dict[str, Any]:
    """워크플로우 내 각 노드별 성능 요약을 제공합니다."""
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        backend_log.info("Retrieving node performance summary",
                       metadata={"workflow_name": workflow_name,
                               "workflow_id": workflow_id})

        controller = get_performance_controller(request)
        summary_data = controller.get_node_performance_summary(workflow_name, workflow_id)

        response_data = {
            "success": True,
            "data": summary_data
        }

        node_count = len(summary_data) if isinstance(summary_data, list) else 0
        backend_log.success("Successfully retrieved node performance summary",
                          metadata={"workflow_name": workflow_name,
                                  "workflow_id": workflow_id,
                                  "node_count": node_count})

        return response_data

    except Exception as e:
        backend_log.error("Failed to get node performance summary", exception=e,
                        metadata={"workflow_name": workflow_name,
                                "workflow_id": workflow_id})
        raise HTTPException(status_code=500, detail=f"Failed to get node performance summary: {str(e)}") from e

@router.get("/counts/{workflow_name}/{workflow_id}")
async def get_node_log_counts_route(
    request: Request,
    workflow_name: str,
    workflow_id: str
):
    """워크플로우 내 각 노드별 로그 개수를 조회합니다."""
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    backend_log.info("Retrieving node log counts",
                   metadata={"workflow_name": workflow_name,
                           "workflow_id": workflow_id})

    try:
        controller = get_performance_controller(request)
        count_data = controller.get_node_log_counts(workflow_name, workflow_id)

        response_data = {"success": True, "data": count_data}

        node_count = len(count_data) if isinstance(count_data, list) else 0
        backend_log.success("Successfully retrieved node log counts",
                          metadata={"workflow_name": workflow_name,
                                  "workflow_id": workflow_id,
                                  "node_count": node_count})

        return response_data

    except Exception as e:
        backend_log.error("Failed to retrieve node log counts", exception=e,
                        metadata={"workflow_name": workflow_name,
                                "workflow_id": workflow_id})
        raise HTTPException(status_code=500, detail=f"Failed to get node log counts: {str(e)}")

@router.get("/charts/pie/{workflow_name}/{workflow_id}")
async def get_pie_chart_data_route(
    request: Request,
    workflow_name: str,
    workflow_id: str,
    limit: int = Query(10, description="분석할 최근 로그 개수")
):
    """노드별 평균 처리 시간 분포를 파이 차트용 데이터로 반환합니다."""
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    backend_log.info("Retrieving pie chart data",
                   metadata={"workflow_name": workflow_name,
                           "workflow_id": workflow_id,
                           "limit": limit})

    try:
        controller = get_performance_controller(request)
        chart_data = controller.get_pie_chart_data(workflow_name, workflow_id, limit)

        response_data = {"success": True, "data": chart_data}

        data_points = len(chart_data) if isinstance(chart_data, list) else 0
        backend_log.success("Successfully retrieved pie chart data",
                          metadata={"workflow_name": workflow_name,
                                  "workflow_id": workflow_id,
                                  "data_points": data_points})

        return response_data

    except Exception as e:
        backend_log.error("Failed to retrieve pie chart data", exception=e,
                        metadata={"workflow_name": workflow_name,
                                "workflow_id": workflow_id})
        raise HTTPException(status_code=500, detail=f"Failed to get pie chart data: {str(e)}")

@router.get("/charts/bar/{workflow_name}/{workflow_id}")
async def get_bar_chart_data_route(
    request: Request,
    workflow_name: str,
    workflow_id: str,
    limit: int = Query(10, description="분석할 최근 로그 개수")
):
    """노드별 평균 성능 지표를 바 차트용 데이터로 반환합니다."""
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    backend_log.info("Retrieving bar chart data",
                   metadata={"workflow_name": workflow_name,
                           "workflow_id": workflow_id,
                           "limit": limit})

    try:
        controller = get_performance_controller(request)
        chart_data = controller.get_bar_chart_data(workflow_name, workflow_id, limit)

        response_data = {"success": True, "data": chart_data}

        data_points = len(chart_data) if isinstance(chart_data, list) else 0
        backend_log.success("Successfully retrieved bar chart data",
                          metadata={"workflow_name": workflow_name,
                                  "workflow_id": workflow_id,
                                  "data_points": data_points})

        return response_data

    except Exception as e:
        backend_log.error("Failed to retrieve bar chart data", exception=e,
                        metadata={"workflow_name": workflow_name,
                                "workflow_id": workflow_id})
        raise HTTPException(status_code=500, detail=f"Failed to get bar chart data: {str(e)}")

@router.get("/charts/line/{workflow_name}/{workflow_id}")
async def get_line_chart_data_route(
    request: Request,
    workflow_name: str,
    workflow_id: str,
    limit: int = Query(10, description="가져올 최근 로그 개수")
):
    """시간 흐름에 따른 성능 지표를 라인 차트용 데이터로 반환합니다."""
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    backend_log.info("Retrieving line chart data",
                   metadata={"workflow_name": workflow_name,
                           "workflow_id": workflow_id,
                           "limit": limit})

    try:
        controller = get_performance_controller(request)
        chart_data = controller.get_line_chart_data(workflow_name, workflow_id, limit)

        response_data = {"success": True, "data": chart_data}

        data_points = len(chart_data) if isinstance(chart_data, list) else 0
        backend_log.success("Successfully retrieved line chart data",
                          metadata={"workflow_name": workflow_name,
                                  "workflow_id": workflow_id,
                                  "data_points": data_points})

        return response_data

    except Exception as e:
        backend_log.error("Failed to retrieve line chart data", exception=e,
                        metadata={"workflow_name": workflow_name,
                                "workflow_id": workflow_id})
        raise HTTPException(status_code=500, detail=f"Failed to get line chart data: {str(e)}")

@router.delete("/cleanup")
async def cleanup_old_performance_data(
    request: Request,
    days_to_keep: int = 30
) -> Dict[str, Any]:
    """지정된 일수보다 오래된 성능 데이터를 삭제합니다."""
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        backend_log.info("Starting old performance data cleanup",
                       metadata={"days_to_keep": days_to_keep})

        controller = get_performance_controller(request)
        success = controller.delete_old_performance_data(days_to_keep)

        response_data = {
            "success": success,
            "message": f"Old performance data cleanup {'completed' if success else 'failed'}"
        }

        if success:
            backend_log.success("Successfully completed old performance data cleanup",
                              metadata={"days_to_keep": days_to_keep})
        else:
            backend_log.warn("Performance data cleanup completed with issues",
                           metadata={"days_to_keep": days_to_keep})

        return response_data

    except Exception as e:
        backend_log.error("Failed to cleanup old performance data", exception=e,
                        metadata={"days_to_keep": days_to_keep})
        raise HTTPException(status_code=500, detail=f"Failed to cleanup old performance data: {str(e)}") from e

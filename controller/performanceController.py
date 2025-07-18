"""
성능 데이터 관련 컨트롤러 및 라우터
"""
import json
import logging
from typing import List, Dict, Optional, Any
from fastapi import APIRouter, HTTPException, Request, Query
from service.database.connection import AppDatabaseManager
from service.database.models.performance import NodePerformance
from collections import defaultdict

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

class PerformanceController:
    def __init__(self, db_manager: AppDatabaseManager):
        self.db_manager = db_manager

    def _get_recent_performance_logs(self, workflow_name: str, workflow_id: str, limit: int) -> List[Dict[str, Any]]:
        """
        특정 워크플로우의 최근 성능 로그를 시간순(오름차순)으로 정렬하여 가져옵니다.
        LIMIT을 먼저 적용하여 성능을 최적화합니다.
        """
        try:
            db_type = self.db_manager.config_db_manager.db_type
            
            if db_type == "postgresql":
                # PostgreSQL은 서브쿼리에 별칭(alias)이 필요합니다. (e.g., AS recent_logs)
                query = """
                    SELECT * FROM (
                        SELECT * FROM node_performance
                        WHERE workflow_name = %s AND workflow_id = %s
                        ORDER BY timestamp DESC
                        LIMIT %s
                    ) AS recent_logs
                    ORDER BY timestamp ASC
                """
                params = (workflow_name, workflow_id, limit)
            else: # sqlite
                query = """
                    SELECT * FROM (
                        SELECT * FROM node_performance
                        WHERE workflow_name = ? AND workflow_id = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    )
                    ORDER BY timestamp ASC
                """
                params = (workflow_name, workflow_id, limit)
            
            results = self.db_manager.config_db_manager.execute_query(query, params)
            return [dict(row) for row in results] if results else []

        except Exception as e:
            print(f"Error fetching recent performance logs: {e}")
            return []
    
    def get_performance_data(self, workflow_name: str = None, workflow_id: str = None, 
                           node_id: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """성능 데이터를 조회합니다."""
        try:
            conditions = []
            params = []
            param_counter = 1
            
            # 데이터베이스 타입 확인
            db_type = self.db_manager.config_db_manager.db_type
            
            if workflow_name:
                if db_type == "postgresql":
                    conditions.append(f"workflow_name = %s")
                else:
                    conditions.append("workflow_name = ?")
                params.append(workflow_name)
                param_counter += 1
            
            if workflow_id:
                if db_type == "postgresql":
                    conditions.append(f"workflow_id = %s")
                else:
                    conditions.append("workflow_id = ?")
                params.append(workflow_id)
                param_counter += 1
                
            if node_id:
                if db_type == "postgresql":
                    conditions.append(f"node_id = %s")
                else:
                    conditions.append("node_id = ?")
                params.append(node_id)
                param_counter += 1
            
            where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
            
            if db_type == "postgresql":
                limit_clause = f"LIMIT %s"
            else:
                limit_clause = "LIMIT ?"
            
            query = f"""
                SELECT * FROM node_performance
                {where_clause}
                ORDER BY timestamp DESC
                {limit_clause}
            """
            params.append(limit)
            
            results = self.db_manager.config_db_manager.execute_query(query, params)
            return [dict(row) for row in results] if results else []
            
        except Exception as e:
            print(f"Error fetching performance data: {e}")
            return []
    
    def get_performance_average(self, workflow_name: str, workflow_id: str) -> Dict[str, Any]:
        """동일한 workflow_name과 workflow_id를 가진 성능 데이터들의 평균을 계산합니다."""
        try:
            db_type = self.db_manager.config_db_manager.db_type
            
            if db_type == "postgresql":
                query = """
                    SELECT 
                        COUNT(*) as execution_count,
                        AVG(processing_time_ms) as avg_processing_time_ms,
                        AVG(cpu_usage_percent) as avg_cpu_usage_percent,
                        AVG(ram_usage_mb) as avg_ram_usage_mb,
                        AVG(CASE WHEN gpu_usage_percent IS NOT NULL THEN gpu_usage_percent END) as avg_gpu_usage_percent,
                        AVG(CASE WHEN gpu_memory_mb IS NOT NULL THEN gpu_memory_mb END) as avg_gpu_memory_mb,
                        MIN(timestamp) as first_execution,
                        MAX(timestamp) as last_execution
                    FROM node_performance 
                    WHERE workflow_name = %s AND workflow_id = %s
                """
            else:
                query = """
                    SELECT 
                        COUNT(*) as execution_count,
                        AVG(processing_time_ms) as avg_processing_time_ms,
                        AVG(cpu_usage_percent) as avg_cpu_usage_percent,
                        AVG(ram_usage_mb) as avg_ram_usage_mb,
                        AVG(CASE WHEN gpu_usage_percent IS NOT NULL THEN gpu_usage_percent END) as avg_gpu_usage_percent,
                        AVG(CASE WHEN gpu_memory_mb IS NOT NULL THEN gpu_memory_mb END) as avg_gpu_memory_mb,
                        MIN(timestamp) as first_execution,
                        MAX(timestamp) as last_execution
                    FROM node_performance 
                    WHERE workflow_name = ? AND workflow_id = ?
                """
            
            results = self.db_manager.config_db_manager.execute_query(query, [workflow_name, workflow_id])
            
            if results and results[0]['execution_count'] > 0:  # execution_count > 0
                row = results[0]
                return {
                    "workflow_name": workflow_name,
                    "workflow_id": workflow_id,
                    "execution_count": row['execution_count'],
                    "average_performance": {
                        "processing_time_ms": round(row['avg_processing_time_ms'], 2) if row['avg_processing_time_ms'] else 0,
                        "cpu_usage_percent": round(row['avg_cpu_usage_percent'], 2) if row['avg_cpu_usage_percent'] else 0,
                        "ram_usage_mb": round(row['avg_ram_usage_mb'], 2) if row['avg_ram_usage_mb'] else 0,
                        "gpu_usage_percent": round(row['avg_gpu_usage_percent'], 2) if row['avg_gpu_usage_percent'] else None,
                        "gpu_memory_mb": round(row['avg_gpu_memory_mb'], 2) if row['avg_gpu_memory_mb'] else None
                    },
                    "execution_period": {
                        "first_execution": row['first_execution'],
                        "last_execution": row['last_execution']
                    }
                }
            else:
                return {
                    "workflow_name": workflow_name,
                    "workflow_id": workflow_id,
                    "execution_count": 0,
                    "message": "No performance data found"
                }
                
        except Exception as e:
            print(f"Error calculating performance average: {e}")
            return {
                "error": f"Failed to calculate performance average: {str(e)}"
            }
    
    def get_node_performance_summary(self, workflow_name: str, workflow_id: str) -> Dict[str, Any]:
        """워크플로우 내 각 노드별 성능 요약을 제공합니다."""
        try:
            db_type = self.db_manager.config_db_manager.db_type
            
            if db_type == "postgresql":
                query = """
                    SELECT 
                        node_id,
                        node_name,
                        COUNT(*) as execution_count,
                        AVG(processing_time_ms) as avg_processing_time_ms,
                        AVG(cpu_usage_percent) as avg_cpu_usage_percent,
                        AVG(ram_usage_mb) as avg_ram_usage_mb,
                        MIN(processing_time_ms) as min_processing_time_ms,
                        MAX(processing_time_ms) as max_processing_time_ms
                    FROM node_performance 
                    WHERE workflow_name = %s AND workflow_id = %s
                    GROUP BY node_id, node_name
                    ORDER BY avg_processing_time_ms DESC
                """
            else:
                query = """
                    SELECT 
                        node_id,
                        node_name,
                        COUNT(*) as execution_count,
                        AVG(processing_time_ms) as avg_processing_time_ms,
                        AVG(cpu_usage_percent) as avg_cpu_usage_percent,
                        AVG(ram_usage_mb) as avg_ram_usage_mb,
                        MIN(processing_time_ms) as min_processing_time_ms,
                        MAX(processing_time_ms) as max_processing_time_ms
                    FROM node_performance 
                    WHERE workflow_name = ? AND workflow_id = ?
                    GROUP BY node_id, node_name
                    ORDER BY avg_processing_time_ms DESC
                """
            
            results = self.db_manager.config_db_manager.execute_query(query, [workflow_name, workflow_id])
            
            nodes_summary = []
            for row in results:
                nodes_summary.append({
                    "node_id": row['node_id'],
                    "node_name": row['node_name'],
                    "execution_count": row['execution_count'],
                    "avg_processing_time_ms": round(row['avg_processing_time_ms'], 2) if row['avg_processing_time_ms'] else 0,
                    "avg_cpu_usage_percent": round(row['avg_cpu_usage_percent'], 2) if row['avg_cpu_usage_percent'] else 0,
                    "avg_ram_usage_mb": round(row['avg_ram_usage_mb'], 2) if row['avg_ram_usage_mb'] else 0,
                    "min_processing_time_ms": round(row['min_processing_time_ms'], 2) if row['min_processing_time_ms'] else 0,
                    "max_processing_time_ms": round(row['max_processing_time_ms'], 2) if row['max_processing_time_ms'] else 0
                })
            
            return {
                "workflow_name": workflow_name,
                "workflow_id": workflow_id,
                "nodes_summary": nodes_summary,
                "total_nodes": len(nodes_summary)
            }
            
        except Exception as e:
            print(f"Error getting node performance summary: {e}")
            return {
                "error": f"Failed to get node performance summary: {str(e)}"
            }
        
    def get_node_log_counts(self, workflow_name: str, workflow_id: str) -> Dict[str, Any]:
        """워크플로우의 각 노드별 로그 개수를 조회합니다."""
        try:
            db_type = self.db_manager.config_db_manager.db_type
            param_char = "%s" if db_type == "postgresql" else "?"

            query = f"""
                SELECT
                    node_name,
                    COUNT(*) as log_count
                FROM node_performance
                WHERE workflow_name = {param_char} AND workflow_id = {param_char}
                GROUP BY node_name
                ORDER BY log_count DESC
            """
            params = (workflow_name, workflow_id)
            results = self.db_manager.config_db_manager.execute_query(query, params)

            if not results:
                return {"workflow_name": workflow_name, "workflow_id": workflow_id, "counts": [], "message": "No logs found for this workflow."}

            counts = [{"node_name": row['node_name'], "count": row['log_count']} for row in results]

            return {
                "workflow_name": workflow_name,
                "workflow_id": workflow_id,
                "node_counts": counts
            }

        except Exception as e:
            print(f"Error getting node log counts: {e}")
            return {"error": f"Failed to get node log counts: {str(e)}"}

    def get_pie_chart_data(self, workflow_name: str, workflow_id: str, limit: int) -> Dict[str, Any]:
        """파이 차트 데이터를 생성합니다."""
        logs = self._get_recent_performance_logs(workflow_name, workflow_id, limit)
        if not logs:
            return {"labels": [], "datasets": []}

        node_times = defaultdict(lambda: {'total_time': 0.0, 'count': 0})
        for log in logs:
            node_name = log.get('node_name')
            if node_name:
                node_times[node_name]['total_time'] += safe_float(log.get('processing_time_ms'))
                node_times[node_name]['count'] += 1
        
        labels = list(node_times.keys())
        avg_times = [node_times[name]['total_time'] / node_times[name]['count'] for name in labels]

        return {
            'title': f'Node Average Processing Time (Last {limit} logs)',
            'labels': labels,
            'datasets': [{'label': 'Average Processing Time (ms)', 'data': avg_times}]
        }
    
    def get_bar_chart_data(self, workflow_name: str, workflow_id: str, limit: int) -> Dict[str, Any]:
        """바 차트 데이터를 생성합니다."""
        logs = self._get_recent_performance_logs(workflow_name, workflow_id, limit)
        if not logs:
            return {"processingTime": {}, "cpuUsage": {}}

        node_metrics = defaultdict(lambda: {'time': [], 'cpu': []})
        for log in logs:
            node_name = log.get('node_name')
            if node_name:
                node_metrics[node_name]['time'].append(safe_float(log.get('processing_time_ms')))
                node_metrics[node_name]['cpu'].append(safe_float(log.get('cpu_usage_percent')))

        labels = list(node_metrics.keys())
        avg_time = [sum(m['time']) / len(m['time']) if m['time'] else 0 for m in node_metrics.values()]
        avg_cpu = [sum(m['cpu']) / len(m['cpu']) if m['cpu'] else 0 for m in node_metrics.values()]
        
        return {
            'processingTime': {
                'title': f'Average Processing Time (Last {limit} logs)',
                'labels': labels,
                'datasets': [{'label': 'Avg. Time (ms)', 'data': avg_time}]
            },
            'cpuUsage': {
                'title': f'Average CPU Usage (Last {limit} logs)',
                'labels': labels,
                'datasets': [{'label': 'Avg. CPU (%)', 'data': avg_cpu}]
            }
        }
    
    def get_line_chart_data(self, workflow_name: str, workflow_id: str, limit: int) -> Dict[str, Any]:
        """라인 차트 데이터를 생성합니다."""
        logs = self._get_recent_performance_logs(workflow_name, workflow_id, limit)
        if not logs:
            return {"cpuOverTime": {}, "processingTimeOverTime": {}}

        datasets_cpu = defaultdict(list)
        datasets_time = defaultdict(list)

        for log in logs:
            node_name = log.get('node_name')
            if node_name:
                timestamp = log.get('timestamp').isoformat() if hasattr(log.get('timestamp'), 'isoformat') else str(log.get('timestamp')) # type: ignore
                datasets_cpu[node_name].append({'x': timestamp, 'y': safe_float(log.get('cpu_usage_percent'))})
                datasets_time[node_name].append({'x': timestamp, 'y': safe_float(log.get('processing_time_ms'))})

        return {
            'cpuOverTime': {
                'title': f'CPU Usage Over Time (Last {limit} logs)',
                'datasets': [{'label': name, 'data': data} for name, data in datasets_cpu.items()]
            },
            'processingTimeOverTime': {
                'title': f'Processing Time Over Time (Last {limit} logs)',
                'datasets': [{'label': name, 'data': data} for name, data in datasets_time.items()]
            }
        }
    
    def delete_old_performance_data(self, days_to_keep: int = 30) -> bool:
        """지정된 일수보다 오래된 성능 데이터를 삭제합니다."""
        try:
            query = """
                DELETE FROM node_performance 
                WHERE timestamp < datetime('now', '-{} days')
            """.format(days_to_keep)
            
            self.db_manager.config_db_manager.execute_query(query)
            return True
            
        except Exception as e:
            print(f"Error deleting old performance data: {e}")
            return False



def get_performance_controller(request: Request):
    """성능 컨트롤러 의존성 주입"""
    if hasattr(request.app.state, 'app_db') and request.app.state.app_db:
        return PerformanceController(request.app.state.app_db)
    else:
        raise HTTPException(status_code=500, detail="Database not available")

@router.get("/data")
async def get_performance_data(
    request: Request,
    workflow_name: Optional[str] = None,
    workflow_id: Optional[str] = None,
    node_id: Optional[str] = None,
    limit: int = 100
) -> Dict[str, Any]:
    """성능 데이터를 조회합니다."""
    try:
        controller = get_performance_controller(request)
        data = controller.get_performance_data(workflow_name, workflow_id, node_id, limit)
        
        return {
            "success": True,
            "data": data,
            "count": len(data)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch performance data: {str(e)}") from e

@router.get("/average/{workflow_name}/{workflow_id}")
async def get_performance_average(
    request: Request,
    workflow_name: str,
    workflow_id: str
) -> Dict[str, Any]:
    """동일한 workflow_name과 workflow_id를 가진 성능 데이터들의 평균을 계산합니다."""
    try:
        controller = get_performance_controller(request)
        average_data = controller.get_performance_average(workflow_name, workflow_id)
        
        return {
            "success": True,
            "data": average_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to calculate performance average: {str(e)}") from e

@router.get("/summary/{workflow_name}/{workflow_id}")
async def get_node_performance_summary(
    request: Request,
    workflow_name: str,
    workflow_id: str
) -> Dict[str, Any]:
    """워크플로우 내 각 노드별 성능 요약을 제공합니다."""
    try:
        controller = get_performance_controller(request)
        summary_data = controller.get_node_performance_summary(workflow_name, workflow_id)
        
        return {
            "success": True,
            "data": summary_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get node performance summary: {str(e)}") from e

@router.get("/counts/{workflow_name}/{workflow_id}")
async def get_node_log_counts_route(
    request: Request,
    workflow_name: str,
    workflow_id: str
):
    """워크플로우 내 각 노드별 로그 개수를 조회합니다."""
    controller = get_performance_controller(request)
    count_data = controller.get_node_log_counts(workflow_name, workflow_id)
    return {"success": True, "data": count_data}

@router.get("/charts/pie/{workflow_name}/{workflow_id}")
async def get_pie_chart_data_route(
    request: Request,
    workflow_name: str,
    workflow_id: str,
    limit: int = Query(10, description="분석할 최근 로그 개수")
):
    """노드별 평균 처리 시간 분포를 파이 차트용 데이터로 반환합니다."""
    controller = get_performance_controller(request)
    chart_data = controller.get_pie_chart_data(workflow_name, workflow_id, limit)
    return {"success": True, "data": chart_data}

@router.get("/charts/bar/{workflow_name}/{workflow_id}")
async def get_bar_chart_data_route(
    request: Request,
    workflow_name: str,
    workflow_id: str,
    limit: int = Query(10, description="분석할 최근 로그 개수")
):
    """노드별 평균 성능 지표를 바 차트용 데이터로 반환합니다."""
    controller = get_performance_controller(request)
    chart_data = controller.get_bar_chart_data(workflow_name, workflow_id, limit)
    return {"success": True, "data": chart_data}

@router.get("/charts/line/{workflow_name}/{workflow_id}")
async def get_line_chart_data_route(
    request: Request,
    workflow_name: str,
    workflow_id: str,
    limit: int = Query(10, description="가져올 최근 로그 개수")
):
    """시간 흐름에 따른 성능 지표를 라인 차트용 데이터로 반환합니다."""
    controller = get_performance_controller(request)
    chart_data = controller.get_line_chart_data(workflow_name, workflow_id, limit)
    return {"success": True, "data": chart_data}

@router.delete("/cleanup")
async def cleanup_old_performance_data(
    request: Request,
    days_to_keep: int = 30
) -> Dict[str, Any]:
    """지정된 일수보다 오래된 성능 데이터를 삭제합니다."""
    try:
        controller = get_performance_controller(request)
        success = controller.delete_old_performance_data(days_to_keep)
        
        return {
            "success": success,
            "message": f"Old performance data cleanup {'completed' if success else 'failed'}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cleanup old performance data: {str(e)}") from e

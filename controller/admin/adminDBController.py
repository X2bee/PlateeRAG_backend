import logging
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
from fastapi import APIRouter, Request, HTTPException
from controller.admin.adminHelper import manager_section_access
from controller.helper.singletonHelper import get_db_manager
from controller.admin.adminBaseController import validate_superuser
from service.database.logger_helper import create_logger

logger = logging.getLogger("admin-db-controller")
router = APIRouter(prefix="/database", tags=["Admin"])

class QueryRequest(BaseModel):
    query: str
    params: Optional[List[Any]] = None

class QueryResponse(BaseModel):
    success: bool
    error: Optional[str] = None
    data: List[Dict[str, Any]] = []
    row_count: Optional[int] = None

class TableListResponse(BaseModel):
    success: bool
    tables: List[Dict[str, Any]] = []
    error: Optional[str] = None

@router.get("/tables", response_model=TableListResponse)
async def get_table_list(request: Request):
    """
    데이터베이스의 모든 테이블 목록을 반환하는 API
    """
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, val_superuser.get("user_id"), request)
    section_access = manager_section_access(app_db, val_superuser.get("user_id"), ["database"])
    if not section_access:
        backend_log.warn(f"User {val_superuser.get('user_id')} attempted to access database without permission")
        raise HTTPException(
            status_code=403,
            detail="Database permissions access required"
        )

    try:
        tables = app_db.get_table_list()
        backend_log.success("Successfully retrieved database table list",
                          metadata={"table_count": len(tables)})
        logger.info("Retrieved %d tables from database", len(tables))
        return TableListResponse(
            success=True,
            tables=tables
        )

    except Exception as e:
        backend_log.error("Error getting table list", exception=e)
        logger.error("Error getting table list: %s", str(e))
        return TableListResponse(
            success=False,
            tables=[],
            error=str(e)
        )

@router.post("/query", response_model=QueryResponse)
async def execute_query(request: Request, query_request: QueryRequest):
    """
    임의의 SQL 쿼리를 실행하는 API
    보안상 SELECT 쿼리만 허용됩니다.
    """
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, val_superuser.get("user_id"), request)
    section_access = manager_section_access(app_db, val_superuser.get("user_id"), ["database"])
    if not section_access:
        backend_log.warn(f"User {val_superuser.get('user_id')} attempted to access database without permission")
        raise HTTPException(
            status_code=403,
            detail="Database permissions access required"
        )

    try:
        # 파라미터 튜플로 변환
        params = None
        if query_request.params:
            params = tuple(query_request.params)

        # 쿼리 실행
        result = app_db.execute_raw_query(query_request.query, params)

        if result["success"]:
            backend_log.success("Query executed successfully",
                              metadata={"query": query_request.query[:100], "row_count": result.get('row_count', 0)})
            logger.info("Query executed successfully, returned %d rows", result.get('row_count', 0))
        else:
            backend_log.warn("Query execution failed",
                           metadata={"query": query_request.query[:100], "error": result['error']})
            logger.warning("Query execution failed: %s", result['error'])

        return QueryResponse(
            success=result["success"],
            error=result["error"],
            data=result["data"],
            row_count=result.get("row_count")
        )

    except Exception as e:
        backend_log.error("Error executing query", exception=e,
                         metadata={"query": query_request.query[:100]})
        logger.error("Error executing query: %s", str(e))
        return QueryResponse(
            success=False,
            error=str(e),
            data=[],
            row_count=0
        )

@router.get("/table/{table_name}/structure")
async def get_table_structure(request: Request, table_name: str):
    """
    특정 테이블의 구조(컬럼 정보)를 반환하는 API
    """
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, val_superuser.get("user_id"), request)
    section_access = manager_section_access(app_db, val_superuser.get("user_id"), ["database"])
    if not section_access:
        backend_log.warn(f"User {val_superuser.get('user_id')} attempted to access database without permission")
        raise HTTPException(
            status_code=403,
            detail="Database permissions access required"
        )

    try:
        db_type = app_db.config_db_manager.db_type

        if db_type == "postgresql":
            query = """
                SELECT
                    column_name,
                    data_type,
                    is_nullable,
                    column_default,
                    character_maximum_length,
                    numeric_precision,
                    numeric_scale
                FROM information_schema.columns
                WHERE table_name = %s
                ORDER BY ordinal_position
            """
            params = (table_name,)
        else:  # SQLite
            query = f"PRAGMA table_info({table_name})"
            params = None

        result = app_db.execute_raw_query(query, params)

        if result["success"]:
            logger.info("Retrieved structure for table: %s", table_name)
            return {
                "success": True,
                "table_name": table_name,
                "columns": result["data"],
                "error": None
            }
        else:
            logger.warning("Failed to get table structure: %s", result['error'])
            return {
                "success": False,
                "table_name": table_name,
                "columns": [],
                "error": result["error"]
            }

    except Exception as e:
        logger.error("Error getting table structure for %s: %s", table_name, str(e))
        return {
            "success": False,
            "table_name": table_name,
            "columns": [],
            "error": str(e)
        }

@router.get("/table/{table_name}/sample")
async def get_table_sample_data(request: Request, table_name: str, limit: int = 100):
    """
    특정 테이블의 샘플 데이터를 반환하는 API
    """
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, val_superuser.get("user_id"), request)
    section_access = manager_section_access(app_db, val_superuser.get("user_id"), ["database"])
    if not section_access:
        backend_log.warn(f"User {val_superuser.get('user_id')} attempted to access database without permission")
        raise HTTPException(
            status_code=403,
            detail="Database permissions access required"
        )

    try:
        # limit 값 검증
        if limit < 1 or limit > 1000:
            limit = 100

        db_type = app_db.config_db_manager.db_type

        if db_type == "postgresql":
            query = f"SELECT * FROM {table_name} LIMIT %s"
            params = (limit,)
        else:  # SQLite
            query = f"SELECT * FROM {table_name} LIMIT ?"
            params = (limit,)

        result = app_db.execute_raw_query(query, params)

        if result["success"]:
            logger.info("Retrieved %d sample rows from table: %s", result.get('row_count', 0), table_name)
            return {
                "success": True,
                "table_name": table_name,
                "data": result["data"],
                "row_count": result.get("row_count", 0),
                "limit": limit,
                "error": None
            }
        else:
            logger.warning("Failed to get sample data: %s", result['error'])
            return {
                "success": False,
                "table_name": table_name,
                "data": [],
                "row_count": 0,
                "limit": limit,
                "error": result["error"]
            }

    except Exception as e:
        logger.error("Error getting sample data for %s: %s", table_name, str(e))
        return {
            "success": False,
            "table_name": table_name,
            "data": [],
            "row_count": 0,
            "limit": limit,
            "error": str(e)
        }

@router.get("/database/info")
async def get_database_info(request: Request):
    """
    데이터베이스 기본 정보를 반환하는 API
    """
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, val_superuser.get("user_id"), request)
    section_access = manager_section_access(app_db, val_superuser.get("user_id"), ["database"])
    if not section_access:
        backend_log.warn(f"User {val_superuser.get('user_id')} attempted to access database without permission")
        raise HTTPException(
            status_code=403,
            detail="Database permissions access required"
        )

    try:
        db_type = app_db.config_db_manager.db_type

        # 기본 데이터베이스 정보
        db_info = {
            "database_type": db_type,
            "connection_status": "connected" if app_db.config_db_manager.connection else "disconnected"
        }

        # 데이터베이스별 추가 정보
        if db_type == "postgresql":
            version_query = "SELECT version() as version"
            version_result = app_db.execute_raw_query(version_query)
            if version_result["success"] and version_result["data"]:
                db_info["version"] = version_result["data"][0]["version"]
        elif db_type == "sqlite":
            version_query = "SELECT sqlite_version() as version"
            version_result = app_db.execute_raw_query(version_query)
            if version_result["success"] and version_result["data"]:
                db_info["version"] = f"SQLite {version_result['data'][0]['version']}"

        # 테이블 개수
        tables = app_db.get_table_list()
        db_info["table_count"] = len(tables)

        logger.info("Retrieved database info successfully")
        return {
            "success": True,
            "database_info": db_info,
            "error": None
        }

    except Exception as e:
        logger.error("Error getting database info: %s", str(e))
        return {
            "success": False,
            "database_info": {},
            "error": str(e)
        }

@router.get("/table/{table_name}/count")
async def get_table_row_count(request: Request, table_name: str):
    """
    특정 테이블의 행 개수를 반환하는 API
    """
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, val_superuser.get("user_id"), request)
    section_access = manager_section_access(app_db, val_superuser.get("user_id"), ["database"])
    if not section_access:
        backend_log.warn(f"User {val_superuser.get('user_id')} attempted to access database without permission")
        raise HTTPException(
            status_code=403,
            detail="Database permissions access required"
        )

    try:
        if not table_name.replace('_', '').replace('-', '').isalnum():
            return {
                "success": False,
                "table_name": table_name,
                "row_count": 0,
                "error": "Invalid table name format"
            }

        count_query = f"SELECT COUNT(*) as count FROM {table_name}"
        result = app_db.execute_raw_query(count_query)

        if result["success"] and result["data"]:
            row_count = result["data"][0]["count"]
            logger.info("Retrieved row count for table %s: %d", table_name, row_count)
            return {
                "success": True,
                "table_name": table_name,
                "row_count": row_count,
                "error": None
            }
        else:
            logger.warning("Failed to get row count for table %s: %s", table_name, result.get('error'))
            return {
                "success": False,
                "table_name": table_name,
                "row_count": 0,
                "error": result.get("error", "Unknown error")
            }

    except Exception as e:
        logger.error("Error getting row count for table %s: %s", table_name, str(e))
        return {
            "success": False,
            "table_name": table_name,
            "row_count": 0,
            "error": str(e)
        }

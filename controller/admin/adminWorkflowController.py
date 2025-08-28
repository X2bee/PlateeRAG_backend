import logging
import json
from fastapi import APIRouter, Request, HTTPException
from controller.helper.singletonHelper import get_db_manager
from controller.admin.adminBaseController import validate_superuser

from service.database.models.executor import ExecutionIO

logger = logging.getLogger("admin-controller")
router = APIRouter(prefix="/workflow", tags=["Admin"])

def extract_result_from_json(json_string):
    """
    Extract the 'result' field from a JSON string.
    If parsing fails or 'result' is not found, return the original string.
    """
    if not json_string:
        return json_string

    try:
        data = json.loads(json_string)
        return data.get("result", json_string)
    except (json.JSONDecodeError, TypeError):
        return json_string

def process_io_logs_efficient(io_logs):
    """
    Efficiently process io_logs using map and dictionary unpacking.
    """
    def process_single_log(log):
        log_dict = {k: v for k, v in log.__dict__.items() if not k.startswith('_')}
        log_dict.update({
            "input_data": extract_result_from_json(log.input_data),
            "output_data": extract_result_from_json(log.output_data)
        })
        return log_dict

    return list(map(process_single_log, io_logs))

@router.get("/all-io-logs")
async def get_all_workflows(request: Request, page: int = 1, page_size: int = 250):
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )

    try:
        # 페이지 번호 검증
        if page < 1:
            page = 1
        if page_size < 1 or page_size > 1000:
            page_size = 250

        offset = (page - 1) * page_size

        app_db = get_db_manager(request)

        io_logs = app_db.find_all(ExecutionIO, limit=page_size, offset=offset)
        processed_logs = process_io_logs_efficient(io_logs)

        return {
            "io_logs": processed_logs,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "offset": offset,
                "total_returned": len(processed_logs)
            }
        }
    except Exception as e:
        logger.error("Error fetching all IO logs: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        ) from e

"""
Admin User Token Dashboard Controller
Handles user token usage statistics and analytics
"""
import logging
import json
from datetime import datetime, date, timezone
from typing import Optional, Dict, List, Any
from fastapi import APIRouter, Request, HTTPException, Query
from controller.helper.singletonHelper import get_db_manager
from controller.admin.adminBaseController import validate_superuser
from service.database.logger_helper import create_logger
from controller.admin.adminHelper import get_manager_groups, get_manager_accessible_users, manager_section_access, get_manager_accessible_workflows_ids

from service.database.models.executor import ExecutionIO
from service.database.models.user import User
from service.utils.token_counter import count_io_tokens

logger = logging.getLogger("admin-user-token-controller")
router = APIRouter(prefix="/user-token", tags=["Admin User Token"])

# "parsed date filters" 와 "database datetime values"가 mismatch되는 error 해결 위해 다음 코드 추가
def normalize_datetime_for_comparison(dt):
    """
    Normalize datetime for comparison by ensuring timezone awareness.
    If datetime is naive, assume it's UTC.
    """
    if dt is None:
        return None

    if isinstance(dt, str):
        # Parse string datetime
        try:
            dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
        except ValueError:
            try:
                dt = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S.%f")
            except ValueError:
                try:
                    dt = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    return dt  # Return as-is if can't parse

    # If datetime is naive, make it timezone-aware (assume UTC)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt

@router.get("/usage")
async def get_user_token_usage(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    start_date: Optional[str] = Query(None, description="Start date in YYYY-MM-DD format"),
    end_date: Optional[str] = Query(None, description="End date in YYYY-MM-DD format")
):
    """
    Get token usage statistics per user with pagination and date filtering.
    """
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, val_superuser.get("user_id"), request)
    section_access = manager_section_access(app_db, val_superuser.get("user_id"), ["user-token-dashboard"])
    if not section_access:
        backend_log.warn(f"User {val_superuser.get('user_id')} attempted to access user token dashboard without permission")
        raise HTTPException(
            status_code=403,
            detail="User token dashboard access required"
        )

    try:
        start_datetime = None
        end_datetime = None

        if start_date:
            try:
                start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
                # Make timezone-aware (UTC) to match database timestamps
                start_datetime = start_datetime.replace(tzinfo=timezone.utc)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid start_date format. Use YYYY-MM-DD")

        if end_date:
            try:
                end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
                # Add one day to include the entire end date
                end_datetime = end_datetime.replace(hour=23, minute=59, second=59)
                # Make timezone-aware (UTC) to match database timestamps
                end_datetime = end_datetime.replace(tzinfo=timezone.utc)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid end_date format. Use YYYY-MM-DD")

        if val_superuser.get("user_type") != "superuser":
            manager_accessible_users = get_manager_accessible_users(app_db, val_superuser.get("user_id"))
            user_id_to_username = {user.id: user.username for user in manager_accessible_users}
            manager_accessible_user_ids = list(user_id_to_username.keys())
            all_logs = app_db.find_by_condition(ExecutionIO, {"user_id__in__": manager_accessible_user_ids})
        else:
            all_users = app_db.find_all(User)
            user_id_to_username = {user.id: user.username for user in all_users}
            all_logs = app_db.find_all(ExecutionIO)

        # First pass: identify users who had activity in the date range (if specified)
        users_in_date_range = set()
        if start_datetime or end_datetime:
            for log in all_logs:
                user_id = log.user_id
                if user_id is None:
                    continue

                log_date = normalize_datetime_for_comparison(log.created_at)
                in_range = True
                if start_datetime and log_date < start_datetime:
                    in_range = False
                if end_datetime and log_date > end_datetime:
                    in_range = False

                if in_range:
                    users_in_date_range.add(user_id)

        # Second pass: calculate complete statistics for all users (or filtered users)
        user_stats = {}

        for log in all_logs:
            user_id = log.user_id
            if user_id is None:
                continue  # Skip logs without user_id

            # If date filtering is active, only include users who had activity in the date range
            if (start_datetime or end_datetime) and user_id not in users_in_date_range:
                continue

            if user_id not in user_stats:
                username = user_id_to_username.get(user_id, None)
                user_stats[user_id] = {
                    'user_id': user_id,
                    'username': username,
                    'total_interactions': 0,
                    'total_input_tokens': 0,
                    'total_output_tokens': 0,
                    'total_tokens': 0,
                    'workflow_usage': {},
                    'first_interaction': None,
                    'last_interaction': None
                }

            # Count tokens for this interaction
            try:
                token_counts = count_io_tokens(log.input_data, log.output_data)
            except Exception as e:
                logger.warning(f"Failed to count tokens for log {log.id}: {e}")
                token_counts = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}

            # Update user statistics
            stats = user_stats[user_id]
            stats['total_interactions'] += 1
            stats['total_input_tokens'] += token_counts['input_tokens']
            stats['total_output_tokens'] += token_counts['output_tokens']
            stats['total_tokens'] += token_counts['total_tokens']

            # Track workflow usage
            workflow_name = log.workflow_name or 'Unknown'
            if workflow_name not in stats['workflow_usage']:
                stats['workflow_usage'][workflow_name] = {'usage_count': 0, 'total_tokens': 0, 'interactions': 0, 'input_tokens': 0, 'output_tokens': 0}
            stats['workflow_usage'][workflow_name]['usage_count'] += 1
            stats['workflow_usage'][workflow_name]['total_tokens'] += token_counts['total_tokens']
            stats['workflow_usage'][workflow_name]['interactions'] += 1
            stats['workflow_usage'][workflow_name]['input_tokens'] += token_counts['input_tokens']
            stats['workflow_usage'][workflow_name]['output_tokens'] += token_counts['output_tokens']

            # Track interaction dates (complete history, not just filtered range)
            created_at = normalize_datetime_for_comparison(log.created_at)
            if stats['first_interaction'] is None or created_at < stats['first_interaction']:
                stats['first_interaction'] = created_at
            if stats['last_interaction'] is None or created_at > stats['last_interaction']:
                stats['last_interaction'] = created_at



        # Convert to list and add derived fields
        user_list = []
        for user_id, stats in user_stats.items():
            # Find most used workflow
            if stats['workflow_usage']:
                most_used_workflow = max(stats['workflow_usage'].items(), key=lambda x: x[1]['usage_count'])
                stats['most_used_workflow'] = most_used_workflow[0]
                stats['workflow_usage_count'] = most_used_workflow[1]
            else:
                stats['most_used_workflow'] = None
                stats['workflow_usage_count'] = 0

            # Calculate averages
            if stats['total_interactions'] > 0:
                stats['average_input_tokens'] = round(stats['total_input_tokens'] / stats['total_interactions'], 2)
                stats['average_output_tokens'] = round(stats['total_output_tokens'] / stats['total_interactions'], 2)
            else:
                stats['average_input_tokens'] = 0
                stats['average_output_tokens'] = 0

            # Format dates
            if stats['first_interaction']:
                stats['first_interaction'] = stats['first_interaction'].isoformat()
            else:
                stats['first_interaction'] = None

            if stats['last_interaction']:
                stats['last_interaction'] = stats['last_interaction'].isoformat()
            else:
                stats['last_interaction'] = None

            # Remove workflow_usage dict (not needed in response)
            # del stats['workflow_usage']

            user_list.append(stats)

        # Sort by total tokens (descending)
        user_list.sort(key=lambda x: x['total_tokens'], reverse=True)

        # Apply pagination
        total_users = len(user_list)
        total_pages = (total_users + page_size - 1) // page_size
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_users = user_list[start_idx:end_idx]

        return {
            "users": paginated_users,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_users": total_users,
                "total_pages": total_pages
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching user token usage: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        ) from e

@router.get("/summary")
async def get_token_usage_summary(
    request: Request,
    start_date: Optional[str] = Query(None, description="Start date in YYYY-MM-DD format"),
    end_date: Optional[str] = Query(None, description="End date in YYYY-MM-DD format")
):
    """
    Get overall token usage summary statistics.
    """
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, val_superuser.get("user_id"), request)
    section_access = manager_section_access(app_db, val_superuser.get("user_id"), ["user-token-dashboard"])
    if not section_access:
        backend_log.warn(f"User {val_superuser.get('user_id')} attempted to access user token dashboard without permission")
        raise HTTPException(
            status_code=403,
            detail="User token dashboard access required"
        )

    try:
        # Parse date filters (same logic as main endpoint)
        start_datetime = None
        end_datetime = None

        if start_date:
            try:
                start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
                start_datetime = start_datetime.replace(tzinfo=timezone.utc)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid start_date format. Use YYYY-MM-DD")

        if end_date:
            try:
                end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
                end_datetime = end_datetime.replace(hour=23, minute=59, second=59)
                end_datetime = end_datetime.replace(tzinfo=timezone.utc)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid end_date format. Use YYYY-MM-DD")

        # Get all logs
        if val_superuser.get("user_type") != "superuser":
            manager_accessible_users = get_manager_accessible_users(app_db, val_superuser.get("user_id"))
            manager_accessible_user_ids = [user.id for user in manager_accessible_users]
            all_logs = app_db.find_by_condition(ExecutionIO, {"user_id__in__": manager_accessible_user_ids})
        else:
            all_logs = app_db.find_all(ExecutionIO)

        # Filter logs by date range if specified
        filtered_logs = []
        if start_datetime or end_datetime:
            for log in all_logs:
                log_date = normalize_datetime_for_comparison(log.created_at)

                in_range = True
                if start_datetime and log_date < start_datetime:
                    in_range = False
                if end_datetime and log_date > end_datetime:
                    in_range = False

                if in_range:
                    filtered_logs.append(log)
        else:
            filtered_logs = all_logs

        # Calculate summary statistics
        total_interactions = len(filtered_logs)
        total_input_tokens = 0
        total_output_tokens = 0
        unique_users = set()
        unique_workflows = set()

        for log in filtered_logs:
            if log.user_id:
                unique_users.add(log.user_id)
            if log.workflow_name:
                unique_workflows.add(log.workflow_name)

            try:
                token_counts = count_io_tokens(log.input_data, log.output_data)
                total_input_tokens += token_counts['input_tokens']
                total_output_tokens += token_counts['output_tokens']
            except Exception as e:
                logger.warning(f"Failed to count tokens for log {log.id}: {e}")

        total_tokens = total_input_tokens + total_output_tokens
        avg_tokens_per_interaction = round(total_tokens / total_interactions, 2) if total_interactions > 0 else 0

        return {
            "summary": {
                "total_interactions": total_interactions,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_tokens": total_tokens,
                "unique_users": len(unique_users),
                "unique_workflows": len(unique_workflows),
                "average_tokens_per_interaction": avg_tokens_per_interaction
            },
            "date_range": {
                "start_date": start_date,
                "end_date": end_date
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching token usage summary: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        ) from e

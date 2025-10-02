from service.database import AppDatabaseManager
from service.database.models.user import User
from service.database.models.workflow import WorkflowMeta
from controller.utils.section_config import available_admin_sections

def get_manager_groups(app_db: AppDatabaseManager, manager_id):
    manager_account = app_db.find_by_condition(User, {"id": manager_id})
    manager_groups = manager_account[0].groups if manager_account else []
    admin_groups = [group for group in manager_groups if group.endswith("__admin__")]
    admin_groups = [group.replace("__admin__", "") for group in admin_groups]
    return list(set(admin_groups))

def get_manager_accessible_users(app_db: AppDatabaseManager, manager_id):
    admin_groups = get_manager_groups(app_db, manager_id)
    if not admin_groups:
        return []

    all_users = app_db.find_by_condition(User, {"user_type__in__": ['admin', 'standard']}, limit=10000)
    own_admin_account = app_db.find_by_condition(User, {"id": manager_id, "user_type": "admin"}, limit=1)

    filtered_users = [own_admin_account[0]] if own_admin_account else []
    for user in all_users:
        user_groups = user.groups if user.groups else []
        user_normal_groups = [g for g in user_groups if not g.endswith("__admin__")]
        user_groups_set = set(user_normal_groups)

        if user_groups_set and user_groups_set.issubset(set(admin_groups)):
            filtered_users.append(user)

    return filtered_users

def manager_section_access(app_db: AppDatabaseManager, manager_id, section_name):
    db_user_info = app_db.find_by_condition(User, {"id": manager_id}, limit=1)
    if not db_user_info:
        return False

    db_user_info = db_user_info[0]
    user_type = db_user_info.user_type
    user_available_admin_sections = db_user_info.available_admin_sections

    if user_type == "superuser":
        return True

    if isinstance(section_name, str):
        section_names = [section_name]
    else:
        section_names = section_name

    return any(section in user_available_admin_sections for section in section_names)

def get_manager_accessible_workflows_ids(app_db, manager_id):
    admin_groups = get_manager_groups(app_db, manager_id)
    workflow_ids = []
    my_workflow_metas = app_db.find_by_condition(WorkflowMeta, {"user_id": manager_id}, select_columns=["workflow_id"])
    workflow_ids.extend([meta.workflow_id for meta in my_workflow_metas])

    if admin_groups:
        workflow_metas = app_db.find_by_condition(WorkflowMeta, {"share_group__in__": admin_groups}, select_columns=["workflow_id"])
        workflow_ids.extend([meta.workflow_id for meta in workflow_metas])

    all_users = app_db.find_by_condition(
        User,
        {"user_type__in__": ['admin', 'standard']},
        limit=10000
    )

    filtered_users = []
    for user in all_users:
        user_groups = user.groups if user.groups else []
        user_normal_groups = [g for g in user_groups if not g.endswith("__admin__")]
        user_groups_set = set(user_normal_groups)

        if user_groups_set and user_groups_set.issubset(set(admin_groups)):
            filtered_users.append(user)

    filtered_user_ids = [user.id for user in filtered_users]
    if filtered_user_ids:
        user_workflow_metas = app_db.find_by_condition(WorkflowMeta, {"user_id__in__": filtered_user_ids}, select_columns=["workflow_id"])
        workflow_ids.extend([meta.workflow_id for meta in user_workflow_metas])

    return list(set(workflow_ids))

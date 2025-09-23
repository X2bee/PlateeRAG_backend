from typing import Optional
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import logging
import gc
from controller.helper.singletonHelper import get_rag_service, get_config_composer, get_db_manager
from controller.helper.controllerHelper import extract_user_id_from_request
from service.database.models.vectordb import VectorDB, VectorDBFolders
from service.database.logger_helper import create_logger

logger = logging.getLogger("folder-controller")
router = APIRouter(prefix="/folder", tags=["folder"])

class CreateFolderRequest(BaseModel):
    folder_name: str
    parent_collection_id: int = None
    parent_folder_id: Optional[int] = None
    parent_folder_name: Optional[str] = None

class DeleteFolderRequest(BaseModel):
    folder_path: str
    collection_id: int

@router.post("/create")
async def create_folder(request: Request, create_folder_request: CreateFolderRequest):
    """Create a new folder"""
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        backend_log.info("Starting folder creation",
                        metadata={"folder_name": create_folder_request.folder_name,
                                "parent_collection_id": create_folder_request.parent_collection_id,
                                "parent_folder_id": create_folder_request.parent_folder_id})

        collection = app_db.find_by_condition(VectorDB, {'id': create_folder_request.parent_collection_id, 'user_id': user_id})
        if not collection:
            backend_log.warn("Parent collection not found or access denied",
                           metadata={"parent_collection_id": create_folder_request.parent_collection_id})
            raise HTTPException(status_code=404, detail="Parent collection not found or access denied")

        collection = collection[0]
        parent_collection_name = collection.collection_name
        parent_collection_make_name = collection.collection_make_name

        root_directory_path = f"/{parent_collection_make_name}"

        if create_folder_request.parent_folder_id:
            parent_folder = app_db.find_by_condition(VectorDBFolders, {'id': create_folder_request.parent_folder_id, 'user_id': user_id})
            if not parent_folder:
                backend_log.warn("Parent folder not found or access denied",
                               metadata={"parent_folder_id": create_folder_request.parent_folder_id})
                raise HTTPException(status_code=404, detail="Parent folder not found or access denied")
            parent_folder = parent_folder[0]
            current_directory_full_path = f"{parent_folder.full_path}/{create_folder_request.folder_name}"

            existing_folder = app_db.find_by_condition(VectorDBFolders, {'full_path': current_directory_full_path, 'user_id': user_id})
            if existing_folder:
                backend_log.warn("Folder with same name already exists",
                               metadata={"full_path": current_directory_full_path})
                raise HTTPException(status_code=400, detail="Folder with the same name already exists")

            existing_folders = app_db.find_by_condition(VectorDBFolders, {'parent_folder_id': parent_folder.id, 'user_id': user_id})
            number_of_siblings = len(existing_folders)

            vectordb_folder = VectorDBFolders(
                user_id=user_id,
                collection_name=parent_collection_name,
                collection_make_name=parent_collection_make_name,
                collection_id=create_folder_request.parent_collection_id,
                folder_name=create_folder_request.folder_name,
                parent_folder_name=parent_folder.folder_name,
                parent_folder_id=parent_folder.id,
                is_root=False,
                full_path=current_directory_full_path,
                order_index=number_of_siblings + 1
            )
            app_db.insert(vectordb_folder)

            backend_log.success("Subfolder created successfully",
                              metadata={"folder_name": create_folder_request.folder_name,
                                      "full_path": current_directory_full_path,
                                      "parent_folder_name": parent_folder.folder_name,
                                      "is_root": False,
                                      "order_index": number_of_siblings + 1})

        else:
            current_directory_full_path = f"{root_directory_path}/{create_folder_request.folder_name}"
            existing_folder = app_db.find_by_condition(VectorDBFolders, {'full_path': current_directory_full_path, 'user_id': user_id})
            if existing_folder:
                backend_log.warn("Root folder with same name already exists",
                               metadata={"full_path": current_directory_full_path})
                raise HTTPException(status_code=400, detail="Folder with the same name already exists")
            existing_root_folders = app_db.find_by_condition(VectorDBFolders, {'collection_id': create_folder_request.parent_collection_id, 'is_root': True, 'user_id': user_id})
            number_of_root_folders = len(existing_root_folders)

            vectordb_folder = VectorDBFolders(
                user_id=user_id,
                collection_name=parent_collection_name,
                collection_make_name=parent_collection_make_name,
                collection_id=create_folder_request.parent_collection_id,
                folder_name=create_folder_request.folder_name,
                parent_folder_name=None,
                parent_folder_id=None,
                is_root=True,
                full_path=current_directory_full_path,
                order_index=number_of_root_folders + 1
            )
            app_db.insert(vectordb_folder)

            backend_log.success("Root folder created successfully",
                              metadata={"folder_name": create_folder_request.folder_name,
                                      "full_path": current_directory_full_path,
                                      "collection_name": parent_collection_name,
                                      "is_root": True,
                                      "order_index": number_of_root_folders + 1})

        return {"success": True, "message": "Folder created successfully"}

    except HTTPException:
        raise
    except Exception as e:
        backend_log.error("Folder creation failed", exception=e,
                         metadata={"folder_name": create_folder_request.folder_name,
                                 "parent_collection_id": create_folder_request.parent_collection_id,
                                 "parent_folder_id": create_folder_request.parent_folder_id})
        logger.error("Error creating folder: %s", e)
        raise HTTPException(status_code=500, detail="Failed to create folder") from e

@router.delete("/delete")
async def delete_folder(request: Request, delete_folder_request: DeleteFolderRequest):
    """Delete a folder"""
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        backend_log.info("Starting folder deletion",
                        metadata={"folder_path": delete_folder_request.folder_path,
                                "collection_id": delete_folder_request.collection_id})

        folders_to_delete = app_db.find_by_condition(VectorDBFolders,
            {
                'full_path__like__': delete_folder_request.folder_path,
                'collection_id': delete_folder_request.collection_id
            }
        )

        collection_find = app_db.find_by_condition(VectorDB, {'id': delete_folder_request.collection_id})

        if not folders_to_delete or not collection_find:
            backend_log.warn("No folders found to delete",
                           metadata={"folder_path": delete_folder_request.folder_path,
                                   "collection_id": delete_folder_request.collection_id})

            raise HTTPException(status_code=404, detail="No folders found to delete or access denied")

        folders_count = len(folders_to_delete)

        app_db.delete_by_condition(VectorDBFolders,
            {'full_path__like__': delete_folder_request.folder_path,
             'collection_id': delete_folder_request.collection_id})

        backend_log.success("Folder(s) deleted successfully",
                          metadata={"folder_path": delete_folder_request.folder_path,
                                  "collection_id": delete_folder_request.collection_id,
                                  "folders_deleted_count": folders_count})

        return {"success": True, "message": "Folder deleted successfully"}

    except Exception as e:
        backend_log.error("Folder deletion failed", exception=e,
                         metadata={"folder_path": delete_folder_request.folder_path,
                                 "collection_id": delete_folder_request.collection_id})
        logger.error("Error deleting folder: %s", e)
        raise HTTPException(status_code=500, detail="Failed to delete folder") from e

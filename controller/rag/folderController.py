from typing import Optional
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import logging
import gc
from controller.helper.singletonHelper import get_rag_service, get_config_composer, get_db_manager
from controller.helper.controllerHelper import extract_user_id_from_request
from service.database.models.vectordb import VectorDB, VectorDBFolders

logger = logging.getLogger("folder-controller")
router = APIRouter(prefix="/folder", tags=["folder"])

class CreateFolderRequest(BaseModel):
    folder_name: str
    parent_collection_id: int = None
    parent_folder_id: Optional[int] = None
    parent_folder_name: Optional[str] = None

@router.post("/create")
async def create_folder(request: Request, create_folder_request: CreateFolderRequest):
    """Create a new folder"""
    user_id = extract_user_id_from_request(request)
    
    app_db = get_db_manager(request)
    try:
        collection = app_db.find_by_condition(VectorDB, {'id': create_folder_request.parent_collection_id, 'user_id': user_id})
        if not collection:
            raise HTTPException(status_code=404, detail="Parent collection not found or access denied")

        collection = collection[0]
        parent_collection_name = collection.collection_name
        parent_collection_make_name = collection.collection_make_name

        root_directory_path = f"/{parent_collection_make_name}"

        if create_folder_request.parent_folder_id:
            parent_folder = app_db.find_by_condition(VectorDBFolders, {'id': create_folder_request.parent_folder_id, 'user_id': user_id})
            if not parent_folder:
                raise HTTPException(status_code=404, detail="Parent folder not found or access denied")
            parent_folder = parent_folder[0]
            current_directory_full_path = f"{parent_folder.full_path}/{create_folder_request.folder_name}"

            existing_folder = app_db.find_by_condition(VectorDBFolders, {'full_path': current_directory_full_path, 'user_id': user_id})
            if existing_folder:
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

        else:
            current_directory_full_path = f"{root_directory_path}/{create_folder_request.folder_name}"
            existing_folder = app_db.find_by_condition(VectorDBFolders, {'full_path': current_directory_full_path, 'user_id': user_id})
            if existing_folder:
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

        return {"success": True, "message": "Folder created successfully"}
    except Exception as e:
        logger.error("Error creating folder: %s", e)
        raise HTTPException(status_code=500, detail="Failed to create folder") from e

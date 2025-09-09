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
    parent_folder_id: int = None
    parent_folder_name: str = None

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

        





        return {"success": True, "message": "Folder created successfully"}
    except Exception as e:
        logger.error("Error creating folder: %s", e)
        raise HTTPException(status_code=500, detail="Failed to create folder") from e

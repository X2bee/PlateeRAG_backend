from fastapi import APIRouter, HTTPException, Request
from controller.helper.controllerHelper import extract_user_id_from_request
from huggingface_hub import HfApi
from controller.helper.singletonHelper import get_config_composer, get_db_manager
from service.database.logger_helper import create_logger

router = APIRouter(
    prefix="/api/huggingface",
    tags=["node"],
    responses={404: {"description": "Not found"}},
)

@router.get("/models")
async def get_models(request: Request):
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        config_composer = get_config_composer(request)

        hugging_face_user_id = config_composer.get_config_by_name("HUGGING_FACE_USER_ID").value
        hugging_face_hub_token = config_composer.get_config_by_name("HUGGING_FACE_HUB_TOKEN").value

        if not hugging_face_user_id:
            backend_log.warn("HUGGING_FACE_USER_ID not configured")
            raise HTTPException(status_code=422, detail="HUGGING_FACE_USER_ID_NOT_CONFIGURED")
        if not hugging_face_hub_token:
            backend_log.warn("HUGGING_FACE_HUB_TOKEN not configured")
            raise HTTPException(status_code=422, detail="HUGGING_FACE_HUB_TOKEN_NOT_CONFIGURED")

        api = HfApi(token=hugging_face_hub_token)
        formatted_models = []

        trending_models = api.list_models(limit=20, sort="trending_score", task='text-generation')
        for model in trending_models:
            main_fields = {
                "id": model.id,
                "author": model.author,
                "private": model.private,
                "downloads": model.downloads,
                "created_at": model.created_at.isoformat() if model.created_at else None
            }

            additional_info = {}
            for attr_name in dir(model):
                if not attr_name.startswith('_') and attr_name not in ['id', 'author', 'private', 'downloads', 'created_at']:
                    try:
                        attr_value = getattr(model, attr_name)
                        if not callable(attr_value):
                            if hasattr(attr_value, 'isoformat'):
                                additional_info[attr_name] = attr_value.isoformat()
                            else:
                                additional_info[attr_name] = attr_value
                    except:
                        pass

            formatted_model = {**main_fields, "additional_info": additional_info}
            formatted_models.append(formatted_model)

        models = api.list_models(author=hugging_face_user_id)
        for model in models:
            main_fields = {
                "id": model.id,
                "author": model.author,
                "private": model.private,
                "downloads": model.downloads,
                "created_at": model.created_at.isoformat() if model.created_at else None
            }

            additional_info = {}
            for attr_name in dir(model):
                if not attr_name.startswith('_') and attr_name not in ['id', 'author', 'private', 'downloads', 'created_at']:
                    try:
                        attr_value = getattr(model, attr_name)
                        if not callable(attr_value):
                            if hasattr(attr_value, 'isoformat'):
                                additional_info[attr_name] = attr_value.isoformat()
                            else:
                                additional_info[attr_name] = attr_value
                    except:
                        pass

            formatted_model = {**main_fields, "additional_info": additional_info}
            formatted_models.append(formatted_model)

        backend_log.success("Successfully retrieved Hugging Face models",
                          metadata={"total_models": len(formatted_models),
                                  "trending_count": len(list(trending_models)),
                                  "user_models_count": len(list(models)),
                                  "hugging_face_user_id": hugging_face_user_id})

        return {"models": formatted_models}

    except HTTPException:
        raise
    except Exception as e:
        backend_log.error("Error retrieving Hugging Face models", exception=e,
                         metadata={"hugging_face_user_id": hugging_face_user_id if 'hugging_face_user_id' in locals() else None})
        raise HTTPException(status_code=500, detail=f"Failed to retrieve models: {str(e)}")

@router.get("/datasets")
async def get_datasets(request: Request):
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        config_composer = get_config_composer(request)

        hugging_face_user_id = config_composer.get_config_by_name("HUGGING_FACE_USER_ID").value
        hugging_face_hub_token = config_composer.get_config_by_name("HUGGING_FACE_HUB_TOKEN").value

        if not hugging_face_user_id:
            backend_log.warn("HUGGING_FACE_USER_ID not configured")
            raise HTTPException(status_code=422, detail="HUGGING_FACE_USER_ID_NOT_CONFIGURED")
        if not hugging_face_hub_token:
            backend_log.warn("HUGGING_FACE_HUB_TOKEN not configured")
            raise HTTPException(status_code=422, detail="HUGGING_FACE_HUB_TOKEN_NOT_CONFIGURED")

        api = HfApi(token=hugging_face_hub_token)
        formatted_datasets = []

        trending_datasets = api.list_datasets(limit=20, sort="trending_score")
        for dataset in trending_datasets:
            main_fields = {
                "id": dataset.id,
                "author": dataset.author,
                "private": dataset.private,
                "downloads": dataset.downloads,
                "created_at": dataset.created_at.isoformat() if dataset.created_at else None
            }

            additional_info = {}
            for attr_name in dir(dataset):
                if not attr_name.startswith('_') and attr_name not in ['id', 'author', 'private', 'downloads', 'created_at']:
                    try:
                        attr_value = getattr(dataset, attr_name)
                        if not callable(attr_value):
                            if hasattr(attr_value, 'isoformat'):
                                additional_info[attr_name] = attr_value.isoformat()
                            else:
                                additional_info[attr_name] = attr_value
                    except:
                        pass

            formatted_dataset = {**main_fields, "additional_info": additional_info}
            formatted_datasets.append(formatted_dataset)

        datasets = api.list_datasets(author=hugging_face_user_id)
        for dataset in datasets:
            # Extract main fields
            main_fields = {
                "id": dataset.id,
                "author": dataset.author,
                "private": dataset.private,
                "downloads": dataset.downloads,
                "created_at": dataset.created_at.isoformat() if dataset.created_at else None
            }

            # Create additional_info with all other fields
            additional_info = {}
            for attr_name in dir(dataset):
                if not attr_name.startswith('_') and attr_name not in ['id', 'author', 'private', 'downloads', 'created_at']:
                    try:
                        attr_value = getattr(dataset, attr_name)
                        if not callable(attr_value):
                            # Convert datetime objects to ISO format
                            if hasattr(attr_value, 'isoformat'):
                                additional_info[attr_name] = attr_value.isoformat()
                            else:
                                additional_info[attr_name] = attr_value
                    except:
                        pass

            formatted_dataset = {**main_fields, "additional_info": additional_info}
            formatted_datasets.append(formatted_dataset)

        backend_log.success("Successfully retrieved Hugging Face datasets",
                          metadata={"total_datasets": len(formatted_datasets),
                                  "trending_count": len(list(trending_datasets)),
                                  "user_datasets_count": len(list(datasets)),
                                  "hugging_face_user_id": hugging_face_user_id})

        return {"datasets": formatted_datasets}

    except HTTPException:
        raise
    except Exception as e:
        backend_log.error("Error retrieving Hugging Face datasets", exception=e,
                         metadata={"hugging_face_user_id": hugging_face_user_id if 'hugging_face_user_id' in locals() else None})
        raise HTTPException(status_code=500, detail=f"Failed to retrieve datasets: {str(e)}")

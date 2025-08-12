from fastapi import APIRouter, HTTPException, Request
from controller.controller_helper import extract_user_id_from_request
from huggingface_hub import HfApi

router = APIRouter(
    prefix="/api/huggingface",
    tags=["node"],
    responses={404: {"description": "Not found"}},
)

def get_config_composer(request: Request):
    if hasattr(request.app.state, 'config_composer') and request.app.state.config_composer:
        return request.app.state.config_composer
    else:
        from config.config_composer import config_composer
        return config_composer

@router.get("/models")
async def get_models(request: Request):
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")
    config_composer = get_config_composer(request)
    if not config_composer:
        raise HTTPException(status_code=500, detail="Config composer not found in request state")

    hugging_face_user_id = config_composer.get_config_by_name("HUGGING_FACE_USER_ID").value
    hugging_face_hub_token = config_composer.get_config_by_name("HUGGING_FACE_HUB_TOKEN").value

    if not hugging_face_user_id:
        raise HTTPException(status_code=422, detail="HUGGING_FACE_USER_ID_NOT_CONFIGURED")
    if not hugging_face_hub_token:
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

    return {"models": formatted_models}

@router.get("/datasets")
async def get_datasets(request: Request):
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")
    config_composer = get_config_composer(request)
    if not config_composer:
        raise HTTPException(status_code=500, detail="Config composer not found in request state")

    hugging_face_user_id = config_composer.get_config_by_name("HUGGING_FACE_USER_ID").value
    hugging_face_hub_token = config_composer.get_config_by_name("HUGGING_FACE_HUB_TOKEN").value

    if not hugging_face_user_id:
        raise HTTPException(status_code=422, detail="HUGGING_FACE_USER_ID_NOT_CONFIGURED")
    if not hugging_face_hub_token:
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

    return {"datasets": formatted_datasets}

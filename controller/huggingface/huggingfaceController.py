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
    models = api.list_models(author=hugging_face_user_id)

    formatted_models = []
    for model in models:
        formatted_model = {
            "id": model.id,
            "author": model.author,
            "private": model.private,
            "downloads": model.downloads,
            "trending_score": model.trending_score,
            "created_at": model.created_at.isoformat() if model.created_at else None
        }
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
    datasets = api.list_datasets(author=hugging_face_user_id)

    formatted_datasets = []
    for dataset in datasets:
        formatted_dataset = {
            "id": dataset.id,
            "author": dataset.author,
            "private": dataset.private,
            "downloads": dataset.downloads,
            "trending_score": dataset.trending_score,
            "created_at": dataset.created_at.isoformat() if dataset.created_at else None
        }
        formatted_datasets.append(formatted_dataset)

    return {"datasets": formatted_datasets}

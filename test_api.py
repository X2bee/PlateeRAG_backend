import requests

API_URL = "http://192.168.219.101:8000/api/workflow/deploy/execute/based_id"

def query(payload):
    response = requests.post(API_URL, json=payload)
    return response.json()

output = query({
    "workflow_name": "Workflow_vllm_hf",
    "workflow_id": "workflow_195af1fd3b72f102c426f13c1e9375cf819dcb10",
    "input_data": "hf에서 데이터 셋의 정보를 가져와 줘.",
    "interaction_id": "default",
    "selected_collection": "string",
    "additional_params": {
        "api_loader/APICallingTool-1754720502413": {
            "dataset_path": "str",
            "hugging_face_user_id": "str",
            "hugging_face_token": "str"
        }
    },
    "user_id": "1"
})

print(output)

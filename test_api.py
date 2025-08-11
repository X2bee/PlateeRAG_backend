import requests

API_URL = "http://localhost:8000/api/workflow/deploy/execute/based_id"

def query(payload):
    response = requests.post(API_URL, json=payload)
    return response.json()

output = query({
    "workflow_name": "Workflow-test",
    "workflow_id": "workflow_f8529d8932d17d20dcbc96627819bea321aa89a3",
    "input_data": "안녕하세요",
    "interaction_id": "default",
    "selected_collection": "string",
    # "additional_params": {
    #     "api_loader/APICallingTool-1754637013411": {
    #         "return_dict": "bool"
    #     }
    # },
    "user_id": "17"
})

print(output)

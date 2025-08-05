import os
import json
import copy
import hashlib

from typing import Dict, Any

from controller.workflow.utils import extract_collection_name
from fastapi import HTTPException

async def _workflow_parameter_helper(request_body, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Updates the workflow data by setting the interaction ID in the parameters of nodes
    with a function ID of 'memory', if applicable.

    Args:
        request_body: An object containing the interaction ID to be applied.
        workflow_data: A dictionary representing the workflow's nodes and their parameters.

    Returns:
        The updated workflow data with the interaction ID applied where necessary.
    """
    if (request_body.interaction_id) and (request_body.interaction_id != "default"):
        for node in workflow_data.get('nodes', []):
            if node.get('data', {}).get('functionId') == 'memory':
                parameters = node.get('data', {}).get('parameters', [])
                for parameter in parameters:
                    if parameter.get('id') == 'interaction_id':
                        parameter['value'] = request_body.interaction_id

    return workflow_data

async def _default_workflow_parameter_helper(request, request_body, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Updates the workflow data with default parameters based on the request body.

    This function modifies the `workflow_data` dictionary by setting specific parameter values
    for nodes in the workflow. It updates the `collection_name` parameter for nodes with the
    `document_loaders` function ID and the `interaction_id` parameter for nodes with the
    `memory` function ID, based on the values provided in the `request_body`.

    Parameters:
        request_body: An object containing the request data. It should have attributes
            `selected_collection` (optional) and `interaction_id` (optional).
        workflow_data: A dictionary representing the workflow structure. It contains a list
            of nodes, each of which may have a `data` dictionary with `functionId` and `parameters`.

    Returns:
        A dictionary representing the updated workflow data with modified parameters.
    """
    config_composer = request.app.state.config_composer
    if not config_composer:
        raise HTTPException(status_code=500, detail="Config composer not available")

    llm_provider = config_composer.get_config_by_name("DEFAULT_LLM_PROVIDER").value

    if llm_provider == "openai":
        model = config_composer.get_config_by_name("OPENAI_MODEL_DEFAULT").value
        url = config_composer.get_config_by_name("OPENAI_API_BASE_URL").value
    elif llm_provider == "vllm":
        model = config_composer.get_config_by_name("VLLM_MODEL_NAME").value
        url = config_composer.get_config_by_name("VLLM_API_BASE_URL").value
    else:
        raise HTTPException(status_code=500, detail="Unsupported LLM provider")

    for node in workflow_data.get('nodes', []):
        if node.get('data', {}).get('functionId') == 'agents':
            parameters = node.get('data', {}).get('parameters', [])
            for parameter in parameters:
                if parameter.get('id') == 'model':
                    parameter['value'] = model
                if parameter.get('id') == 'base_url':
                    parameter['value'] = url

    if request_body.selected_collections:
        constant_folder = os.path.join(os.getcwd(), "constants")
        collection_file_path = os.path.join(constant_folder, "collection_node_template.json")
        edge_template_path = os.path.join(constant_folder, "base_edge_template.json")
        with open(collection_file_path, 'r', encoding='utf-8') as f:
            collection_node_template = json.load(f)
        with open(edge_template_path, 'r', encoding='utf-8') as f:
            edge_template = json.load(f)

        for collection in request_body.selected_collections:
            # UUID 부분을 제거하고 깨끗한 컬렉션 이름 추출
            collection_name = extract_collection_name(collection)
            coleection_code = hashlib.sha1(collection_name.encode('utf-8')).hexdigest()[:8]

            print(f"Adding collection node for: {collection} (clean name: {collection_name})")
            collection_node = copy.deepcopy(collection_node_template)
            edge = copy.deepcopy(edge_template)
            collection_node['id'] = f"document_loaders_{collection}"
            collection_node['data']['parameters'][0]['value'] = collection # collection에서 collection_name으로 수정함. uuid 제거 위해서 수정
            collection_node['data']['parameters'][1]['value'] = f"retrieval_search_tool_for_{coleection_code}"
            collection_node['data']['parameters'][2]['value'] = f"Use when a search is needed for the given question related to {collection_name}."
            workflow_data['nodes'].append(collection_node)

            edge_id = f"{collection_node['id']}:tools-default_agents:tools-{coleection_code}"
            edge['id'] = edge_id
            edge['source']['nodeId'] = collection_node['id']
            workflow_data['edges'].append(edge)

    if (request_body.interaction_id) and (request_body.interaction_id != "default"):
        for node in workflow_data.get('nodes', []):
            if node.get('data', {}).get('functionId') == 'memory':
                parameters = node.get('data', {}).get('parameters', [])
                for parameter in parameters:
                    if parameter.get('id') == 'interaction_id':
                        parameter['value'] = request_body.interaction_id

    return workflow_data
import logging
import requests
from typing import Any, Dict
from pydantic import BaseModel, create_model, Field
from editor.node_composer import Node
from editor.utils.helper.parse_helper import parse_param_value
from langchain.agents import tool
import json
logger = logging.getLogger(__name__)

class APICallingTool(Node):
    categoryId = "xgen"
    functionId = "api_loader"
    nodeId = "api_loader/APICallingTool"
    nodeName = "API Calling Tool"
    description = "API 호출을 위한 Tool을 전달"
    tags = ["api", "rag", "setup"]

    inputs = [
        {"id": "args_schema", "name": "ArgsSchema", "type": "BaseModel"},
    ]
    outputs = [
        {"id": "tools", "name": "Tools", "type": "TOOL"},
    ]

    parameters = [
        {"id": "tool_name", "name": "Tool Name", "type": "STR", "value": "api_calling_tool", "required": True},
        {"id": "description", "name": "Description", "type": "STR", "value": "Use this tool when you need to call an external API to retrieve specific data or perform an operation. Call this tool when the user requests information that requires an API call to external services.", "required": True, "description": "이 도구를 언제 사용하여야 하는지 설명합니다. AI는 해당 설명을 통해, 해당 도구를 언제 호출해야할지 결정할 수 있습니다."},
        {"id": "api_endpoint", "name": "API Endpoint", "type": "STR", "value": "", "required": True, "description": "해당 도구의 실행으로 호출할 API의 엔드포인트 URL입니다."},
        {"id": "method", "name": "HTTP Method", "type": "STR", "value": "GET", "required": True, "options": [
            {"value": "GET", "label": "GET"},
            {"value": "POST", "label": "POST"},
            {"value": "PUT", "label": "PUT"},
            {"value": "DELETE", "label": "DELETE"},
            {"value": "PATCH", "label": "PATCH"}
        ]},
        {"id": "timeout", "name": "Timeout (seconds)", "type": "INT", "value": 30, "min": 1, "max": 300},
    ]

    def execute(self, tool_name, description, api_endpoint, method="GET", timeout=30, args_schema: BaseModel=None, *args, **kwargs):
        description = description + "\n명시적인 요청이 없다면, return_dict를 False로 하여 STR 형태의 응답을 받으려고 시도하십시오."
        additional_params = kwargs.get("additional_params", {})
        def create_api_tool():
            if args_schema is None:
                # 빈 스키마를 명시적으로 생성
                from pydantic import BaseModel
                class EmptySchema(BaseModel):
                    pass
                actual_args_schema = EmptySchema
            else:
                actual_args_schema = args_schema

            @tool(tool_name, description=description, args_schema=actual_args_schema)
            def api_tool(return_dict: bool = False, **kwargs) -> str:
                logger.info(f"Creating API tool with name: {tool_name}, endpoint: {api_endpoint}, method: {method}, timeout: {timeout}")

                if not kwargs:
                    kwargs = {}
                request_data = kwargs if kwargs else {}

                if additional_params and additional_params != {}:
                    parsed_additional_params = {}
                    for key, value in additional_params.items():
                        parsed_additional_params[key] = parse_param_value(value)
                    request_data.update(parsed_additional_params)
                    logger.info(f"Additional parameters provided: {additional_params}")
                    logger.info(f"Request data after merging additional parameters: {request_data}")

                try:
                    # HTTP 메서드에 따라 요청 방식 결정
                    method_upper = method.upper()

                    # 공통 헤더 설정
                    headers = {
                        'Content-Type': 'application/json',
                        'User-Agent': 'PlateeRAG-APICallingTool/1.0'
                    }

                    # API 호출
                    if method_upper == "GET":
                        params = request_data if request_data else None
                        logger.info(f"Making GET request to {api_endpoint} with params: {params}")
                        response = requests.get(
                            api_endpoint,
                            params=params,
                            headers=headers,
                            timeout=timeout
                        )
                        logger.info(f"GET request to {api_endpoint} completed with status code: {response}")
                    elif method_upper in ["POST", "PUT", "PATCH"]:
                        json_data = request_data if request_data else None
                        logger.info(f"Making {method_upper} request to {api_endpoint} with data: {json_data}")
                        response = requests.request(
                            method_upper,
                            api_endpoint,
                            json=json_data,
                            headers=headers,
                            timeout=timeout
                        )
                    elif method_upper == "DELETE":
                        params = request_data if request_data else None
                        logger.info(f"Making DELETE request to {api_endpoint} with params: {params}")
                        response = requests.delete(
                            api_endpoint,
                            params=params,
                            headers=headers,
                            timeout=timeout
                        )
                    else:
                        return f"Unsupported HTTP method: {method}"

                    # 응답 상태 코드 확인
                    if response.status_code == 200:
                        try:
                            result = response.json()
                            logger.info(f"API call successful: {result}")
                            if return_dict:
                                return result
                            else:
                                return str(json.dumps(result, ensure_ascii=False, indent=2))
                        except ValueError:
                            logger.info(f"API call successful, but response is not JSON: {response.text}")
                            return response.text
                    else:
                        logger.error(f"API call failed with status code {response.status_code}. Response: {response.text}")
                        return response.text

                except requests.exceptions.Timeout:
                    return f"API call timed out after {timeout} seconds"
                except requests.exceptions.ConnectionError:
                    return f"Failed to connect to API endpoint: {api_endpoint}"
                except requests.exceptions.RequestException as e:
                    return f"Request failed: {str(e)}"
                except (ValueError, TypeError) as e:
                    return f"Invalid data format: {str(e)}"

            return api_tool

        return create_api_tool()

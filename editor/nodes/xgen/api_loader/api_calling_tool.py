import logging
import requests
from typing import Any, Dict
from pydantic import BaseModel, create_model, Field
from editor.node_composer import Node
from editor.utils.helper.parse_helper import parse_param_value
from langchain.agents import tool
import json
import re
logger = logging.getLogger(__name__)

class APICallingTool(Node):
    categoryId = "xgen"
    functionId = "api_loader"
    nodeId = "api_loader/APICallingTool"
    nodeName = "API Calling Tool"
    description = "API 호출을 위한 Tool을 전달"
    tags = ["api", "rag", "setup"]

    inputs = [
        {"id": "args_schema", "name": "ArgsSchema", "type": "InputSchema"},
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
        {"id": "enable_response_filtering", "name": "Enable Response Filtering", "type": "BOOL", "value": False, "description": "JSON 응답에서 특정 데이터만 추출하여 반환할지 여부를 설정합니다."},
        {"id": "response_filter_path", "name": "Response Filter Path", "type": "STR", "value": "", "description": "JSON 응답에서 추출할 데이터의 경로를 설정합니다. (예: 'payload.searchDataList')"},
        {"id": "response_filter_fields", "name": "Response Filter Fields", "type": "STR", "value": "", "description": "각 객체에서 추출할 필드들을 콤마로 구분하여 입력합니다. (예: 'goodsNm,salePrc')"},
    ]

    @staticmethod
    def get_nested_value(data, path):
        """JSON 경로를 따라가서 중첩된 값을 추출합니다."""
        if not path:
            return data

        keys = path.split('.')
        current = data

        try:
            for key in keys:
                if isinstance(current, dict):
                    current = current[key]
                else:
                    return None
            return current
        except (KeyError, TypeError):
            return None

    @staticmethod
    def filter_response_data(response_data, filter_path, filter_fields):
        """응답 데이터에서 지정된 경로와 필드만 추출합니다."""
        # 필터 필드 파싱 (콤마로 구분)
        fields = []
        if filter_fields and isinstance(filter_fields, str):
            fields = [field.strip() for field in filter_fields.split(',') if field.strip()]

        # 필터 경로로 데이터 추출
        extracted_data = response_data
        if filter_path and filter_path.strip():
            extracted_data = APICallingTool.get_nested_value(response_data, filter_path)
            if extracted_data is None:
                logger.warning(f"Filter path '{filter_path}' not found in response")
                extracted_data = response_data

        # 필터 필드가 없으면 추출된 데이터 그대로 반환
        if not fields:
            return extracted_data

        # 배열인 경우 각 객체에서 지정된 필드만 추출
        if isinstance(extracted_data, list):
            filtered_list = []
            for item in extracted_data:
                if isinstance(item, dict):
                    filtered_item = {field: item.get(field) for field in fields if field in item}
                    filtered_list.append(filtered_item)
                else:
                    # 객체가 아닌 경우 그대로 추가
                    filtered_list.append(item)
            return filtered_list

        # 단일 객체인 경우
        elif isinstance(extracted_data, dict):
            return {field: extracted_data.get(field) for field in fields if field in extracted_data}

        # 배열도 객체도 아닌 경우 그대로 반환
        return extracted_data

    def execute(self, tool_name, description, api_endpoint, method="GET", timeout=30,
                enable_response_filtering=False, response_filter_path="", response_filter_fields="",
                args_schema: BaseModel=None, *args, **kwargs):
        description = description + "\n명시적인 요청이 없다면, return_dict를 False로 하여 STR 형태의 응답을 받으려고 시도하십시오."
        additional_params = kwargs.get("additional_params", {})
        def create_api_tool():
            if args_schema is None:
                # OpenAI API 호환을 위한 기본 스키마 생성
                from pydantic import BaseModel
                class DefaultSchema(BaseModel):
                    return_dict: bool = Field(default=False, description="Whether to return the response as a dictionary or string")
                actual_args_schema = DefaultSchema
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

                endpoint = api_endpoint
                if request_data:
                    placeholder_pattern = r'\{([^}]+)\}'
                    placeholders = re.findall(placeholder_pattern, endpoint)

                    for placeholder in placeholders:
                        if placeholder in request_data:
                            value = request_data.pop(placeholder)
                            endpoint = endpoint.replace(f'{{{placeholder}}}', str(value))

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
                        logger.info(f"Making GET request to {endpoint} with params: {params}")
                        response = requests.get(
                            endpoint,
                            params=params,
                            headers=headers,
                            timeout=timeout
                        )
                        logger.info(f"GET request to {endpoint} completed with status code: {response}")
                    elif method_upper in ["POST", "PUT", "PATCH"]:
                        # HTTP body handling:
                        # - If kwargs contains 'body' (dict or JSON string) or keys ending with '_body',
                        #   they will be used as the request body.
                        # - If kwargs contains '__form__' or 'as_form' truthy value, send as x-www-form-urlencoded.
                        send_as_form = False
                        if isinstance(request_data, dict):
                            # support explicit override via __form__ kwarg
                            if request_data.pop('__form__', None):
                                send_as_form = True
                            # or if Schema Provider passed body_type=FORM
                            elif 'body_type' in request_data and str(request_data.get('body_type')).upper() == 'FORM':
                                send_as_form = True
                            # remove control key so it is not sent as payload
                            request_data.pop('body_type', None)

                        # Extract explicit body fields
                        body = None
                        if isinstance(request_data, dict):
                            if 'body' in request_data:
                                body = request_data.pop('body')
                                # try to parse JSON string
                                if isinstance(body, str):
                                    try:
                                        body = json.loads(body)
                                    except Exception:
                                        pass
                            else:
                                # collect keys that end with '_body' and merge into a single body dict
                                body_keys = [k for k in list(request_data.keys()) if k.endswith('_body')]
                                if body_keys:
                                    body = {}
                                    for k in body_keys:
                                        v = request_data.pop(k)
                                        key_name = k[:-5] if len(k) > 5 else k
                                        if isinstance(v, str):
                                            try:
                                                parsed_v = json.loads(v)
                                                v = parsed_v
                                            except Exception:
                                                pass
                                        body[key_name] = v

                        if send_as_form:
                            # form-encoding: application/x-www-form-urlencoded
                            headers['Content-Type'] = 'application/x-www-form-urlencoded'
                            if body is not None:
                                # flatten body values for form encoding (serialize nested structures)
                                form_data = {}
                                for bk, bv in (body.items() if isinstance(body, dict) else []):
                                    if isinstance(bv, (dict, list)):
                                        form_data[bk] = json.dumps(bv, ensure_ascii=False)
                                    else:
                                        form_data[bk] = '' if bv is None else str(bv)
                                # include any remaining params as form fields
                                for rk, rv in request_data.items():
                                    if isinstance(rv, (dict, list)):
                                        form_data[rk] = json.dumps(rv, ensure_ascii=False)
                                    else:
                                        form_data[rk] = '' if rv is None else str(rv)
                                logger.info(f"Making {method_upper} request to {endpoint} with form-encoded data: {form_data}")
                                response = requests.request(
                                    method_upper,
                                    endpoint,
                                    data=form_data,
                                    headers=headers,
                                    timeout=timeout
                                )
                            else:
                                logger.info(f"Making {method_upper} request to {endpoint} with form-encoded data: {request_data}")
                                response = requests.request(
                                    method_upper,
                                    endpoint,
                                    data=request_data if request_data else None,
                                    headers=headers,
                                    timeout=timeout
                                )
                        else:
                            # JSON body sending
                            json_payload = body if body is not None else (request_data if request_data else None)
                            logger.info(f"Making {method_upper} request to {endpoint} with JSON body: {json_payload}")
                            response = requests.request(
                                method_upper,
                                endpoint,
                                json=json_payload,
                                headers=headers,
                                timeout=timeout
                            )
                    elif method_upper == "DELETE":
                        params = request_data if request_data else None
                        logger.info(f"Making DELETE request to {endpoint} with params: {params}")
                        response = requests.delete(
                            endpoint,
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

                            # 응답 필터링 적용 (filter가 활성화되어 있고, path 또는 fields 중 하나라도 있으면 적용)
                            if enable_response_filtering and (response_filter_path or response_filter_fields):
                                logger.info(f"Applying response filtering with path: '{response_filter_path}', fields: '{response_filter_fields}'")
                                filtered_result = APICallingTool.filter_response_data(
                                    result, response_filter_path, response_filter_fields
                                )
                                result = filtered_result
                                logger.info(f"API call successful (filtered): {result}")
                            else:
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
                    return f"Failed to connect to API endpoint: {endpoint}"
                except requests.exceptions.RequestException as e:
                    return f"Request failed: {str(e)}"
                except (ValueError, TypeError) as e:
                    return f"Invalid data format: {str(e)}"

            return api_tool

        return create_api_tool()

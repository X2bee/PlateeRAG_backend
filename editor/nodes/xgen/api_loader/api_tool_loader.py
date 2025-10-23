import logging
import requests
from typing import Any, Dict
from pydantic import BaseModel, create_model, Field
from editor.node_composer import Node
from editor.utils.helper.parse_helper import parse_param_value
from langchain_core.tools import tool
import json
import re
from editor.utils.helper.async_helper import sync_run_async
from editor.utils.helper.service_helper import AppServiceManager
from fastapi import Request
from controller.tools.toolStorageController import simple_list_tools
from service.database.models.tools import Tools

logger = logging.getLogger(__name__)

class APICallingTool(Node):
    categoryId = "xgen"
    functionId = "api_loader"
    nodeId = "api_loader/APIToolLoader"
    nodeName = "API Tool Loader"
    description = "API Tool Storage에서 API 호출 도구를 로드합니다. 외부 API를 호출하여 특정 데이터를 검색하거나 작업을 수행해야 할 때 이 도구를 사용하십시오."
    tags = ["api", "rag", "setup"]

    inputs = [
    ]
    outputs = [
        {"id": "tools", "name": "Tools", "type": "TOOL"},
    ]

    parameters = [
        {"id": "tool_id", "name": "API Tool ID", "type": "STR", "value": "Select Tool", "required": True, "is_api": True, "api_name": "api_collection", "options": []},
    ]

    def api_collection(self, request: Request) -> Dict[str, Any]:
        tools = sync_run_async(simple_list_tools(request))
        return [{"value": tool.get("id"), "label": f"{tool.get("function_name")}({tool.get("username", "unknown")})"} for tool in tools]

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

    def execute(self, tool_id, *args, **kwargs):
        app_db = AppServiceManager.get_db_manager()
        tool_data = app_db.find_by_id(Tools, tool_id)
        if not tool_data:
            logger.error(f"Tool with ID {tool_id} not found")
            return {"error": "Tool not found"}, 404

        description = tool_data.description + "\n명시적인 요청이 없다면, return_dict를 False로 하여 STR 형태의 응답을 받으려고 시도하십시오."

        # API 정보 추출
        tool_name = tool_data.function_id
        api_endpoint = tool_data.api_url
        method = tool_data.api_method or 'GET'
        timeout = tool_data.api_timeout or 30
        body_type = tool_data.body_type or 'application/json'
        is_query_string = tool_data.is_query_string or False

        # api_header 파싱
        api_headers = tool_data.api_header
        if isinstance(api_headers, str):
            api_headers = json.loads(api_headers) if api_headers else {}
        if not api_headers:
            api_headers = {}

        # static_body 파싱
        static_body = tool_data.static_body
        if isinstance(static_body, str):
            static_body = json.loads(static_body) if static_body else {}
        if not static_body:
            static_body = {}

        # body_type에 따라 Content-Type 헤더 정합성 검증 및 설정
        method_upper = method.upper()
        if method_upper != 'GET':
            expected_content_type = None

            # body_type에 따른 예상 Content-Type 결정
            if body_type not in ['multipart/form-data', 'url-params']:
                expected_content_type = body_type

            # 헤더 정합성 검증 및 수정
            current_content_type = api_headers.get('Content-Type') or api_headers.get('content-type')

            if expected_content_type:
                if current_content_type != expected_content_type:
                    logger.warning(
                        f"Content-Type mismatch for tool '{tool_name}': "
                        f"header has '{current_content_type}', but body_type is '{body_type}'. "
                        f"Setting Content-Type to '{expected_content_type}'"
                    )
                    # 기존 Content-Type 제거 (대소문자 구분 없이)
                    api_headers = {k: v for k, v in api_headers.items() if k.lower() != 'content-type'}
                    # 새 Content-Type 설정
                    api_headers['Content-Type'] = expected_content_type
            elif body_type == 'url-params':
                # url-params인 경우 Content-Type 제거
                api_headers = {k: v for k, v in api_headers.items() if k.lower() != 'content-type'}

        # response filter 정보
        enable_response_filtering = tool_data.response_filter or False
        response_filter_path = tool_data.response_filter_path or ""
        response_filter_fields = tool_data.response_filter_field or ""

        # api_body를 기반으로 ArgsSchema 동적 생성
        api_body_schema = tool_data.api_body
        if isinstance(api_body_schema, str):
            api_body_schema = json.loads(api_body_schema) if api_body_schema else {}

        args_schema = None
        if api_body_schema and isinstance(api_body_schema, dict) and 'properties' in api_body_schema:
            properties = api_body_schema.get('properties', {})

            # properties가 비어있으면 기본 스키마 사용
            if not properties:
                class DefaultSchema(BaseModel):
                    return_dict: bool = Field(default=False, description="Whether to return the response as a dictionary or string")
                args_schema = DefaultSchema
            else:
                # JSON Schema에서 Pydantic 모델 생성
                fields = {}
                required_fields = api_body_schema.get('required', [])

                for field_name, field_info in properties.items():
                    field_type = field_info.get('type', 'string')
                    field_description = field_info.get('description', '')
                    field_enum = field_info.get('enum', None)

                    # 타입 매핑
                    type_mapping = {
                        'string': str,
                        'integer': int,
                        'number': float,
                        'boolean': bool,
                        'array': list,
                        'object': dict
                    }
                    python_type = type_mapping.get(field_type, str)

                    # Field 생성
                    field_kwargs = {'description': field_description}

                    # enum이 있으면 추가
                    if field_enum:
                        from typing import Literal
                        # Literal 타입으로 enum 제약 추가
                        if python_type == str:
                            python_type = Literal[tuple(field_enum)]

                    # required 체크
                    if field_name in required_fields:
                        fields[field_name] = (python_type, Field(**field_kwargs))
                    else:
                        fields[field_name] = (python_type, Field(default=None, **field_kwargs))

                # Pydantic 모델 생성
                args_schema = create_model('DynamicAPISchema', **fields)
        else:
            class DefaultSchema(BaseModel):
                return_dict: bool = Field(default=False, description="Whether to return the response as a dictionary or string")
            args_schema = DefaultSchema

        additional_params = kwargs.get("additional_params", {})

        def create_api_tool():
            actual_args_schema = args_schema

            @tool(tool_name, description=description, args_schema=actual_args_schema)
            def api_tool(return_dict: bool = False, **tool_kwargs) -> str:
                logger.info(f"Calling API tool: {tool_name}, endpoint: {api_endpoint}, method: {method}, body_type: {body_type}")

                # AI가 제공한 파라미터 추출
                ai_provided_params = tool_kwargs if tool_kwargs else {}

                # static_body를 기본값으로 사용하고 AI 파라미터를 추가/병합
                merged_data = dict(static_body)  # static_body 복사
                merged_data.update(ai_provided_params)  # AI 파라미터로 업데이트

                logger.info(f"Static body: {static_body}")
                logger.info(f"AI provided params: {ai_provided_params}")
                logger.info(f"Merged data: {merged_data}")

                # additional_params 병합
                if additional_params and additional_params != {}:
                    parsed_additional_params = {}
                    for key, value in additional_params.items():
                        parsed_additional_params[key] = parse_param_value(value)
                    merged_data.update(parsed_additional_params)
                    logger.info(f"After additional params: {merged_data}")

                # URL placeholder 처리
                endpoint = api_endpoint

                # is_query_string이 True인 경우, URL의 placeholder를 merged_data로 교체
                if is_query_string and merged_data:
                    logger.info(f"Processing query string placeholders in URL: {endpoint}")

                    # URL에서 {key} 패턴을 찾아서 merged_data의 값으로 치환
                    for key, value in list(merged_data.items()):
                        placeholder = f"{{{key}}}"
                        if placeholder in endpoint:
                            endpoint = endpoint.replace(placeholder, str(value))
                            logger.info(f"Replaced {placeholder} with {value}")
                            # URL에 포함되었으므로 merged_data에서 제거
                            merged_data.pop(key)

                    logger.info(f"Final URL with replaced query string: {endpoint}")

                # 일반적인 URL placeholder 처리 (is_query_string이 False인 경우)
                elif merged_data:
                    placeholder_pattern = r'\{([^}]+)\}'
                    placeholders = re.findall(placeholder_pattern, endpoint)

                    for placeholder in placeholders:
                        if placeholder in merged_data:
                            value = merged_data.pop(placeholder)
                            endpoint = endpoint.replace(f'{{{placeholder}}}', str(value))
                            logger.info(f"Replaced placeholder '{placeholder}' with '{value}' in URL")

                try:
                    # 헤더 준비 (api_headers 복사)
                    headers = dict(api_headers)
                    if not headers.get('User-Agent'):
                        headers['User-Agent'] = 'PlateeRAG-APICallingTool/1.0'

                    logger.info(f"Request headers: {headers}")
                    logger.info(f"Request method: {method_upper}, body_type: {body_type}")

                    # body_type에 따라 요청 방식 결정
                    if method_upper == "GET":
                        # GET 요청: URL 파라미터로 전송
                        params = merged_data if merged_data else None
                        logger.info(f"Making GET request to {endpoint} with params: {params}")
                        response = requests.get(
                            endpoint,
                            params=params,
                            headers=headers,
                            timeout=timeout
                        )
                    else:
                        # GET이 아닌 경우: body_type에 따라 처리
                        if body_type == 'application/json':
                            # JSON 형식
                            logger.info(f"Making {method_upper} request (JSON) to {endpoint} with data: {merged_data}")
                            response = requests.request(
                                method_upper,
                                endpoint,
                                json=merged_data if merged_data else None,
                                headers=headers,
                                timeout=timeout
                            )

                        elif body_type == 'application/x-www-form-urlencoded':
                            # Form URL encoded
                            logger.info(f"Making {method_upper} request (form-urlencoded) to {endpoint} with data: {merged_data}")
                            response = requests.request(
                                method_upper,
                                endpoint,
                                data=merged_data if merged_data else None,
                                headers=headers,
                                timeout=timeout
                            )

                        elif body_type == 'multipart/form-data':
                            # Multipart form data
                            # Content-Type은 requests가 자동으로 설정하므로 제거
                            headers_without_ct = {k: v for k, v in headers.items() if k.lower() != 'content-type'}
                            files = {k: (None, str(v)) for k, v in merged_data.items()} if merged_data else None
                            logger.info(f"Making {method_upper} request (multipart) to {endpoint} with files: {files}")
                            response = requests.request(
                                method_upper,
                                endpoint,
                                files=files,
                                headers=headers_without_ct,
                                timeout=timeout
                            )

                        elif body_type == 'url-params':
                            # URL 파라미터로 전송 (body 없음)
                            params = merged_data if merged_data else None
                            logger.info(f"Making {method_upper} request (url-params) to {endpoint} with params: {params}")
                            response = requests.request(
                                method_upper,
                                endpoint,
                                params=params,
                                headers=headers,
                                timeout=timeout
                            )

                        elif body_type in ['application/xml', 'text/plain', 'text/html', 'text/csv']:
                            # 텍스트 기반 형식 (data로 전송)
                            data_str = str(merged_data) if merged_data else None
                            logger.info(f"Making {method_upper} request ({body_type}) to {endpoint} with data: {data_str}")
                            response = requests.request(
                                method_upper,
                                endpoint,
                                data=data_str,
                                headers=headers,
                                timeout=timeout
                            )

                        else:
                            # 기본값: JSON
                            logger.info(f"Making {method_upper} request (default JSON) to {endpoint} with data: {merged_data}")
                            response = requests.request(
                                method_upper,
                                endpoint,
                                json=merged_data if merged_data else None,
                                headers=headers,
                                timeout=timeout
                            )

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

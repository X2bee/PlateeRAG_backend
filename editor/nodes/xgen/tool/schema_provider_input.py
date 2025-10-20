from editor.node_composer import Node
import json
from typing import Dict, Any, Type
from pydantic import BaseModel, create_model, Field

class InputSchemaProviderNode(Node):
    categoryId = "xgen"
    functionId = "tools"
    nodeId = "input_schema_provider"
    nodeName = "Schema Provider(Input)"
    description = "사용자 입력에 따라 동적으로 Pydantic 모델을 생성하는 노드입니다."
    tags = ["input", "json", "dict", "parameter", "source", "user_input", "key_value"]

    inputs = []
    outputs = [
        {"id": "args_schema", "name": "ArgsSchema", "type": "InputSchema"},
    ]
    # 기본적으로 빈 파라미터를 사용하던 기존 동작에 더해,
    # Request Body 관련 설정용 파라미터를 노드 설정에 자동으로 추가합니다.
    parameters = [
        # Body type: JSON or Form
        {"id": "body_type", "name": "Body Type", "type": "STR", "value": "JSON", "options": [{"value": "JSON", "label": "JSON"}, {"value": "FORM", "label": "Form"}], "description": "Choose JSON to edit the request body as JSON, or Form to send application/x-www-form-urlencoded."},
        # (삭제) `as_form`은 더 이상 필요하지 않습니다. Body Type으로 전송 방식(JSON/Form)을 제어합니다.
        # 안내용 필드: Request Body 전용 필드 사용 안내
        {"id": "request_body_note", "name": "Request Body (use *_body keys)", "type": "STR", "value": "", "description": "Add fields with suffix '_body' to include them in the POST body (e.g. payload_body, metadata_body)."}
    ]

    # 팝업 에디터용 필드: Agent의 system prompt처럼 Body 편집창을 열 수 있도록 합니다.
    # UI가 `expandable`을 지원하면 팝업 형태로 편집 가능해집니다.
    parameters.append({"id": "request_body_editor", "name": "Request Body Editor", "type": "STR", "value": "", "expandable": True, "description": "Open the editor to edit JSON body for POST requests."})

    def execute(self, *args, **kwargs) -> BaseModel:
        if not kwargs:
            # 빈 스키마 반환
            class EmptySchema(BaseModel):
                pass
            return EmptySchema

        fields = {}

        # key와 key_description을 매칭하기 위한 딕셔너리
        key_values = {}
        key_descriptions = {}

        # kwargs를 순회하면서 key와 key_description을 분류
        # method에 따라 Request Body 관련 필드 노출/생성 여부 결정
        method_val = None
        for k in ('method', 'http_method', 'api_method'):
            if k in kwargs:
                method_val = kwargs.get(k)
                break
        method_upper = None
        if isinstance(method_val, str):
            method_upper = method_val.upper()
        # body_type에 따른 노출 제어 (JSON/FORM)
        body_type_val = None
        if 'body_type' in kwargs:
            try:
                body_type_val = str(kwargs.get('body_type')).upper()
            except Exception:
                body_type_val = None

        for param_id, param_value in kwargs.items():
            # 내부 안내용 필드는 스키마에 포함시키지 않음
            if param_id == 'request_body_note':
                continue
            # GET 방식이면 Request Body 관련 설정은 스키마에 포함시키지 않음
            if method_upper == 'GET' and (param_id == 'as_form' or param_id == 'request_body_editor' or param_id == 'request_body_note' or param_id.endswith('_body')):
                continue
            # body_type이 FORM이면 Request Body Editor 및 안내는 포함시키지 않음 (폼 전송은 개별 필드/키로 처리)
            if body_type_val == 'FORM' and (param_id == 'request_body_editor' or param_id == 'request_body_note'):
                continue

            if param_value is not None:
                if param_id.endswith('_description'):
                    key_name = param_id[:-12]
                    key_descriptions[key_name] = param_value
                else:
                    # 일반 key
                    key_values[param_id] = param_value

        for key_name, key_value in key_values.items():
            field_type = self._infer_type(key_value)
            description = key_descriptions.get(key_name, "")
            fields[key_name] = (field_type, Field(description=description))

        ArgsSchema = create_model('DynamicArgsSchema', **fields)
        return ArgsSchema

    def _infer_type(self, value: Any) -> Type:
        if isinstance(value, str):
            value_lower = value.lower().strip()

            if value_lower in ['str', 'string']:
                return str
            elif value_lower in ['int', 'integer']:
                return int
            elif value_lower in ['float', 'double']:
                return float
            elif value_lower in ['bool', 'boolean']:
                return bool
            elif value_lower in ['any']:
                return Any

            parsed_value = self._parse_value(value)
            return type(parsed_value)

        return type(value)

    def _parse_value(self, value: str) -> Any:
        if not isinstance(value, str):
            return value
        if not value.strip():
            return ""
        value = value.strip()
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        if value.lower() in ('null', 'none'):
            return None
        if value.startswith('[') and value.endswith(']'):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [self._parse_value(str(item)) if isinstance(item, str) else item for item in parsed]
                return parsed
            except (json.JSONDecodeError, ValueError):
                return value

        if value.startswith('{') and value.endswith('}'):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, ValueError):
                return value
        if value.lstrip('-').isdigit():
            return int(value)

        try:
            float_val = float(value)
            if float_val.is_integer():
                return int(float_val)
            return float_val
        except ValueError:
            pass

        if 'e' in value.lower():
            try:
                float_val = float(value)
                if float_val.is_integer():
                    return int(float_val)
                return float_val
            except ValueError:
                pass

        if value.lower().startswith('0x'):
            try:
                return int(value, 16)
            except ValueError:
                pass
        if value.lower().startswith('0o'):
            try:
                return int(value, 8)
            except ValueError:
                pass
        if value.lower().startswith('0b'):
            try:
                return int(value, 2)
            except ValueError:
                pass
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            return value[1:-1]
        return value

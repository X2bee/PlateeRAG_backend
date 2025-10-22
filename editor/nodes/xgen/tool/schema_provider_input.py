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
    parameters = [
        {"id": "body_type", "name": "Body Type", "type": "STR", "value": "JSON", "options": [{"value": "JSON", "label": "JSON"}, {"value": "FORM", "label": "Form"}], "description": "Choose JSON to edit the request body as JSON, or Form to send application/x-www-form-urlencoded."},
        {"id": "request_body_note", "name": "Request Body (use *_body keys)", "type": "STR", "value": "", "description": "Add fields with suffix '_body' to include them in the POST body (e.g. payload_body, metadata_body)."}
    ]

    parameters.append({"id": "request_body_editor", "name": "Request Body Editor", "type": "STR", "value": "", "expandable": True, "description": "Open the editor to edit JSON body for POST requests."})

    def execute(self, *args, **kwargs) -> BaseModel:
        if not kwargs:
            class EmptySchema(BaseModel):
                pass
            return EmptySchema

        fields = {}
        key_values = {}
        key_descriptions = {}

        method_val = None
        for k in ('method', 'http_method', 'api_method'):
            if k in kwargs:
                method_val = kwargs.get(k)
                break
        method_upper = None
        if isinstance(method_val, str):
            method_upper = method_val.upper()
        
        body_type_val = None
        if 'body_type' in kwargs:
            try:
                body_type_val = str(kwargs.get('body_type')).upper()
            except Exception:
                body_type_val = None

        for param_id, param_value in kwargs.items():
            if param_id == 'request_body_note':
                continue
            if method_upper == 'GET' and (param_id == 'request_body_editor' or param_id == 'request_body_note' or param_id.endswith('_body')):
                continue
            if body_type_val == 'FORM' and (param_id == 'request_body_editor' or param_id == 'request_body_note'):
                continue

            if param_value is not None:
                if param_id.endswith('_description'):
                    key_name = param_id[:-12]
                    key_descriptions[key_name] = param_value
                else:
                    key_values[param_id] = param_value

        for key_name, key_value in key_values.items():
            field_type = self._infer_type(key_value)
            description = key_descriptions.get(key_name, "")
            # body_type은 기본값을 강제로 포함시켜, 에이전트가 값을 누락해도 전송 방식이 유지되도록 함
            if key_name == 'body_type':
                fields[key_name] = (str, Field(default=str(key_value).upper(), description=description))
            else:
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

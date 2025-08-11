from editor.node_composer import Node
import json
from typing import Dict, Any, Type
from pydantic import BaseModel, create_model, Field

class SchemaProviderNode(Node):
    categoryId = "xgen"
    functionId = "tools"
    nodeId = "schema_provider"
    nodeName = "Schema Provider"
    description = "BaseModel을 사용하여 handle_id가 true인 파라미터들의 id를 키로, 입력된 값을 값으로 하는 JSON 딕셔너리를 출력하는 노드입니다."
    tags = ["input", "json", "dict", "parameter", "source", "start_node", "user_input", "key_value"]

    inputs = []
    outputs = [
        {"id": "args_schema", "name": "ArgsSchema", "type": "BaseModel"},
    ]
    parameters = [
        # {"id": "key", "name": "Key", "type": "STR", "value": "value", "required": True, "handle_id": True},
    ]

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
        for param_id, param_value in kwargs.items():
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

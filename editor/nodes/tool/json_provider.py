from editor.node_composer import Node
import json
from typing import Dict, Any

class JsonProviderNode(Node):
    categoryId = "utilities"
    functionId = "tools"
    nodeId = "json_provider"
    nodeName = "JSON Provider"
    description = "handle_id가 true인 파라미터들의 id를 키로, 입력된 값을 값으로 하는 JSON 딕셔너리를 출력하는 노드입니다."
    tags = ["input", "json", "dict", "parameter", "source", "start_node", "user_input", "key_value"]

    inputs = []
    outputs = [
        {"id": "json", "name": "JSON", "type": "DICT"},
    ]
    parameters = [
        {"id": "key", "name": "Key", "type": "STR", "value": "value", "required": True, "handle_id": True},
    ]

    def execute(self, *args, **kwargs) -> Dict[str, Any]:
        result = {}
        for param_id, param_value in kwargs.items():
            if param_value is not None:
                parsed_value = self._parse_value(param_value)
                result[param_id] = parsed_value

        return result

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

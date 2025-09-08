from editor.node_composer import Node
import json
from typing import Dict, Any, Type, Optional
from pydantic import BaseModel, ValidationError

#Notice: 이거 바꾸면 io_logs 파싱 로직도 바꿔야 함.

class InputStringNode(Node):
    categoryId = "xgen"
    functionId = "startnode"
    nodeId = "input_string"
    nodeName = "Input String"
    description = "사용자가 설정한 문자열 값을 출력하는 입력 노드입니다. 워크플로우에서 텍스트 데이터의 시작점으로 사용됩니다. ArgsSchema가 제공되면 입력값들을 스키마에 따라 검증합니다."
    tags = ["input", "string", "text", "parameter", "source", "start_node", "user_input"]

    inputs = [
        {"id": "args_schema", "name": "ArgsSchema", "type": "InputSchema", "required": False},
    ]
    outputs = [
        {"id": "text", "name": "Text", "type": "STR"},
    ]
    parameters = [
        {"id": "input_str", "name": "INPUT", "type": "STR", "value": "", "required": False},
        {"id": "use_stt", "name": "Use STT", "type": "BOOL", "value": False, "required": False, "optional": True},
    ]

    def execute(self, input_str: str, args_schema: Optional[BaseModel] = None, use_stt: bool = False, **kwargs) -> str:
        #TODO STT 관련 Node 레벨에서 처리할 것이 있으면 구현.
        if use_stt:
            pass
        else:
            pass

        kwargs_result = {}
        if args_schema is not None:
            # args_schema에 정의된 필드명들 가져오기
            schema_fields = set(args_schema.model_fields.keys())

            # kwargs에서 schema에 정의되지 않은 키들 찾기
            invalid_keys = set(kwargs.keys()) - schema_fields

            if invalid_keys:
                # 정의되지 않은 키가 있는 경우에만 오류 반환
                error_details = []
                for key in invalid_keys:
                    error_details.append({
                        "field": key,
                        "message": f"Field '{key}' is not defined in the schema",
                        "input": kwargs.get(key, "N/A")
                    })

                return f"Input: {input_str}\n\nValidation Error: " + json.dumps(error_details, ensure_ascii=False)

            # 스키마에 정의된 키들만 필터링해서 사용 (partial validation)
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in schema_fields}

            try:
                # 부분적 검증 시도 (스키마에 정의된 키들만)
                validated_data = args_schema(**filtered_kwargs)
                kwargs_result = validated_data.model_dump()
            except ValidationError as e:
                # 타입 변환 등의 오류가 있는 경우에만 오류 반환
                error_details = []
                for error in e.errors():
                    error_details.append({
                        "field": error.get("loc", ["unknown"])[0] if error.get("loc") else "unknown",
                        "message": error.get("msg", "Validation error"),
                        "input": error.get("input", "N/A")
                    })

                return f"Input: {input_str}\n\nValidation Error: " + json.dumps(error_details, ensure_ascii=False)
        else:
            # ArgsSchema가 없는 경우 기존 로직 사용
            for param_id, param_value in kwargs.items():
                if param_value is not None:
                    parsed_value = self._parse_value(param_value)
                    kwargs_result[param_id] = parsed_value

        # input_str이 존재하고 빈 문자열이 아닌 경우
        if input_str and input_str.strip():
            if kwargs_result:
                parsed_input_str = f"""Input: {input_str}

parameters: {json.dumps(kwargs_result, ensure_ascii=False)}"""
                return parsed_input_str
            else:
                return input_str

        # input_str이 없거나 빈 문자열이지만 kwargs가 존재하는 경우
        elif (not input_str or input_str.strip() == "") and kwargs_result:
            return json.dumps(kwargs_result, ensure_ascii=False)

        # 둘 다 없는 경우 에러 출력
        else:
            return "Error: No input string or parameters provided. At least one of them must be provided."


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

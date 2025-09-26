import logging
import inspect
from typing import Any, Dict, List, Optional
from editor.node_composer import Node
from langchain.agents import tool
from langchain_core.tools import BaseTool
import json

logger = logging.getLogger(__name__)

class HierarchyToolsNode(Node):
    categoryId = "xgen"
    functionId = "tools"
    nodeId = "tools/hierarchy_tools"
    nodeName = "Hierarchy Tools"
    description = "여러 도구를 계층적으로 실행할 수 있는 Tool을 생성합니다"
    tags = ["hierarchy", "tools", "rag", "setup"]

    inputs = [
        {"id": "tools", "name": "Tools", "type": "TOOL", "multi": True}
    ]
    outputs = [
        {"id": "tools", "name": "Tools", "type": "TOOL"},
    ]

    parameters = [
        {"id": "tool_name", "name": "Tool Name", "type": "STR", "value": "hierarchy_tool", "required": True},
        {"id": "description", "name": "Description", "type": "STR", "value": "계층적으로 여러 도구를 실행할 수 있는 도구입니다. 인자 없이 호출하면 사용 가능한 도구 목록을 반환하고, child_tool_name과 arguments를 제공하면 해당 도구를 실행합니다.", "required": True, "expandable": True, "description": "이 도구를 언제 사용하여야 하는지 설명합니다."},
    ]

    def _extract_tool_info(self, langchain_tool: BaseTool) -> Dict[str, Any]:
        """도구에서 정보를 추출합니다."""
        try:
            tool_info = {
                "name": langchain_tool.name,
                "description": langchain_tool.description,
                "parameters": {}
            }

            # Pydantic 모델에서 매개변수 정보 추출
            if hasattr(langchain_tool, 'args_schema') and langchain_tool.args_schema:
                schema = langchain_tool.args_schema.schema()
                properties = schema.get('properties', {})
                required = schema.get('required', [])

                for param_name, param_info in properties.items():
                    tool_info["parameters"][param_name] = {
                        "type": param_info.get('type', 'string'),
                        "description": param_info.get('description', ''),
                        "required": param_name in required,
                        "default": param_info.get('default')
                    }
            else:
                # fallback: inspect 모듈로 함수 시그니처 분석
                if hasattr(langchain_tool, 'func'):
                    sig = inspect.signature(langchain_tool.func)
                    for param_name, param in sig.parameters.items():
                        if param_name not in ['self', 'args', 'kwargs']:
                            tool_info["parameters"][param_name] = {
                                "type": "string",
                                "description": "",
                                "required": param.default == inspect.Parameter.empty,
                                "default": param.default if param.default != inspect.Parameter.empty else None
                            }

            return tool_info
        except (AttributeError, TypeError, ValueError) as e:
            logger.error("도구 정보 추출 중 오류 발생: %s", e)
            return {
                "name": getattr(langchain_tool, 'name', 'unknown'),
                "description": getattr(langchain_tool, 'description', ''),
                "parameters": {}
            }

    def _format_tools_list(self, tools_info: List[Dict[str, Any]]) -> str:
        """도구 목록을 Agent가 이해하기 쉬운 형식으로 포맷팅합니다."""
        if not tools_info:
            return "No available tools."

        tools_schemas = {}

        for tool_info in tools_info:
            # JSON 스키마 형식으로 변환
            schema = {
                'description': tool_info['description'],
                'type': 'object',
                'properties': {},
                'required': []
            }

            # 매개변수 정보를 properties로 변환
            for param_name, param_info in tool_info['parameters'].items():
                schema['properties'][param_name] = {
                    'title': param_name.title(),
                    'type': param_info['type'],
                    'description': param_info['description']
                }

                # 필수 매개변수 추가
                if param_info['required']:
                    schema['required'].append(param_name)

            # 도구명을 키로 하여 스키마 저장
            tools_schemas[tool_info['name']] = schema

        # 영어 프롬프트와 함께 JSON 스키마 반환
        prompt = f"""Available tools for hierarchy_tool:

To use a tool, call hierarchy_tool with the following format:
- List all tools: hierarchy_tool()
- Execute specific tool: hierarchy_tool(child_tool_name='tool_name', required_param='value', ...)

Tool schemas:
{json.dumps(tools_schemas, ensure_ascii=False, indent=2)}

Important notes:
1. child_tool_name must be one of the tool names listed above
2. All required parameters must be provided
3. Parameter names and values must be exact"""

        return prompt

    def execute(self, tools, tool_name, description, *args, **kwargs):
        # tools가 단일 도구인 경우 리스트로 변환
        if not isinstance(tools, list):
            tools = [tools] if tools else []

        # 도구 정보 추출
        tools_info = []
        tools_dict = {}

        for langchain_tool in tools:
            if isinstance(langchain_tool, BaseTool):
                info = self._extract_tool_info(langchain_tool)
                tools_info.append(info)
                tools_dict[langchain_tool.name] = langchain_tool

        def create_hierarchy_tool():
            from pydantic import BaseModel, Field
            from typing import Any

            class HierarchyToolInput(BaseModel):
                child_tool_name: Optional[str] = Field(None, description="실행할 하위 도구의 이름")

                class Config:
                    extra = "allow"  # 추가 필드 허용

            @tool(tool_name, description=description, args_schema=HierarchyToolInput)
            def hierarchy_tool(**input_data) -> str:
                """
                계층적 도구 실행기

                Args:
                    child_tool_name: 실행할 하위 도구의 이름 (선택적)
                    **tool_kwargs: 하위 도구에 전달할 인자들

                Returns:
                    str: 도구 목록 또는 실행 결과
                """
                # input_data에서 child_tool_name 추출
                child_tool_name = input_data.pop('child_tool_name', None)
                kwargs = input_data  # 나머지 파라미터들

                logger.info("hierarchy_tool 호출됨 - child_tool_name: %s, kwargs: %s", child_tool_name, kwargs)

                try:
                    # 1. child_tool_name이 없는 경우: 도구 목록 반환
                    if (child_tool_name not in tools_dict) or (not child_tool_name):
                        return self._format_tools_list(tools_info)

                    target_tool = tools_dict[child_tool_name]
                    logger.info("도구 '%s' 실행 - 전달된 인자: %s", child_tool_name, kwargs)

                    try:
                        if hasattr(target_tool, 'invoke'):
                            result = target_tool.invoke(kwargs)
                        elif hasattr(target_tool, 'run'):
                            if kwargs:
                                result = target_tool.run(**kwargs)
                            else:
                                result = target_tool.run()
                        else:
                            # 직접 함수 호출
                            result = target_tool(**kwargs)

                        # 결과를 문자열로 변환
                        if isinstance(result, str):
                            return result
                        elif isinstance(result, dict):
                            return json.dumps(result, ensure_ascii=False, indent=2)
                        else:
                            return str(result)

                    except (AttributeError, TypeError, ValueError) as tool_error:
                        error_msg = f"도구 '{child_tool_name}' 실행 중 오류 발생: {str(tool_error)}"
                        logger.error("도구 실행 중 오류: %s", tool_error)
                        return error_msg

                except (AttributeError, TypeError, ValueError, KeyError) as e:
                    error_msg = f"계층적 도구 실행 중 오류 발생: {str(e)}"
                    logger.error("계층적 도구 실행 중 오류: %s", e)
                    return error_msg

            return hierarchy_tool

        return create_hierarchy_tool()

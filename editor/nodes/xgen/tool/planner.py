from editor.node_composer import Node
import json
from typing import Dict, Any, Type, Optional
from pydantic import BaseModel, ValidationError
from langchain_core.tools import BaseTool

class PlannerNode(Node):
    categoryId = "xgen"
    functionId = "tools"
    nodeId = "tools/agent_planner"
    nodeName = "Agent Planner"
    description = "에이전트의 도구 사용 계획을 수립하는 노드입니다."
    tags = []

    inputs = [
        {"id": "plan", "name": "Plan", "type": "PLAN", "required": False},
        {"id": "tools", "name": "Tools", "type": "TOOL", "required": False, "multi": True},

    ]
    outputs = [
        {"id": "plan", "name": "Plan", "type": "PLAN", "required": False},
    ]
    parameters = [
        {"id": "plan_description", "name": "Plan Description", "type": "STR", "value": "", "required": False, "expandable": True, "description": "Agent가 수행하는 작업 계획을 제시합니다."},
    ]

    def execute(self, plan: Dict[str, Any] = None, tools: list[BaseTool] = None, plan_description: str = "", **kwargs) -> str:
        if not isinstance(tools, list):
            tools = [tools]

        if plan is None:
            plan = {"steps": [], "tools": []}

            if tools is None:
                plan['steps'] = [plan_description]

            else:
                plan_description = plan_description + f"다음의 도구들을 사용합니다: {', '.join([tool.name for tool in tools])}"
                plan['tools'] = tools
                plan['steps'] = [plan_description]
        else:
            if tools is None:
                plan['steps'].append(plan_description)
            else:
                plan_description = plan_description + f"다음의 도구들을 사용합니다: {', '.join([tool.name for tool in tools])}"
                plan['tools'].extend(tools)
                plan['steps'].append(plan_description)

        return plan

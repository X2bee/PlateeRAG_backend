import os
import logging
import asyncio
from typing import Optional, Dict, Any
from editor.node_composer import Node
from langchain.agents import tool
from editor.utils.helper.service_helper import AppServiceManager
from editor.utils.helper.async_helper import sync_run_async
from service.database.models.vectordb import VectorDB
from fastapi import Request
from controller.controller_helper import extract_user_id_from_request

logger = logging.getLogger(__name__)
enhance_prompt = """You are an AI assistant that must strictly follow these guidelines when using the provided document context:

1. ANSWER ONLY BASED ON PROVIDED CONTEXT: Use only the information from the retrieved documents to answer questions. Do not add information from your general knowledge.
2. BE PRECISE AND ACCURATE: Quote specific facts, numbers, and details exactly as they appear in the documents. Include relevant quotes when appropriate.
3. ACKNOWLEDGE LIMITATIONS: If the provided documents do not contain sufficient information to answer the user's question, clearly state "I don't have enough information in the provided documents to answer this question" or "The provided documents don't contain information about [specific topic]."
4. STAY FOCUSED: Answer only what the user asked. Do not provide additional information beyond what was requested unless it's directly relevant to the question.
5. CITE SOURCES: When possible, reference which document number contains the information you're using (e.g., "According to Document 1..." or "As mentioned in Document 2...").
6. BE CONCISE: Provide clear, direct answers without unnecessary elaboration. Focus on delivering exactly what the user needs.

Remember: It's better to say "I don't know" than to provide inaccurate or fabricated information."""

class APICallingTool(Node):
    categoryId = "xgen"
    functionId = "api_loader"
    nodeId = "api_loader/APICallingTool"
    nodeName = "API Calling Tool"
    description = "API 호출을 위한 Tool을 전달"
    tags = ["api", "rag", "setup"]

    inputs = []
    outputs = [
        {"id": "tools", "name": "Tools", "type": "TOOL"},
    ]

    parameters = [
        {"id": "tool_name", "name": "Tool Name", "type": "STR", "value": "api_calling_tool", "required": True},
        {"id": "description", "name": "Description", "type": "STR", "value": "Use this tool when you need to call an external API to retrieve specific data or perform an operation. Call this tool when the user requests information that requires an API call to external services.", "required": True, "description": "이 도구를 언제 사용하여야 하는지 설명합니다. AI는 해당 설명을 통해, 해당 도구를 언제 호출해야할지 결정할 수 있습니다."},
        {"id": "api_endpoint", "name": "API Endpoint", "type": "STR", "value": "", "required": True, "description": "해당 도구의 실행으로 호출할 API의 엔드포인트 URL입니다."},
    ]

    def execute(self, tool_name, description):
        def create_api_tool():
            @tool(tool_name, description=description)
            def api_tool() -> str:
                try:
                    # Placeholder for API call logic
                    return "API call executed successfully."
                except Exception as e:
                    return f"Failed to execute API call: {e}"

            return api_tool

        return create_api_tool()

"""
워크플로우 자동생성 API 엔드포인트
"""
import json
import logging
import time
import re
import requests
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Request
from controller.helper.controllerHelper import extract_user_id_from_request
from controller.helper.singletonHelper import get_db_manager, get_config_composer
from service.database.logger_helper import create_logger
from controller.workflow.models.requests import (
    WorkflowGenerationRequest, 
    WorkflowGenerationResponse,
    AgentNodeInfoRequest
)
from controller.node.nodeController import get_node_list
from editor.node_composer import get_node_registry

logger = logging.getLogger(__name__)


def extract_json_from_response(response_text: str, user_requirements: str = "") -> str:
    """LLM 응답에서 JSON 부분만 추출하는 공통 함수"""
    json_str = response_text.strip()
    # <think> 태그 처리 (닫는 태그가 있든 없든)
    if "<think>" in json_str:
        think_start = json_str.find("<think>")
        if "</think>" in json_str:
            # 닫는 태그가 있는 경우
            think_end = json_str.find("</think>") + 8
            json_str = json_str[:think_start] + json_str[think_end:]
        else:
            # 닫는 태그가 없는 경우 - <think> 이후 모든 내용을 제거하고 JSON만 찾기
            json_str = json_str[:think_start]
        json_str = json_str.strip()
        logger.info(f"<think> 태그 제거 후: {json_str[:100]}")
    
    # 다른 XML 스타일 태그들 제거 (예: <reasoning>, <analysis> 등)
    xml_pattern = r'<[^>]+>.*?</[^>]+>'
    json_str = re.sub(xml_pattern, '', json_str, flags=re.DOTALL)
    json_str = json_str.strip()
    
    # JSON 코드 블록 제거
    if "```json" in json_str:
        start = json_str.find("```json") + 7
        end = json_str.find("```", start)
        if end != -1:
            json_str = json_str[start:end].strip()
    elif "```" in json_str:
        start = json_str.find("```") + 3
        end = json_str.find("```", start)
        if end != -1:
            json_str = json_str[start:end].strip()
    
    # JSON 시작 위치 찾기
    json_start = -1
    for i, char in enumerate(json_str):
        if char in ['{', '[']:
            json_start = i
            break
    
    if json_start > 0:
        json_str = json_str[json_start:].strip()
    elif json_start == -1:
        # JSON이 전혀 없는 경우, 원본에서 다시 찾기
        logger.warning("JSON 시작 문자를 찾을 수 없음. 원본에서 다시 검색...")
        original_json_start = -1
        for i, char in enumerate(response_text):
            if char in ['{', '[']:
                original_json_start = i
                break
        if original_json_start != -1:
            json_str = response_text[original_json_start:].strip()
            logger.info(f"원본에서 찾은 JSON 시작: {json_str[:100]}")
        else:
            logger.error("JSON을 전혀 찾을 수 없음. 기본 노드 선택 JSON 반환")
            # 사용자 요구사항에서 API 개수 추출하여 기본 구조 생성
            numbers = re.findall(r'\d+', user_requirements if user_requirements else response_text)
            api_count = 1
            for num in numbers:
                if int(num) > 1 and int(num) <= 10:
                    api_count = int(num)
                    break
            
            # API 개수에 맞는 기본 노드 선택 JSON 반환
            selected_nodes = []
            if api_count >= 1:
                selected_nodes.append({"index": 8, "reason": "API 호출 도구"})
            if api_count >= 2:
                selected_nodes.append({"index": 13, "reason": "Brave 검색 API"})
            if api_count >= 3:
                selected_nodes.append({"index": 16, "reason": "GitHub API"})
            if api_count >= 4:
                selected_nodes.append({"index": 18, "reason": "Meta 검색 API"})
            
            fallback_json = {
                "selected_nodes": selected_nodes,
                "workflow_description": f"{api_count}개 API를 사용하는 챗봇 워크플로우"
            }
            return json.dumps(fallback_json, ensure_ascii=False)
    
    # JSON 끝 위치 찾기 (중괄호 매칭)
    if json_str.startswith('{'):
        brace_count = 0
        json_end = -1
        for i, char in enumerate(json_str):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    break
        if json_end > 0:
            json_str = json_str[:json_end]
    elif json_str.startswith('['):
        bracket_count = 0
        json_end = -1
        for i, char in enumerate(json_str):
            if char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    json_end = i + 1
                    break
        if json_end > 0:
            json_str = json_str[:json_end]
    
    logger.info(f"최종 추출된 JSON: {json_str[:200]}...")
    
    # JSON 유효성 검사
    try:
        import json as json_module
        json_module.loads(json_str)
        logger.info("JSON 유효성 검사 통과")
    except json_module.JSONDecodeError as e:
        logger.error(f"추출된 JSON이 유효하지 않음: {e}")
        logger.error(f"유효하지 않은 JSON: {json_str}")
        # 기본 구조 반환 (Agent 노드만 사용하는 단순 구조)
        return """{
            "workflow_structure": [
                {
                    "node_index": "agent",
                    "x_position": 400,
                    "connections": []
                }
            ]
        }"""
    
    return json_str

router = APIRouter(
    prefix="/api/workflow/auto-generation",
    tags=["workflow-auto-generation"],
    responses={404: {"description": "Not found"}},
)


def find_agent_node_by_id(agent_node_id: str, available_nodes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Agent 노드 정보 조회"""
    for category in available_nodes:
        if category.get("functions"):
            for func in category["functions"]:
                if func["functionId"] == "agents" and func.get("nodes"):
                    for node in func["nodes"]:
                        if node["id"] == agent_node_id:
                            return node
    return None

def get_compatible_nodes(agent_node: Dict[str, Any], available_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Agent 노드와 호환 가능한 노드들 필터링"""
    compatible_nodes = []
    
    # Agent 노드의 입력/출력 타입 분석
    agent_inputs = {inp.get('type') for inp in agent_node.get('inputs', [])}
    agent_outputs = {out.get('type') for out in agent_node.get('outputs', [])}
    
    for node in available_nodes:
        node_id = node.get('id', '')
        
        # Agent 노드 자체는 제외
        if node_id == agent_node.get('id'):
            continue
            
        # 시작/종료 노드는 항상 포함
        if 'startnode' in node_id or 'endnode' in node_id:
            compatible_nodes.append(node)
            continue
            
        # 입력/출력 타입 호환성 확인
        node_inputs = {inp.get('type') for inp in node.get('inputs', [])}
        node_outputs = {out.get('type') for out in node.get('outputs', [])}
        
        # Agent와 연결 가능한 노드들 (입력/출력 타입이 겹치는 경우)
        if agent_outputs.intersection(node_inputs) or node_outputs.intersection(agent_inputs):
            compatible_nodes.append(node)
    
    return compatible_nodes


def flatten_nodes_from_categories(categories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """카테고리별 중첩된 노드 구조를 평면적인 노드 리스트로 변환"""
    flattened_nodes = []
    
    for category in categories:
        if not category.get("functions"):
            continue
            
        for func in category["functions"]:
            if not func.get("nodes"):
                continue
                
            for node in func["nodes"]:
                # 노드에 카테고리 정보 추가
                node_copy = node.copy()
                node_copy["categoryId"] = category.get("categoryId")
                node_copy["categoryName"] = category.get("categoryName")
                flattened_nodes.append(node_copy)
    
    return flattened_nodes

def call_llm_for_node_selection(user_requirements: str, agent_node: Dict[str, Any], all_nodes: List[Dict[str, Any]], model_name: str, api_base_url: str, api_key: str = None) -> str:
    """1단계: 사용자 요구사항에 맞는 필요한 노드들 선택"""
    
    # 중첩된 구조를 평면적인 노드 리스트로 변환
    flattened_nodes = flatten_nodes_from_categories(all_nodes)
    logger.info(f"평면화된 노드 수: {len(flattened_nodes)}")
    
    # 노드 선택을 위한 프롬프트 생성
    nodes_info = []
    for i, node in enumerate(flattened_nodes):  # 평면화된 노드 사용
        node_info = {
            "index": i,
            "functionId": node.get('functionId', 'Unknown'),
            "id": node.get('id', 'Unknown'),
            "description": node.get('description', 'No description'),
            "tags": node.get('tags', [])
        }
        nodes_info.append(node_info)
    
    nodes_json = json.dumps(nodes_info, ensure_ascii=False, indent=2)
    
    prompt = f"""워크플로우 노드 선택 작업입니다. 사용자 요구사항을 정확히 분석하여 필요한 노드만 선택하세요.

**요구사항**: {user_requirements}

**Agent**: {agent_node.get('nodeName', 'Unknown')} (이미 선택됨)

**노드 목록**:
{nodes_json}

**선택 기준**:
- description과 tags를 모두 고려하여 요구사항과 관련된 노드만 선택
- 각 노드의 functionId, id, description, tags를 종합적으로 판단
- 불필요한 노드는 선택하지 마세요
- 완성된 워크플로우에는 startnode와 endnode는 반드시 선택해야 합니다. 선택된 agent 노드 포트를 참고하여 적절한 startnode와 endnode를 선택하세요.

**응답 형식** (JSON만 응답):
{{
  "selected_nodes": [
    {{
      "node_id": "노드ID",
      "reason": "선택 이유"
    }},
    {{
      "node_id": "노드ID", 
      "reason": "선택 이유"
    }}
  ],
  "workflow_description": "선택한 노드들로 구성할 워크플로우에 대한 간단한 설명"
}}

**중요**: JSON만 응답하세요. 다른 텍스트나 설명은 절대 포함하지 마세요."""
    # LLM API 호출
    try:
        # 헤더 설정 (OpenAI API 키가 있으면 Authorization 헤더 추가)
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            
        response = requests.post(
            f"{api_base_url}/chat/completions",
            json={
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are a workflow node selection expert. CRITICAL: Respond ONLY with valid JSON. Do NOT include any thinking, explanation, or other text. JSON response only."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 1,
                "max_completion_tokens": 2048,
                "stream": False
            },
            headers=headers,
            timeout=300
        )
        
        if response.status_code != 200:
            error_text = response.text
            raise HTTPException(status_code=500, detail=f"노드 선택 단계에서 모델 호출 실패: {error_text}")
        
        result = response.json()
        llm_output = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        return llm_output
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"노드 선택 단계 오류: {str(e)}")

def parse_node_selection_response(selection_response: str, all_nodes: List[Dict[str, Any]], user_requirements: str = "") -> List[Dict[str, Any]]:
    """노드 선택 응답을 파싱하여 선택된 노드들 반환 (ID 기반으로 전체 노드 정보 조회)"""
    try:
        # 중첩된 구조를 평면적인 노드 리스트로 변환
        flattened_nodes = flatten_nodes_from_categories(all_nodes)
        
        # 공통 JSON 추출 함수 사용
        json_str = extract_json_from_response(selection_response, user_requirements)
        
        logger.info(f"파싱할 JSON 문자열: {json_str[:200]}...")
        selection_data = json.loads(json_str)
        selected_nodes = []
        
        # 평면화된 노드들을 ID로 매핑
        node_id_map = {node.get('id'): node for node in flattened_nodes}
        
        # 선택된 노드들 수집
        for selection in selection_data.get("selected_nodes", []):
            # 먼저 index로 시도 (기존 방식 호환성)
            index = selection.get("index")
            node_id = selection.get("node_id")  # 새로운 ID 기반 방식
            
            reason = selection.get("reason", "")
            selected_node = None
            
            # ID 기반으로 노드 찾기 (우선)
            if node_id and node_id in node_id_map:
                selected_node = node_id_map[node_id].copy()
            # 인덱스 기반으로 노드 찾기 (호환성)
            elif index is not None and 0 <= index < len(flattened_nodes):
                selected_node = flattened_nodes[index].copy()
            
            if selected_node:
                selected_node["selection_reason"] = reason
                selected_nodes.append(selected_node)
                logger.info(f"✓ {selected_node.get('nodeName', 'Unknown')} ({selected_node.get('id', 'Unknown')})")
        
        workflow_description = selection_data.get("workflow_description", "")
        logger.info(f"워크플로우 설명: {workflow_description}")
        
        # 선택된 노드들 요약 로그
        if selected_nodes:
            selected_summary = []
            for node in selected_nodes:
                node_name = node.get('nodeName', 'Unknown')
                node_id = node.get('id', 'Unknown')
                selected_summary.append(f"{node_name}({node_id})")
            logger.info(f"✅ 선택된 노드들: {', '.join(selected_summary)}")
        else:
            logger.warning("⚠️ 선택된 노드가 없습니다.")
        
        return selected_nodes
        
    except Exception as e:
        logger.error(f"노드 선택 응답 파싱 실패: {str(e)}")
        logger.error(f"파싱 실패한 응답: {selection_response[:500]}")
        raise HTTPException(status_code=500, detail=f"노드 선택 결과 파싱 실패: {str(e)}")

def find_best_end_node(all_nodes: List[Dict[str, Any]], agent_node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Agent 노드의 출력 타입에 맞는 최적의 종료 노드 찾기"""
    
    # 중첩된 구조를 평면적인 노드 리스트로 변환
    flattened_nodes = flatten_nodes_from_categories(all_nodes)
    
    # Agent 노드의 출력 타입 확인
    agent_output_types = set()
    for output in agent_node.get('outputs', []):
        output_type = output.get('type', '')
        agent_output_types.add(output_type)
    
    logger.info(f"Agent 노드 출력 타입: {agent_output_types}")
    
    # 종료 노드들 수집
    end_nodes = []
    for node in flattened_nodes:
        if node.get('functionId') == 'endnode':
            end_nodes.append(node)
    
    # 최적의 종료 노드 선택
    best_node = None
    
    # 1. STREAM STR 출력이 있으면 Print Any (Stream) 우선
    if 'STREAM STR' in agent_output_types:
        for node in end_nodes:
            node_name = node.get('nodeName', '').lower()
            if 'stream' in node_name:
                logger.info(f"STREAM 출력에 맞는 종료 노드 선택: {node.get('nodeName')}")
                return node
    
    # 2. 일반적인 Print Any 선택
    for node in end_nodes:
        node_name = node.get('nodeName', '').lower()
        if 'print any' in node_name and 'stream' not in node_name:
            logger.info(f"일반 출력에 맞는 종료 노드 선택: {node.get('nodeName')}")
            return node
    
    # 3. 첫 번째 종료 노드 사용
    if end_nodes:
        logger.info(f"기본 종료 노드 선택: {end_nodes[0].get('nodeName')}")
        return end_nodes[0]
    
    return None

def ensure_essential_nodes(selected_nodes: List[Dict[str, Any]], all_nodes: List[Dict[str, Any]], agent_node: Dict[str, Any]) -> List[Dict[str, Any]]:
    """선택된 노드에 시작노드 또는 종료노드가 없으면 all_nodes에서 찾아서 추가"""
    # 중첩된 구조를 평면적인 노드 리스트로 변환
    flattened_nodes = flatten_nodes_from_categories(all_nodes)
    
    # 현재 선택된 노드들 functionId 모음
    selected_function_ids = {node.get('functionId') for node in selected_nodes}
    
    # 시작노드 없는 경우에만 찾음
    start_node = None
    if 'startnode' not in selected_function_ids:
        for node in flattened_nodes:
            if node.get('functionId') == 'startnode':
                start_node = node
                logger.info(f"시작 노드 발견: {node.get('nodeName')}")
                break
        
        if not start_node:
            logger.warning("시작 노드를 찾을 수 없습니다.")
    else:
        logger.info(f"✅ AI가 이미 시작 노드를 선택했으므로 추가하지 않음")
         
    # 종료노드 없는 경우에만 찾음
    end_node = None
    if 'endnode' not in selected_function_ids:
        end_node = find_best_end_node(all_nodes, agent_node)
        logger.info(f"🔍 종료 노드 검색: {'endnode' not in selected_function_ids}")
    else:
        logger.info(f"✅ AI가 이미 종료 노드를 선택했으므로 추가하지 않음")

    # 결과 노드 리스트
    final_nodes = selected_nodes.copy()
    
    if start_node:
        final_nodes.insert(0, start_node)
        logger.info(f"✅ 필수 시작 노드 추가: {start_node.get('nodeName')} (ID: {start_node.get('id')})")
    else:
        logger.info("ℹ️ 시작 노드는 AI가 선택한 것을 사용합니다.")
        
    if end_node:
        final_nodes.append(end_node)
        logger.info(f"✅ 필수 종료 노드 추가: {end_node.get('nodeName')} (ID: {end_node.get('id')})")
    else:
        logger.info("ℹ️ 종료 노드는 AI가 선택한 것을 사용합니다.")
    
    logger.info(f"📋 최종 노드 목록 ({len(final_nodes)}개): {[node.get('nodeName') for node in final_nodes]}")
    
    return final_nodes






def call_llm_for_edge_creation(nodes: List[Dict[str, Any]], agent_node: Dict[str, Any], model_name: str, api_base_url: str, api_key: str = None) -> List[Dict[str, Any]]:
    """AI를 통한 엣지 생성: 노드 정보를 보고 최적의 연결 결정"""
    
    # 노드 정보 수집
    nodes_info = []
    for node in nodes:
        node_data = node.get('data', {})
        nodes_info.append({
            "id": node.get('id'),
            "name": node_data.get('nodeName'),
            "functionId": node_data.get('functionId'),
            "inputs": [{"id": inp.get('id'), "type": inp.get('type')} for inp in node_data.get('inputs', [])],
            "outputs": [{"id": out.get('id'), "type": out.get('type')} for out in node_data.get('outputs', [])]
        })
    
    # Agent 노드 정보
    agent_info = {
        "id": "agent",
        "name": agent_node.get('nodeName'),
        "functionId": agent_node.get('functionId'),
        "inputs": [{"id": inp.get('id'), "type": inp.get('type')} for inp in agent_node.get('inputs', [])],
        "outputs": [{"id": out.get('id'), "type": out.get('type')} for out in agent_node.get('outputs', [])]
    }
    
    nodes_json = json.dumps(nodes_info, ensure_ascii=False, indent=2)
    agent_json = json.dumps(agent_info, ensure_ascii=False, indent=2)
    
    prompt = f"""워크플로우 노드 연결 작업입니다. 주어진 노드들의 입력/출력 포트 정보를 분석하여 논리적인 연결을 생성하세요.

**Agent 노드**:
{agent_json}

**연결할 노드들**:
{nodes_json}

**연결 가이드**:
1. **기본 흐름**: Input → Agent → Output
2. **도구 연결**: API/Tool 노드의 tools 출력 → Agent의 tools 입력
3. **텍스트 연결**: Input 노드의 text 출력 → Agent의 text 입력
4. **스트림 연결**: Agent의 stream 출력 → Print 노드의 input 입력
5. **포트 타입 매칭**: 동일하거나 호환되는 타입끼리 연결
6. **논리적 순서**: 데이터가 논리적으로 흐를 수 있는 방향으로 연결

**연결 규칙**:
- 출력 포트 → 입력 포트 방향으로만 연결
- 포트 타입이 호환되는 경우에만 연결 (STR ↔ STR, TOOL ↔ TOOL, STREAM STR ↔ input 등)
- Agent 노드는 중앙 허브 역할 (대부분의 노드가 Agent와 연결)
- 시작 노드는 Agent의 text 입력에 연결
- 도구 노드들은 Agent의 tools 입력에 연결
- Agent의 최종 출력은 Print 노드의 input에 연결

**응답 형식** (JSON만):
{{
  "edges": [
    {{
      "source_node_id": "소스노드ID",
      "source_port": "출력포트ID",
      "target_node_id": "타겟노드ID", 
      "target_port": "입력포트ID"
    }}
  ]
}}

**중요**: JSON만 응답하고 다른 텍스트는 포함하지 마세요."""
    
    logger.info("=" * 80)
    logger.info("AI를 통한 엣지 생성 시작")
    logger.info(f"모델: {model_name}")
    logger.info(f"연결할 노드 수: {len(nodes)}")
    logger.info("=" * 80)
    
    # LLM API 호출
    try:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            
        response = requests.post(
            f"{api_base_url}/chat/completions",
            json={
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are a workflow connection expert. Analyze node input/output ports and create logical connections. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 1,
                "max_completion_tokens": 2048,
                "stream": False
            },
            headers=headers,
            timeout=300
        )
        
        if response.status_code != 200:
            error_text = response.text
            logger.error(f"엣지 생성 LLM API 호출 실패: {response.status_code}")
            logger.error(f"에러 응답 내용: {error_text}")
            raise HTTPException(status_code=500, detail=f"엣지 생성 단계에서 모델 호출 실패: {error_text}")
        
        result = response.json()
        llm_output = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        logger.info("=" * 80)
        logger.info("AI 엣지 생성 응답 받음")
        logger.info(f"응답 길이: {len(llm_output)} 문자")
        logger.info("AI가 생성한 엣지 연결:")
        logger.info(llm_output)
        logger.info("=" * 80)
        
        # JSON 추출 및 파싱
        json_str = extract_json_from_response(llm_output)
        edges_data = json.loads(json_str)
        
        # 엣지 생성
        edges = []
        timestamp = str(int(time.time() * 1000))
        
        for i, edge_info in enumerate(edges_data.get("edges", [])):
            source_node_id = edge_info.get("source_node_id")
            source_port = edge_info.get("source_port")
            target_node_id = edge_info.get("target_node_id")
            target_port = edge_info.get("target_port")
            
            if source_node_id and target_node_id and source_port and target_port:
                edge_id = f"edge-{source_node_id}:{source_port}-{target_node_id}:{target_port}-{timestamp}-{i}"
                edges.append({
                "id": edge_id,
                "source": {
                        "nodeId": source_node_id,
                    "portId": source_port,
                        "portType": "output"
                },
                "target": {
                        "nodeId": target_node_id,
                    "portId": target_port,
                    "portType": "input"
                }
                })
                logger.info(f"엣지 생성: {source_node_id}:{source_port} → {target_node_id}:{target_port}")
        
        return edges
        
    except Exception as e:
        logger.error(f"엣지 생성 LLM 호출 오류: {str(e)}")
        # 실패 시 기본 연결 생성
        return create_fallback_edges(nodes, agent_node)

def create_fallback_edges(nodes: List[Dict[str, Any]], agent_node: Dict[str, Any]) -> List[Dict[str, Any]]:
    """AI 실패 시 기본 엣지 생성"""
    edges = []
    timestamp = str(int(time.time() * 1000))
    
    # Agent 노드 찾기
    agent_node_id = None
    for node in nodes:
        if node.get('data', {}).get('functionId') == 'agents':
            agent_node_id = node.get('id')
            break
    
    if not agent_node_id:
        return edges
    
    # 기본 연결 생성
    for i, node in enumerate(nodes):
        node_data = node.get('data', {})
        node_id = node.get('id')
        function_id = node_data.get('functionId')
        
        if function_id == 'startnode':
            # Input → Agent
            edges.append({
                "id": f"edge-{node_id}:text-{agent_node_id}:text-{timestamp}-{i}",
                "source": {
                    "nodeId": node_id,
                    "portId": "text",
                    "portType": "output"
                },
                "target": {
                    "nodeId": agent_node_id,
                    "portId": "text",
                    "portType": "input"
                }
            })
        elif function_id == 'api_loader':
            # API Tool → Agent
            edges.append({
                "id": f"edge-{node_id}:tools-{agent_node_id}:tools-{timestamp}-{i}",
                "source": {
                    "nodeId": node_id,
                    "portId": "tools",
                    "portType": "output"
                },
                "target": {
                    "nodeId": agent_node_id,
                    "portId": "tools",
                    "portType": "input"
                }
            })
        elif function_id == 'endnode':
            # Agent → Print
            edges.append({
                "id": f"edge-{agent_node_id}:stream-{node_id}:input-{timestamp}-{i}",
                "source": {
                    "nodeId": agent_node_id,
                    "portId": "stream",
                    "portType": "output"
                },
                "target": {
                    "nodeId": node_id,
                    "portId": "input",
                    "portType": "input"
                }
            })
    
    return edges

def ensure_start_end_connections(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]], agent_node: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Start/End 노드 연결을 보장하는 함수"""
    timestamp = str(int(time.time() * 1000))
    
    # Agent 노드 ID 찾기
    agent_node_id = None
    for node in nodes:
        if node.get('data', {}).get('functionId') == 'agents':
            agent_node_id = node.get('id')
            break
    
    if not agent_node_id:
        logger.warning("Agent 노드를 찾을 수 없어 Start/End 연결을 보장할 수 없습니다.")
        return edges
    
    # 기존 엣지에서 연결된 노드 ID 수집
    connected_node_ids = set()
    for edge in edges:
        connected_node_ids.add(edge.get('source', {}).get('nodeId'))
        connected_node_ids.add(edge.get('target', {}).get('nodeId'))
    
    # Start 노드 연결 확인 및 추가
    start_node = None
    for node in nodes:
        node_data = node.get('data', {})
        if node_data.get('functionId') == 'startnode':
            start_node = node
            break
    
    if start_node and start_node.get('id') not in connected_node_ids:
        logger.info(f"🔗 Start 노드 연결 추가: {start_node.get('data', {}).get('nodeName')}")
        edges.append({
            "id": f"edge-start-{timestamp}",
            "source": {
                "nodeId": start_node.get('id'),
                "portId": "text",
                "portType": "output"
            },
            "target": {
                "nodeId": agent_node_id,
                "portId": "text",
                "portType": "input"
            }
        })
    
    # End 노드 연결 확인 및 추가
    end_node = None
    for node in nodes:
        node_data = node.get('data', {})
        if node_data.get('functionId') == 'endnode':
            end_node = node
            break
    
    if end_node and end_node.get('id') not in connected_node_ids:
        logger.info(f"🔗 End 노드 연결 추가: {end_node.get('data', {}).get('nodeName')}")
        edges.append({
            "id": f"edge-end-{timestamp}",
            "source": {
                "nodeId": agent_node_id,
                "portId": "stream",
                "portType": "output"
            },
            "target": {
                "nodeId": end_node.get('id'),
                "portId": "stream",
                "portType": "input"
            }
        })
    
    logger.info(f"🔗 Start/End 연결 보장 완료: 총 {len(edges)}개 엣지")
    return edges

def build_workflow_from_structure(agent_node: Dict[str, Any], selected_nodes: List[Dict[str, Any]], all_nodes: List[Dict[str, Any]], model_name: str, api_base_url: str, api_key: str = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """워크플로우 생성: selected_nodes ID로 all_nodes에서 전체 객체 받아와서 워크플로우 구성"""
    try:
        logger.info("🏗️ 워크플로우 생성 시작")
        
        # 1. 전체 노드들을 ID로 매핑
        all_nodes_map = {}
        for category in all_nodes:
            if category.get("functions"):
                for func in category["functions"]:
                    if func.get("nodes"):
                        for node in func["nodes"]:
                            all_nodes_map[node.get('id')] = node
        
        logger.info(f"🗂️ 전체 노드 매핑: {len(all_nodes_map)}개")
        
        # 2. selected_nodes에서 발굴한 ID 값으로 all_nodes에서 객체 전체를 받아옴
        full_selected_nodes = []
        for selected_node in selected_nodes:
            node_id = selected_node.get('id')
            if node_id in all_nodes_map:
                full_node = all_nodes_map[node_id].copy()
                logger.info(f"🔍 {full_node.get('nodeName')} ({node_id})")
                full_selected_nodes.append(full_node)
            else:
                logger.warning(f"노드를 찾을 수 없음: {node_id}")
                full_selected_nodes.append(selected_node)
        
        # 3. Agent 노드도 전체 정보로 교체
        agent_node_id = agent_node.get('id')
        if agent_node_id in all_nodes_map:
            full_agent_node = all_nodes_map[agent_node_id].copy()
            logger.info(f"🤖 Agent: {full_agent_node.get('nodeName')} ({agent_node_id})")
        else:
            full_agent_node = agent_node.copy()
            logger.warning(f"Agent 노드를 찾을 수 없음: {agent_node_id}")
        
        # 4. 선택된 모든 노드들을 워크플로우에 포함
        logger.info(f"📊 선택된 노드 수: {len(full_selected_nodes)}개")
        
        # 5. 노드 생성 (선택된 모든 노드 포함)
        timestamp = str(int(time.time() * 1000))
        nodes = []
        node_id_map = {}
        
        # Agent 노드 먼저 생성 (사용자 뷰포트 중심을 기준으로 배치)
        agent_original_id = full_agent_node['id']
        agent_node_id = f"{agent_original_id}-{timestamp}-agent"
        node_id_map[agent_original_id] = agent_node_id
        node_id_map["agent"] = agent_node_id  # 호환성을 위해 유지

        # 기본 기준점 설정: 사용자가 보고 있는 캔버스의 viewport_center가 있으면 그 좌표를 기준으로 사용
        base_x = 0
        base_y = 0
        try:
            if context and isinstance(context, dict):
                vc = context.get("viewport_center")
                if vc and isinstance(vc, dict) and "x" in vc and "y" in vc:
                    base_x = vc.get("x", base_x)
                    base_y = vc.get("y", base_y)
        except Exception:
            # 실패 시 기본값 사용
            pass

        nodes.append({
            "id": agent_node_id,
            "data": full_agent_node.copy(),
            "position": {"x": base_x, "y": base_y},
            "isExpanded": True
        })
        logger.info(f"🤖 Agent: {full_agent_node.get('nodeName')} @ ({base_x},{base_y})")

        # 나머지 선택된 노드들 생성
        # 배치 규칙 (viewport_center 기준):
        # - startnode: agent 기준 더 왼쪽(원래 -300 → -500), 상단
        # - endnode: agent 기준 오른쪽 상단
        # - 기타 노드: startnode 아래에 수직 정렬
        vertical_counter = 0
        # start_x를 더 왼쪽으로 조정 (-500)
        start_x = base_x - 700
        start_y = base_y - 150
        end_x = base_x + 600
        end_y = base_y - 150

        for i, node in enumerate(full_selected_nodes):
            # Agent 노드는 이미 생성했으므로 제외
            if node.get('id') == agent_original_id:
                continue

            original_id = node['id']
            node_id = f"{original_id}-{timestamp}-{i}"
            node_id_map[original_id] = node_id

            func_id = node.get('functionId')
            if func_id == 'startnode':
                position = {"x": start_x, "y": start_y}
                logger.info(f"📥 Input: {node.get('nodeName')} @ ({start_x},{start_y})")
            elif func_id == 'endnode':
                position = {"x": end_x, "y": end_y}
                logger.info(f"📤 Print: {node.get('nodeName')} @ ({end_x},{end_y})")
            else:
                # 초기 오프셋을 300으로 늘리고 각 노드 간 간격을 300으로 증가시킵니다.
                position = {"x": start_x, "y": start_y + 310 + (vertical_counter * 350)}
                vertical_counter += 1
                logger.info(f"🔧 Tool (vertical): {node.get('nodeName')} @ ({position['x']},{position['y']})")

            nodes.append({
                "id": node_id,
                "data": node.copy(),
                "position": position,
                "isExpanded": True
            })
        
        logger.info(f"🎯 총 {len(nodes)}개 노드 생성 완료")
        
        # 뷰포트 보정 비활성화: 생성한 노드 좌표를 그대로 사용합니다.
        # 이전 구현: context의 viewport_center에 따라 전체 노드에 오프셋(dx, dy)을 적용하였음.
        # 보정이 필요 없으므로 해당 로직을 제거함.

        # 6. AI를 통한 엣지 생성 (노드 정보 기반 연결 결정)
        edges = call_llm_for_edge_creation(nodes, full_agent_node, model_name, api_base_url, api_key)
        
        logger.info(f"🔗 총 {len(edges)}개 엣지 생성 완료")
        
        # 6-1. Start/End 노드 연결 보장
        edges = ensure_start_end_connections(nodes, edges, full_agent_node)
        
        # 8. 최종 워크플로우 반환
        return {
            "nodes": nodes,
            "edges": edges
        }
        
    except Exception as e:
        import traceback
        logger.error(f"워크플로우 생성 실패: {str(e)}")
        logger.error(f"상세 에러 정보: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"워크플로우 생성 실패: {str(e)}")

def generate_ai_optimized_workflow(user_requirements: str, agent_node: Dict[str, Any], user_id: Optional[str], config_composer, selected_model: Optional[str] = None, provided_workflow_name: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """실제 LLM 모델을 호출하여 워크플로우 생성 (모델 정보 없으면 실패)

    provided_workflow_name: 프론트엔드에서 전달된 워크플로우 이름이 있으면 사용합니다.
    """
    
    # 1. 전체 노드 정보 조회
    try:
        all_nodes = get_node_list(user_id=user_id)
        logger.info(f"전체 노드 카테고리 수: {len(all_nodes)}")
    except Exception as e:
        logger.error(f"노드 정보 조회 실패: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"노드 정보를 조회할 수 없습니다: {str(e)}"
        )
    
    # 3. Agent 노드 타입에 따른 모델 정보 가져오기 (필수)
    try:
        agent_node_id = agent_node.get('id', '')
        logger.info(f"선택된 Agent 노드 ID: {agent_node_id}")
        
        if 'openai' in agent_node_id.lower():
            model_name = selected_model if selected_model else config_composer.get_config_by_name("OPENAI_MODEL_DEFAULT").value
            api_base_url = config_composer.get_config_by_name("OPENAI_API_BASE_URL").value
            api_key = config_composer.get_config_by_name("OPENAI_API_KEY").value
            
            if not model_name or not api_base_url:
                raise ValueError("OpenAI 모델 정보가 설정되지 않았습니다.")
            if not api_key:
                raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
                
            logger.info(f"OpenAI 모델 정보 조회 성공: {model_name} @ {api_base_url}")
            if selected_model:
                logger.info(f"사용자가 선택한 모델 사용: {selected_model}")
            
        else:
            model_name = selected_model if selected_model else config_composer.get_config_by_name("VLLM_MODEL_NAME").value
            api_base_url = config_composer.get_config_by_name("VLLM_API_BASE_URL").value
            api_key = None  # VLLM은 API 키가 필요 없음
            
            if not model_name or not api_base_url:
                raise ValueError("VLLM 모델 정보가 설정되지 않았습니다.")
                
            logger.info(f"VLLM 모델 정보 조회 성공: {model_name} @ {api_base_url}")
            if selected_model:
                logger.info(f"사용자가 선택한 모델 사용: {selected_model}")
        
    except Exception as e:
        logger.error(f"모델 정보 조회 실패: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"모델 정보를 가져올 수 없습니다. 시스템 설정을 확인해주세요. (에러: {str(e)})"
        )
    
    # 4. 1단계: 필요한 노드들 선택
    try:
        node_selection_output = call_llm_for_node_selection(
            user_requirements,
            agent_node,
            all_nodes,
            model_name,
            api_base_url,
            api_key
        )
        
        selected_nodes = parse_node_selection_response(node_selection_output, all_nodes, user_requirements)
    except Exception as e:
        logger.error(f"노드 선택 실패: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"노드 선택 중 오류가 발생했습니다: {str(e)}"
        )
    
    # 필수 노드 확인 및 추가
    selected_nodes = ensure_essential_nodes(selected_nodes, all_nodes, agent_node)
    
    logger.info(f"✅ 1단계 완료: {len(selected_nodes)}개 노드 선택")
    
    # 워크플로우 생성
    workflow = build_workflow_from_structure(agent_node, selected_nodes, all_nodes, model_name, api_base_url, api_key, context)
    
    # Agent 노드에 실제 모델 정보 적용
    for node in workflow.get("nodes", []):
        if node.get("data", {}).get("functionId") == "agents":
            if node["data"].get("parameters"):
                for param in node["data"]["parameters"]:
                    if param.get("id") == "model":
                        param["value"] = model_name
                        logger.info(f"Agent 노드 모델 설정: {model_name}")
                    elif param.get("id") == "base_url":
                        param["value"] = api_base_url
                        logger.info(f"Agent 노드 API URL 설정: {api_base_url}")
                    # OpenAI Agent인 경우 추가 설정이 필요할 수 있음
                    elif 'openai' in agent_node_id.lower() and param.get("id") == "api_key":
                        param["value"] = api_key
                        logger.info(f"Agent 노드 API 키 설정: {'*' * (len(api_key) - 4) + api_key[-4:] if api_key else 'None'}")
    
    logger.info(f"✅ 2단계 완료: 워크플로우 생성 ({len(workflow.get('nodes', []))}개 노드, {len(workflow.get('edges', []))}개 엣지)")
    
    # 6. 워크플로우 메타데이터 추가
    timestamp = str(int(time.time() * 1000))
    # 프론트에서 워크플로우 이름을 제공한 경우 우선 사용, 그렇지 않으면 자동 생성
    if provided_workflow_name and isinstance(provided_workflow_name, str) and provided_workflow_name.strip():
        workflow_name = provided_workflow_name.strip()
        logger.info(f"사용자 지정 워크플로우 이름 사용: {workflow_name}")
    else:
        workflow_name = f"AI_Generated_{agent_node.get('nodeName', 'Workflow').replace(' ', '_')}_{timestamp}"
    
    # 캔버스 컨텍스트에서 뷰포트 정보 가져오기
    canvas_context = context or {}
    current_view = canvas_context.get("current_view", {"x": 0, "y": 0, "scale": 1})
    
    logger.info(f"워크플로우 뷰포트 설정: {current_view}")
    
    workflow_data = {
        "workflow_name": workflow_name,
        "workflow_id": f"workflow_{timestamp}",
        "view": current_view,  # 현재 캔버스의 뷰포트 정보 사용
        "nodes": workflow.get("nodes", []),
        "edges": workflow.get("edges", []),
        "interaction_id": "default"
    }
    
    # 최종 워크플로우 정보 출력
    logger.info("🎉 최종 워크플로우 생성 완료!")
    logger.info(f"📋 워크플로우 이름: {workflow_name}")
    logger.info(f"🔢 노드 수: {len(workflow_data['nodes'])}개")
    logger.info(f"🔗 엣지 수: {len(workflow_data['edges'])}개")
    
    # 노드 목록 요약
    if workflow_data['nodes']:
        node_summary = []
        for node in workflow_data['nodes']:
            node_name = node.get('data', {}).get('nodeName', 'Unknown')
            node_id = node.get('id', 'Unknown')
            node_summary.append(f"{node_name}({node_id.split('-')[0]})")
        logger.info(f"📦 생성된 노드들: {', '.join(node_summary)}")
    
    # 엣지 목록 요약
    if workflow_data['edges']:
        edge_summary = []
        for edge in workflow_data['edges']:
            source = edge.get('source', {}).get('nodeId', 'Unknown')
            target = edge.get('target', {}).get('nodeId', 'Unknown')
            edge_summary.append(f"{source.split('-')[0]}→{target.split('-')[0]}")
        logger.info(f"🔗 생성된 엣지들: {', '.join(edge_summary)}")
    
    # 전체 워크플로우 JSON 출력
    logger.info("=" * 80)
    logger.info("📄 생성된 워크플로우 JSON 전체:")
    logger.info("=" * 80)
    try:
        import json
        workflow_json = json.dumps(workflow_data, ensure_ascii=False, indent=2)
        logger.info(workflow_json)
    except Exception as e:
        logger.error(f"워크플로우 JSON 출력 실패: {str(e)}")
        logger.info(f"워크플로우 데이터: {workflow_data}")
    logger.info("=" * 80)
    
    return {
        "workflow_data": workflow_data,
        "model_info": {
            "model_name": model_name,
            "api_base_url": api_base_url
        }
    }

@router.get("/agent-node-info/{agent_node_id}")
async def get_agent_node_info(request: Request, agent_node_id: str):
    """Agent 노드 정보 조회"""
    try:
        user_id = extract_user_id_from_request(request)
        app_db = get_db_manager(request)
        backend_log = create_logger(app_db, user_id, request)
        
        # 사용 가능한 노드 목록 조회
        available_nodes = get_node_list(user_id=user_id)
        
        # Agent 노드 정보 찾기
        agent_node = find_agent_node_by_id(agent_node_id, available_nodes)
        
        if not agent_node:
            raise HTTPException(status_code=404, detail=f"Agent 노드를 찾을 수 없습니다: {agent_node_id}")
        
        # 호환 가능한 노드들 조회
        compatible_nodes = get_compatible_nodes(agent_node, available_nodes)
        
        backend_log.info("Agent 노드 정보 조회 완료", 
                        metadata={"agent_node_id": agent_node_id, "compatible_nodes_count": len(compatible_nodes)})
        return {
            "success": True,
            "agent_node": agent_node,
            "compatible_nodes": compatible_nodes,
            "compatible_nodes_count": len(compatible_nodes)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent 노드 정보 조회 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agent 노드 정보 조회에 실패했습니다: {str(e)}")

@router.post("/generate", response_model=WorkflowGenerationResponse)
async def generate_workflow_with_ai(request: Request):
    """범용적 백엔드 로직으로 워크플로우 자동 생성

    본 엔드포인트는 `application/json` 또는 `application/x-www-form-urlencoded`
    타입의 POST 요청을 모두 지원합니다. form-data로 전송된 경우 내부에서
    파싱하여 `WorkflowGenerationRequest` 모델로 변환합니다.
    """
    try:
        # 요청 바디를 JSON 또는 form으로 처리
        content_type = (request.headers.get("content-type") or "").lower()
        if "application/x-www-form-urlencoded" in content_type or "form" in content_type:
            form = await request.form()
            # form은 MultiDict 형태이므로 dict로 변환
            form_data = {k: v for k, v in form.items()}
            # context 같은 필드가 JSON 문자열로 전달될 수 있으므로 시도적으로 파싱
            if "context" in form_data and isinstance(form_data["context"], str):
                try:
                    form_data["context"] = json.loads(form_data["context"])
                except Exception:
                    # 파싱 실패하면 문자열 그대로 사용
                    pass
            generation_request = WorkflowGenerationRequest(**form_data)
        else:
            body = await request.json()
            generation_request = WorkflowGenerationRequest(**body)

        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID not found in request")
        
        app_db = get_db_manager(request)
        config_composer = get_config_composer(request)
        
        # 1. Agent 노드 찾기 (노드 정보는 generate_ai_optimized_workflow에서 조회)
        available_nodes = get_node_list(user_id=user_id)
        agent_node = find_agent_node_by_id(generation_request.agent_node_id, available_nodes)
        if not agent_node:
            raise HTTPException(status_code=404, detail=f"Agent 노드를 찾을 수 없습니다: {generation_request.agent_node_id}")
        
        # 2. AI 최적화 워크플로우 생성 (사용자 요구사항 기반)
        # 프론트에서 전달된 workflow_name이 있으면 전달하여 사용하도록 함
        generation_result = generate_ai_optimized_workflow(
            generation_request.user_requirements,
            agent_node,
            user_id,
            config_composer,
            generation_request.selected_model,
            generation_request.workflow_name,
            generation_request.context
        )
        
        generated_workflow = generation_result["workflow_data"]
        model_info = generation_result["model_info"]
        
        # 워크플로우 메타데이터 생성 (저장 없이 로드용)
        node_count = len(generated_workflow["nodes"])
        edge_count = len(generated_workflow["edges"])
        
        return WorkflowGenerationResponse(
            success=True,
            message=f"AI 최적화 워크플로우가 성공적으로 생성되었습니다. (모델: {model_info['model_name']})",
            workflow_data=generated_workflow,
            workflow_id=None,  # 저장하지 않으므로 ID 없음
            workflow_name=generated_workflow.get("workflow_name"),
            generated_nodes_count=node_count,
            generated_edges_count=edge_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"워크플로우 자동생성 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"워크플로우 자동생성에 실패했습니다: {str(e)}")


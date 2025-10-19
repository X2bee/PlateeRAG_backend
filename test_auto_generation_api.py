#!/usr/bin/env python3
"""
워크플로우 자동생성 API 테스트 스크립트
"""
import requests
import json
import sys

# API 기본 URL
BASE_URL = "http://localhost:8000"

def test_agent_node_info():
    """Agent 노드 정보 조회 테스트"""
    print("🔍 Agent 노드 정보 조회 테스트...")
    
    # 사용 가능한 Agent 노드 ID (예시)
    agent_node_id = "agents/vllm_stream"  # 실제 존재하는 Agent 노드 ID로 변경 필요
    
    url = f"{BASE_URL}/api/workflow/auto-generation/agent-node-info/{agent_node_id}"
    
    try:
        response = requests.get(url)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Agent 노드 정보 조회 성공!")
            print(f"Agent 노드: {data.get('agent_node', {}).get('nodeName', 'Unknown')}")
            print(f"호환 가능한 노드 수: {data.get('compatible_nodes_count', 0)}")
            return data.get('agent_node')
        else:
            print(f"❌ Agent 노드 정보 조회 실패: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ 요청 실패: {e}")
        return None

def test_workflow_generation(agent_node_id):
    """워크플로우 자동생성 테스트"""
    print("\n🚀 워크플로우 자동생성 테스트...")
    
    url = f"{BASE_URL}/api/workflow/auto-generation/generate"
    
    payload = {
        "agent_node_id": agent_node_id,
        "user_requirements": "간단한 채팅 봇을 만들어주세요. 사용자 입력을 받아서 AI가 응답하는 워크플로우를 구성해주세요.",
        "workflow_name": "Test_Auto_Generated_ChatBot",
        "context": {
            "purpose": "테스트용 채팅 봇",
            "complexity": "simple"
        }
    }
    
    try:
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ 워크플로우 자동생성 성공!")
            print(f"워크플로우 ID: {data.get('workflow_id')}")
            print(f"워크플로우 이름: {data.get('workflow_name')}")
            print(f"생성된 노드 수: {data.get('generated_nodes_count')}")
            print(f"생성된 엣지 수: {data.get('generated_edges_count')}")
            
            # 생성된 워크플로우 데이터 저장
            if data.get('workflow_data'):
                with open('generated_workflow.json', 'w', encoding='utf-8') as f:
                    json.dump(data['workflow_data'], f, indent=2, ensure_ascii=False)
                print("💾 생성된 워크플로우가 'generated_workflow.json'에 저장되었습니다.")
            
            return data.get('workflow_id')
        else:
            print(f"❌ 워크플로우 자동생성 실패: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ 요청 실패: {e}")
        return None

def test_workflow_load(workflow_id):
    """생성된 워크플로우 로드 테스트"""
    print(f"\n📥 워크플로우 로드 테스트 (ID: {workflow_id})...")
    
    url = f"{BASE_URL}/api/workflow/auto-generation/load/{workflow_id}"
    
    try:
        response = requests.get(url)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ 워크플로우 로드 성공!")
            print(f"워크플로우 이름: {data.get('workflow_data', {}).get('workflow_name', 'Unknown')}")
            print(f"노드 수: {len(data.get('workflow_data', {}).get('nodes', []))}")
            print(f"엣지 수: {len(data.get('workflow_data', {}).get('edges', []))}")
        else:
            print(f"❌ 워크플로우 로드 실패: {response.text}")
            
    except Exception as e:
        print(f"❌ 요청 실패: {e}")

def main():
    """메인 테스트 함수"""
    print("🧪 워크플로우 자동생성 API 테스트 시작")
    print("=" * 50)
    
    # 1. Agent 노드 정보 조회
    agent_node = test_agent_node_info()
    if not agent_node:
        print("❌ Agent 노드 정보 조회 실패로 테스트 중단")
        sys.exit(1)
    
    agent_node_id = agent_node.get('id')
    
    # 2. 워크플로우 자동생성
    workflow_id = test_workflow_generation(agent_node_id)
    if not workflow_id:
        print("❌ 워크플로우 자동생성 실패로 테스트 중단")
        sys.exit(1)
    
    # 3. 생성된 워크플로우 로드
    test_workflow_load(workflow_id)
    
    print("\n" + "=" * 50)
    print("🎉 모든 테스트 완료!")

if __name__ == "__main__":
    main()

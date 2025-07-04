import json
from src.node_composer import (
    run_discovery,
    generate_json_spec,
    NODE_REGISTRY
)

if __name__ == "__main__":
    print("\n\n--- 🚀 실행 테스트 시작 🚀 ---")
    run_discovery()
    print(f"\n✅ Step 4: 노드 등록 결과 확인")
    print(f" -> 총 {len(NODE_REGISTRY)}개의 노드가 레지스트리에 등록되었습니다.")
    print(" -> 등록된 노드 상세 정보:")
    print(json.dumps(NODE_REGISTRY, indent=2, ensure_ascii=False))

    output_filename = "test_export_nodes.json"
    print(f"\n✅ Step 5: '{output_filename}' 파일 생성 시작")
    generate_json_spec(output_path=output_filename) # JSON 생성 함수 호출

    print(f"\n✅ Step 6: 생성된 '{output_filename}' 파일 내용 확인")
    try:
        with open(output_filename, "r", encoding="utf-8") as f:
            generated_json_content = f.read()
            print("--- 파일 내용 시작 ---")
            print(generated_json_content)
            print("--- 파일 내용 종료 ---")
        
        # 테스트 후 생성된 파일 삭제
        # os.remove(output_filename)
    except FileNotFoundError:
        print(f" -> 🔴 에러: JSON 파일 '{output_filename}'이 생성되지 않았습니다.")
    
    print("\n--- ✨ 실행 테스트 성공적으로 종료 ✨ ---")
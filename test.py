from controller.nodeController import export_nodes, list_nodes
import asyncio

# 비동기 함수들을 실행하고 조율하는 메인 비동기 함수
async def main():
    print("작업을 시작합니다.")

    nodes = await list_nodes()
    print(f"조회된 노드 목록: {nodes}") 

    # status = await export_nodes()
    # print(f"내보내기 결과: {status}")

    print("모든 작업이 완료되었습니다.")

# 프로그램을 실행하기 위한 진입점
if __name__ == "__main__":
    # asyncio.run()을 사용해 최상위 비동기 함수인 main()을 실행합니다.
    asyncio.run(main())
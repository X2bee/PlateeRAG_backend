from controller.nodeController import export_nodes, list_nodes
import asyncio
from src.workflow_executor import WorkflowExecutor

# # 비동기 함수들을 실행하고 조율하는 메인 비동기 함수
# async def main():
#     print("작업을 시작합니다.")

#     nodes = await list_nodes()
#     print(f"조회된 노드 목록: {nodes}") 

#     # status = await export_nodes()
#     # print(f"내보내기 결과: {status}")

#     print("모든 작업이 완료되었습니다.")

# # 프로그램을 실행하기 위한 진입점
# if __name__ == "__main__":
#     # asyncio.run()을 사용해 최상위 비동기 함수인 main()을 실행합니다.
#     asyncio.run(main())
    
if __name__ == "__main__":
    mock_workflow_data = {
        "view": {
            "x": -7127.456772387501,
            "y": -3360.42560329336,
            "scale": 0.85309348359375
        },
        "nodes": [
            {
            "id": "math/input_int-1751932725883",
            "data": {
                "id": "math/input_int",
                "nodeName": "Input Integer",
                "inputs": [],
                "outputs": [
                {
                    "id": "result",
                    "name": "Result",
                    "type": "INT"
                }
                ],
                "parameters": [
                {
                    "id": "input_int",
                    "name": "Integer",
                    "type": "INT",
                    "value": 10,
                    "step": 1,
                    "min": -2147483648,
                    "max": 2147483647
                }
                ]
            },
            "position": {
                "x": 8767.876511619374,
                "y": 4119.318597079693
            }
            },
            {
            "id": "math/add_integers-1751932726810",
            "data": {
                "id": "math/add_integers",
                "nodeName": "Add Integers",
                "inputs": [
                {
                    "id": "a",
                    "name": "A",
                    "type": "INT",
                    "multi": False
                },
                {
                    "id": "b",
                    "name": "B",
                    "type": "INT",
                    "multi": False
                }
                ],
                "outputs": [
                {
                    "id": "result",
                    "name": "Result",
                    "type": "INT"
                }
                ],
                "parameters": []
            },
            "position": {
                "x": 9206.971654363102,
                "y": 4277.140920285132
            }
            },
            {
            "id": "math/input_int-1751933108563",
            "data": {
                "id": "math/input_int",
                "nodeName": "Input Integer",
                "inputs": [],
                "outputs": [
                {
                    "id": "result",
                    "name": "Result",
                    "type": "INT"
                }
                ],
                "parameters": [
                {
                    "id": "input_int",
                    "name": "Integer",
                    "type": "INT",
                    "value": 50,
                    "step": 1,
                    "min": -2147483648,
                    "max": 2147483647
                }
                ]
            },
            "position": {
                "x": 8766.274641887663,
                "y": 4389.821485351686
            }
            },
            {
            "id": "tools/print_any-1751935004370",
            "data": {
                "id": "tools/print_any",
                "nodeName": "Print Any",
                "inputs": [
                {
                    "id": "input_print",
                    "name": "Print",
                    "type": "ANY",
                    "multi": False
                }
                ],
                "outputs": [],
                "parameters": []
            },
            "position": {
                "x": 9688.805425600438,
                "y": 4302.489321371075
            }
            }
        ],
        "edges": [
            {
            "id": "edge-math/input_int-1751932725883:result-math/add_integers-1751932726810:a-1751932729452",
            "source": {
                "nodeId": "math/input_int-1751932725883",
                "portId": "result",
                "portType": "output",
                "type": "INT"
            },
            "target": {
                "nodeId": "math/add_integers-1751932726810",
                "portId": "a",
                "portType": "input"
            }
            },
            {
            "id": "edge-math/input_int-1751933108563:result-math/add_integers-1751932726810:b-1751933113352",
            "source": {
                "nodeId": "math/input_int-1751933108563",
                "portId": "result",
                "portType": "output",
                "type": "INT"
            },
            "target": {
                "nodeId": "math/add_integers-1751932726810",
                "portId": "b",
                "portType": "input"
            }
            },
            {
            "id": "edge-math/add_integers-1751932726810:result-tools/print_any-1751935004370:input_print-1751935006931",
            "source": {
                "nodeId": "math/add_integers-1751932726810",
                "portId": "result",
                "portType": "output",
                "type": "INT"
            },
            "target": {
                "nodeId": "tools/print_any-1751935004370",
                "portId": "input_print",
                "portType": "input"
            }
            }
        ]
    }
    print("워크플로우 실행기를 생성합니다.")
    executor = WorkflowExecutor(mock_workflow_data)
    
    final_outputs = executor.execute_workflow()
    
    print("\n최종 노드 출력 데이터:")
    import json
    print(json.dumps(final_outputs, indent=2, ensure_ascii=False))
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
            "id": "math/add_integers-1751944056168",
            "data": {
                "functionId": "arithmetic",
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
                "x": 9142.55813973802,
                "y": 4106.731173862441
            }
            },
            {
            "id": "math/input_int-1751944068980",
            "data": {
                "functionId": "arithmetic",
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
                "x": 8572.86676435361,
                "y": 4037.5711097725625
            }
            },
            {
            "id": "math/input_int-1751944070288",
            "data": {
                "functionId": "arithmetic",
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
                    "value": 15,
                    "step": 1,
                    "min": -2147483648,
                    "max": 2147483647
                }
                ]
            },
            "position": {
                "x": 8615.066125493195,
                "y": 4358.755136223856
            }
            },
            {
            "id": "tools/print_any-1751944087053",
            "data": {
                "functionId": "endnode",
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
                "x": 9713.421719598527,
                "y": 4204.024145378708
            }
            }
        ],
        "edges": [
            {
            "id": "edge-math/input_int-1751944068980:result-math/add_integers-1751944056168:a-1751944077203",
            "source": {
                "nodeId": "math/input_int-1751944068980",
                "portId": "result",
                "portType": "output",
                "type": "INT"
            },
            "target": {
                "nodeId": "math/add_integers-1751944056168",
                "portId": "a",
                "portType": "input"
            }
            },
            {
            "id": "edge-math/input_int-1751944070288:result-math/add_integers-1751944056168:b-1751944078597",
            "source": {
                "nodeId": "math/input_int-1751944070288",
                "portId": "result",
                "portType": "output",
                "type": "INT"
            },
            "target": {
                "nodeId": "math/add_integers-1751944056168",
                "portId": "b",
                "portType": "input"
            }
            },
            {
            "id": "edge-math/add_integers-1751944056168:result-tools/print_any-1751944087053:input_print-1751944088992",
            "source": {
                "nodeId": "math/add_integers-1751944056168",
                "portId": "result",
                "portType": "output",
                "type": "INT"
            },
            "target": {
                "nodeId": "tools/print_any-1751944087053",
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
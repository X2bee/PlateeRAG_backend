from typing import Generator, Any
from editor.node_composer import Node

class PrintAnyStreamNode(Node):
    categoryId = "utilities"
    functionId = "endnode"
    nodeId = "tools/print_any_stream"
    nodeName = "Print Any (Stream)"
    description = "임의의 스트리밍 데이터를 입력받아 그대로 반환하는 스트리밍 출력 노드입니다. 워크플로우의 스트리밍 결과를 확인하는데 사용됩니다."
    tags = ["output", "print", "display", "debug", "end_node", "utility", "stream", "any_type"]

    inputs = [
        {"id": "stream", "name": "Stream", "type": "STREAM STR", "multi": False, "required": True, "stream": True},
    ]

    def execute(self, stream: Generator[Any, None, None]) -> Generator[Any, None, None]:
        """
        입력으로 받은 스트림(generator)의 각 항목을 그대로 yield 합니다.
        WorkflowExecutor는 이 generator를 감지하고 최종 응답으로 스트리밍합니다.
        """
        try:
            for chunk in stream:
                yield chunk
        except Exception as e:
            print(f"\n[STREAMING ERROR] 스트림 처리 중 오류 발생: {e}")
            yield f"스트림 처리 중 오류 발생: {e}"

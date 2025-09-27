import json
from typing import Generator, Any
from editor.node_composer import Node

class PrintAnyNode(Node):
    categoryId = "xgen"
    functionId = "endnode"
    nodeId = "tools/print_any"
    nodeName = "Print Any"
    description = "임의의 타입의 데이터를 입력받아 적절한 형태로 변환하여 반환하는 출력 노드입니다. 객체는 JSON 문자열로, 스트림은 자동으로 스트리밍 형태로 변환됩니다."
    tags = ["output", "print", "display", "debug", "end_node", "utility", "any_type", "stream_compatible"]

    inputs = [
        {"id": "input_print", "name": "Print", "type": "ANY", "multi": False, "required": True},
    ]

    def execute(self, *args, **kwargs):
        """
        입력 데이터를 분석하여 적절한 형태로 변환합니다.
        - 문자열, 숫자, 불린: 그대로 반환
        - 딕셔너리, 리스트, 기타 객체: JSON 문자열로 변환
        - Generator(스트림): 자동으로 스트리밍 형태로 반환
        """
        # kwargs에서 input_print 값 추출
        input_print = kwargs.get('input_print')

        try:
            # Generator 객체 감지 및 처리 - 스트림으로 그대로 반환
            if hasattr(input_print, '__iter__') and hasattr(input_print, '__next__'):
                print(f"[PrintAnyNode] Generator 객체 감지, 스트림으로 처리: {type(input_print)}")
                # generator를 그대로 yield해서 스트리밍 처리
                def stream_generator():
                    try:
                        for chunk in input_print:
                            yield chunk
                    except (StopIteration, GeneratorExit, RuntimeError, ValueError) as stream_error:
                        print(f"[PrintAnyNode] 스트림 처리 중 오류: {stream_error}")
                        yield f"스트림 처리 중 오류 발생: {stream_error}"

                return stream_generator()

            # 기본 타입들은 그대로 반환
            if isinstance(input_print, (str, int, float, bool)) or input_print is None:
                return input_print

            # 딕셔너리, 리스트, 기타 객체들은 JSON 문자열로 변환
            if isinstance(input_print, (dict, list)) or hasattr(input_print, '__dict__'):
                return json.dumps(input_print, ensure_ascii=False, indent=2)

            # 기타 경우에는 문자열로 변환
            return str(input_print)

        except (TypeError, ValueError, json.JSONDecodeError) as e:
            # JSON 변환 실패 시 문자열로 변환
            print(f"[PrintAnyNode] JSON 변환 실패, 문자열로 변환: {e}")
            return str(input_print)

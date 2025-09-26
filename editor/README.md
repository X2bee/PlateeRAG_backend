# Parameter 사용 가능인자 정리

1. id (필수)
해당 파라미터의 id 값. 이것을 기준으로 execute 함수가 값을 인식함

2. name (필수)
해당 파라미터 id의 표시값. 이것을 기준으로 해당 값이 워크플로우 에디터에서 표시됨.

3. type (필수)
해당 파라미터의 type. 이것을 기준으로 파라미터 타입을 검사 및 변환함 (TODO: 해당 기능 확실하게 구현해야 함)

INT 및 Float 설정시 프론트에서 숫자값으로 표시하고, BOOL 설정시 True False Selction으로 작동.

4. value (필수)
해당 파라미터의 기본 value 설정. 그냥 비워두어도 상관없으나, 해당 field는 반드시 존재해야함

5. options (옵션: 기본값 [])
해당 값을 Selection의 형태로 표시할지의 여부
만약 Selection으로 표시하기 위해서는 해당 값을 다음과 같이 입력해야만 함.
    "options": [
        {"value": "gpt-oss-20b", "label": "GPT-OSS-20B"},
        {"value": "gpt-oss-120b", "label": "GPT-OSS-120B"},
        {"value": "gpt-3.5-turbo", "label": "GPT-3.5 Turbo"},
        {"value": "gpt-4", "label": "GPT-4"},
        {"value": "gpt-4o", "label": "GPT-4o"},
        {"value": "gpt-5", "label": "GPT-5"},
        {"value": "gpt-5-mini", "label": "GPT-5 Mini"},
    ]

6. required (옵션: 기본값 False)
해당 파라미터가 필수적인지 여부. True 설정시 해당 값이 반드시 입력되어야 함

7. optional (옵션: 기본값 False)
해당 파라미터가 옵션으로 작동하는지 여부. True 설정시 해당 값은 워크플로우 에디터에서 숨김처리 됨 (확장 버튼 눌러야 나타남)

8. expandable (옵션: 기본값 False)
해당 파라미터를 확장 Modal을 통해 입력 가능하게 할지 여부. True 설정시 해당 값은 워크플로우 에디터에서 확장 모달을 통해 입력할 수 있음. (예: 프롬프트와 같이 긴 문장을 입력해야 하는 경우에 사용)

9. handle_id (옵션-고급: 기본값 False)
해당 파라미터의 id를 워크플로우 에디터 레벨에서 편집가능하게 할지 여부. True 설정 시 해당 값의 id는 에디터에서 편집할 수 있음.

10. is_api (옵션-고급: 기본값 False)
해당 파라미터의 변수를 API를 통해 가져올지의 여부. 이 경우 반드시 해당 Node 내부에 API 함수가 정의되어 있어야 하며, 해당 API 함수의 이름이 api_name 파라미터에 입력되어 있어야 함.

--예시--
아래와 같이 함수가 정의되어 있어야 함.

``` python
def api_collection(self, request: Request) -> Dict[str, Any]:
    collections = sync_run_async(list_collections(request))
    return [{"value": collection.get("collection_name"), "label": collection.get("collection_make_name")} for collection in collections]
```

이후 해당 파라미터를 아래와 같이 사용하는 경우, 해당 api 호출을 통해 값을 가져와서 사용하도록 변경됨.

{"id": "collection_name", "name": "Collection Name", "type": "STR", "value": "Select Collection", "required": True, "is_api": True, "api_name": "api_collection", "options": []},
11. api_name (옵션-고급: 기본값 "")
위 is_api가 true인 경우에만 사용. 해당 API로 가져오게 될 함수를 지정해야 함.

12. description (옵션: 기본값 None)
해당 함수를 설명해주는 값.
해당 값이 존재하면 워크플로우 에디터에서 설명을 표시해 줌.

13. dependency (옵션: 기본값 None)
파라미터 의존성을 설정하면 해당 파라미터가 True일때 활성화됨

14. multi (옵션: 기본값 False)
파라미터가 해당 Port에 여러개 연결될 수 있는지에 대한 여부.

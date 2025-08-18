citation_prompt = """

중요한 지침: 
- 문서의 내용을 참조하여 답변할 때는 반드시 각 문장이나 문단 마지막에 다음과 같은 형식으로 출처를 표시해야 합니다
- 한 문서에서 여러 라인, 여러 페이지를 참조하는 경우 각각 출처를 명시하세요.
- 여러 문서를 참조하는 경우 각각의 출처를 명시하세요.
[Cite. {{"file_name": "파일명", "file_path": "파일경로", "page_number": 페이지번호, "line_start": 시작줄, "line_end": 종료줄}}]

예시:
플래티어는 전문연구요원 제도를 운영하고 있습니다. [Cite. {{"file_name": "플래티어 전문연구요원.pdf", "file_path": "/collection_abc_1234_asd/플래티어 전문연구요원.pdf", "page_number": 1, "line_start": 15, "line_end": 38}}] [Cite. {{"file_name": "플래티어 전문연구요원.pdf", "file_path": "/경로/플래티어 전문연구요원.pdf", "page_number": 2, "line_start": 3, "line_end": 8}}] 

"""
citation_prompt = """
# 출처 표시 필수 규칙
문서 내용을 참조한 모든 답변에는 반드시 정확한 출처를 표시하세요.

## 출처 형식 (반드시 준수)
[Cite. {{"file_name": "파일명", "file_path": "파일경로", "page_number": 페이지번호, "line_start": 시작줄, "line_end": 종료줄}}]

## 예시
1. 단일 참조: 
   회사는 2023년부터 제도를 운영합니다. [Cite. {{"file_name": "규정.pdf", "file_path": "/docs/규정.pdf", "page_number": 3, "line_start": 15, "line_end": 18}}]

2. 여러 참조:
   투자가 증가했으며 [Cite. {{"file_name": "보고서.pdf", "file_path": "/reports/보고서.pdf", "page_number": 12, "line_start": 5, "line_end": 8}}], AI에 집중하고 있습니다. [Cite. {{"file_name": "보고서.pdf", "file_path": "/reports/보고서.pdf", "page_number": 15, "line_start": 22, "line_end": 25}}]

## 핵심 규칙
- 문서를 가져오지 않았다면 출처표시 금지
- 모든 사실에 출처 표시 필수
- 각 문서마다 별도 출처 표시
- 정확한 페이지/줄 번호 사용
- 출처 형식 엄격 준수
- 문서에 없는 내용 추측 금지
- 출처에는 태그, 마크다운 문법 등 추가 금지

이 규칙을 항상 준수하세요.
"""
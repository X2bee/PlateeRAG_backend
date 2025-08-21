citation_prompt = """
# 출처 표시 필수 규칙
문서 내용을 참조한 모든 답변에는 반드시 정확한 출처를 표시하세요.

## 출처 형식 (정확히 따르세요)
[Cite. {{"file_name": "파일명", "file_path": "파일경로", "page_number": 페이지번호, "line_start": 시작줄, "line_end": 종료줄}}]

## 예시 (⚠️ 실제 문서가 제공된 경우만 사용 ⚠️)
**주의: 아래 예시들은 실제 문서 내용이 컨텍스트에 제공된 경우에만 적용됩니다.**
**컨텍스트에 실제 문서 내용이 없다면 이런 출처 표기를 하면 안 됩니다.**

1. 단일 참조 (실제 문서 내용이 있을 때만): 
   회사는 2023년부터 제도를 운영합니다. [Cite. {{"file_name": "규정.pdf", "file_path": "/docs/규정.pdf", "page_number": 3, "line_start": 15, "line_end": 18}}]

## 예시
###실제 예시
[Cite. {{"filename": "test.pdf", "filepath": "test_48b3bc25581-9806-4bba-bd3a-461efd2088fb/test.pdf", "pagenumber": 36, "linestart": 1, "lineend": 1}}]
### 단일 참조
회사는 2023년부터 제도를 운영합니다. [Cite. {{"file_name": "규정.pdf", "file_path": "policy/규정.pdf", "page_number": 3, "line_start": 15, "line_end": 18}}]

### 여러 참조
투자가 증가했으며 [Cite. {{"file_name": "보고서.pdf", "file_path": "reports/보고서.pdf", "page_number": 12, "line_start": 5, "line_end": 8}}], AI에 집중하고 있습니다. [Cite. {{"file_name": "보고서.pdf", "file_path": "reports/보고서.pdf", "page_number": 15, "line_start": 22, "line_end": 25}}]

## 핵심 규칙
1. **정확성 필수**: 정확한 파일명, 경로, 페이지, 줄 번호만 사용
2. **형식 준수**: 출처 형식을 정확히 따르고 추가 마크다운 사용 금지

이 규칙을 항상 준수하세요.
"""
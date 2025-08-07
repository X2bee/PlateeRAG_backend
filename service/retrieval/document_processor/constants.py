"""
Document Processor 상수 및 설정
"""

from langchain_text_splitters import Language

# Language mapping for code splitting
LANGCHAIN_CODE_LANGUAGE_MAP = {
    'py': Language.PYTHON, 'js': Language.JS, 'ts': Language.TS,
    'java': Language.JAVA, 'cpp': Language.CPP, 'c': Language.CPP,
    'cs': Language.CSHARP, 'go': Language.GO, 'rs': Language.RUST,
    'php': Language.PHP, 'rb': Language.RUBY, 'swift': Language.SWIFT,
    'kt': Language.KOTLIN, 'scala': Language.SCALA,
    'html': Language.HTML, 'jsx': Language.JS, 'tsx': Language.TS,
}

# File type categories
DOCUMENT_TYPES = ['pdf', 'docx', 'doc', 'pptx', 'ppt']
TEXT_TYPES = ['txt', 'md', 'markdown', 'rtf']
CODE_TYPES = ['py','js','ts','java','cpp','c','h','cs','go','rs',
              'php','rb','swift','kt','scala','dart','r','sql',
              'html','css','jsx','tsx','vue','svelte']
CONFIG_TYPES = ['json','yaml','yml','xml','toml','ini','cfg','conf','properties','env']
DATA_TYPES = ['csv','tsv','xlsx','xls']
SCRIPT_TYPES = ['sh','bat','ps1','zsh','fish']
LOG_TYPES = ['log']
WEB_TYPES = ['htm','xhtml']
IMAGE_TYPES = ['jpg','jpeg','png','gif','bmp','webp']

# Encoding options for text files
ENCODINGS = ['utf-8','utf-8-sig','cp949','euc-kr','latin-1','ascii']

# OCR prompts
OCR_SINGLE_PROMPT = """이 이미지를 정확한 텍스트로 변환해주세요. 다음 규칙을 철저히 지켜주세요:

1. **표 구조 보존**: 표가 있다면 정확한 행과 열 구조를 유지하고, 마크다운 표 형식으로 변환해주세요
2. **레이아웃 유지**: 원본의 레이아웃, 들여쓰기, 줄바꿈을 최대한 보존해주세요
3. **정확한 텍스트**: 모든 문자, 숫자, 기호를 정확히 인식해주세요
4. **구조 정보**: 제목, 부제목, 목록, 단락 구분을 명확히 표현해주세요
5. **특수 형식**: 날짜, 금액, 주소, 전화번호 등의 형식을 정확히 유지해주세요
6. **슬라이드 구조**: 슬라이드 제목, 내용, 차트/그래프 설명을 구분해주세요
7. **섹션 구분**: MarkDown 형식을 철저히 준수해서 섹션 구분을 철저히 나눠주세요.
8. **표 구분**: 각 이미지 내에서 표는 [표 구분]으로 명확히 구분

섹션 구분은 다음과 같이 해주세요 

섹션 1 
[색션 구분]
색션 2 
[표 구분]
표 1
...

**섹션 구분 예시:**
- 서로 다른 주제나 단락 사이  
- 표와 다른 내용 사이
- 각 표는 하나의 [표 구분]을 이룹니다.
단, 표와 설명 텍스는 같은 [표 구분]으로 구분해줘

만약 표가 있다면 다음과 같은 마크다운 형식으로 변환해주세요:
| 항목 | 내용 |
|------|------|
| 데이터1 | 값1 |
| 데이터2 | 값2 |

텍스트만 출력하고, 추가 설명은 하지 마세요."""

def get_batch_ocr_prompt(image_count: int) -> str:
    """배치 OCR용 프롬프트 생성"""
    return f"""다음 {image_count}개의 이미지를 각각 정확한 텍스트로 변환해주세요. 

            **중요한 규칙:**
            1. 각 이미지의 결과를 명확히 구분해주세요
            2. 다음 형식으로 응답해주세요:

            === 이미지 1 ===
            [첫 번째 이미지의 텍스트 내용]

            === 이미지 2 ===
            [두 번째 이미지의 텍스트 내용]

            === 이미지 3 ===
            [세 번째 이미지의 텍스트 내용]

            **변환 규칙:**
            - 표가 있다면 마크다운 표 형식으로 변환
            - 원본의 레이아웃, 들여쓰기, 줄바꿈 보존
            - 모든 문자, 숫자, 기호를 정확히 인식
            - 제목, 부제목, 목록, 단락 구분을 명확히 표현
            - 특수 형식(날짜, 금액 등) 정확히 유지
            - **섹션 구분**: 각 이미지 내에서 문맥적으로 다른 내용 섹션들은 [색션 구분] 으로 명확히 구분
            - **표 구분**: 각 이미지 내에서 표는 [표 구분]으로 명확히 구분

            **섹션 구분 예시:**
            - 서로 다른 주제나 단락 사이 
            - 표와 다른 내용 사이
            - 각 표는 하나의 [표 구분]을 이룹니다.
            단, 표와 설명 텍스는 같은 [표 구분]으로 구분해줘

            **출력 형식 예시:**
            === 이미지 1 ===
            # 제목
            [색션 구분]
            본문 내용이 여기에...

            [색션 구분]
            ## 소제목
            다른 섹션의 내용...

            [표 구분]
            | 표 데이터 |
            |----------|
            | 내용     |

            텍스트만 출력하고, 추가 설명은 하지 마세요."""
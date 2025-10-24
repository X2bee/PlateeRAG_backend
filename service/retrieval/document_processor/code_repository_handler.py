import os
import asyncio
import requests
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import logging
import urllib.parse

logger = logging.getLogger(__name__)

class CodeRepositoryHandler:
    """GitLab 레포지토리 코드를 처리하는 핸들러"""

    # 지원하는 코드 파일 확장자 (Code Assistant와 동일)
    VALID_EXTENSIONS = [
        '.py', '.java', '.html', '.xml', '.yaml', '.yml',
        '.js', '.ts', '.jsx', '.tsx', '.css', '.scss', '.sass',
        '.md', '.markdown', '.json', '.c', '.cpp', '.h', '.hpp',
        '.go', '.rs', '.php', '.rb', '.pl', '.pm', '.lua',
        '.kt', '.swift', '.sql'
    ]

    # 제외할 파일 및 확장자
    EXCLUDED_FILES = [
        'repository_tree.yml', 'requirements.txt',
        'Dockerfile', '.dockerignore', 'Makefile', '.env'
    ]
    EXCLUDED_EXTENSIONS = ['.sh', '.bash', '.bat', '.cmd', '.ini', '.toml']

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}

    async def extract_text_from_repository(
        self,
        gitlab_url: str,
        gitlab_token: str,
        repository_path: str,
        branch: str = "main",
        enable_annotation: bool = False,
        enable_api_extraction: bool = False,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        GitLab 레포지토리에서 코드를 추출하고 처리

        Args:
            gitlab_url: GitLab 인스턴스 URL
            gitlab_token: Personal Access Token
            repository_path: 레포지토리 경로 (예: group/project)
            branch: 브랜치 이름
            enable_annotation: LLM 기반 코드 주석 생성 여부
            enable_api_extraction: API 엔드포인트 추출 여부

        Returns:
            {
                'files': [{
                    'path': str,
                    'content': str,
                    'annotated_content': str (optional),
                    'language': str,
                    'size': int
                }],
                'api_info': [...] (optional),
                'metadata': {...}
            }
        """

        logger.info(f"Starting repository extraction: {repository_path} (branch: {branch})")

        # 진행 상태 업데이트 헬퍼
        def update_progress(status, progress, step, step_num, message, **kwargs):
            if progress_callback:
                progress_callback(
                    status=status,
                    progress=progress,
                    current_step=step,
                    current_step_number=step_num,
                    message=message,
                    **kwargs
                )

        # 1. 레포지토리 파일 트리 가져오기
        update_progress('downloading', 10, 'Fetching repository tree...', 1, 'Retrieving file list from GitLab')

        file_tree = await self._get_repository_tree(
            gitlab_url, gitlab_token, repository_path, branch
        )

        logger.info(f"Retrieved {len(file_tree)} items from repository tree")

        # 2. 코드 파일만 필터링
        update_progress('downloading', 15, 'Filtering code files...', 2, f'Found {len(file_tree)} items')

        code_files = self._filter_code_files(file_tree)

        logger.info(f"Filtered to {len(code_files)} code files")
        update_progress('downloading', 20, 'Downloading files...', 2, f'Downloading {len(code_files)} code files',
                       total_files=len(code_files))

        # 3. 파일 내용 다운로드
        files_data = await self._download_files(
            gitlab_url, gitlab_token, repository_path,
            branch, code_files, progress_callback=lambda idx, total, fname: update_progress(
                'downloading', 20 + (30 * idx / total), f'Downloading files... ({idx}/{total})', 2,
                f'Downloading {fname}', processed_files=idx, total_files=total, current_file=fname
            )
        )

        logger.info(f"Downloaded {len(files_data)} files successfully")
        update_progress('downloading', 50, 'Download complete', 2, f'Downloaded {len(files_data)} files successfully',
                       processed_files=len(files_data), total_files=len(code_files))

        result = {
            'files': files_data,
            'metadata': {
                'repository': repository_path,
                'branch': branch,
                'total_files': len(files_data),
                'gitlab_url': gitlab_url
            }
        }

        # 4. (선택) 코드 어노테이션
        if enable_annotation:
            logger.info("Starting code annotation with LLM...")
            update_progress('annotating', 55, 'Annotating code with LLM...', 3, 'Adding comments to code files')
            try:
                result['files'] = await self._annotate_code_files(files_data)
                logger.info("Code annotation completed")
                update_progress('annotating', 75, 'Annotation complete', 3, 'Code annotation completed')
            except Exception as e:
                logger.error(f"Code annotation failed: {e}")
                update_progress('annotating', 75, 'Annotation skipped', 3, f'Annotation failed: {str(e)}')

        # 5. (선택) API 정보 추출
        if enable_api_extraction:
            logger.info("Extracting API information...")
            update_progress('extracting', 80, 'Extracting API endpoints...', 4, 'Detecting API patterns')
            try:
                result['api_info'] = await self._extract_api_info(files_data)
                logger.info(f"Extracted {len(result.get('api_info', []))} API endpoints")
                update_progress('extracting', 90, 'API extraction complete', 4,
                               f"Extracted {len(result.get('api_info', []))} API endpoints")
            except Exception as e:
                logger.error(f"API extraction failed: {e}")
                update_progress('extracting', 90, 'API extraction skipped', 4, f'Extraction failed: {str(e)}')

        update_progress('completed', 100, 'Processing complete', 7, 'Repository processing completed successfully')
        return result

    async def _get_repository_tree(
        self,
        gitlab_url: str,
        token: str,
        repo_path: str,
        ref: str
    ) -> List[Dict]:
        """GitLab API로 파일 트리 가져오기"""

        encoded_path = urllib.parse.quote(repo_path, safe='')
        api_url = f"{gitlab_url}/api/v4/projects/{encoded_path}/repository/tree"

        headers = {"PRIVATE-TOKEN": token}
        params = {"ref": ref, "recursive": True, "per_page": 100}

        all_items = []
        page = 1
        max_pages = 100  # 안전 장치

        while page <= max_pages:
            params['page'] = page

            try:
                response = requests.get(api_url, headers=headers, params=params, timeout=30)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to fetch repository tree (page {page}): {e}")
                break

            items = response.json()
            if not items:
                break

            all_items.extend(items)

            # 페이지네이션 체크
            if 'x-next-page' not in response.headers or not response.headers['x-next-page']:
                break

            page += 1

        return all_items

    def _filter_code_files(self, file_tree: List[Dict]) -> List[Dict]:
        """코드 파일만 필터링"""
        code_files = []

        for item in file_tree:
            if item.get('type') != 'blob':  # 디렉토리는 스킵
                continue

            path = item.get('path', '')
            file_name = os.path.basename(path)
            ext = os.path.splitext(path)[1]

            # 제외 조건 체크
            if file_name in self.EXCLUDED_FILES:
                continue
            if ext in self.EXCLUDED_EXTENSIONS:
                continue
            if '/target/' in path or path.startswith('target/'):  # Maven 빌드 디렉토리
                continue
            if '/node_modules/' in path or path.startswith('node_modules/'):  # NPM 패키지
                continue
            if '/.git/' in path or path.startswith('.git/'):  # Git 내부
                continue

            # 유효한 확장자만 포함
            if ext in self.VALID_EXTENSIONS:
                code_files.append(item)

        return code_files

    async def _download_files(
        self,
        gitlab_url: str,
        token: str,
        repo_path: str,
        ref: str,
        file_list: List[Dict],
        progress_callback: Optional[Callable] = None
    ) -> List[Dict]:
        """파일 내용 다운로드"""

        encoded_path = urllib.parse.quote(repo_path, safe='')
        files_data = []
        total_files = len(file_list)

        for idx, file_info in enumerate(file_list):
            file_path = file_info.get('path', '')
            if not file_path:
                continue

            encoded_file_path = urllib.parse.quote(file_path, safe='')

            api_url = f"{gitlab_url}/api/v4/projects/{encoded_path}/repository/files/{encoded_file_path}/raw"
            headers = {"PRIVATE-TOKEN": token}
            params = {"ref": ref}

            try:
                response = requests.get(api_url, headers=headers, params=params, timeout=30)
                response.raise_for_status()

                # 인코딩 처리
                content = self._decode_content(response.content)

                files_data.append({
                    'path': file_path,
                    'content': content,
                    'language': self._detect_language(file_path),
                    'size': len(content)
                })

                # 진행 상태 업데이트
                if progress_callback:
                    progress_callback(idx + 1, total_files, file_path)

            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to download {file_path}: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error downloading {file_path}: {e}")
                continue

        return files_data

    def _decode_content(self, content_bytes: bytes) -> str:
        """다중 인코딩 시도"""
        encodings = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr', 'latin-1']

        for encoding in encodings:
            try:
                return content_bytes.decode(encoding)
            except (UnicodeDecodeError, AttributeError):
                continue

        # 최후의 수단
        return content_bytes.decode('utf-8', errors='ignore')

    def _detect_language(self, file_path: str) -> str:
        """파일 확장자로 언어 감지"""
        ext_to_lang = {
            '.py': 'python',
            '.java': 'java',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'jsx',
            '.tsx': 'tsx',
            '.go': 'go',
            '.rs': 'rust',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.php': 'php',
            '.rb': 'ruby',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.dart': 'dart',
            '.r': 'r',
            '.sql': 'sql',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.json': 'json',
            '.xml': 'xml',
            '.html': 'html',
            '.css': 'css',
            '.scss': 'scss',
            '.sass': 'sass',
            '.md': 'markdown',
            '.markdown': 'markdown'
        }

        ext = os.path.splitext(file_path)[1]
        return ext_to_lang.get(ext, 'text')

    async def _annotate_code_files(
        self,
        files_data: List[Dict]
    ) -> List[Dict]:
        """LLM으로 코드 주석 생성 (Code Assistant 방식)"""
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain_openai import ChatOpenAI
        except ImportError:
            logger.warning("langchain not available for code annotation")
            return files_data

        # 텍스트 스플리터 설정
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=200,
            length_function=len
        )

        # LLM 설정
        llm = ChatOpenAI(
            model='gpt-4o-mini',
            temperature=0.2,
            max_tokens=4000
        )

        annotated_files = []

        for idx, file_data in enumerate(files_data):
            try:
                content = file_data['content']
                file_path = file_data['path']
                language = file_data['language']

                logger.info(f"Annotating {idx+1}/{len(files_data)}: {file_path}")

                # 청크로 분할
                chunks = text_splitter.split_text(content)
                annotated_chunks = []

                for chunk in chunks:
                    prompt = f"""다음 {language} 코드에 주석을 달아주세요.
파일 경로: {file_path}

각 함수, 클래스, 그리고 주요 로직에 대한 설명을 추가하되, 코드 구조는 유지하세요.

코드:
```{language}
{chunk}
```

주석이 추가된 코드를 반환해주세요."""

                    response = await llm.ainvoke(prompt)
                    annotated_chunks.append(response.content)

                file_data['annotated_content'] = '\n'.join(annotated_chunks)
                annotated_files.append(file_data)

            except Exception as e:
                logger.error(f"Annotation failed for {file_data['path']}: {e}")
                annotated_files.append(file_data)  # 원본 유지

        return annotated_files

    async def _extract_api_info(
        self,
        files_data: List[Dict]
    ) -> List[Dict]:
        """API 엔드포인트 정보 추출 (간단한 정규식 기반)"""
        import re

        api_info = []

        # Spring Boot 어노테이션 패턴
        spring_patterns = [
            (r'@GetMapping\s*\(\s*["\']([^"\']+)["\']', 'GET'),
            (r'@PostMapping\s*\(\s*["\']([^"\']+)["\']', 'POST'),
            (r'@PutMapping\s*\(\s*["\']([^"\']+)["\']', 'PUT'),
            (r'@DeleteMapping\s*\(\s*["\']([^"\']+)["\']', 'DELETE'),
            (r'@PatchMapping\s*\(\s*["\']([^"\']+)["\']', 'PATCH'),
            (r'@RequestMapping\s*\([^)]*value\s*=\s*["\']([^"\']+)["\']', 'REQUEST')
        ]

        # Express.js 패턴
        express_patterns = [
            (r'router\.get\s*\(\s*["\']([^"\']+)["\']', 'GET'),
            (r'router\.post\s*\(\s*["\']([^"\']+)["\']', 'POST'),
            (r'router\.put\s*\(\s*["\']([^"\']+)["\']', 'PUT'),
            (r'router\.delete\s*\(\s*["\']([^"\']+)["\']', 'DELETE'),
            (r'app\.get\s*\(\s*["\']([^"\']+)["\']', 'GET'),
            (r'app\.post\s*\(\s*["\']([^"\']+)["\']', 'POST'),
        ]

        # FastAPI/Flask 패턴
        python_patterns = [
            (r'@app\.get\s*\(\s*["\']([^"\']+)["\']', 'GET'),
            (r'@app\.post\s*\(\s*["\']([^"\']+)["\']', 'POST'),
            (r'@app\.put\s*\(\s*["\']([^"\']+)["\']', 'PUT'),
            (r'@app\.delete\s*\(\s*["\']([^"\']+)["\']', 'DELETE'),
            (r'@router\.get\s*\(\s*["\']([^"\']+)["\']', 'GET'),
            (r'@router\.post\s*\(\s*["\']([^"\']+)["\']', 'POST'),
        ]

        for file_data in files_data:
            language = file_data['language']
            content = file_data['content']

            # 언어별 패턴 선택
            if language == 'java':
                patterns = spring_patterns
            elif language in ['javascript', 'typescript']:
                patterns = express_patterns
            elif language == 'python':
                patterns = python_patterns
            else:
                continue

            for pattern, method in patterns:
                matches = re.finditer(pattern, content, re.MULTILINE)
                for match in matches:
                    api_info.append({
                        'file': file_data['path'],
                        'method': method,
                        'path': match.group(1),
                        'language': language
                    })

        return api_info


async def extract_text_from_code_repository(
    gitlab_url: str,
    gitlab_token: str,
    repository_path: str,
    branch: str = "main",
    config: Dict[str, Any] = None,
    enable_annotation: bool = False,
    enable_api_extraction: bool = False,
    progress_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    코드 레포지토리에서 텍스트 추출

    Args:
        gitlab_url: GitLab 인스턴스 URL
        gitlab_token: Personal Access Token
        repository_path: 레포지토리 경로 (예: group/project)
        branch: 브랜치 이름
        config: 설정 딕셔너리
        enable_annotation: LLM 기반 코드 주석 생성 여부
        enable_api_extraction: API 엔드포인트 추출 여부
        progress_callback: 진행 상황 콜백 함수 (Optional[Callable])

    Returns:
        {
            'files': [{
                'path': str,
                'content': str,
                'annotated_content': str (optional),
                'language': str,
                'size': int
            }],
            'api_info': [...] (optional),
            'metadata': {...}
        }
    """
    handler = CodeRepositoryHandler(config or {})

    result = await handler.extract_text_from_repository(
        gitlab_url=gitlab_url,
        gitlab_token=gitlab_token,
        repository_path=repository_path,
        branch=branch,
        enable_annotation=enable_annotation,
        enable_api_extraction=enable_api_extraction,
        progress_callback=progress_callback
    )

    # Return the dict directly instead of converting to string
    return result

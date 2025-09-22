"""
Data Manager Helper Functions - 심플 버전
"""

import os
import shutil
import logging
from typing import Dict, List, Any, IO, Tuple
from datetime import datetime
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.csv as csv

logger = logging.getLogger(__name__)


def save_and_load_files(uploaded_files: List[IO], filenames: List[str], manager_id: str) -> pa.Table:
    """
    업로드된 파일들을 저장하고 하나의 테이블로 합쳐서 로드

    Args:
        uploaded_files: 업로드된 파일 객체들
        filenames: 파일명들
        manager_id: 매니저 ID

    Returns:
        pa.Table: 합쳐진 테이블
    """
    # 파일 형식 검증 - 모든 파일이 같은 형식이어야 함
    file_extensions = [filename.split('.')[-1].lower() for filename in filenames]
    unique_extensions = set(file_extensions)

    if len(unique_extensions) > 1:
        raise RuntimeError(f"모든 파일이 같은 형식이어야 합니다. 업로드된 형식: {list(unique_extensions)}")

    file_type = file_extensions[0]
    if file_type not in ['parquet', 'csv']:
        raise RuntimeError(f"지원되지 않는 파일 형식: {file_type}")

    # 저장 경로
    dataset_id = f"dataset_{manager_id}_{int(datetime.now().timestamp())}"
    save_dir = f"/dataset_local_upload/{dataset_id}"
    os.makedirs(save_dir, exist_ok=True)

    tables = []

    for uploaded_file, filename in zip(uploaded_files, filenames):
        # 파일 저장
        file_path = os.path.join(save_dir, filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(uploaded_file, buffer)

        logger.info("파일 저장: %s", file_path)

        # 파일 읽기
        if file_type == 'parquet':
            table = pq.read_table(file_path)
        else:  # csv
            table = csv.read_csv(file_path)

        tables.append(table)
        logger.info("파일 로드: %s (%d행, %d열)", filename, table.num_rows, table.num_columns)

    # 테이블 합치기
    if len(tables) == 1:
        combined_table = tables[0]
    else:
        combined_table = pa.concat_tables(tables)
        logger.info("테이블 %d개 합침: 총 %d행 (형식: %s)", len(tables), combined_table.num_rows, file_type)

    return combined_table, dataset_id


# ========== HuggingFace 전용 함수들 ==========

def classify_dataset_files(repo_files: List[str], split: str = None) -> Tuple[List[str], str]:
    """HF 리포지토리 파일들을 분류"""
    parquet_files = [f for f in repo_files if f.endswith('.parquet')]
    csv_files = [f for f in repo_files if f.endswith('.csv')]

    if parquet_files:
        target_files = parquet_files
        if split:
            split_files = [f for f in parquet_files if split in f.lower()]
            target_files = split_files if split_files else parquet_files
        return target_files, 'parquet'
    elif csv_files:
        target_files = csv_files
        if split:
            split_files = [f for f in csv_files if split in f.lower()]
            target_files = split_files if split_files else csv_files
        return target_files, 'csv'
    else:
        raise RuntimeError("지원되는 데이터 파일이 없습니다")


def download_and_read_file(repo_id: str, filename: str, file_type: str, cache_dir: str) -> Tuple[pa.Table, str]:
    """HF에서 파일 다운로드 및 읽기"""
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type='dataset',
        cache_dir=cache_dir,
    )

    if file_type == 'parquet':
        table = pq.read_table(downloaded_path)
    else:
        table = csv.read_csv(downloaded_path)

    return table, downloaded_path


def process_multiple_files(repo_id: str, target_files: List[str], file_type: str, cache_dir: str) -> Tuple[List[pa.Table], List[str]]:
    """여러 HF 파일들 처리"""
    tables = []
    paths = []

    for filename in target_files:
        table, path = download_and_read_file(repo_id, filename, file_type, cache_dir)
        tables.append(table)
        paths.append(path)

    return tables, paths


def combine_tables(tables: List[pa.Table], target_files: List[str] = None) -> pa.Table:
    """테이블들 합치기"""
    if len(tables) == 1:
        return tables[0]
    return pa.concat_tables(tables)


def determine_file_type_from_filename(filename: str) -> str:
    """파일 타입 판단"""
    if filename.endswith('.parquet'):
        return 'parquet'
    elif filename.endswith('.csv'):
        return 'csv'
    else:
        raise RuntimeError(f"지원되지 않는 파일 형식: {filename}")


def create_result_info(repo_id: str, file_type: str, combined_table: pa.Table, **kwargs) -> Dict[str, Any]:
    """HF 결과 정보 생성"""
    return {
        "success": True,
        "repo_id": repo_id,
        "file_type": file_type,
        "num_rows": combined_table.num_rows,
        "num_columns": combined_table.num_columns,
        "columns": combined_table.column_names,
        "loaded_at": datetime.now().isoformat(),
        **kwargs
    }

"""
Data Manager Helper Functions - 심플 버전
"""

import os
import shutil
import logging
from typing import Dict, List, Any, IO, Tuple
from datetime import datetime
from huggingface_hub import hf_hub_download, create_repo, upload_file
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.compute as pc
import difflib
import re
import ast
import tempfile
import subprocess
import sys

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
    save_dir = f"/plateerag_backend/downloads/dataset_local_upload/{dataset_id}"
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


def generate_dataset_statistics(table: pa.Table) -> Dict[str, Any]:
    """
    pyarrow Table의 기술통계정보 생성

    Args:
        table (pa.Table): 분석할 테이블

    Returns:
        Dict[str, Any]: 기술통계정보
    """
    if table is None or table.num_rows == 0:
        return {
            "success": False,
            "message": "빈 데이터셋",
            "statistics": {}
        }

    try:
        stats = {
            "dataset_info": {
                "total_rows": table.num_rows,
                "total_columns": table.num_columns,
                "column_names": table.column_names
            },
            "column_statistics": {}
        }

        # 각 컬럼별 통계
        for col_name in table.column_names:
            column = table.column(col_name)

            # null count 계산
            null_count = pc.sum(pc.is_null(column)).as_py()
            non_null_count = table.num_rows - null_count

            col_stats = {
                "data_type": str(column.type),
                "null_count": null_count,
                "non_null_count": non_null_count,
                "unique_count": None
            }

            # null 비율
            col_stats["null_percentage"] = (null_count / table.num_rows * 100) if table.num_rows > 0 else 0

            # unique count 계산 (메모리 효율성을 위해 try-except 사용)
            try:
                unique_values = pc.unique(column)
                unique_count = len(unique_values)
                col_stats["unique_count"] = unique_count

                # unique value가 30개 이하면 개별 값 카운트 제공
                if unique_count <= 30:
                    try:
                        # value_counts 계산
                        unique_dict = {}
                        for unique_val in unique_values:
                            if unique_val.is_valid:
                                val = unique_val.as_py()
                                count = pc.sum(pc.equal(column, val)).as_py()
                                unique_dict[str(val)] = count
                        col_stats["unique_dict"] = unique_dict
                    except Exception:
                        col_stats["unique_dict"] = "계산불가"

            except Exception:
                col_stats["unique_count"] = "계산불가"

            # 숫자형 데이터 타입 체크 및 추가 통계
            data_type = str(column.type)
            if any(num_type in data_type.lower() for num_type in ['int', 'float', 'double', 'decimal']):
                try:
                    col_stats["min"] = pc.min(column).as_py()
                    col_stats["max"] = pc.max(column).as_py()
                    col_stats["mean"] = pc.mean(column).as_py()

                    # Q1, Q3 분위수 계산
                    try:
                        q1_result = pc.quantile(column, q=0.25)
                        q3_result = pc.quantile(column, q=0.75)

                        # 결과가 스칼라면 as_py(), 배열이면 첫 번째 요소 사용
                        if hasattr(q1_result, 'as_py'):
                            col_stats["q1"] = q1_result.as_py()
                        else:
                            col_stats["q1"] = q1_result[0].as_py() if len(q1_result) > 0 else "계산불가"

                        if hasattr(q3_result, 'as_py'):
                            col_stats["q3"] = q3_result.as_py()
                        else:
                            col_stats["q3"] = q3_result[0].as_py() if len(q3_result) > 0 else "계산불가"

                    except Exception as e:
                        logger.warning("분위수 계산 실패 (%s): %s", col_name, e)
                        col_stats["q1"] = "계산불가"
                        col_stats["q3"] = "계산불가"

                except Exception as e:
                    logger.warning("숫자형 통계 계산 실패 (%s): %s", col_name, e)
                    pass

            stats["column_statistics"][col_name] = col_stats

        return {
            "success": True,
            "statistics": stats,
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error("통계 생성 실패: %s", e)
        return {
            "success": False,
            "message": f"통계 생성 중 오류: {str(e)}",
            "statistics": {}
        }


def drop_columns_from_table(table: pa.Table, columns_to_drop: List[str]) -> Tuple[pa.Table, Dict[str, Any]]:
    """
    pyarrow Table에서 지정된 컬럼들을 삭제

    Args:
        table (pa.Table): 원본 테이블
        columns_to_drop (List[str]): 삭제할 컬럼명들

    Returns:
        Tuple[pa.Table, Dict[str, Any]]: (수정된 테이블, 결과 정보)
    """
    if table is None:
        raise RuntimeError("테이블이 None입니다")

    if not columns_to_drop:
        raise RuntimeError("삭제할 컬럼이 지정되지 않았습니다")

    existing_columns = set(table.column_names)
    columns_to_drop_set = set(columns_to_drop)

    # 존재하지 않는 컬럼 확인
    missing_columns = columns_to_drop_set - existing_columns

    if missing_columns:
        # 유사한 컬럼명 찾기
        suggestions = {}
        for missing_col in missing_columns:
            # difflib를 사용해 가장 유사한 컬럼명 찾기
            close_matches = difflib.get_close_matches(
                missing_col,
                existing_columns,
                n=3,  # 최대 3개까지
                cutoff=0.3  # 30% 이상 유사도
            )
            if close_matches:
                suggestions[missing_col] = close_matches

        # 오류 메시지 생성
        error_msg = f"존재하지 않는 컬럼: {list(missing_columns)}"
        if suggestions:
            error_msg += "\n혹시 이런 컬럼들을 의도하셨나요?"
            for missing_col, similar_cols in suggestions.items():
                error_msg += f"\n- '{missing_col}' → {similar_cols}"

        raise RuntimeError(error_msg)

    # 남겨둘 컬럼들 결정
    columns_to_keep = [col for col in table.column_names if col not in columns_to_drop_set]

    if not columns_to_keep:
        raise RuntimeError("모든 컬럼을 삭제할 수 없습니다. 최소 1개의 컬럼은 남겨두어야 합니다")

    try:
        # 컬럼 선택으로 새 테이블 생성
        new_table = table.select(columns_to_keep)

        result_info = {
            "success": True,
            "dropped_columns": list(columns_to_drop),
            "remaining_columns": columns_to_keep,
            "original_columns_count": table.num_columns,
            "new_columns_count": new_table.num_columns,
            "rows_count": new_table.num_rows,
            "dropped_at": datetime.now().isoformat()
        }

        logger.info("컬럼 삭제 완료: %d개 컬럼 삭제, %d개 컬럼 남음",
                   len(columns_to_drop), len(columns_to_keep))

        return new_table, result_info

    except Exception as e:
        logger.error("컬럼 삭제 실패: %s", e)
        raise RuntimeError(f"컬럼 삭제 중 오류 발생: {str(e)}")


def replace_column_values(table: pa.Table, column_name: str, old_value: str, new_value: str) -> Tuple[pa.Table, Dict[str, Any]]:
    """
    특정 컬럼에서 문자열 값을 다른 값으로 교체

    Args:
        table (pa.Table): 원본 테이블
        column_name (str): 대상 컬럼명
        old_value (str): 교체할 기존 값
        new_value (str): 새로운 값

    Returns:
        Tuple[pa.Table, Dict[str, Any]]: (수정된 테이블, 결과 정보)
    """
    if table is None:
        raise RuntimeError("테이블이 None입니다")

    if column_name not in table.column_names:
        raise RuntimeError(f"컬럼 '{column_name}'이 존재하지 않습니다")

    try:
        # 기존 컬럼 가져오기
        column = table.column(column_name)

        # 문자열로 변환하여 교체 수행
        str_column = pc.cast(column, pa.string())
        replaced_column = pc.replace_substring_regex(str_column, old_value, new_value)

        # 원래 타입으로 복원 시도
        try:
            final_column = pc.cast(replaced_column, column.type)
        except:
            final_column = replaced_column  # 변환 실패시 문자열 유지

        # 새 테이블 생성
        new_table = table.set_column(table.column_names.index(column_name), column_name, final_column)

        # 교체된 개수 계산
        replaced_count = pc.sum(pc.not_equal(column, final_column)).as_py()

        result_info = {
            "success": True,
            "column_name": column_name,
            "old_value": old_value,
            "new_value": new_value,
            "replaced_count": replaced_count,
            "total_rows": table.num_rows,
            "replaced_at": datetime.now().isoformat()
        }

        logger.info("값 교체 완료: 컬럼 %s에서 %d개 값 교체", column_name, replaced_count)
        return new_table, result_info

    except Exception as e:
        logger.error("값 교체 실패: %s", e)
        raise RuntimeError(f"값 교체 중 오류 발생: {str(e)}")


def apply_column_operation(table: pa.Table, column_name: str, operation: str) -> Tuple[pa.Table, Dict[str, Any]]:
    """
    특정 컬럼에 수치 연산 적용

    Args:
        table (pa.Table): 원본 테이블
        column_name (str): 대상 컬럼명
        operation (str): 연산식 (예: "+4", "*3+4", "+4*3")

    Returns:
        Tuple[pa.Table, Dict[str, Any]]: (수정된 테이블, 결과 정보)
    """
    if table is None:
        raise RuntimeError("테이블이 None입니다")

    if column_name not in table.column_names:
        raise RuntimeError(f"컬럼 '{column_name}'이 존재하지 않습니다")

    try:
        column = table.column(column_name)

        # 숫자 타입인지 확인
        if not any(num_type in str(column.type).lower() for num_type in ['int', 'float', 'double', 'decimal']):
            raise RuntimeError(f"컬럼 '{column_name}'은 숫자 타입이 아닙니다")

        # 연산식 파싱 및 실행
        original_column = column
        result_column = column

        # 연산자와 숫자를 순차적으로 찾아서 적용
        operations = re.findall(r'([+\-*/])(\d+(?:\.\d+)?)', operation)

        if not operations:
            raise RuntimeError(f"잘못된 연산식: {operation}")

        for op, value in operations:
            num_value = float(value) if '.' in value else int(value)

            if op == '+':
                result_column = pc.add(result_column, num_value)
            elif op == '-':
                result_column = pc.subtract(result_column, num_value)
            elif op == '*':
                result_column = pc.multiply(result_column, num_value)
            elif op == '/':
                result_column = pc.divide(result_column, num_value)

        # 새 테이블 생성
        new_table = table.set_column(table.column_names.index(column_name), column_name, result_column)

        result_info = {
            "success": True,
            "column_name": column_name,
            "operation": operation,
            "operations_applied": len(operations),
            "total_rows": table.num_rows,
            "applied_at": datetime.now().isoformat()
        }

        logger.info("연산 적용 완료: 컬럼 %s에 연산 %s 적용", column_name, operation)
        return new_table, result_info

    except Exception as e:
        logger.error("연산 적용 실패: %s", e)
        raise RuntimeError(f"연산 적용 중 오류 발생: {str(e)}")


def remove_null_rows(table: pa.Table, column_name: str = None) -> Tuple[pa.Table, Dict[str, Any]]:
    """
    NULL 값이 있는 row를 제거

    Args:
        table (pa.Table): 원본 테이블
        column_name (str, optional): 특정 컬럼명. None이면 전체 컬럼에서 NULL 체크

    Returns:
        Tuple[pa.Table, Dict[str, Any]]: (수정된 테이블, 결과 정보)
    """
    if table is None:
        raise RuntimeError("테이블이 None입니다")

    try:
        original_rows = table.num_rows

        if column_name is not None:
            # 특정 컬럼에서만 NULL 체크
            if column_name not in table.column_names:
                raise RuntimeError(f"컬럼 '{column_name}'이 존재하지 않습니다")

            # 해당 컬럼에서 NULL이 아닌 행만 필터링
            column = table.column(column_name)
            mask = pc.is_valid(column)
            new_table = pc.filter(table, mask)

            null_rows_removed = original_rows - new_table.num_rows
            filter_info = f"컬럼 '{column_name}'"

        else:
            # 전체 컬럼에서 NULL 체크 - 어느 컬럼이든 NULL이 있으면 해당 row 제거
            mask = None

            for col_name in table.column_names:
                column = table.column(col_name)
                col_mask = pc.is_valid(column)

                if mask is None:
                    mask = col_mask
                else:
                    # AND 연산 - 모든 컬럼이 valid해야 함
                    mask = pc.and_(mask, col_mask)

            new_table = pc.filter(table, mask)
            null_rows_removed = original_rows - new_table.num_rows
            filter_info = "전체 컬럼"

        result_info = {
            "success": True,
            "filter_target": filter_info,
            "original_rows": original_rows,
            "remaining_rows": new_table.num_rows,
            "removed_rows": null_rows_removed,
            "total_columns": new_table.num_columns,
            "filtered_at": datetime.now().isoformat()
        }

        logger.info("NULL row 제거 완료: %s에서 %d개 행 제거 (전체: %d → %d)",
                   filter_info, null_rows_removed, original_rows, new_table.num_rows)

        return new_table, result_info

    except Exception as e:
        logger.error("NULL row 제거 실패: %s", e)
        raise RuntimeError(f"NULL row 제거 중 오류 발생: {str(e)}")


def upload_dataset_to_hf(table: pa.Table, repo_id: str, hf_user_id: str, hub_token: str,
                        filename: str = None, private: bool = False) -> Dict[str, Any]:
    """
    pyarrow Table을 parquet 파일로 저장하고 HuggingFace Hub에 업로드

    Args:
        table (pa.Table): 업로드할 테이블
        repo_id (str): HuggingFace 리포지토리 ID (user/repo-name 형식)
        hf_user_id (str): HuggingFace 사용자 ID
        hub_token (str): HuggingFace Hub 토큰
        filename (str, optional): 업로드할 파일명. None이면 자동 생성
        private (bool): 프라이빗 리포지토리 여부

    Returns:
        Dict[str, Any]: 업로드 결과 정보
    """
    if table is None:
        raise RuntimeError("테이블이 None입니다")

    if not hub_token:
        raise RuntimeError("HuggingFace Hub 토큰이 제공되지 않았습니다")

    if not hf_user_id:
        raise RuntimeError("HuggingFace 사용자 ID가 제공되지 않았습니다")

    try:
        # 임시 디렉토리 생성
        upload_dir = f"/plateerag_backend/downloads/tmp/hf_upload_{int(datetime.now().timestamp())}"
        os.makedirs(upload_dir, exist_ok=True)

        # 파일명 설정
        if filename is None:
            filename = f"dataset_{int(datetime.now().timestamp())}.parquet"

        if not filename.endswith('.parquet'):
            filename += '.parquet'

        temp_parquet_path = os.path.join(upload_dir, filename)

        # parquet 파일로 저장
        pq.write_table(table, temp_parquet_path)
        logger.info("parquet 파일 생성: %s (%d행, %d열)", temp_parquet_path, table.num_rows, table.num_columns)

        # repo_id 검증 및 조정
        if '/' not in repo_id:
            repo_id = f"{hf_user_id}/{repo_id}"

        try:
            # 리포지토리 생성 시도 (이미 존재하면 무시)
            create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=private,
                token=hub_token,
                exist_ok=True
            )
            logger.info("HuggingFace 데이터셋 리포지토리 생성/확인: %s", repo_id)

        except Exception as e:
            logger.warning("리포지토리 생성 실패 (이미 존재할 수 있음): %s", e)

        # 파일 업로드
        upload_result = upload_file(
            path_or_fileobj=temp_parquet_path,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="dataset",
            token=hub_token,
            commit_message=f"Upload dataset {filename} - {table.num_rows} rows, {table.num_columns} columns"
        )

        logger.info("HuggingFace Hub 업로드 완료: %s → %s", temp_parquet_path, upload_result)

        # 파일 크기 계산
        file_size = os.path.getsize(temp_parquet_path)

        result_info = {
            "success": True,
            "repo_id": repo_id,
            "filename": filename,
            "upload_url": upload_result,
            "file_size_mb": file_size / (1024 * 1024),
            "dataset_rows": table.num_rows,
            "dataset_columns": table.num_columns,
            "column_names": table.column_names,
            "private": private,
            "uploaded_at": datetime.now().isoformat()
        }

        # 임시 파일 정리
        try:
            shutil.rmtree(upload_dir)
        except Exception as e:
            logger.warning("임시 디렉토리 삭제 실패: %s", e)

        logger.info("HuggingFace 데이터셋 업로드 완료: %s (%d행, %d열, %.2f MB)",
                   repo_id, table.num_rows, table.num_columns, file_size / (1024 * 1024))

        return result_info

    except Exception as e:
        # 임시 파일 정리 (오류 시)
        try:
            if 'upload_dir' in locals():
                shutil.rmtree(upload_dir)
        except:
            pass

        logger.error("HuggingFace 업로드 실패: %s", e)
        raise RuntimeError(f"HuggingFace 업로드 중 오류 발생: {str(e)}")


def copy_column(table: pa.Table, source_column: str, new_column: str) -> Tuple[pa.Table, Dict[str, Any]]:
    """
    특정 컬럼을 복사하여 새로운 컬럼으로 추가

    Args:
        table (pa.Table): 원본 테이블
        source_column (str): 복사할 원본 컬럼명
        new_column (str): 새로운 컬럼명

    Returns:
        Tuple[pa.Table, Dict[str, Any]]: (수정된 테이블, 결과 정보)
    """
    if table is None:
        raise RuntimeError("테이블이 None입니다")

    if source_column not in table.column_names:
        raise RuntimeError(f"원본 컬럼 '{source_column}'이 존재하지 않습니다")

    if new_column in table.column_names:
        raise RuntimeError(f"새로운 컬럼명 '{new_column}'이 이미 존재합니다")

    try:
        # 원본 컬럼 데이터 가져오기
        source_column_data = table.column(source_column)

        # 새로운 컬럼으로 추가
        new_table = table.append_column(new_column, source_column_data)

        result_info = {
            "success": True,
            "source_column": source_column,
            "new_column": new_column,
            "original_columns": table.num_columns,
            "new_columns": new_table.num_columns,
            "rows_count": new_table.num_rows,
            "copied_at": datetime.now().isoformat()
        }

        logger.info("컬럼 복사 완료: '%s' → '%s' (%d행)", source_column, new_column, table.num_rows)
        return new_table, result_info

    except Exception as e:
        logger.error("컬럼 복사 실패: %s", e)
        raise RuntimeError(f"컬럼 복사 중 오류 발생: {str(e)}")


def rename_column(table: pa.Table, old_name: str, new_name: str) -> Tuple[pa.Table, Dict[str, Any]]:
    """
    특정 컬럼의 이름을 변경

    Args:
        table (pa.Table): 원본 테이블
        old_name (str): 기존 컬럼명
        new_name (str): 새로운 컬럼명

    Returns:
        Tuple[pa.Table, Dict[str, Any]]: (수정된 테이블, 결과 정보)
    """
    if table is None:
        raise RuntimeError("테이블이 None입니다")

    if old_name not in table.column_names:
        raise RuntimeError(f"컬럼 '{old_name}'이 존재하지 않습니다")

    if new_name in table.column_names:
        raise RuntimeError(f"새로운 컬럼명 '{new_name}'이 이미 존재합니다")

    try:
        # 컬럼명 변경
        column_names = list(table.column_names)
        old_index = column_names.index(old_name)
        column_names[old_index] = new_name

        # 새로운 스키마 생성
        new_schema = pa.schema([
            pa.field(column_names[i], table.schema.field(i).type, table.schema.field(i).nullable)
            for i in range(len(column_names))
        ])

        # 새로운 테이블 생성
        new_table = pa.Table.from_arrays(
            [table.column(i) for i in range(table.num_columns)],
            schema=new_schema
        )

        result_info = {
            "success": True,
            "old_name": old_name,
            "new_name": new_name,
            "total_columns": new_table.num_columns,
            "rows_count": new_table.num_rows,
            "renamed_at": datetime.now().isoformat()
        }

        logger.info("컬럼 이름 변경 완료: '%s' → '%s'", old_name, new_name)
        return new_table, result_info

    except Exception as e:
        logger.error("컬럼 이름 변경 실패: %s", e)
        raise RuntimeError(f"컬럼 이름 변경 중 오류 발생: {str(e)}")


def format_columns_string(table: pa.Table, column_names: List[str], template: str, new_column: str) -> Tuple[pa.Table, Dict[str, Any]]:
    """
    여러 컬럼의 값들을 문자열 템플릿에 삽입하여 새로운 컬럼 생성

    Args:
        table (pa.Table): 원본 테이블
        column_names (List[str]): 사용할 컬럼명들
        template (str): 문자열 템플릿 (예: "{col1}_aiaiaiai_{col2}")
        new_column (str): 새로운 컬럼명

    Returns:
        Tuple[pa.Table, Dict[str, Any]]: (수정된 테이블, 결과 정보)
    """
    if table is None:
        raise RuntimeError("테이블이 None입니다")

    if not column_names:
        raise RuntimeError("컬럼명이 지정되지 않았습니다")

    if new_column in table.column_names:
        raise RuntimeError(f"새로운 컬럼명 '{new_column}'이 이미 존재합니다")

    # 컬럼 존재 확인
    missing_columns = [col for col in column_names if col not in table.column_names]
    if missing_columns:
        raise RuntimeError(f"존재하지 않는 컬럼: {missing_columns}")

    try:
        # 각 컬럼의 데이터를 문자열로 변환
        column_data = {}
        for col_name in column_names:
            column = table.column(col_name)
            # 문자열로 변환 (NULL 값은 빈 문자열로 처리)
            str_column = pc.cast(column, pa.string())
            # NULL을 빈 문자열로 교체
            str_column = pc.fill_null(str_column, "")
            column_data[col_name] = str_column

        # 템플릿에 컬럼명이 포함되어 있는지 확인
        template_columns = []
        for col_name in column_names:
            if f"{{{col_name}}}" in template:
                template_columns.append(col_name)

        if not template_columns:
            raise RuntimeError("템플릿에 지정된 컬럼명이 포함되지 않았습니다")

        # 행별로 템플릿 적용
        result_values = []
        for i in range(table.num_rows):
            # 각 행의 값들을 딕셔너리로 구성
            row_values = {}
            for col_name in template_columns:
                value = column_data[col_name][i].as_py()
                row_values[col_name] = str(value) if value is not None else ""

            # 템플릿에 값 삽입
            formatted_string = template.format(**row_values)
            result_values.append(formatted_string)

        # 새로운 컬럼 생성
        result_array = pa.array(result_values, type=pa.string())

        # 테이블에 컬럼 추가
        new_table = table.append_column(new_column, result_array)

        result_info = {
            "success": True,
            "used_columns": template_columns,
            "template": template,
            "new_column": new_column,
            "original_columns": table.num_columns,
            "new_columns": new_table.num_columns,
            "rows_processed": table.num_rows,
            "formatted_at": datetime.now().isoformat()
        }

        logger.info("컬럼 문자열 포맷팅 완료: %s → '%s' (%d행)",
                   template_columns, new_column, table.num_rows)
        return new_table, result_info

    except Exception as e:
        logger.error("컬럼 문자열 포맷팅 실패: %s", e)
        raise RuntimeError(f"컬럼 문자열 포맷팅 중 오류 발생: {str(e)}")


def calculate_columns_operation(table: pa.Table, col1: str, col2: str, operation: str, new_column: str) -> Tuple[pa.Table, Dict[str, Any]]:
    """
    두 컬럼 간 사칙연산을 수행하여 새로운 컬럼 생성
    문자열과 숫자 타입에 따라 특별한 처리 로직 적용

    Args:
        table (pa.Table): 원본 테이블
        col1 (str): 첫 번째 컬럼명
        col2 (str): 두 번째 컬럼명
        operation (str): 연산자 (+, -, *, /)
        new_column (str): 새로운 컬럼명

    Returns:
        Tuple[pa.Table, Dict[str, Any]]: (수정된 테이블, 결과 정보)
    """
    if table is None:
        raise RuntimeError("테이블이 None입니다")

    if new_column in table.column_names:
        raise RuntimeError(f"새로운 컬럼명 '{new_column}'이 이미 존재합니다")

    # 컬럼 존재 확인
    missing_columns = [col for col in [col1, col2] if col not in table.column_names]
    if missing_columns:
        raise RuntimeError(f"존재하지 않는 컬럼: {missing_columns}")

    if operation not in ['+', '-', '*', '/']:
        raise RuntimeError(f"지원되지 않는 연산자: {operation}")

    try:
        column1 = table.column(col1)
        column2 = table.column(col2)

        # 컬럼 타입 확인
        col1_type = str(column1.type).lower()
        col2_type = str(column2.type).lower()

        # 숫자 타입인지 확인
        def is_numeric_type(type_str):
            return any(num_type in type_str for num_type in ['int', 'float', 'double', 'decimal'])

        col1_is_numeric = is_numeric_type(col1_type)
        col2_is_numeric = is_numeric_type(col2_type)

        # 연산 결과 배열
        result_values = []

        # 행별로 연산 수행
        for i in range(table.num_rows):
            val1 = column1[i].as_py()
            val2 = column2[i].as_py()

            # NULL 값 처리
            if val1 is None or val2 is None:
                result_values.append(None)
                continue

            try:
                if operation == '+':
                    if col1_is_numeric and col2_is_numeric:
                        # 숫자 + 숫자 = 숫자 덧셈
                        result = float(val1) + float(val2)
                        result_values.append(result)
                    else:
                        # 문자열 연결: {col1}{col2}
                        result = str(val1) + str(val2)
                        result_values.append(result)

                elif operation == '-':
                    if col1_is_numeric and col2_is_numeric:
                        # 숫자 - 숫자 = 숫자 빼기
                        result = float(val1) - float(val2)
                        result_values.append(result)
                    else:
                        # 문자열에서는 빼기 연산 불가
                        raise ValueError("문자열 타입에서는 빼기 연산을 지원하지 않습니다")

                elif operation == '*':
                    if col1_is_numeric and col2_is_numeric:
                        # 숫자 * 숫자 = 숫자 곱셈
                        result = float(val1) * float(val2)
                        result_values.append(result)
                    elif not col1_is_numeric and col2_is_numeric:
                        # 문자열 * 숫자 = 문자열 반복
                        result = str(val1) * int(val2)
                        result_values.append(result)
                    elif col1_is_numeric and not col2_is_numeric:
                        # 숫자 * 문자열 = 문자열 반복
                        result = str(val2) * int(val1)
                        result_values.append(result)
                    else:
                        # 문자열 * 문자열은 불가
                        raise ValueError("두 문자열 간의 곱셈은 지원하지 않습니다")

                elif operation == '/':
                    if col1_is_numeric and col2_is_numeric:
                        if float(val2) == 0:
                            raise ValueError("0으로 나눌 수 없습니다")
                        result = float(val1) / float(val2)
                        result_values.append(result)
                    else:
                        # 문자열에서는 나누기 연산 불가
                        raise ValueError("문자열 타입에서는 나누기 연산을 지원하지 않습니다")

            except Exception as e:
                logger.warning("행 %d에서 연산 실패: %s", i, e)
                result_values.append(None)

        # 결과 타입 결정
        if all(isinstance(v, (int, float)) for v in result_values if v is not None):
            # 모두 숫자면 float 타입
            result_array = pa.array(result_values, type=pa.float64())
            result_type = "numeric"
        else:
            # 문자열이 포함되면 string 타입
            result_array = pa.array(result_values, type=pa.string())
            result_type = "string"

        # 테이블에 컬럼 추가
        new_table = table.append_column(new_column, result_array)

        result_info = {
            "success": True,
            "col1": col1,
            "col2": col2,
            "operation": operation,
            "new_column": new_column,
            "col1_type": col1_type,
            "col2_type": col2_type,
            "result_type": result_type,
            "original_columns": table.num_columns,
            "new_columns": new_table.num_columns,
            "rows_processed": table.num_rows,
            "calculated_at": datetime.now().isoformat()
        }

        logger.info("컬럼 연산 완료: %s %s %s → '%s' (%s타입, %d행)",
                   col1, operation, col2, new_column, result_type, table.num_rows)
        return new_table, result_info

    except Exception as e:
        logger.error("컬럼 연산 실패: %s", e)
        raise RuntimeError(f"컬럼 연산 중 오류 발생: {str(e)}")


def execute_safe_callback(table: pa.Table, callback_code: str) -> Tuple[pa.Table, Dict[str, Any]]:
    """
    AST를 사용하여 사용자 정의 PyArrow 코드를 안전하게 실행
    exec을 사용하지 않고 임시 파일과 subprocess를 활용

    Args:
        table (pa.Table): 원본 테이블
        callback_code (str): 실행할 PyArrow 코드

    Returns:
        Tuple[pa.Table, Dict[str, Any]]: (수정된 테이블, 실행 결과 정보)
    """
    if table is None:
        raise RuntimeError("테이블이 None입니다")

    if not callback_code or not callback_code.strip():
        raise RuntimeError("실행할 코드가 지정되지 않았습니다")

    # 허용된 안전한 함수/속성명들
    ALLOWED_NAMES = {
        # PyArrow Table 메서드들
        'table', 'select', 'filter', 'take', 'slice', 'sort_by', 'group_by',
        'append_column', 'add_column', 'set_column', 'remove_column', 'column',
        'rename_columns', 'cast', 'drop_duplicates', 'column_names', 'num_rows', 'num_columns',

        # PyArrow Compute 함수들
        'pc', 'pa', 'add', 'subtract', 'multiply', 'divide', 'power',
        'equal', 'not_equal', 'greater', 'less', 'greater_equal', 'less_equal',
        'and_', 'or_', 'not_', 'is_null', 'is_valid', 'fill_null',
        'strptime', 'strftime', 'extract', 'starts_with', 'ends_with',
        'match_substring', 'replace_substring', 'replace_substring_regex',
        'length', 'upper', 'lower', 'sum', 'mean', 'min', 'max', 'count',
        'stddev', 'variance', 'quantile', 'unique', 'value_counts',
        'concat_tables', 'array', 'field', 'schema',

        # 안전한 내장함수들
        'len', 'range', 'enumerate', 'zip', 'list', 'dict', 'tuple', 'set',
        'str', 'int', 'float', 'bool', 'type', 'isinstance', 'min', 'max',
        'abs', 'round', 'sorted', 'reversed', 'all', 'any', 'print',

        # 기본 타입과 연산
        'result', 'True', 'False', 'None'
    }

    try:
        # 1. AST를 사용한 코드 안전성 검증
        try:
            parsed = ast.parse(callback_code)
        except SyntaxError as e:
            raise RuntimeError(f"코드 문법 오류: {str(e)}")

        # 2. AST 노드 검증 - 위험한 노드 차단
        for node in ast.walk(parsed):
            # 함수/메서드 호출 검증
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id not in ALLOWED_NAMES:
                        raise RuntimeError(f"허용되지 않는 함수: {node.func.id}")
                elif isinstance(node.func, ast.Attribute):
                    # 메서드 호출은 허용 (table.select 등)
                    if node.func.attr not in ALLOWED_NAMES:
                        raise RuntimeError(f"허용되지 않는 메서드: {node.func.attr}")

            # 변수명 검증
            elif isinstance(node, ast.Name):
                if isinstance(node.ctx, (ast.Store, ast.Load)):
                    if node.id not in ALLOWED_NAMES:
                        # 사용자 정의 변수는 허용 (result, temp_var 등)
                        if not (node.id.startswith(('temp_', 'result', 'filtered', 'new_', 'updated_')) or
                               node.id.replace('_', '').isalnum()):
                            raise RuntimeError(f"허용되지 않는 변수명: {node.id}")

            # 위험한 노드 타입들 차단
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                raise RuntimeError("import 구문은 허용되지 않습니다")

            # 속성 접근 검증 (__builtins__ 등 차단)
            elif isinstance(node, ast.Attribute):
                if node.attr.startswith('_'):
                    raise RuntimeError(f"private 속성 접근은 허용되지 않습니다: {node.attr}")

        # 3. 추가 문자열 패턴 검사 (이중 보안)
        dangerous_patterns = [
            '__builtins__', '__globals__', '__locals__', 'exec(', 'eval(',
            'compile(', 'open(', 'file(', '__import__', 'getattr(', 'setattr(',
            'subprocess', 'os.', 'sys.'
        ]

        code_lower = callback_code.lower()
        for pattern in dangerous_patterns:
            if pattern.lower() in code_lower:
                raise RuntimeError(f"허용되지 않는 패턴: {pattern}")

        # 3. 임시 파일을 사용한 안전한 실행
        execution_start = datetime.now()
        original_rows = table.num_rows
        original_columns = table.num_columns
        original_column_names = table.column_names.copy()

        # 임시 디렉토리 기본 경로 설정
        base_temp_dir = "/plateerag_backend/downloads/tmp"
        os.makedirs(base_temp_dir, exist_ok=True)

        with tempfile.TemporaryDirectory(dir=base_temp_dir) as temp_dir:
            # 입력 테이블을 임시 parquet 파일로 저장
            input_file = os.path.join(temp_dir, "input_table.parquet")
            output_file = os.path.join(temp_dir, "output_table.parquet")

            pq.write_table(table, input_file)

            # 사용자 코드를 적절히 들여쓰기 처리
            indented_code = ""
            for line in callback_code.split('\n'):
                if line.strip():  # 빈 줄이 아닌 경우에만 들여쓰기 추가
                    indented_code += "    " + line + "\n"
                else:
                    indented_code += "\n"

            # 실행할 파이썬 스크립트 생성
            script_content = f'''import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
import sys

try:
    # 입력 테이블 로드
    table = pq.read_table("{input_file}")

    # 사용자 코드 실행
{indented_code}
    # 결과 테이블 결정
    if 'result' in locals() and isinstance(result, pa.Table):
        final_table = result
    else:
        final_table = table

    # 결과 저장
    pq.write_table(final_table, "{output_file}")

    # 성공 표시
    print("SUCCESS")

except Exception as e:
    print(f"ERROR: {{str(e)}}")
    sys.exit(1)
'''

            script_file = os.path.join(temp_dir, "callback_script.py")
            with open(script_file, 'w', encoding='utf-8') as f:
                f.write(script_content)

            # subprocess로 격리된 환경에서 실행
            try:
                result = subprocess.run([
                    sys.executable, script_file
                ], capture_output=True, text=True, timeout=30)  # 30초 타임아웃

                if result.returncode != 0:
                    error_output = result.stderr.strip() or result.stdout.strip()
                    raise RuntimeError(f"코드 실행 실패: {error_output}")

                if "SUCCESS" not in result.stdout:
                    raise RuntimeError("코드 실행이 완료되지 않았습니다")

                # 결과 테이블 로드
                if not os.path.exists(output_file):
                    raise RuntimeError("결과 테이블이 생성되지 않았습니다")

                result_table = pq.read_table(output_file)

                if not isinstance(result_table, pa.Table):
                    raise RuntimeError("올바른 PyArrow Table이 반환되지 않았습니다")

            except subprocess.TimeoutExpired:
                raise RuntimeError("코드 실행 시간 초과 (30초)")
            except FileNotFoundError:
                raise RuntimeError("Python 인터프리터를 찾을 수 없습니다")

        execution_time = (datetime.now() - execution_start).total_seconds()

        # 4. 결과 정보 생성
        result_info = {
            "success": True,
            "original_rows": original_rows,
            "original_columns": original_columns,
            "original_column_names": original_column_names,
            "final_rows": result_table.num_rows,
            "final_columns": result_table.num_columns,
            "final_column_names": result_table.column_names,
            "execution_time_seconds": execution_time,
            "rows_changed": result_table.num_rows - original_rows,
            "columns_changed": result_table.num_columns - original_columns,
            "executed_at": execution_start.isoformat(),
            "code_executed": callback_code,
            "execution_method": "subprocess_isolation"
        }

        logger.info("사용자 콜백 코드 실행 완료 (격리환경): %d행 → %d행, %d열 → %d열 (%.3f초)",
                   original_rows, result_table.num_rows,
                   original_columns, result_table.num_columns, execution_time)

        return result_table, result_info

    except Exception as e:
        logger.error("AST 기반 콜백 코드 실행 실패: %s", e)
        raise RuntimeError(f"콜백 코드 실행 중 오류 발생: {str(e)}")

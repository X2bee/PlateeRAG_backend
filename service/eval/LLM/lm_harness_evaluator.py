import subprocess
import json
import os
import glob
import datetime
from typing import Dict, List, Any, Optional
import logging

def run_lm_eval(
    job_name: str,
    model: str,
    tasks: str,
    gpu_count: int = 1,
    batch_size: int = 32,
    max_length: int = 8096,
    output_dir: str = "./eval_results",
    log_dir: str = "./eval_job_data",
    base_model: bool = False
) -> Dict[str, Any]:
    """
    LM Evaluation Harness를 실행하고 결과를 반환하는 함수
    
    Args:
        model: 모델 이름 (예: "google/gemma-3-4b-it")
        tasks: 평가할 태스크 리스트 (예: ["mmlu_pro_ko_biology", "hellaswag"])
        gpu_count: 사용할 GPU 개수
        batch_size: 배치 크기
        max_length: 최대 길이
        output_dir: 결과 저장 디렉토리
        log_dir: 로그 저장 디렉토리
        base_model: Base Model 평가 여부
    
    Returns:
        Dict: 평가 결과 딕셔너리
    """
    
    # 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 현재 시간으로 로그 파일명 생성
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{job_name}.txt")
    eval_output_dir = output_dir
    
    # 로깅 설정 - 기존 핸들러 제거 후 새로 설정
    logger = logging.getLogger(f"lm_eval_{timestamp}")
    logger.setLevel(logging.INFO)
    
    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 파일 핸들러 추가
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # 콘솔 핸들러 추가
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 모델 이름에서 안전한 디렉토리명 생성
    safe_model_name = model.replace("/", "__").replace(":", "__")
    
    # GPU 설정
    if gpu_count > 1:
        model_args = f"pretrained={model},parallelize=True,max_length={max_length}"
    else:
        model_args = f"pretrained={model},max_length={max_length}"
    
    # lm_eval 명령어 구성
    if base_model:
        cmd = [
            "lm_eval",
            "--model", "hf",
            "--model_args", model_args,
            "--tasks", tasks,
            "--device", "cuda",
            "--batch_size", str(batch_size)
        ]
    else:
        cmd = [
            "lm_eval",
            "--model", "hf",
            "--model_args", model_args,
            "--tasks", tasks,
            "--device", "cuda",
            "--batch_size", str(batch_size),
            "--output_path", eval_output_dir  # 수정: base_model에 따라 다른 output_dir 사용
        ]
    
    model_type = "Base Model" if base_model else "Main Model"
    logger.info(f"시작: LM Evaluation ({model_type})")
    logger.info(f"모델: {model}")
    logger.info(f"태스크: {tasks}")
    logger.info(f"GPU 개수: {gpu_count}")
    logger.info(f"배치 크기: {batch_size}")
    logger.info(f"출력 디렉토리: {eval_output_dir}")
    logger.info(f"로그 파일: {log_file}")
    logger.info(f"실행 명령어: {' '.join(cmd)}")
    
    try:
        logger.info("평가 실행 중...")
        
        # subprocess를 사용해서 실시간 로그 출력
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # stderr도 stdout으로 리다이렉트
            text=True,
            encoding='utf-8',
            bufsize=1,  # 라인 버퍼링
            universal_newlines=True
        )
        
        # 실시간 로그 출력 및 저장
        output_lines = []
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                line = output.strip()
                logger.info(f"[lm_eval] {line}")
                output_lines.append(line)
        
        # 프로세스 완료 대기
        return_code = process.poll()
        
        if return_code != 0:
            logger.error(f"프로세스가 오류로 종료됨 (return code: {return_code})")
            logger.error("전체 출력:")
            for line in output_lines:
                logger.error(f"  {line}")
            return {"error": f"평가 프로세스 실패 (return code: {return_code})"}
        
        logger.info("평가 완료!")
        
        # 결과 파일 찾기 - 여러 패턴 시도
        logger.info("결과 파일 검색 중...")
        result_patterns = [
            os.path.join(eval_output_dir, safe_model_name, "results_*.json"),
            os.path.join(eval_output_dir, f".__{safe_model_name}", "results_*.json"),
            os.path.join(eval_output_dir, f".__eval__minio__model__{safe_model_name}", "results_*.json"),
            os.path.join(eval_output_dir, f"._{safe_model_name}", "results_*.json")
        ]
        
        result_files = []
        found_pattern = None
        for pattern in result_patterns:
            logger.info(f"패턴 시도: {pattern}")
            files = glob.glob(pattern)
            if files:
                result_files.extend(files)
                found_pattern = pattern
                logger.info(f"파일 발견: {files}")
                break
        
        # 패턴으로 못 찾으면 전체 검색
        if not result_files:
            logger.info("패턴으로 찾지 못해 전체 디렉토리 검색 중...")
            result_files = glob.glob(f"{eval_output_dir}/**/results_*.json", recursive=True)
            if result_files:
                logger.info(f"전체 검색으로 발견된 파일들: {result_files}")
        
        if not result_files:
            logger.error("결과 파일을 찾을 수 없습니다.")
            logger.error("시도한 패턴들:")
            for pattern in result_patterns:
                logger.error(f"  {pattern}")
            
            # 디렉토리 구조 출력
            logger.error(f"실제 {eval_output_dir} 디렉토리 구조:")
            for root, dirs, files in os.walk(eval_output_dir):
                logger.error(f"  디렉토리: {root}")
                for d in dirs:
                    logger.error(f"    폴더: {d}")
                for f in files:
                    logger.error(f"    파일: {f}")
            
            return {"error": "결과 파일을 찾을 수 없습니다"}
        
        # 가장 최근 결과 파일 선택
        latest_result_file = max(result_files, key=os.path.getctime)
        logger.info(f"결과 파일 로드: {latest_result_file}")
        
        # JSON 파일 읽기
        with open(latest_result_file, 'r', encoding='utf-8') as f:
            full_results = json.load(f)
        
        logger.info(f"결과 파일 내용 키들: {list(full_results.keys())}")
        
        # Base Model일 경우 결과를 별도 파일로도 저장
        
        # 'results' 키의 내용만 추출
        if 'results' in full_results:
            results = full_results['results']
            logger.info("결과 추출 완료")
            
            # 결과 요약 로그
            logger.info(f"=== {model_type} 평가 결과 요약 ===")
            for task, metrics in results.items():
                logger.info(f"태스크: {task}")
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            logger.info(f"  {metric}: {value:.4f}")
                        else:
                            logger.info(f"  {metric}: {value}")
                else:
                    logger.info(f"  결과: {metrics}")
            
            return results
        else:
            logger.error("결과 파일에 'results' 키가 없습니다")
            logger.error(f"사용 가능한 키들: {list(full_results.keys())}")
            return {"error": "결과 파일 형식 오류"}
            
    except Exception as e:
        logger.error(f"예상치 못한 오류: {str(e)}")
        logger.error(f"오류 타입: {type(e).__name__}")
        import traceback
        logger.error(f"트레이스백: {traceback.format_exc()}")
        return {"error": f"예상치 못한 오류: {str(e)}"}
    
    finally:
        # 핸들러 정리
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        logger.info(f"로그가 저장되었습니다: {log_file}")
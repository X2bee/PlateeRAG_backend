import os
import json
import time
import datetime
from zoneinfo import ZoneInfo

import psutil
from service.database.connection import AppDatabaseManager

# 환경변수에서 타임존 가져오기 (기본값: 서울 시간)
TIMEZONE = ZoneInfo(os.getenv('TIMEZONE', 'Asia/Seoul'))

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

class PerformanceLogger:
    """
    워크플로우 노드의 성능을 측정하고 JSON 형식으로 로깅하는 클래스
    - 처리 시간
    - CPU, RAM 사용량 (프로세스 기준)
    - GPU 사용량 (NVIDIA GPU만 해당, 선택 사항)
    """

    MAX_LOG_ITEMS = 10      # 컬렉션(dict, list 등)의 최대 기록 항목 수
    MAX_LOG_STR_LEN = 150

    def __init__(self, workflow_name: str, workflow_id: str, node_id: str, node_name: str, user_id: str = None, db_manager: AppDatabaseManager = None):
        self.workflow_name = workflow_name
        self.workflow_id = workflow_id
        self.node_id = node_id
        self.node_name = node_name
        self.user_id = user_id
        self.db_manager = db_manager
        self._process = psutil.Process(os.getpid())
        self._start_time = None
        self._start_cpu_times = None
        self._start_ram_usage = None
        self._gpu_handles = []

        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                self._gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(device_count)]
            except pynvml.NVMLError:
                # NVML 초기화 실패 시 GPU 로깅 비활성화
                self._gpu_handles = []

    def __enter__(self):
        """컨텍스트 시작: 처리 시간, 자원 사용향 측정 시작"""
        self._start_time = time.perf_counter()
        if self._process:
            self._start_cpu_times = self._process.cpu_times()
            self._start_ram_usage = self._process.memory_info().rss
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 종료: GPU 리소스 정리"""
        if PYNVML_AVAILABLE and self._gpu_handles:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass # 종료 시 에러는 무시

    def _get_system_usage(self):
        """CPU, RAM, GPU 사용량을 계산합니다."""
        # RAM
        end_ram_usage = self._process.memory_info().rss
        ram_usage_mb = round((end_ram_usage - self._start_ram_usage) / (1024 * 1024), 2)

        # CPU
        end_cpu_times = self._process.cpu_times()
        cpu_time_diff = (end_cpu_times.user - self._start_cpu_times.user) + \
                        (end_cpu_times.system - self._start_cpu_times.system)

        elapsed_time = time.perf_counter() - self._start_time
        cpu_usage_percent = 0.0
        if elapsed_time > 0:
            cpu_usage_percent = round((cpu_time_diff / elapsed_time) * 100 / psutil.cpu_count(logical=True), 2)


        # GPU (NVIDIA)
        gpu_stats = {'gpu_usage_percent': 'N/A', 'gpu_memory_mb': 'N/A'}
        if self._gpu_handles:
            try:
                total_util, total_mem = 0, 0
                for handle in self._gpu_handles:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    total_util += util.gpu
                    total_mem += mem.used
                gpu_stats['gpu_usage_percent'] = round(total_util / len(self._gpu_handles), 2)
                gpu_stats['gpu_memory_mb'] = round(total_mem / (1024 * 1024), 2)
            except pynvml.NVMLError:
                pass

        return {
            "cpu_usage_percent": cpu_usage_percent,
            "ram_usage_mb": ram_usage_mb,
            "gpu_usage_percent": gpu_stats['gpu_usage_percent'],
            "gpu_memory_mb": gpu_stats['gpu_memory_mb']
        }

    def log(self, input_data: dict, output_data: any):
        """
        성능 정보를 최종적으로 계산하고 데이터베이스에 기록합니다.
        """
        processing_time_ms = round((time.perf_counter() - self._start_time) * 1000, 2)

        system_usage = self._get_system_usage()
        timestamp = datetime.datetime.now(TIMEZONE).isoformat()

        # 데이터베이스에 저장
        if self.db_manager:
            self._save_to_database(timestamp, processing_time_ms, system_usage, input_data, output_data)

    def _save_to_database(self, timestamp: str, processing_time_ms: float, system_usage: dict, input_data: dict, output_data: any):
        """성능 데이터를 데이터베이스에 저장"""
        try:
            # 직접 SQL 삽입으로 시도
            db_type = self.db_manager.config_db_manager.db_type

            if db_type == "postgresql":
                query = """
                INSERT INTO node_performance (
                    workflow_name, workflow_id, node_id, node_name, user_id, timestamp,
                    processing_time_ms, cpu_usage_percent, ram_usage_mb,
                    gpu_usage_percent, gpu_memory_mb, input_data, output_data
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
            else:
                query = """
                INSERT INTO node_performance (
                    workflow_name, workflow_id, node_id, node_name, user_id, timestamp,
                    processing_time_ms, cpu_usage_percent, ram_usage_mb,
                    gpu_usage_percent, gpu_memory_mb, input_data, output_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """

            params = (
                self.workflow_name,
                self.workflow_id,
                self.node_id,
                self.node_name,
                self.user_id,
                timestamp,
                processing_time_ms,
                system_usage.get('cpu_usage_percent', 0.0),
                system_usage.get('ram_usage_mb', 0.0),
                system_usage.get('gpu_usage_percent') if system_usage.get('gpu_usage_percent') != 'N/A' else None,
                system_usage.get('gpu_memory_mb') if system_usage.get('gpu_memory_mb') != 'N/A' else None,
                json.dumps(self._summarize_data(input_data), ensure_ascii=False),
                json.dumps(self._summarize_data(output_data), ensure_ascii=False)
            )

            self.db_manager.config_db_manager.execute_query(query, params)

        except Exception as e:
            # 데이터베이스 저장 실패는 조용히 처리 (성능 로깅 실패로 애플리케이션을 중단시키지 않음)
            pass

    def _summarize_data(self, data: any):
        """데이터를 로깅에 적합하게 요약합니다."""
        if isinstance(data, (int, float, bool)) or data is None:
            return data

        if isinstance(data, str):
            if len(data) > self.MAX_LOG_STR_LEN:
                return data[:self.MAX_LOG_STR_LEN] + '...'
            return data

        if isinstance(data, dict):
            # 딕셔너리 항목 개수가 임계값을 초과하면 요약 정보만 반환
            if len(data) > self.MAX_LOG_ITEMS:
                return {"type": "dict", "size": len(data), "info": "truncated"}
            # 임계값 이내이면 각 값(value)을 재귀적으로 요약하여 새로운 딕셔너리 생성
            return {str(k): self._summarize_data(v) for k, v in data.items()}

        if isinstance(data, (list, tuple, set)):
            # 리스트/튜플/셋 항목 개수가 임계값을 초과하면 요약 정보만 반환
            if len(data) > self.MAX_LOG_ITEMS:
                return {"type": type(data).__name__, "size": len(data), "info": "truncated"}
            # 임계값 이내이면 각 항목(item)을 재귀적으로 요약하여 새로운 리스트 생성
            return [self._summarize_data(item) for item in data]
        try:
            return str(data)
        except Exception:
            return f"<{type(data).__name__} object (un-stringifiable)>"

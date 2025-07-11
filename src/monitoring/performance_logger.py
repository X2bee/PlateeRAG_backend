import os 
import json
import time
import datetime
import threading
from logging import getLogger, FileHandler, Formatter, INFO

import psutil

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

logger = getLogger('performance')
logger.setLevel(INFO)

log_dir = os.path.join(os.getcwd(), "logs")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not logger.handlers:
    handler = FileHandler(os.path.join(log_dir, 'performance.log'), encoding='utf-8')
    formatter = Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class PerformanceLogger:
    """
    워크플로우 노드의 성능을 측정하고 JSON 형식으로 로깅하는 클래스
    - 처리 시간
    - CPU, RAM 사용량 (프로세스 기준)
    - GPU 사용량 (NVIDIA GPU만 해당, 선택 사항)
    """

    MAX_LOG_ITEMS = 10      # 컬렉션(dict, list 등)의 최대 기록 항목 수
    MAX_LOG_STR_LEN = 150

    def __init__(self, workflow_id: str, node_id: str, node_name: str):
        self.workflow_id = workflow_id
        self.node_id = node_id
        self.node_name = node_name
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
        """컨텍스트 종료: 로깅은 log 메소드를 통해 명시적으로 호출"""
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
        성능 정보를 최종적으로 계산하고 로그 파일에 기록합니다.
        """
        processing_time_ms = round((time.perf_counter() - self._start_time) * 1000, 2)
        
        system_usage = self._get_system_usage()

        log_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "workflow_id": self.workflow_id,
            "node_id": self.node_id,
            "node_name": self.node_name,
            "input": self._summarize_data(input_data),
            "output": self._summarize_data(output_data),
            "processing_time_ms": processing_time_ms,
            **system_usage
        }
        logger.info(json.dumps(log_entry, ensure_ascii=False))
        
    def _summarize_data(self, data: any, max_len=100):
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
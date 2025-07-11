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
        self._start_cpu_times = self._process.cpu_times()
        self._start_ram_usage = self._process.memory_info().rss
        return self
    
    def _exit_(self):
        """컨텍스트 종료: 로깅은 log 메소드를 통해 명시적으로 호출"""
        if PYNVML_AVAILABLE and self._gpu_handles:
            pynvml.nvmlShutdown()

    def _get_system_usage(self):
        """CPU, RAM, GPU 사용량을 계산합니다."""
        # RAM
        end_ram_usage = self._process.memory_info().rss
        ram_usage_mb = round((end_ram_usage - self._start_ram_usage) / (1024 * 1024), 2)

        # CPU
        end_cpu_times = self._process.cpu_times()
        cpu_time_diff = (end_cpu_times.user - self._start_cpu_times.user) + \
                        (end_cpu_times.system - self._start_cpu_times.system)
        # 멀티 코어 환경을 고려하여 논리 코어 수로 나눔
        cpu_usage_percent = round((cpu_time_diff / (time.perf_counter() - self._start_time)) * 100 / psutil.cpu_count(), 2)


        # GPU (NVIDIA)
        gpu_stats = {'gpu_usage_percent': 'N/A', 'gpu_memory_mb': 'N/A'}
        if PYNVML_AVAILABLE and self._gpu_handles:
            try:
                total_gpu_util = 0
                total_mem_used = 0
                for handle in self._gpu_handles:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    total_gpu_util += util.gpu
                    total_mem_used += mem.used
                
                gpu_stats['gpu_usage_percent'] = round(total_gpu_util / len(self._gpu_handles), 2) if self._gpu_handles else 0
                gpu_stats['gpu_memory_mb'] = round(total_mem_used / (1024 * 1024), 2)
            except pynvml.NVMLError:
                # 에러 발생 시 N/A 처리
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

        # Input/Output 데이터 요약
        # 너무 큰 데이터가 로그에 남지 않도록 요약 처리
        summarized_input = self._summarize_data(input_data)
        summarized_output = self._summarize_data(output_data)

        log_entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z",
            "workflow_id": self.workflow_id,
            "node_id": self.node_id,
            "node_name": self.node_name,
            "input": summarized_input,
            "output": summarized_output,
            "processing_time_ms": processing_time_ms,
            **system_usage
        }

        logger.info(json.dumps(log_entry, ensure_ascii=False))
        
    def _summarize_data(self, data: any, max_len=100) -> any:
        """데이터를 로깅하기에 적합한 형태로 요약합니다."""
        if isinstance(data, str):
            return data[:max_len] + '...' if len(data) > max_len else data
        if isinstance(data, (dict, list)):
            # 간단하게 타입과 크기/길이만 반환
            return {"type": str(type(data)), "size": len(data)}
        # 그 외의 타입은 문자열로 변환
        return str(data)
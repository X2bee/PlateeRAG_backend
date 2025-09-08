import logging
import psutil
import asyncio
import json
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from controller.helper.singletonHelper import get_config_composer, get_vector_manager, get_rag_service, get_document_processor, get_db_manager
from controller.admin.adminBaseController import validate_superuser

try:
    import pynvml
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

logger = logging.getLogger("admin-system-controller")
router = APIRouter(prefix="/system", tags=["Admin"])

# Response Models
class CPUInfo(BaseModel):
    usage_percent: float
    core_count: int
    frequency_current: float
    frequency_max: float
    load_average: List[float]

class MemoryInfo(BaseModel):
    total: int
    available: int
    percent: float
    used: int
    free: int

class GPUInfo(BaseModel):
    name: str
    memory_total: int
    memory_used: int
    memory_free: int
    memory_percent: float
    utilization: float
    temperature: Optional[float]

class NetworkInfo(BaseModel):
    interface: str
    is_up: bool
    bytes_sent: int
    bytes_recv: int
    packets_sent: int
    packets_recv: int

class DiskInfo(BaseModel):
    device: str
    mountpoint: str
    fstype: str
    total: int
    used: int
    free: int
    percent: float

class SystemMonitorResponse(BaseModel):
    cpu: CPUInfo
    memory: MemoryInfo
    gpu: List[GPUInfo]
    network: List[NetworkInfo]
    disk: List[DiskInfo]
    uptime: float

# Utility Functions
def get_cpu_info() -> CPUInfo:
    """Get CPU information"""
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    cpu_freq = psutil.cpu_freq()
    load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0.0, 0.0, 0.0]

    return CPUInfo(
        usage_percent=cpu_percent,
        core_count=cpu_count,
        frequency_current=cpu_freq.current if cpu_freq else 0.0,
        frequency_max=cpu_freq.max if cpu_freq else 0.0,
        load_average=list(load_avg)
    )

def get_memory_info() -> MemoryInfo:
    """Get memory information"""
    memory = psutil.virtual_memory()
    return MemoryInfo(
        total=memory.total,
        available=memory.available,
        percent=memory.percent,
        used=memory.used,
        free=memory.free
    )

def get_gpu_info() -> List[GPUInfo]:
    """Get GPU information"""
    gpu_list = []

    # Check if GPU monitoring is available
    if not GPU_AVAILABLE:
        logger.debug("NVML library not available, GPU monitoring disabled")
        return gpu_list

    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        # If no GPU devices found, return empty list
        if device_count == 0:
            logger.debug("No GPU devices found")
            return gpu_list

        for i in range(device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                # Get device name (handle both string and bytes return types)
                raw_name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(raw_name, bytes):
                    name = raw_name.decode('utf-8')
                else:
                    name = str(raw_name)

                # Memory info
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_total = memory_info.total
                memory_used = memory_info.used
                memory_free = memory_info.free
                memory_percent = (memory_used / memory_total) * 100 if memory_total > 0 else 0.0

                # Utilization
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    utilization = util.gpu
                except:
                    utilization = 0.0

                # Temperature
                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    temperature = None

                gpu_list.append(GPUInfo(
                    name=name,
                    memory_total=memory_total,
                    memory_used=memory_used,
                    memory_free=memory_free,
                    memory_percent=memory_percent,
                    utilization=utilization,
                    temperature=temperature
                ))
            except Exception as e:
                logger.warning(f"Failed to get info for GPU {i}: {e}")
                continue

    except pynvml.NVMLError as e:
        logger.debug(f"NVML error (GPU not available): {e}")
    except Exception as e:
        logger.debug(f"GPU monitoring not available: {e}")

    return gpu_list

def get_network_info() -> List[NetworkInfo]:
    """Get network interface information"""
    network_list = []

    try:
        # Get network interface stats
        net_io = psutil.net_io_counters(pernic=True)
        addrs = psutil.net_if_addrs()
        stats = psutil.net_if_stats()

        for interface, io_stats in net_io.items():
            if interface in stats:
                is_up = stats[interface].isup
                network_list.append(NetworkInfo(
                    interface=interface,
                    is_up=is_up,
                    bytes_sent=io_stats.bytes_sent,
                    bytes_recv=io_stats.bytes_recv,
                    packets_sent=io_stats.packets_sent,
                    packets_recv=io_stats.packets_recv
                ))
    except Exception as e:
        logger.warning(f"Failed to get network info: {e}")

    return network_list

def get_disk_info() -> List[DiskInfo]:
    """Get disk information"""
    disk_list = []

    try:
        partitions = psutil.disk_partitions()
        for partition in partitions:
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_list.append(DiskInfo(
                    device=partition.device,
                    mountpoint=partition.mountpoint,
                    fstype=partition.fstype,
                    total=usage.total,
                    used=usage.used,
                    free=usage.free,
                    percent=usage.percent
                ))
            except PermissionError:
                # This can happen on Windows
                continue
    except Exception as e:
        logger.warning(f"Failed to get disk info: {e}")

    return disk_list

def get_system_uptime() -> float:
    """Get system uptime in seconds"""
    try:
        return psutil.boot_time()
    except Exception as e:
        logger.warning(f"Failed to get uptime: {e}")
        return 0.0

@router.get("/status", response_model=SystemMonitorResponse)
async def get_system_status(request: Request):
    """Get comprehensive system monitoring information"""
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )

    try:
        # Gather all system information
        cpu_info = get_cpu_info()
        memory_info = get_memory_info()
        gpu_info = get_gpu_info()
        network_info = get_network_info()
        disk_info = get_disk_info()
        uptime = get_system_uptime()

        return SystemMonitorResponse(
            cpu=cpu_info,
            memory=memory_info,
            gpu=gpu_info,
            network=network_info,
            disk=disk_info,
            uptime=uptime
        )
    except Exception as e:
        logger.error(f"Failed to get system monitor info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve system monitoring information: {str(e)}"
        )

async def generate_system_status():
    """Generator function for SSE system status updates"""
    while True:
        try:
            # Gather all system information
            cpu_info = get_cpu_info()
            memory_info = get_memory_info()
            gpu_info = get_gpu_info()
            network_info = get_network_info()
            disk_info = get_disk_info()
            uptime = get_system_uptime()

            status_data = SystemMonitorResponse(
                cpu=cpu_info,
                memory=memory_info,
                gpu=gpu_info,
                network=network_info,
                disk=disk_info,
                uptime=uptime
            )

            # Convert to JSON and send as SSE format
            json_data = status_data.model_dump_json()
            yield f"data: {json_data}\n\n"

            # Wait for 1.5 seconds before next update
            await asyncio.sleep(1.5)

        except Exception as e:
            logger.error(f"Error in SSE system status generation: {e}")
            error_data = {"error": f"Failed to retrieve system monitoring information: {str(e)}"}
            yield f"data: {json.dumps(error_data)}\n\n"
            await asyncio.sleep(1.5)

@router.get("/status/stream")
async def stream_system_status(request: Request):
    """Stream system monitoring information via Server-Sent Events"""
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )

    return StreamingResponse(
        generate_system_status(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )

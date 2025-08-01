
import sys
import re

# 텍스트 파싱 시뮬레이션 - 실제 VastAI 출력 형태
sample_output = """
 ID      STATUS  GPU_NAME  GPU_MODEL  GPU_CORES  GPU_RAM  DPH_TOTAL  SSH_ADDR     SSH_PORT  
24513164 running RTX4090   Nvidia     120        24       0.5        257.6       256
"""

lines = sample_output.strip().split("\n")
print("원본 출력:")
for i, line in enumerate(lines):
    print(f"Line {i}: {repr(line)}")

# 인스턴스 라인 찾기
instance_id = "24513164"
instance_line = None
for line in lines:
    if str(instance_id) in line:
        instance_line = line
        print(f"\n발견된 인스턴스 라인: {repr(instance_line)}")
        break

if instance_line:
    parts = instance_line.split()
    print(f"\n분리된 필드들:")
    for i, part in enumerate(parts):
        print(f"  parts[{i}]: {repr(part)}")
    
    # 현재 파싱 로직
    if len(parts) > 7:
        ssh_addr = parts[7]
        ssh_port = parts[8] if len(parts) > 8 else "unknown"
        print(f"\n현재 파싱 결과:")
        print(f"  SSH 주소: {repr(ssh_addr)}")
        print(f"  SSH 포트: {repr(ssh_port)}")
        print(f"  문제: SSH 주소가 완전한 IP가 아님!")


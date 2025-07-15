#!/bin/bash

set -e  # 오류 발생 시 스크립트 종료

# 1. Qdrant 이미지 최신 상태로 유지
echo "[INFO] Pulling latest Qdrant image..."
docker pull qdrant/qdrant

# 2. qdrant_data 디렉토리 생성 (없으면)
mkdir -p "$(pwd)/qdrant_data"

# 3. 기존 컨테이너 확인 및 실행 여부 결정
if docker ps -a --format '{{.Names}}' | grep -q '^qdrant-dev$'; then
  if [ "$(docker inspect -f '{{.State.Running}}' qdrant-dev)" == "true" ]; then
    echo "[INFO] Qdrant container is already running."
  else
    echo "[INFO] Starting existing Qdrant container..."
    docker start qdrant-dev
  fi
else
  echo "[INFO] Creating and starting new Qdrant container..."
  docker run -d \
    --name qdrant-dev \
    -p 6333:6333 \
    -v "$(pwd)/qdrant_data:/qdrant/storage" \
    qdrant/qdrant
fi

# 4. Qdrant 기동 확인 (최대 10초 대기)
echo "[INFO] Waiting for Qdrant to be healthy..."
for i in {1..10}; do
  if curl -s http://localhost:6333/healthz | grep -q "healthz check passed"; then
    echo "[INFO] Qdrant is healthy!"
    break
  fi
  echo "[INFO] Waiting... ($i)"
  sleep 1
done

echo "[INFO] Running main.py"
python main.py

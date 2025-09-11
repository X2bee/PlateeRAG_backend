
FROM python:3.12-slim

WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    wget \
    git \
    libreoffice \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# pip 업그레이드 및 uv 설치
RUN pip install --upgrade pip uv

COPY ./pyproject.toml .
RUN uv sync

RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')"
RUN uv run python -c "from transformers import WhisperProcessor; WhisperProcessor.from_pretrained('openai/whisper-small')"
RUN uv run python -c "from transformers import WhisperForConditionalGeneration; WhisperForConditionalGeneration.from_pretrained('openai/whisper-small')"

# 애플리케이션 코드 복사
COPY . .
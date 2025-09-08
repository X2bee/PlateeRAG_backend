
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    rustc cargo espeak-ng ffmpeg \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade uv

COPY ./pyproject.toml .

RUN uv sync

COPY ./service/tts/zonos ./service/tts/zonos

RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')"
RUN uv run python -c "from transformers import WhisperProcessor; WhisperProcessor.from_pretrained('openai/whisper-small')"
RUN uv run python -c "from transformers import WhisperForConditionalGeneration; WhisperForConditionalGeneration.from_pretrained('openai/whisper-small')"
RUN uv run python -c "from service.tts.zonos.model import Zonos; Zonos.from_pretrained('Zyphra/Zonos-v0.1-transformer', device='cpu')"

COPY . .
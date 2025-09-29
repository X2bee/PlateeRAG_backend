
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

RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')"
# RUN uv run python -c "from transformers import WhisperProcessor; WhisperProcessor.from_pretrained('openai/whisper-small')"
# RUN uv run python -c "from transformers import WhisperForConditionalGeneration; WhisperForConditionalGeneration.from_pretrained('openai/whisper-small')"
RUN uv run python -c "from transformers import Qwen3ForCausalLM; Qwen3ForCausalLM.from_pretrained('Qwen/Qwen3Guard-Gen-0.6B', revision='0a0af2ce53c6724d17e3b9ccb53f322e479566a3')"

COPY . .
# Let BASE_IMAGE be overridden at build time (mirrors, nexus proxy, etc.)
ARG BASE_IMAGE=python:3.12-slim

# ---- Base (shared) ----------------------------------------------------------
FROM ${BASE_IMAGE} AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_SYSTEM_PYTHON=1
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git \
 && rm -rf /var/lib/apt/lists/*
ENV VENV_PATH=/opt/venv
ENV PATH="${VENV_PATH}/bin:${PATH}"

# ---- Builder (compilers here only) ------------------------------------------
FROM base AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential rustc cargo libpq-dev \
 && rm -rf /var/lib/apt/lists/*

# uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV UV_BIN=/root/.local/bin/uv

WORKDIR /app
# Copy dependency metadata first (best cache)
COPY pyproject.toml ./
# If you later commit uv.lock, add: COPY uv.lock ./

# Create venv & install deps into it via wheelhouse
RUN python -m venv ${VENV_PATH} \
 && ${UV_BIN} lock \                                           # <-- generate lock if absent
 && ${UV_BIN} export --format=requirements-txt --locked > /tmp/requirements.txt \
 && ${VENV_PATH}/bin/pip install --upgrade pip wheel \
 && ${VENV_PATH}/bin/pip wheel --no-cache-dir --wheel-dir /wheelhouse -r /tmp/requirements.txt \
 && ${VENV_PATH}/bin/pip install --no-cache-dir --no-index --find-links=/wheelhouse -r /tmp/requirements.txt

# Now copy application
COPY . .

# ---- Runtime (tiny) ---------------------------------------------------------
FROM ${BASE_IMAGE} AS runtime
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg espeak-ng libpq5 \
 && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN addgroup --system --gid 1001 app && \
    adduser  --system --uid 1001 --ingroup app --home /app app
WORKDIR /app

# Venv and app from builder
ENV VENV_PATH=/opt/venv
ENV PATH="${VENV_PATH}/bin:${PATH}"

COPY --from=builder ${VENV_PATH} ${VENV_PATH}
COPY --from=builder /app /app

# Model/cache dirs (we will mount a PVC here in K8s)
ENV HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    XDG_CACHE_HOME=/app/.cache \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000
USER app
CMD ["python","-m","uvicorn","main:app","--host","0.0.0.0","--port","8000"]

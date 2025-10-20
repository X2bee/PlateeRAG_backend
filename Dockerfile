# You can override BASE_IMAGE at build time to use a mirror/Nexus proxy
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

# venv path shared across stages
ENV VENV_PATH=/opt/venv
ENV PATH="${VENV_PATH}/bin:${PATH}"

# ---- Builder (only compilers here) ------------------------------------------
FROM base AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential rustc cargo libpq-dev \
 && rm -rf /var/lib/apt/lists/*

# install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV UV_BIN=/root/.local/bin/uv

WORKDIR /app

# Copy dependency metadata first for better layer caching
COPY pyproject.toml ./
# If you have a lockfile, uncomment the next line:
# COPY uv.lock ./

# Create venv and install deps (deterministic if uv.lock is present)
ENV VENV_PATH=/opt/venv
RUN python -m venv ${VENV_PATH} && \
    ${UV_BIN} lock && \
    ${UV_BIN} sync --locked --python ${VENV_PATH}/bin/python && \
    ${VENV_PATH}/bin/pip install --no-cache-dir "uvicorn[standard]>=0.30" 

# Now copy your application code
COPY . .

# ---- Runtime (slim) ---------------------------------------------------------
FROM ${BASE_IMAGE} AS runtime
ARG DEBIAN_FRONTEND=noninteractive
# runtime-only libs (drop what you don’t need)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg espeak-ng libpq5 \
 && rm -rf /var/lib/apt/lists/*

# non-root user
RUN addgroup --system --gid 1001 app && \
    adduser  --system --uid 1001 --ingroup app --home /app app
WORKDIR /app

ENV VENV_PATH=/opt/venv
ENV PATH="${VENV_PATH}/bin:${PATH}"
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# bring venv and app from builder
COPY --from=builder ${VENV_PATH} ${VENV_PATH}
COPY --from=builder /app /app

EXPOSE 8000
USER app

# Adjust if your entrypoint/module differs
CMD ["python","-m","uvicorn","main:app","--host","0.0.0.0","--port","8000"]

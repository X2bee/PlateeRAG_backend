# ------------------------------------------------------------
# Build-time args (can be overridden by CI)
# ------------------------------------------------------------
ARG BASE_IMAGE=python:3.12-slim

# Optional groups to install from pyproject optional-dependencies
# e.g. UV_GROUPS="rag,jp-core,ml"
ARG UV_GROUPS=""

# If you need Torch CPU wheels, set to 1
ARG INSTALL_TORCH=0
ARG TORCH_VER=2.7.1
ARG TORCHAUDIO_VER=2.7.1

# ============================================================
# Base (shared)
# ============================================================
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

# ============================================================
# Builder (compilers only)
# ============================================================
FROM base AS builder

# Build tools needed to compile native deps
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential rustc cargo libpq-dev \
 && rm -rf /var/lib/apt/lists/*

# Install uv (resolver/installer)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV UV_BIN=/root/.local/bin/uv

WORKDIR /app

# Copy dependency metadata first for better caching.
# If you commit uv.lock later, add:  COPY uv.lock ./
COPY pyproject.toml ./

# Bring in build args (so they’re visible in this stage)
ARG UV_GROUPS
ARG INSTALL_TORCH
ARG TORCH_VER
ARG TORCHAUDIO_VER

# Create venv and install *base* deps locked.
# Then optionally install selected optional groups from pyproject.
RUN python -m venv ${VENV_PATH} \
 && ${UV_BIN} lock \
 && ${UV_BIN} sync --locked --python ${VENV_PATH}/bin/python \
 && if [ -n "${UV_GROUPS}" ]; then \
      for g in $(echo "${UV_GROUPS}" | tr ',' ' '); do \
        echo ">> Installing optional group: ${g}"; \
        ${UV_BIN} sync --locked --group "${g}" --python ${VENV_PATH}/bin/python; \
      done; \
    fi

# If you actually need Torch, install CPU-only wheels (smaller) from the official CPU index.
RUN if [ "${INSTALL_TORCH}" = "1" ]; then \
      ${VENV_PATH}/bin/pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==${TORCH_VER} torchaudio==${TORCHAUDIO_VER}; \
    else \
      echo ">> Skipping Torch installation (INSTALL_TORCH=${INSTALL_TORCH})"; \
    fi

# Now copy the application code
COPY . .

# (Optional sanity: ensure uvicorn is importable since it’s in base deps)
# RUN ${VENV_PATH}/bin/python -c "import uvicorn; print('uvicorn OK')"

# ============================================================
# Runtime (slim)
# ============================================================
FROM ${BASE_IMAGE} AS runtime
ARG DEBIAN_FRONTEND=noninteractive

# Only runtime libs (no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg espeak-ng libpq5 \
 && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN addgroup --system --gid 1001 app && \
    adduser  --system --uid 1001 --ingroup app --home /app app

WORKDIR /app
ENV VENV_PATH=/opt/venv
ENV PATH="${VENV_PATH}/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy the virtualenv and app from builder
COPY --from=builder ${VENV_PATH} ${VENV_PATH}
COPY --from=builder /app /app

EXPOSE 8000
USER app

# FastAPI entrypoint
CMD ["python","-m","uvicorn","main:app","--host","0.0.0.0","--port","8000"]

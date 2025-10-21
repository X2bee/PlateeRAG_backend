# ================== Build-time args ==================
ARG BASE_IMAGE=python:3.12-slim

# Optional groups from [project.optional-dependencies] (leave empty if none)
ARG UV_GROUPS=""

# Torch CPU wheels (0 = skip, 1 = install CPU-only)
ARG INSTALL_TORCH=0
ARG TORCH_VER=2.7.1
ARG TORCHAUDIO_VER=2.7.1

# ================== Base ==================
FROM ${BASE_IMAGE} AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git \
 && rm -rf /var/lib/apt/lists/*

ENV VENV_PATH=/opt/venv
ENV PATH="${VENV_PATH}/bin:${PATH}"

# ================== Builder (has compilers) ==================
FROM base AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential rustc cargo libpq-dev \
 && rm -rf /var/lib/apt/lists/*

# uv installer
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV UV_BIN=/root/.local/bin/uv

WORKDIR /app

# Copy dependency metadata first (for caching)
COPY pyproject.toml ./
# If you maintain a lockfile, also:
# COPY uv.lock ./

# Bring args into this stage
ARG UV_GROUPS
ARG INSTALL_TORCH
ARG TORCH_VER
ARG TORCHAUDIO_VER

# Create venv, install locked base deps, optional groups
RUN python -m venv ${VENV_PATH} \
 && ${UV_BIN} lock \
 && ${UV_BIN} sync --locked --python ${VENV_PATH}/bin/python \
 && if [ -n "${UV_GROUPS}" ]; then \
      for g in $(echo "${UV_GROUPS}" | tr ',' ' '); do \
        echo ">> Installing optional group: ${g}"; \
        ${UV_BIN} sync --locked --group "${g}" --python ${VENV_PATH}/bin/python; \
      done; \
    fi

# Optional: Torch CPU (only if you actually need Torch at runtime)
RUN if [ "${INSTALL_TORCH}" = "1" ]; then \
      ${VENV_PATH}/bin/pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==${TORCH_VER} torchaudio==${TORCHAUDIO_VER}; \
    else \
      echo ">> Skipping Torch (INSTALL_TORCH=${INSTALL_TORCH})"; \
    fi

# Safety net: ensure uvicorn & fastapi are present even if pyproject changes
RUN ${VENV_PATH}/bin/pip install --no-cache-dir "uvicorn>=0.38.0" "fastapi==0.116.1"

# Sanity checks: fail the build if imports don’t work
RUN ${VENV_PATH}/bin/python - <<'PY'
import sys
print("VENVPY:", sys.executable)
import uvicorn, fastapi
print("OK uvicorn:", uvicorn.__version__, " fastapi:", fastapi.__version__)
PY

# Now copy the rest of the app
COPY . .

# ================== Runtime (slim) ==================
FROM ${BASE_IMAGE} AS runtime
ARG DEBIAN_FRONTEND=noninteractive
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
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Copy venv and app from builder
COPY --from=builder ${VENV_PATH} ${VENV_PATH}
COPY --from=builder /app /app

EXPOSE 8000
USER app

# If your app object is in main.py -> app
CMD ["python","-m","uvicorn","main:app","--host","0.0.0.0","--port","8000"]

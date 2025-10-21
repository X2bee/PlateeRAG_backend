# ------------------------------------------------------------
# Build-time args (override from CI if needed)
# ------------------------------------------------------------
ARG BASE_IMAGE=python:3.12-slim

# Optional groups from [project.optional-dependencies] in pyproject.toml
# e.g. UV_GROUPS="rag,jp-core" or "" for none
ARG UV_GROUPS=""

# Torch CPU install toggle (0=skip, 1=install CPU-only wheels)
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
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git \
 && rm -rf /var/lib/apt/lists/*

ENV VENV_PATH=/opt/venv
ENV PATH="${VENV_PATH}/bin:${PATH}"

# ============================================================
# Builder (has compilers)
# ============================================================
FROM base AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential rustc cargo libpq-dev \
 && rm -rf /var/lib/apt/lists/*

# Install uv (fast resolver/installer)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV UV_BIN=/root/.local/bin/uv

WORKDIR /app

# Copy dependency metadata first (better cache). Add uv.lock if you have it.
COPY pyproject.toml ./
# COPY uv.lock ./

# Bring build args into this stage
ARG UV_GROUPS
ARG INSTALL_TORCH
ARG TORCH_VER
ARG TORCHAUDIO_VER

# Create venv and install base deps (locked).
# Then optionally install selected dependency groups.
RUN python -m venv ${VENV_PATH} \
 && ${UV_BIN} lock \
 && ${UV_BIN} sync --locked --python ${VENV_PATH}/bin/python \
 && if [ -n "${UV_GROUPS}" ]; then \
      for g in $(echo "${UV_GROUPS}" | tr ',' ' '); do \
        echo ">> Installing optional group: ${g}"; \
        ${UV_BIN} sync --locked --group "${g}" --python ${VENV_PATH}/bin/python; \
      done; \
    fi

# Optional: Torch CPU wheels (smaller than CUDA)
RUN if [ "${INSTALL_TORCH}" = "1" ]; then \
      ${VENV_PATH}/bin/pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==${TORCH_VER} torchaudio==${TORCHAUDIO_VER}; \
    else \
      echo ">> Skipping Torch installation (INSTALL_TORCH=${INSTALL_TORCH})"; \
    fi

# Ensure uvicorn is present even if pyproject is changed later;
# also sanity-check it's importable in THIS venv.
RUN ${VENV_PATH}/bin/pip install --no-cache-dir "uvicorn>=0.38.0" \
 && ${VENV_PATH}/bin/python - <<'PY'
import sys
print("PYTHON:", sys.executable)
import uvicorn
print("UVICORN:", uvicorn.__version__)
PY

# Now copy the application code
COPY . .

# ============================================================
# Runtime (slim, no compilers)
# ============================================================
FROM ${BASE_IMAGE} AS runtime
ARG DEBIAN_FRONTEND=noninteractive

# Only runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg espeak-ng libpq5 \
 && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN addgroup --system --gid 1001 app && \
    adduser  --system --uid 1001 --ingroup app --home /app app

WORKDIR /app

# Use the venv by default
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

# Either form is fine; PATH points to venv/bin so both work:
# CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]
CMD ["python","-m","uvicorn","main:app","--host","0.0.0.0","--port","8000"]

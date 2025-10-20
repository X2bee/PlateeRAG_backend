# ---- Base (shared) -----------------------------------------------------------
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_SYSTEM_PYTHON=1
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git \
 && rm -rf /var/lib/apt/lists/*

# One place for the venv; both stages will use it
ENV VENV_PATH=/opt/venv
ENV PATH="${VENV_PATH}/bin:${PATH}"

# ---- Builder (compilers here only) ------------------------------------------
FROM base AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential rustc cargo libpq-dev \
 && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV UV_BIN=/root/.local/bin/uv

WORKDIR /app

# Copy only files needed to resolve deps first (best layer cache)
COPY pyproject.toml ./
# COPY uv.lock ./

# Create the venv now so uv/pip installs go there
RUN python -m venv ${VENV_PATH}

# Resolve & install deps into the venv (offline-capable wheelhouse)
# If you have uv.lock, add:  --locked
RUN ${UV_BIN} export --format=requirements-txt --frozen > /tmp/requirements.txt \
 && ${VENV_PATH}/bin/pip install --upgrade pip wheel \
 && ${VENV_PATH}/bin/pip wheel --no-cache-dir --wheel-dir /wheelhouse -r /tmp/requirements.txt \
 && ${VENV_PATH}/bin/pip install --no-cache-dir --no-index --find-links=/wheelhouse -r /tmp/requirements.txt

# Now copy your application
COPY . .

# If you have a local package you need editable install for, uncomment:
# RUN ${VENV_PATH}/bin/pip install --no-cache-dir -e .

# ---- Runtime (tiny) ---------------------------------------------------------
FROM python:3.12-slim AS runtime
ARG DEBIAN_FRONTEND=noninteractive

# Runtime-only system libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg espeak-ng libpq5 \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VENV_PATH=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

# Non-root user
RUN addgroup --system --gid 1001 app && \
    adduser  --system --uid 1001 --ingroup app --home /app app

WORKDIR /app

# Copy venv & app from builder
COPY --from=builder ${VENV_PATH} ${VENV_PATH}
COPY --from=builder /app /app

# Configure cache dirs to a writable path (we’ll mount a PVC here in K8s)
ENV HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    XDG_CACHE_HOME=/app/.cache

EXPOSE 8000
USER app

# Default command (adjust if your entrypoint differs)
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
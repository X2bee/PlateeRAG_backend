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

# Copy all files first
COPY . .

# Create venv and install dependencies
ENV VENV_PATH=/opt/venv
RUN python -m venv ${VENV_PATH}

# Activate venv and install dependencies
RUN . ${VENV_PATH}/bin/activate && \
    ${UV_BIN} pip install --python ${VENV_PATH}/bin/python -e .

# Verify installation
RUN ${VENV_PATH}/bin/python -c "import fastapi; print('FastAPI installed successfully')"

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

# Change ownership to app user
RUN chown -R app:app ${VENV_PATH} /app

EXPOSE 8000
USER app

# Verify installation in runtime
RUN python -c "import fastapi; print('FastAPI available in runtime')"

# Adjust if your entrypoint/module differs
CMD ["python","-m","uvicorn","main:app","--host","0.0.0.0","--port","8000"]

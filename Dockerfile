# Optimized Dockerfile with minimal dependencies
ARG BASE_IMAGE=python:3.12-slim

# ---- Builder stage ----------------------------------------------------------
FROM ${BASE_IMAGE} AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

ARG DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements-minimal.txt ./

# Create virtual environment and install minimal dependencies
ENV VENV_PATH=/opt/venv
RUN python -m venv ${VENV_PATH}
ENV PATH="${VENV_PATH}/bin:${PATH}"

# Install only essential dependencies
RUN pip install --no-cache-dir -r requirements-minimal.txt && \
    # Clean up to reduce size
    find ${VENV_PATH} -type f -name "*.pyc" -delete && \
    find ${VENV_PATH} -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find ${VENV_PATH} -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find ${VENV_PATH} -name "*.dist-info" -exec rm -rf {} + 2>/dev/null || true

# Copy application code
COPY . .

# ---- Runtime stage ----------------------------------------------------------
FROM ${BASE_IMAGE} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ARG DEBIAN_FRONTEND=noninteractive

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN addgroup --system --gid 1001 app && \
    adduser --system --uid 1001 --ingroup app --home /app app

WORKDIR /app

# Set up virtual environment path
ENV VENV_PATH=/opt/venv
ENV PATH="${VENV_PATH}/bin:${PATH}"

# Copy virtual environment and application from builder
COPY --from=builder --chown=app:app ${VENV_PATH} ${VENV_PATH}
COPY --from=builder --chown=app:app /app /app

# Switch to non-root user
USER app

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
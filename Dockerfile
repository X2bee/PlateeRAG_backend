# Simple Dockerfile without virtual environment
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN addgroup --system --gid 1001 app && \
    adduser --system --uid 1001 --ingroup app --home /app app

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements-minimal.txt ./
RUN pip install --no-cache-dir -r requirements-minimal.txt && \
    pip cache purge

# Copy application code
COPY --chown=app:app . .

# Switch to non-root user
USER app

# Verify installation
RUN python -c "import fastapi; print('FastAPI version:', fastapi.__version__)" && \
    python -c "import uvicorn; print('Uvicorn version:', uvicorn.__version__)"

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
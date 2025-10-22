# --- Builder stage: install build tools & Python deps into a venv ---
FROM python:3.12.11-slim AS builder

ARG DEBIAN_FRONTEND=noninteractive
# Set to "true" if you need LibreOffice, fonts, etc (will increase final image size dramatically)
ARG INSTALL_OFFICE=false

# Install build and runtime system packages needed to build wheels (adjust as needed)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       curl \
       git \
       ca-certificates \
       gcc \
       libffi-dev \
       libssl-dev \
       libbz2-dev \
       liblzma-dev \
       libreadline-dev \
       zlib1g-dev \
       pkg-config \
       # nodejs may be needed for some build steps (optional)
       curl \
    && if [ "$INSTALL_OFFICE" = "true" ]; then \
          apt-get install -y --no-install-recommends poppler-utils \
            libreoffice-writer libreoffice-calc libreoffice libreoffice-l10n-ko \
            fonts-nanum fonts-nanum-extra nodejs npm rustc cargo; \
       fi \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for build/install steps (files will be owned by this user)
ARG APP_USER=app
ARG APP_UID=1000
RUN useradd -m -u ${APP_UID} ${APP_USER}

WORKDIR /src

# Copy only dependency manifests first to leverage Docker cache
# Adjust filenames if your project uses pyproject.toml or pyproject.yaml naming differences
COPY pyproject.toml pyproject.yaml* uv.lock* ./

# Create a venv and install runtime deps into it
ENV VENV_PATH=/opt/venv
RUN python -m venv ${VENV_PATH} \
    && ${VENV_PATH}/bin/pip install --upgrade pip setuptools wheel

# Install uv (the CLI you use) and then sync dependencies via uv
# (keeps dependencies inside venv)
RUN ${VENV_PATH}/bin/pip install uv \
    && ${VENV_PATH}/bin/uv sync

# Copy project files (source)
# Copy everything needed for the app. Use .dockerignore to avoid copying .git, tests, etc.
COPY --chown=${APP_USER}:${APP_USER} . /src

# If your app needs any build-step (npm build, compiling assets), do it here as the app user:
USER ${APP_USER}
# Example: if you need node/npm to build frontend assets, run them here:
# RUN if [ -f package.json ]; then npm ci && npm run build; fi

# Make sure venv executables are available
ENV PATH="${VENV_PATH}/bin:${PATH}"

# --- Final stage: minimal runtime image ---
FROM python:3.12.11-slim AS runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG INSTALL_OFFICE=false

# Install only minimal runtime system packages (and optionally heavy office deps)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates \
       libffi7 \
       libssl3 \
       # add other shared libs needed by wheels your app uses if you know them
    && if [ "$INSTALL_OFFICE" = "true" ]; then \
          apt-get install -y --no-install-recommends poppler-utils \
            libreoffice-writer libreoffice-calc libreoffice libreoffice-l10n-ko \
            fonts-nanum fonts-nanum-extra nodejs npm rustc cargo; \
       fi \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (keep uid consistent)
ARG APP_USER=app
ARG APP_UID=1000
RUN useradd -m -u ${APP_UID} ${APP_USER}

WORKDIR /app

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
# Copy app source (only what's needed)
COPY --from=builder --chown=${APP_USER}:${APP_USER} /src /app

ENV PATH="/opt/venv/bin:${PATH}"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose the port your app listens to (adjust if your app uses another port)
EXPOSE 8000

USER ${APP_USER}

# Default command (same as your entrypoint: uses 'uv')
# If you want explicit host/port, you can change the command.
CMD ["uv", "run", "python", "main.py", "--host", "0.0.0.0", "--port", "8000"]
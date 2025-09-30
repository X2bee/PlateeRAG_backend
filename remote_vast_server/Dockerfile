FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    rustc cargo espeak-ng ffmpeg \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade uv

# Copy only resources needed to run the proxy server
COPY config ./config
COPY controller ./controller
COPY constants ./constants
COPY service ./service
COPY remote_vast_server ./remote_vast_server

# Prepare the runtime script
RUN chmod +x remote_vast_server/run_remote_vast.sh

CMD ["bash", "remote_vast_server/run_remote_vast.sh"]

#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/app}"
VAST_ENV="${VAST_ENV:-${PROJECT_ROOT}/.vast-env}"
REMOTE_VAST_HOST="${REMOTE_VAST_HOST:-0.0.0.0}"
REMOTE_VAST_PORT="${REMOTE_VAST_PORT:-9000}"

if [ ! -d "$PROJECT_ROOT" ]; then
  echo "[remote-vast] Project root '$PROJECT_ROOT' does not exist." >&2
  exit 1
fi

cd "$PROJECT_ROOT"

if [ ! -d "${VAST_ENV}" ]; then
  echo "[remote-vast] Creating uv virtual environment at ${VAST_ENV}" >&2
  uv venv "${VAST_ENV}"
fi

echo "[remote-vast] Installing proxy dependencies" >&2
uv pip install --python "${VAST_ENV}/bin/python" -r remote_vast_server/requirements.txt

export PYTHONPATH="${PYTHONPATH:-/app}"
export CONFIG_INCLUDE_ONLY="${CONFIG_INCLUDE_ONLY:-database,vast}"

exec "${VAST_ENV}/bin/uvicorn" remote_vast_server.main:app --host "${REMOTE_VAST_HOST}" --port "${REMOTE_VAST_PORT}"

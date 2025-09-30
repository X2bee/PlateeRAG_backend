#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VAST_ENV="${VAST_ENV:-${PROJECT_ROOT}/.vast-env}"
REMOTE_VAST_HOST="${REMOTE_VAST_HOST:-0.0.0.0}"
REMOTE_VAST_PORT="${REMOTE_VAST_PORT:-9000}"

if [ ! -d "${VAST_ENV}" ]; then
  echo "[remote-vast] Creating uv virtual environment at ${VAST_ENV}" >&2
  uv venv "${VAST_ENV}"
  uv pip install --python "${VAST_ENV}/bin/python" .
fi

echo "[remote-vast] Ensuring vastai package is installed" >&2
uv pip install --python "${VAST_ENV}/bin/python" vastai

exec "${VAST_ENV}/bin/uvicorn" remote_vast_server.main:app --host "${REMOTE_VAST_HOST}" --port "${REMOTE_VAST_PORT}"

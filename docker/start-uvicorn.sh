#!/usr/bin/env sh
set -eu

APP_MODULE="${UVICORN_APP:-mom_service.main:app}"
HOST="${UVICORN_HOST:-0.0.0.0}"
PORT="${UVICORN_PORT:-8000}"
RELOAD="${UVICORN_RELOAD:-false}"

if [ "$RELOAD" = "true" ]; then
  exec uvicorn "$APP_MODULE" \
    --host "$HOST" \
    --port "$PORT" \
    --reload \
    --reload-dir /app \
    --reload-include "*.py" \
    --reload-include "*.yaml" \
    --reload-include "*.yml"
fi

exec uvicorn "$APP_MODULE" \
  --host "$HOST" \
  --port "$PORT" \
  --workers "${UVICORN_WORKERS:-1}"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BACKEND_PORT="${COGNIX_BACKEND_PORT:-8000}"
FRONTEND_PORT="${COGNIX_FRONTEND_PORT:-5173}"

if ! command -v python >/dev/null 2>&1; then
  echo "Python is required. Install Python 3.11+ and run this script again." >&2
  exit 1
fi

if ! command -v node >/dev/null 2>&1; then
  echo "Node.js is required. Install Node.js 20+ and run this script again." >&2
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required. Install npm and run this script again." >&2
  exit 1
fi

mkdir -p \
  data/raw \
  data/processed \
  data/chroma \
  data/logs \
  wiki/outputs/analysis \
  wiki/_health \
  wiki/_indexes

if [ ! -d ".venv" ]; then
  echo "Creating Python virtual environment..."
  python -m venv .venv
fi

backend_deps_ready() {
  ".venv/bin/python" - <<'PY'
import importlib.util

modules = [
    "fastapi",
    "uvicorn",
    "pydantic",
    "pydantic_settings",
    "httpx",
    "multipart",
    "pypdf",
    "chromadb",
    "watchdog",
    "apscheduler",
]
missing = [module for module in modules if importlib.util.find_spec(module) is None]
if missing:
    print(", ".join(missing))
    raise SystemExit(1)
PY
}

if [ "${COGNIX_UPDATE_DEPS:-0}" = "1" ]; then
  echo "Installing/updating backend dependencies..."
  ".venv/bin/python" -m pip install -e backend
elif MISSING_BACKEND_DEPS="$(backend_deps_ready)"; then
  echo "Backend dependencies already installed."
else
  echo "Missing backend dependencies: ${MISSING_BACKEND_DEPS}"
  echo "Installing/updating backend dependencies..."
  ".venv/bin/python" -m pip install -e backend
fi

if [ ! -d "frontend/node_modules" ]; then
  echo "Installing frontend dependencies..."
  (cd frontend && npm install)
fi

cleanup() {
  if [ -n "${BACKEND_PID:-}" ] && kill -0 "$BACKEND_PID" >/dev/null 2>&1; then
    kill "$BACKEND_PID" >/dev/null 2>&1 || true
  fi
  if [ -n "${FRONTEND_PID:-}" ] && kill -0 "$FRONTEND_PID" >/dev/null 2>&1; then
    kill "$FRONTEND_PID" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT INT TERM

echo "Starting Cognix backend on http://127.0.0.1:${BACKEND_PORT}"
".venv/bin/python" -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port "$BACKEND_PORT" &
BACKEND_PID="$!"

echo "Starting Cognix web UI on http://127.0.0.1:${FRONTEND_PORT}"
(cd frontend && npm run dev -- --host 127.0.0.1 --port "$FRONTEND_PORT") &
FRONTEND_PID="$!"

cat <<EOF

Cognix is starting.

Web UI:  http://127.0.0.1:${FRONTEND_PORT}
Backend: http://127.0.0.1:${BACKEND_PORT}

Press Ctrl+C to stop both services.
EOF

wait "$BACKEND_PID" "$FRONTEND_PID"

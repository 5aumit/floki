#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
PYTHON="$ROOT/mlflow_agentic_env/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "Environment not found at $ROOT/mlflow_agentic_env" >&2
  echo "Run: bash setup_env.sh" >&2
  exit 1
fi

exec "$PYTHON" "$ROOT/src/app.py" --tui "$@"

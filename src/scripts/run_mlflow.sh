#!/bin/bash
# Start MLflow UI against Floki's bundled sample database.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATABASE="$ROOT/data/mlflow.db"

if [[ ! -f "$DATABASE" ]]; then
  echo "Sample database not found: $DATABASE" >&2
  exit 1
fi

exec mlflow ui --backend-store-uri "sqlite:///$DATABASE" --port 5000

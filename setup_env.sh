#!/bin/bash
# Create or update the project-local conda env at ./mlflow_agentic_env
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

ENV_PATH="$ROOT/mlflow_agentic_env"

if [[ ! -d "$ENV_PATH/bin" ]]; then
  echo "Creating conda env at $ENV_PATH ..."
  conda env create -f environment.yml -p "$ENV_PATH"
else
  echo "Updating conda env at $ENV_PATH ..."
  conda env update -f environment.yml -p "$ENV_PATH" --prune
fi

echo "Installing pip requirements ..."
"$ENV_PATH/bin/pip" install -r requirements.txt

echo "Verifying TUI dependencies ..."
"$ENV_PATH/bin/python" -c "
from textual import on, work
import pydantic
print('textual + pydantic OK')
"

echo "Done. Activate with:"
echo "  conda activate $ENV_PATH"

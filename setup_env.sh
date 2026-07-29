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

echo "Verifying application and test dependencies ..."
"$ENV_PATH/bin/python" -c "
    from textual import on, work
    import pydantic
    import pytest
    print('textual + pydantic + pytest OK')
"

echo "Done. Activate with:"
echo "  conda activate $ENV_PATH"

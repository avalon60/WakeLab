#!/usr/bin/env bash
#
# Author: Clive Bostock
# Date: 15-May-2026
# Purpose: Run the ONNX external data repackaging utility from Poetry or a local .venv.
# Usage: ./bin/newname-model.sh --source-onnx model.onnx --target-onnx out/model.onnx --target-data out/model.data

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_SCRIPT="$SCRIPT_DIR/newname_model.py"

if command -v poetry >/dev/null 2>&1; then
  exec poetry --project "$PROJECT_ROOT" run python3 "$PYTHON_SCRIPT" "$@"
fi

VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

if [[ -x "$VENV_PYTHON" ]]; then
  exec "$VENV_PYTHON" "$PYTHON_SCRIPT" "$@"
fi

cat >&2 <<EOF
Error: unable to run the ONNX repackaging utility.

Neither Poetry nor a project-local virtual environment was found.
Install Poetry or create a virtual environment at:
  $PROJECT_ROOT/.venv
EOF
exit 1

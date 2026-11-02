#!/usr/bin/env bash
# Configure an existing Vertex endpoint, or deploy Qwen when none is configured.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -x "$ROOT/.venv-vertex/bin/python" ]]; then
  VERTEX_PY="$ROOT/.venv-vertex/bin/python"
else
  VERTEX_PY="$(command -v python3)"
fi

exec "$VERTEX_PY" "$ROOT/scripts/vertex_deploy.py"

#!/usr/bin/env bash
# Start the Streamlit application using an available project environment.
#
# Usage:
#   bash scripts/run_app.sh
#   bash scripts/run_app.sh --server.port 8502
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -x "$PROJECT_DIR/.venv/bin/streamlit" ]]; then
  STREAMLIT="$PROJECT_DIR/.venv/bin/streamlit"
elif [[ -x "$PROJECT_DIR/venv/bin/streamlit" ]]; then
  STREAMLIT="$PROJECT_DIR/venv/bin/streamlit"
else
  echo "ERROR: No Streamlit virtual environment was found." >&2
  echo "Create one with a supported CPython version, then install requirements:" >&2
  echo "  python3.12 -m venv venv" >&2
  echo "  source venv/bin/activate" >&2
  echo "  python -m pip install -r requirements.txt" >&2
  exit 1
fi

exec "$STREAMLIT" run "$PROJECT_DIR/app.py" "$@"

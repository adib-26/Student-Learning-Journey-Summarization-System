#!/usr/bin/env bash
#
# Start the local vLLM inference server (vllm-metal) for the
# Educational Data Analytics Platform.
#
# The Streamlit app talks to this server over the OpenAI-compatible API at
# http://127.0.0.1:8000/v1 (override with LLM_BASE_URL on the app side).
#
# First run: installs vllm-metal via the official installer into
# ~/.venv-vllm-metal (bundles a native arm64 Python 3.12, vLLM core, and the
# Metal GPU plugin), then downloads the model from Hugging Face (~2 GB).
# Later runs: starts serving within seconds.
#
# Usage:
#   bash scripts/start_vllm.sh
#
# Tunables (environment variables):
#   LLM_MODEL                      model to serve (default: Qwen2.5-3B-Instruct 4-bit MLX)
#   VLLM_PORT                      server port (default 8000)
#   VLLM_MAX_MODEL_LEN             context window cap, bounds KV-cache memory (default 4096)
#   VLLM_GPU_MEMORY_UTILIZATION    fraction of unified memory vLLM may claim (default 0.55)
#   VLLM_MAX_NUM_SEQS              max concurrent sequences (default 1)
#   VLLM_HOST                      bind address (default 127.0.0.1; local-only)
#   VLLM_EXTRA_ARGS                extra args appended to `vllm serve`
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ENV_FILE="$PROJECT_DIR/.env"

load_project_llm_settings() {
  # Do not source .env: it is configuration data, not shell code.  This
  # deliberately accepts only the server settings this script owns and leaves
  # an explicitly exported shell value untouched.
  local env_file="$1"
  local line key value first_char last_char value_length

  [[ -r "$env_file" ]] || return 0

  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "$line" || "${line:0:1}" == "#" ]] && continue

    if [[ ! "$line" =~ ^([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=(.*)$ ]]; then
      continue
    fi
    key="${BASH_REMATCH[1]}"
    value="${BASH_REMATCH[2]}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"

    case "$key" in
      LLM_MODEL|VLLM_PORT|VLLM_MAX_MODEL_LEN|VLLM_GPU_MEMORY_UTILIZATION|VLLM_MAX_NUM_SEQS|VLLM_HOST|VLLM_METAL_MEMORY_FRACTION|VLLM_ALLOW_CPU) ;;
      *) continue ;;
    esac

    # Preserve a variable that the user explicitly exported, even if it is
    # intentionally empty.  Indirect expansion works on macOS's Bash 3.2.
    [[ "${!key+x}" == "x" ]] && continue

    # Support the ordinary quoted .env form without expanding $, backticks,
    # command substitutions, or other shell syntax from the file.
    value_length=${#value}
    if (( value_length >= 2 )); then
      first_char="${value:0:1}"
      last_char="${value:$((value_length - 1)):1}"
      if [[ ( "$first_char" == '"' && "$last_char" == '"' ) || ( "$first_char" == "'" && "$last_char" == "'" ) ]]; then
        value="${value:1:$((value_length - 2))}"
      fi
    fi

    export "$key=$value"
  done < "$env_file"
}

load_project_llm_settings "$PROJECT_ENV_FILE"

# The upstream installer always creates this fixed location.  Supporting a
# VLLM_VENV_DIR override here made post-install verification fail misleadingly.
VENV_DIR="$HOME/.venv-vllm-metal"
MODEL="${LLM_MODEL:-mlx-community/Qwen2.5-3B-Instruct-4bit}"
PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
GPU_MEM_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.55}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1}"
HOST="${VLLM_HOST:-127.0.0.1}"

# ``auto`` lets the Metal plugin honour --gpu-memory-utilization while using
# its normal paged-attention allocator.
export VLLM_METAL_MEMORY_FRACTION="${VLLM_METAL_MEMORY_FRACTION:-auto}"

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_memory_fraction() {
  awk -v value="$1" 'BEGIN { exit !(value > 0 && value <= 1) }'
}

if ! is_positive_integer "$PORT" || ! is_positive_integer "$MAX_MODEL_LEN" || ! is_positive_integer "$MAX_NUM_SEQS"; then
  echo "ERROR: VLLM_PORT, VLLM_MAX_MODEL_LEN, and VLLM_MAX_NUM_SEQS must be positive integers." >&2
  exit 2
fi
if ! is_memory_fraction "$GPU_MEM_UTIL"; then
  echo "ERROR: VLLM_GPU_MEMORY_UTILIZATION must be a number greater than 0 and at most 1." >&2
  exit 2
fi
if [[ "$MAX_NUM_SEQS" -gt 1 ]]; then
  echo "WARNING: MAX_NUM_SEQS=$MAX_NUM_SEQS may cause swapping or long queues on an 8 GB Mac." >&2
fi
if awk -v value="$GPU_MEM_UTIL" 'BEGIN { exit !(value > 0.60) }'; then
  echo "WARNING: GPU memory utilization above 0.60 can make macOS and Streamlit unresponsive on 8 GB." >&2
fi

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "ERROR: this script targets Apple Silicon macOS (detected: $(uname -sm))." >&2
  echo "On Linux/NVIDIA, install vLLM normally: pip install vllm" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 1. One-time install via the official vllm-metal installer.
# ---------------------------------------------------------------------------
if [[ ! -x "$VENV_DIR/bin/vllm" ]]; then
  echo "==> vllm-metal not found; running one-time install into $VENV_DIR ..."
  curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
  if [[ ! -x "$VENV_DIR/bin/vllm" ]]; then
    echo "ERROR: install finished but $VENV_DIR/bin/vllm is missing." >&2
    exit 1
  fi
fi

PY="$VENV_DIR/bin/python"

# ---------------------------------------------------------------------------
# 2. Guard against transformers >= 5.13, which silently breaks Metal platform
#    detection: vLLM falls back to the non-optimized CPU backend with no error.
# ---------------------------------------------------------------------------
if ! "$PY" - <<'PYEOF'
import sys
from importlib.metadata import version
from packaging.version import Version
sys.exit(1 if Version(version("transformers")) >= Version("5.13") else 0)
PYEOF
then
  echo "==> Pinning transformers<5.13 (5.13+ silently disables Metal acceleration)"
  if command -v uv >/dev/null 2>&1; then
    VIRTUAL_ENV="$VENV_DIR" uv pip install 'transformers<5.13'
  else
    "$PY" -m pip install 'transformers<5.13'
  fi
fi

# ---------------------------------------------------------------------------
# 3. Verify the Metal platform actually registered (not a CPU fallback).
# ---------------------------------------------------------------------------
if ! "$PY" - <<'PYEOF'
from vllm.platforms import current_platform
name = type(current_platform).__name__
print(f"==> vLLM platform: {name}")
raise SystemExit(0 if "Metal" in name else 1)
PYEOF
then
  if [[ "${VLLM_ALLOW_CPU:-0}" != "1" ]]; then
    echo "ERROR: Metal plugin did not register; refusing to serve on the slow CPU backend." >&2
    echo "Re-run the installer, or set VLLM_ALLOW_CPU=1 to override." >&2
    exit 1
  fi
  echo "WARNING: continuing on CPU backend (VLLM_ALLOW_CPU=1). Expect very slow inference." >&2
fi

# ---------------------------------------------------------------------------
# 4. Serve.
# ---------------------------------------------------------------------------
echo "==> Serving $MODEL on http://$HOST:$PORT/v1"
echo "    (first run downloads the model from Hugging Face, ~2 GB)"
echo "    M1/8 GB profile: max context $MAX_MODEL_LEN, $MAX_NUM_SEQS active sequence(s), Metal memory fraction $GPU_MEM_UTIL"

# shellcheck disable=SC2086
exec "$VENV_DIR/bin/vllm" serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  ${VLLM_EXTRA_ARGS:-}

#!/usr/bin/env bash
#
# Start one local MLX-VLM server for both Qwen analysis and GLM-OCR.
#
# This is the recommended M1/8 GB profile for PDFs and images. No model is
# preloaded: each request selects its model, and MLX-VLM replaces the prior
# text-generation model before loading the next one. This avoids keeping Qwen
# and GLM-OCR resident together in unified memory.
#
# The process is offline-only. Both configured models must already be in the
# local Hugging Face cache; uploads can never trigger a download through this
# launcher. It deliberately binds to loopback only.
#
# Do not run this alongside scripts/start_vllm.sh: both use port 8000 by
# default. The Streamlit app must use LLM_BASE_URL and GLM_OCR_BASE_URL that
# point at this server's /v1 endpoint.
#
# Usage:
#   bash scripts/start_mlx_vlm.sh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ENV_FILE="$PROJECT_DIR/.env"

load_project_mlx_settings() {
  # Never source .env: it is configuration data, not shell code. Only values
  # this launcher owns (or needs to preflight) are accepted, and an explicitly
  # exported environment variable always wins.
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
      LLM_MODEL|LLM_BASE_URL|GLM_OCR_MODEL|GLM_OCR_MODE|GLM_OCR_BASE_URL|GLM_OCR_ALLOW_MODEL_DOWNLOAD|MLX_VLM_PYTHON|MLX_VLM_HOST|MLX_VLM_PORT|MLX_VLM_VISION_CACHE_SIZE|MLX_VLM_PREFILL_STEP_SIZE|MLX_VLM_MAX_TOKENS|MLX_VLM_MAX_KV_SIZE|MLX_VLM_LOG_LEVEL|MLX_VLM_TRUST_REMOTE_CODE) ;;
      *) continue ;;
    esac
    [[ "${!key+x}" == "x" ]] && continue

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

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage: bash scripts/start_mlx_vlm.sh

Start the local, offline MLX-VLM server for the configured Qwen and GLM-OCR
models. It intentionally accepts no extra server arguments; use the bounded
MLX_VLM_* settings in .env instead.
EOF
}

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_truthy() {
  case "$1" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

if [[ $# -gt 0 ]]; then
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      die "Unknown option: $1"
      ;;
  esac
fi

load_project_mlx_settings "$PROJECT_ENV_FILE"

[[ -n "${HOME:-}" ]] || die "HOME is not set, so the vllm-metal Python location cannot be determined."
PYTHON="${MLX_VLM_PYTHON:-$HOME/.venv-vllm-metal/bin/python}"
HOST="${MLX_VLM_HOST:-127.0.0.1}"
PORT="${MLX_VLM_PORT:-8000}"
VISION_CACHE_SIZE="${MLX_VLM_VISION_CACHE_SIZE:-1}"
PREFILL_STEP_SIZE="${MLX_VLM_PREFILL_STEP_SIZE:-512}"
MAX_TOKENS="${MLX_VLM_MAX_TOKENS:-1024}"
MAX_KV_SIZE="${MLX_VLM_MAX_KV_SIZE:-4096}"
LOG_LEVEL="${MLX_VLM_LOG_LEVEL:-INFO}"
TRUST_REMOTE_CODE="${MLX_VLM_TRUST_REMOTE_CODE:-1}"
QWEN_MODEL="${LLM_MODEL:-mlx-community/Qwen2.5-3B-Instruct-4bit}"
GLM_MODEL="${GLM_OCR_MODEL:-mlx-community/GLM-OCR-bf16}"
if [[ "$HOST" == "::1" ]]; then
  EXPECTED_BASE_URL="http://[::1]:${PORT}/v1"
else
  EXPECTED_BASE_URL="http://${HOST}:${PORT}/v1"
fi
EFFECTIVE_LLM_BASE_URL="${LLM_BASE_URL:-$EXPECTED_BASE_URL}"
EFFECTIVE_GLM_BASE_URL="${GLM_OCR_BASE_URL:-$EXPECTED_BASE_URL}"
EFFECTIVE_GLM_MODE="${GLM_OCR_MODE:-http}"
case "$EFFECTIVE_GLM_MODE" in
  auto) EFFECTIVE_GLM_MODE="http" ;;
  force) EFFECTIVE_GLM_MODE="http-force" ;;
esac

[[ -x "$PYTHON" ]] || die "MLX-VLM Python is unavailable: ${PYTHON}. Run bash scripts/start_vllm.sh once to provision ~/.venv-vllm-metal, or set MLX_VLM_PYTHON."
[[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]] || die "This launcher requires Apple Silicon macOS (detected: $(uname -sm))."
case "$HOST" in
  127.0.0.1|localhost|::1) ;;
  *) die "MLX_VLM_HOST must be loopback-only (127.0.0.1, localhost, or ::1)." ;;
esac
for numeric_name in PORT VISION_CACHE_SIZE PREFILL_STEP_SIZE MAX_TOKENS MAX_KV_SIZE; do
  numeric_value="${!numeric_name}"
  is_positive_integer "$numeric_value" || die "${numeric_name} must be a positive whole number."
done
[[ "$PORT" -le 65535 ]] || die "MLX_VLM_PORT must be at most 65535."
case "$LOG_LEVEL" in
  DEBUG|INFO|WARNING|ERROR|CRITICAL) ;;
  *) die "MLX_VLM_LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR, or CRITICAL." ;;
esac
[[ -n "$QWEN_MODEL" && -n "$GLM_MODEL" ]] || die "LLM_MODEL and GLM_OCR_MODEL cannot be empty."

# Explicitly prevent both the hub client and Transformers from falling back to
# the network. This remains in force even if a user accidentally sets
# GLM_OCR_ALLOW_MODEL_DOWNLOAD=1 in .env.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

cache_report=""
if ! cache_report="$("$PYTHON" - "$QWEN_MODEL" "$GLM_MODEL" <<'PY'
import sys
from importlib.util import find_spec

if find_spec("mlx_vlm") is None:
    print("mlx_vlm is not installed in the configured Python", file=sys.stderr)
    raise SystemExit(2)
if find_spec("huggingface_hub") is None:
    print("huggingface_hub is not installed in the configured Python", file=sys.stderr)
    raise SystemExit(3)

from huggingface_hub import scan_cache_dir

wanted = tuple(sys.argv[1:])
try:
    repositories = {repo.repo_id: repo for repo in scan_cache_dir().repos}
except Exception as exc:
    print(f"Cannot inspect the local model cache: {type(exc).__name__}", file=sys.stderr)
    raise SystemExit(4)

missing = []
for model_id in wanted:
    repo = repositories.get(model_id)
    if repo is None:
        missing.append(model_id)
        continue
    files = {entry.file_name for revision in repo.revisions for entry in revision.files}
    if "config.json" not in files or not any(name.endswith(".safetensors") for name in files):
        missing.append(model_id)
if missing:
    print("Missing or incomplete local model cache: " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(5)

def human_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.1f} {unit}"
        amount /= 1024

for model_id in wanted:
    repo = repositories[model_id]
    print(f"  {model_id} ({human_bytes(repo.size_on_disk)})")
PY
)"; then
  die "Both Qwen and GLM-OCR must be cached locally before this offline server can start. No model was downloaded."
fi

if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  die "Port ${PORT} is already in use. Stop scripts/start_vllm.sh (or the other local server) before starting MLX-VLM."
fi

if [[ "$EFFECTIVE_LLM_BASE_URL" != "$EXPECTED_BASE_URL" ]]; then
  printf 'WARNING: LLM_BASE_URL is %s, but this server is %s. The app may not use this server.\n' "$EFFECTIVE_LLM_BASE_URL" "$EXPECTED_BASE_URL" >&2
fi
if [[ "$EFFECTIVE_GLM_BASE_URL" != "$EXPECTED_BASE_URL" ]]; then
  printf 'WARNING: GLM_OCR_BASE_URL is %s, but this server is %s. GLM-OCR HTTP mode may not use this server.\n' "$EFFECTIVE_GLM_BASE_URL" "$EXPECTED_BASE_URL" >&2
fi
if [[ "$EFFECTIVE_GLM_MODE" != "http" && "$EFFECTIVE_GLM_MODE" != "http-force" ]]; then
  printf 'WARNING: GLM_OCR_MODE=%s uses the subprocess fallback, not this shared server.\n' "$EFFECTIVE_GLM_MODE" >&2
fi
if is_truthy "${GLM_OCR_ALLOW_MODEL_DOWNLOAD:-0}"; then
  printf 'WARNING: GLM_OCR_ALLOW_MODEL_DOWNLOAD is set, but this launcher remains offline by design.\n' >&2
fi

server_args=(
  --host "$HOST"
  --port "$PORT"
  --vision-cache-size "$VISION_CACHE_SIZE"
  --prefill-step-size "$PREFILL_STEP_SIZE"
  --max-tokens "$MAX_TOKENS"
  --max-kv-size "$MAX_KV_SIZE"
  --log-level "$LOG_LEVEL"
)
if is_truthy "$TRUST_REMOTE_CODE"; then
  # GLM-OCR's local cached model needs the MLX-VLM remote-code flag. Offline
  # mode means this does not fetch or execute newly downloaded model code.
  server_args+=(--trust-remote-code)
fi

printf '==> Starting shared MLX-VLM at %s\n' "$EXPECTED_BASE_URL"
printf '    Offline-only; no model is preloaded. One text-generation cache swaps Qwen and GLM-OCR per request.\n'
printf '    Cached models:\n%s\n' "$cache_report"
printf '    Do not also run scripts/start_vllm.sh on port %s.\n' "$PORT"

exec "$PYTHON" -m mlx_vlm.server "${server_args[@]}"

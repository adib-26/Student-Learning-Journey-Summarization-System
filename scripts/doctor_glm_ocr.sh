#!/usr/bin/env bash
#
# Read-only GLM-OCR preflight for the shared MLX-VLM endpoint (recommended)
# or the explicit short-lived subprocess fallback.
#
# It never starts or stops a server, loads a model, sends a document, changes
# packages, or downloads model files. In HTTP mode it makes only GET /health,
# which does not select or load a model.
#
# Usage:
#   bash scripts/doctor_glm_ocr.sh
#   bash scripts/doctor_glm_ocr.sh --endpoint http://127.0.0.1:8000/v1
#   bash scripts/doctor_glm_ocr.sh --model mlx-community/GLM-OCR-bf16
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ENV_FILE="$PROJECT_DIR/.env"
WORKER_PATH="$PROJECT_DIR/scripts/glm_ocr_worker.py"
DEFAULT_MODEL="mlx-community/GLM-OCR-bf16"
DEFAULT_ENDPOINT="http://127.0.0.1:8000/v1"

load_project_glm_ocr_settings() {
  # Treat .env as configuration data, never shell source. Only this script's
  # allow-listed settings are read and an exported shell value wins.
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
      GLM_OCR_ENABLED|GLM_OCR_MODE|GLM_OCR_MODEL|GLM_OCR_PYTHON|MLX_VLM_PYTHON|GLM_OCR_BASE_URL|GLM_OCR_ALLOW_REMOTE|GLM_OCR_ALLOW_MODEL_DOWNLOAD|GLM_OCR_HEALTH_TIMEOUT_SECONDS|GLM_OCR_MAX_PAGES|GLM_OCR_MAX_TOKENS) ;;
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
  printf 'FAIL: %s\n' "$*" >&2
  exit 1
}

is_truthy() {
  case "$1" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

is_positive_number() {
  awk -v value="$1" 'BEGIN { exit !(value > 0) }'
}

usage() {
  cat <<'EOF'
Usage: bash scripts/doctor_glm_ocr.sh [options]

Read-only GLM-OCR preflight. In the recommended GLM_OCR_MODE=http profile it
checks the configured MLX-VLM endpoint with GET /health and confirms the local
model cache; it does not load a model. In subprocess mode it additionally
checks the fallback worker's syntax.

Options:
  --endpoint URL     Override GLM_OCR_BASE_URL for this check.
  --python PATH      Override GLM_OCR_PYTHON for this check.
  --model MODEL_ID   Override GLM_OCR_MODEL for this check.
  -h, --help         Show this help text.
EOF
}

require_option_value() {
  local option_name="$1"
  [[ $# -ge 2 && -n "$2" ]] || die "Missing a value for ${option_name}."
}

load_project_glm_ocr_settings "$PROJECT_ENV_FILE"

[[ -n "${HOME:-}" ]] || die "HOME is not set, so the vllm-metal Python location cannot be determined."
default_python="$HOME/.venv-vllm-metal/bin/python"
ocr_python="${GLM_OCR_PYTHON:-${MLX_VLM_PYTHON:-$default_python}}"
model="${GLM_OCR_MODEL:-$DEFAULT_MODEL}"
mode="${GLM_OCR_MODE:-http}"
base_url="${GLM_OCR_BASE_URL:-$DEFAULT_ENDPOINT}"
case "$mode" in
  auto) mode="http" ;;
  force) mode="http-force" ;;
esac

while [[ $# -gt 0 ]]; do
  case "$1" in
    --endpoint)
      require_option_value "$1" "${2:-}"
      base_url="$2"
      shift 2
      ;;
    --python)
      require_option_value "$1" "${2:-}"
      ocr_python="$2"
      shift 2
      ;;
    --model)
      require_option_value "$1" "${2:-}"
      model="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      die "Unknown option: $1"
      ;;
  esac
done

if ! is_truthy "${GLM_OCR_ENABLED:-1}" || [[ "$mode" == "off" || "$mode" == "disabled" || "$mode" == "0" ]]; then
  printf 'PASS: GLM-OCR is disabled by configuration; no model or endpoint check was needed.\n'
  exit 0
fi

[[ -x "$ocr_python" ]] || die "Configured MLX Python is not executable: ${ocr_python}. Run bash scripts/start_vllm.sh once to provision it, or set GLM_OCR_PYTHON."
[[ -n "$model" ]] || die "GLM_OCR_MODEL cannot be empty."

transport="subprocess"
case "$mode" in
  http|http-force) transport="http" ;;
esac
check_worker="0"
if [[ "$transport" == "subprocess" ]]; then
  [[ -f "$WORKER_PATH" ]] || die "GLM-OCR subprocess worker is missing: ${WORKER_PATH}"
  check_worker="1"
fi

readiness=""
if ! readiness="$("$ocr_python" - "$model" "$WORKER_PATH" "$check_worker" <<'PY'
import ast
import sys
from importlib.util import find_spec
from pathlib import Path

model_id, worker_path, check_worker = sys.argv[1], Path(sys.argv[2]), sys.argv[3] == "1"
missing = [name for name in ("mlx", "mlx_vlm", "huggingface_hub") if find_spec(name) is None]
if missing:
    print("Missing runtime package(s): " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(2)

if check_worker:
    try:
        ast.parse(worker_path.read_text(encoding="utf-8"), filename=str(worker_path))
    except (OSError, SyntaxError) as exc:
        print(f"Cannot read GLM-OCR worker: {exc}", file=sys.stderr)
        raise SystemExit(3)

try:
    from huggingface_hub import scan_cache_dir
    report = scan_cache_dir()
except Exception as exc:
    print(f"Cannot inspect the local Hugging Face cache: {type(exc).__name__}", file=sys.stderr)
    raise SystemExit(4)

repositories = [repo for repo in report.repos if repo.repo_id == model_id]
if not repositories:
    print(f"Model is not cached locally: {model_id}", file=sys.stderr)
    raise SystemExit(5)

repository = repositories[0]
files = {file.file_name for revision in repository.revisions for file in revision.files}
if not any(name.endswith(".safetensors") for name in files) or "config.json" not in files:
    print(f"Cached model is incomplete: {model_id}", file=sys.stderr)
    raise SystemExit(6)

def human_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.1f} {unit}"
        amount /= 1024

print(f"Python: {sys.executable} ({sys.version.split()[0]})")
print("MLX + MLX-VLM: available (not loaded)")
print(f"Cached model: {model_id} ({human_bytes(repository.size_on_disk)}, {len(files)} file(s))")
PY
)"; then
  die "GLM-OCR preflight did not pass. The app will safely use its native/Tesseract fallback until this is fixed."
fi

printf 'GLM-OCR preflight (read-only)\n'
printf '%s\n' "$readiness"
printf 'Mode: %s; transport: %s; pages: %s; max tokens: %s\n' \
  "$mode" "$transport" "${GLM_OCR_MAX_PAGES:-1}" "${GLM_OCR_MAX_TOKENS:-1024}"

if [[ "$transport" == "http" ]]; then
  [[ -n "$base_url" ]] || die "GLM_OCR_BASE_URL cannot be empty in HTTP mode."
  endpoint_root="${base_url%/}"
  if [[ "$endpoint_root" == */v1 ]]; then
    endpoint_root="${endpoint_root%/v1}"
  fi
  case "$endpoint_root" in
    http://127.0.0.1*|http://localhost*|http://[[]::1[]]*|https://127.0.0.1*|https://localhost*|https://[[]::1[]]*) ;;
    *)
      is_truthy "${GLM_OCR_ALLOW_REMOTE:-0}" || die "Refusing a non-loopback GLM_OCR_BASE_URL without GLM_OCR_ALLOW_REMOTE=1."
      ;;
  esac
  command -v curl >/dev/null 2>&1 || die "curl is required for the shared endpoint health check."
  health_timeout="${GLM_OCR_HEALTH_TIMEOUT_SECONDS:-1.5}"
  is_positive_number "$health_timeout" || die "GLM_OCR_HEALTH_TIMEOUT_SECONDS must be greater than zero."
  health_url="$endpoint_root/health"
  # A classic vLLM /health response can also return 200. Require MLX-VLM's
  # own runtime marker so this doctor never green-lights the unsafe two-server
  # profile by mistake. The payload is not printed because it can list models.
  health_response="$(curl --silent --show-error --connect-timeout "$health_timeout" --max-time "$health_timeout" --write-out $'\n%{http_code}' "$health_url" || true)"
  health_status="${health_response##*$'\n'}"
  health_payload="${health_response%$'\n'*}"
  [[ "$health_status" =~ ^2[0-9][0-9]$ ]] || die "Cannot reach shared MLX-VLM health endpoint at ${health_url}. Start it with: bash scripts/start_mlx_vlm.sh"
  [[ "$health_payload" == *'"loaded_model"'* || "$health_payload" == *'"loaded_models"'* || "$health_payload" == *'"apc_enabled"'* ]] || die "Endpoint at ${health_url} is not the expected MLX-VLM server. Stop legacy vLLM and start: bash scripts/start_mlx_vlm.sh"
  printf 'Shared MLX-VLM endpoint: %s (GET /health returned %s; no model was loaded)\n' "$base_url" "$health_status"
else
  printf 'Fallback worker: %s\n' "$WORKER_PATH"
fi

case "${GLM_OCR_ALLOW_MODEL_DOWNLOAD:-0}" in
  1|true|TRUE|yes|YES|on|ON)
    printf 'WARNING: GLM_OCR_ALLOW_MODEL_DOWNLOAD is enabled. The recommended shared launcher still forces offline mode.\n'
    ;;
  *)
    printf 'Offline policy: configured — GLM-OCR model downloads are disabled.\n'
    ;;
esac

if [[ "$transport" == "http" ]]; then
  printf 'PASS: shared MLX-VLM GLM-OCR endpoint is ready; no model was loaded.\n'
else
  printf 'PASS: GLM-OCR subprocess fallback is ready; no model was loaded.\n'
fi

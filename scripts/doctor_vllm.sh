#!/usr/bin/env bash
#
# Read-only readiness check for a local OpenAI-compatible vLLM endpoint.
#
# This script does not start, stop, install, or reconfigure vLLM.  It only
# calls GET /models and sends one tiny, deterministic chat completion.  It is
# deliberately small enough to run safely alongside the 4-bit 3B model on an
# 8 GB Apple-silicon Mac.
#
# Usage:
#   bash scripts/doctor_vllm.sh
#   bash scripts/doctor_vllm.sh --endpoint http://127.0.0.1:8000/v1
#   bash scripts/doctor_vllm.sh --model mlx-community/Qwen2.5-3B-Instruct-4bit
#
# Configuration (environment variables):
#   LLM_BASE_URL                     OpenAI-compatible base URL
#                                    (default: http://127.0.0.1:8000/v1)
#   LLM_MODEL                        Model ID to validate. If unset, the first
#                                    model advertised by /models is used.
#   LLM_API_KEY                      Optional API key for a protected server.
#   VLLM_DOCTOR_TIMEOUT_SECONDS      Total timeout per HTTP request (default: 45)
#   VLLM_DOCTOR_CONNECT_TIMEOUT_SECONDS
#                                    Connection timeout (default: 5)
#   VLLM_DOCTOR_PYTHON               Python executable used only for JSON parsing
#
# Exit status is 0 only when both API checks succeed.

set -o nounset
set -o pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ENV_FILE="$PROJECT_DIR/.env"

load_project_llm_settings() {
    # Do not source .env: it is configuration data, not shell code.  Keep the
    # readiness check aligned with the locally selected Qwen model, but do not
    # let it execute values or overwrite exported shell settings.
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
            LLM_BASE_URL|LLM_MODEL|LLM_API_KEY|VLLM_DOCTOR_TIMEOUT_SECONDS|VLLM_DOCTOR_CONNECT_TIMEOUT_SECONDS) ;;
            *) continue ;;
        esac

        # Preserve a variable that the user explicitly exported, even if it is
        # intentionally empty.  Indirect expansion works on macOS's Bash 3.2.
        [[ "${!key+x}" == "x" ]] && continue

        # Support ordinary quoted .env values without expanding any shell
        # syntax embedded in configuration data.
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

readonly DEFAULT_BASE_URL="http://127.0.0.1:8000/v1"
readonly DEFAULT_TIMEOUT_SECONDS=45
readonly DEFAULT_CONNECT_TIMEOUT_SECONDS=5

base_url="${LLM_BASE_URL:-$DEFAULT_BASE_URL}"
configured_model="${LLM_MODEL:-}"
request_timeout="${VLLM_DOCTOR_TIMEOUT_SECONDS:-$DEFAULT_TIMEOUT_SECONDS}"
connect_timeout="${VLLM_DOCTOR_CONNECT_TIMEOUT_SECONDS:-$DEFAULT_CONNECT_TIMEOUT_SECONDS}"
models_body=""
completion_body=""
json_python=""

die() {
    printf 'FAIL: %s\n' "$*" >&2
    exit 1
}

cleanup() {
    [[ -n "$models_body" ]] && rm -f -- "$models_body"
    [[ -n "$completion_body" ]] && rm -f -- "$completion_body"
}

usage() {
    cat <<'EOF'
Usage: bash scripts/doctor_vllm.sh [options]

Read-only check of an OpenAI-compatible vLLM endpoint. The endpoint defaults
to LLM_BASE_URL or http://127.0.0.1:8000/v1.

Options:
  --endpoint URL       Override LLM_BASE_URL for this check.
  --model MODEL_ID     Override LLM_MODEL for this check.
  --timeout SECONDS    Override VLLM_DOCTOR_TIMEOUT_SECONDS for this check.
  -h, --help           Show this help text.
EOF
}

require_option_value() {
    local option_name="$1"
    [[ $# -ge 2 && -n "$2" ]] || die "Missing a value for ${option_name}."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --endpoint)
            require_option_value "$1" "${2:-}"
            base_url="$2"
            shift 2
            ;;
        --model)
            require_option_value "$1" "${2:-}"
            configured_model="$2"
            shift 2
            ;;
        --timeout)
            require_option_value "$1" "${2:-}"
            request_timeout="$2"
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

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_positive_integer "$request_timeout" || die "Timeout must be a positive whole number of seconds."
is_positive_integer "$connect_timeout" || die "Connect timeout must be a positive whole number of seconds."

# A base URL ending in / is accepted, but query strings and fragments are not
# meaningful here because this script appends /models and /chat/completions.
while [[ "$base_url" == */ ]]; do
    base_url="${base_url%/}"
done
case "$base_url" in
    http://*|https://*) ;;
    *) die "Endpoint must begin with http:// or https:// (got: ${base_url})." ;;
esac

if ! command -v curl >/dev/null 2>&1; then
    die "curl is required but was not found in PATH."
fi

find_json_python() {
    local candidate
    local candidate_path
    local configured_python="${VLLM_DOCTOR_PYTHON:-}"
    local venv_dir="${VLLM_VENV_DIR:-}"

    if [[ -z "$venv_dir" && -n "${HOME:-}" ]]; then
        venv_dir="${HOME}/.venv-vllm-metal"
    fi

    if [[ -n "$configured_python" ]]; then
        [[ -x "$configured_python" ]] || die "VLLM_DOCTOR_PYTHON is not executable: ${configured_python}"
        "$configured_python" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)' >/dev/null 2>&1 \
            || die "VLLM_DOCTOR_PYTHON must be Python 3.8 or newer."
        json_python="$configured_python"
        return
    fi

    # vllm-metal installs its own Python, so prefer it when available.
    if [[ -n "$venv_dir" && -x "$venv_dir/bin/python" ]] \
        && "$venv_dir/bin/python" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)' >/dev/null 2>&1; then
        json_python="$venv_dir/bin/python"
        return
    fi

    for candidate in python3 python; do
        candidate_path="$(command -v "$candidate" 2>/dev/null || true)"
        if [[ -n "$candidate_path" ]] \
            && "$candidate_path" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)' >/dev/null 2>&1; then
            json_python="$candidate_path"
            return
        fi
    done

    die "A Python 3 executable is required to parse the OpenAI JSON response. Set VLLM_DOCTOR_PYTHON if needed."
}

json_model_ids() {
    "$json_python" -c '
import json
import sys

try:
    payload = json.load(sys.stdin)
    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError("missing list-valued data field")
    model_ids = [item["id"] for item in data if isinstance(item, dict) and isinstance(item.get("id"), str)]
    if not model_ids:
        raise ValueError("no model IDs in data field")
except (json.JSONDecodeError, TypeError, ValueError) as exc:
    print(f"Could not parse /models response: {exc}", file=sys.stderr)
    raise SystemExit(1)

print("\n".join(model_ids))
'
}

json_request_body() {
    "$json_python" -c '
import json
import sys

print(json.dumps({
    "model": sys.argv[1],
    "messages": [
        {
            "role": "system",
            "content": "You are a health-check service. Reply with exactly READY and nothing else.",
        },
        {"role": "user", "content": "Health check."},
    ],
    "temperature": 0,
    "top_p": 1,
    "max_tokens": 4,
    "seed": 0,
    "stream": False,
}))
' "$1"
}

json_completion_text() {
    "$json_python" -c '
import json
import sys

try:
    payload = json.load(sys.stdin)
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("missing choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("missing assistant message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("empty assistant response")
except (json.JSONDecodeError, AttributeError, TypeError, ValueError) as exc:
    print(f"Could not parse chat completion response: {exc}", file=sys.stderr)
    raise SystemExit(1)

# JSON encoding keeps any model-generated control characters harmless in a
# terminal while retaining the useful response text in the report.
print(json.dumps(content.strip(), ensure_ascii=False))
'
}

seconds_to_milliseconds() {
    awk -v seconds="$1" 'BEGIN { printf "%.0f", seconds * 1000 }'
}

http_status_is_success() {
    [[ "$1" =~ ^2[0-9][0-9]$ ]]
}

trap cleanup EXIT

find_json_python

if (( connect_timeout > request_timeout )); then
    connect_timeout="$request_timeout"
fi

models_body="$(mktemp "${TMPDIR:-/tmp}/vllm-doctor-models.XXXXXX")" || die "Could not create a temporary response file."
completion_body="$(mktemp "${TMPDIR:-/tmp}/vllm-doctor-completion.XXXXXX")" || die "Could not create a temporary response file."

curl_headers=(-H "Accept: application/json")
if [[ -n "${LLM_API_KEY:-}" ]]; then
    # Do not print this header or the key anywhere in the report.
    curl_headers+=(-H "Authorization: Bearer ${LLM_API_KEY}")
fi

printf 'vLLM readiness check (read-only)\n'
printf 'Endpoint: %s\n' "$base_url"
printf 'Timeout: %ss per request\n\n' "$request_timeout"

if ! models_metrics="$(
    curl --silent --show-error \
        --connect-timeout "$connect_timeout" \
        --max-time "$request_timeout" \
        "${curl_headers[@]}" \
        --output "$models_body" \
        --write-out '%{http_code} %{time_total}' \
        "$base_url/models"
)"; then
    die "Cannot reach ${base_url}/models. Ensure the vLLM server is running, then retry (for this project: bash scripts/start_vllm.sh)."
fi

models_status="${models_metrics%% *}"
models_seconds="${models_metrics#* }"
if ! http_status_is_success "$models_status"; then
    auth_hint=""
    if [[ "$models_status" == "401" || "$models_status" == "403" ]]; then
        auth_hint=" Set LLM_API_KEY if this server was started with --api-key."
    fi
    die "GET ${base_url}/models returned HTTP ${models_status}.${auth_hint}"
fi

if ! model_ids="$(json_model_ids < "$models_body")"; then
    die "GET ${base_url}/models returned an unexpected JSON payload."
fi

models_latency_ms="$(seconds_to_milliseconds "$models_seconds")"
printf 'GET /models: OK (HTTP %s, %s ms)\n' "$models_status" "$models_latency_ms"
printf 'Models advertised by server:\n'
while IFS= read -r advertised_model; do
    [[ -n "$advertised_model" ]] && printf '  - %s\n' "$advertised_model"
done <<< "$model_ids"

if [[ -n "$configured_model" ]]; then
    selected_model="$configured_model"
    if ! grep -F -x -q -- "$selected_model" <<< "$model_ids"; then
        die "Configured model '${selected_model}' is not advertised by /models. Set LLM_MODEL to one of the IDs above."
    fi
else
    selected_model="$(sed -n '1p' <<< "$model_ids")"
    printf 'Model selected automatically: %s\n' "$selected_model"
fi

if ! request_body="$(json_request_body "$selected_model")"; then
    die "Could not construct the health-check request."
fi

printf 'POST /chat/completions: testing %s with a 4-token deterministic prompt...\n' "$selected_model"
if ! completion_metrics="$(
    curl --silent --show-error \
        --connect-timeout "$connect_timeout" \
        --max-time "$request_timeout" \
        "${curl_headers[@]}" \
        -H "Content-Type: application/json" \
        --request POST \
        --data-binary "$request_body" \
        --output "$completion_body" \
        --write-out '%{http_code} %{time_total}' \
        "$base_url/chat/completions"
)"; then
    die "Could not complete a short inference request within ${request_timeout}s. Check the vLLM server logs and available unified memory."
fi

completion_status="${completion_metrics%% *}"
completion_seconds="${completion_metrics#* }"
if ! http_status_is_success "$completion_status"; then
    auth_hint=""
    if [[ "$completion_status" == "401" || "$completion_status" == "403" ]]; then
        auth_hint=" Set LLM_API_KEY if this server was started with --api-key."
    fi
    die "POST ${base_url}/chat/completions returned HTTP ${completion_status}.${auth_hint}"
fi

if ! completion_text="$(json_completion_text < "$completion_body")"; then
    die "The server returned HTTP ${completion_status}, but not a usable OpenAI chat completion."
fi

completion_latency_ms="$(seconds_to_milliseconds "$completion_seconds")"
printf 'POST /chat/completions: OK (HTTP %s, %s ms)\n' "$completion_status" "$completion_latency_ms"
printf 'Completion response: %s\n\n' "$completion_text"
printf 'PASS: OpenAI-compatible vLLM endpoint is ready.\n'

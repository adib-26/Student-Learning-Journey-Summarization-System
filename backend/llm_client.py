"""
Secure client for local vLLM and Vertex AI Model Garden endpoints.

The application talks to an OpenAI-compatible API. In the primary deployment,
Qwen3.6 runs in Vertex AI's managed vLLM serving container and requests use
short-lived Google OAuth tokens. Local and Docker vLLM remain compatible. The
client adds application-level resilience:

- request timeouts and bounded retries with exponential backoff (SDK-level)
- a cooldown circuit breaker shared across all client instances, so a down
  server fails fast instead of stalling every Streamlit rerun
- a process-wide admission lock that bounds work submitted by one app worker
- a lightweight model-identity health check used at app startup

Every call returns ``None`` on failure so existing rule-based fallbacks
(draft summaries, empty extraction results) take over gracefully.

Configuration (environment variables):
    LLM_BASE_URL                  OpenAI-compatible endpoint (default http://127.0.0.1:8000/v1)
    LLM_MODEL                     served model name (default Qwen/Qwen3-VL-8B-Instruct)
    LLM_API_KEY                   only needed if the server was started with --api-key
    LLM_TIMEOUT_SECONDS           per-request timeout (default 120)
    LLM_MAX_RETRIES               SDK retry attempts for transient errors (default 2)
    LLM_BREAKER_COOLDOWN_SECONDS  fast-fail window after the server is unreachable (default 60)
    LLM_MAX_TOKENS                completion token cap (default 512)
    LLM_TEMPERATURE               sampling temperature (default 0.2)
    LLM_ENABLE_THINKING           Qwen reasoning mode (default false)
    LLM_QUEUE_TIMEOUT_SECONDS     how long a request may wait for this app's
                                  inference slot (default 30)
"""
import os
import time
import logging
import threading
import math
from pathlib import Path
from typing import Optional

import streamlit as st
from openai import OpenAI, APIConnectionError, APIStatusError

from backend.gcp_access import google_access_token, is_vertex_endpoint

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - deployment fallback if dependencies are incomplete
    load_dotenv = None

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

SERVICE_UNAVAILABLE_MSG = "⚠️ Sorry, the AI summary service is temporarily unavailable"

# The app and NVIDIA vLLM server may run in separate containers or terminals.
# Loading the project-local file keeps model identity consistent for native
# launches. Explicit process/container settings still win via override=False.
_PROJECT_DOTENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def _load_project_dotenv() -> None:
    """Load project LLM settings without logging file contents or secrets."""
    if load_dotenv is None:
        return
    try:
        load_dotenv(dotenv_path=_PROJECT_DOTENV_PATH, override=False)
    except Exception:
        # An unreadable optional .env must not prevent deterministic defaults
        # from starting the local UI.  Do not include exception details here:
        # they can contain a filesystem path or accidental configuration data.
        logger.warning("Project .env could not be loaded; using environment/default settings")


_load_project_dotenv()

# HTTP statuses that indicate the server itself is unhealthy (trip the breaker),
# as opposed to a bad individual request (e.g. 400 prompt too long).
_BREAKER_STATUS_CODES = {429, 500, 502, 503, 504}

# One process-wide gate for model requests from this app worker. vLLM owns
# cross-process scheduling and batching; this lock only prevents duplicate
# Streamlit reruns from submitting redundant work.
_LOCAL_INFERENCE_LOCK = threading.Lock()


def _read_float_setting(name: str, default: float, minimum: float = 0.0) -> float:
    """Read an environment setting without letting a typo crash the UI."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %s", name, raw_value, default)
        return default
    if not math.isfinite(value) or value < minimum:
        logger.warning("Ignoring %s=%r; it must be at least %s", name, raw_value, minimum)
        return default
    return value


def _read_int_setting(name: str, default: int, minimum: int = 1) -> int:
    """Read a positive integer setting without letting a typo crash the UI."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %s", name, raw_value, default)
        return default
    if value < minimum:
        logger.warning("Ignoring %s=%r; it must be at least %s", name, raw_value, minimum)
        return default
    return value


def _read_bool_setting(name: str, default: bool) -> bool:
    """Read a boolean environment setting with a safe fallback."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    logger.warning("Ignoring invalid %s=%r; using %s", name, raw_value, default)
    return default


def get_configured_model() -> str:
    """Model name as configured, for audit logs and UI labels."""
    return os.getenv("LLM_MODEL", DEFAULT_MODEL)


class SecureLLMClient:
    """Resilient wrapper around the local vLLM OpenAI-compatible endpoint."""

    # Circuit-breaker state is class-level: several modules keep their own
    # client instance, and one unreachable server should silence all of them.
    _unavailable_until: float = 0.0
    _warned: bool = False
    # Bound duplicate work from one application process. NVIDIA vLLM handles
    # global scheduling and continuous batching at the serving layer.
    _inference_lock = _LOCAL_INFERENCE_LOCK

    def __init__(self):
        self.base_url = os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL)
        self.model = get_configured_model()
        self._vertex = is_vertex_endpoint(self.base_url)
        # vLLM only enforces a key when launched with --api-key. Vertex uses a
        # short-lived Google token, which is fetched lazily immediately before
        # each request so a missing login cannot break module import/startup.
        self.api_key = "vertex-adc-pending" if self._vertex else os.getenv(
            "LLM_API_KEY", "not-needed"
        )
        self.timeout = _read_float_setting("LLM_TIMEOUT_SECONDS", 120.0, minimum=1.0)
        self.max_retries = _read_int_setting("LLM_MAX_RETRIES", 2, minimum=0)
        self.breaker_cooldown = _read_float_setting(
            "LLM_BREAKER_COOLDOWN_SECONDS", 60.0, minimum=1.0
        )
        self.max_tokens = _read_int_setting("LLM_MAX_TOKENS", 512, minimum=1)
        self.temperature = _read_float_setting("LLM_TEMPERATURE", 0.2, minimum=0.0)
        # Qwen3.6 enables reasoning by default. These short extraction and
        # summary tasks are faster and more reliable in instruct mode because
        # the completion budget is reserved for the answer rather than hidden
        # reasoning. Operators can still opt in for other workloads.
        self.enable_thinking = _read_bool_setting("LLM_ENABLE_THINKING", False)
        self.queue_timeout = _read_float_setting(
            "LLM_QUEUE_TIMEOUT_SECONDS", 30.0, minimum=0.0
        )

        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def _refresh_vertex_token(self) -> None:
        if not self._vertex:
            return
        self.api_key = google_access_token()
        self._client.api_key = self.api_key

    # ------------------------------------------------------------------
    # Circuit breaker
    # ------------------------------------------------------------------
    @property
    def service_unavailable(self) -> bool:
        """True while the breaker is open (server was recently unreachable)."""
        if time.monotonic() < SecureLLMClient._unavailable_until:
            return True
        if SecureLLMClient._unavailable_until:
            # Start the next outage as a fresh incident so the UI can display
            # one warning again instead of remaining permanently suppressed.
            SecureLLMClient.reset_circuit_breaker()
        return False

    @classmethod
    def reset_circuit_breaker(cls) -> None:
        cls._unavailable_until = 0.0
        cls._warned = False

    def _trip_breaker(self, reason: str) -> None:
        SecureLLMClient._unavailable_until = time.monotonic() + self.breaker_cooldown
        logger.error(
            "LLM server unavailable (%s); pausing calls for %.0fs",
            reason, self.breaker_cooldown,
        )
        if not SecureLLMClient._warned:
            SecureLLMClient._warned = True
            try:
                st.warning(SERVICE_UNAVAILABLE_MSG, icon="⚠️")
            except Exception:
                pass  # usable outside a Streamlit script run

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def health_check(self) -> bool:
        """Verify that the configured endpoint can serve this model."""
        try:
            self._refresh_vertex_token()
            probe = self._client.with_options(timeout=30.0, max_retries=0)
            if self._vertex:
                # Vertex self-deployed Model Garden endpoints guarantee Chat
                # Completions, but they do not guarantee an OpenAI /models
                # route. A one-token completion validates DNS, OAuth, routing,
                # the deployed model, and replica readiness together.
                response = probe.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "Reply OK"}],
                    temperature=0.0,
                    max_tokens=1,
                    extra_body={
                        "chat_template_kwargs": {
                            "enable_thinking": self.enable_thinking,
                        }
                    },
                )
                return bool(response.choices)

            response = probe.models.list()
            data = response.get("data", []) if isinstance(response, dict) else getattr(response, "data", [])
            model_ids = [
                item.get("id") if isinstance(item, dict) else getattr(item, "id", None)
                for item in data
            ]
            model_ids = [model_id for model_id in model_ids if isinstance(model_id, str)]
            if not model_ids:
                logger.warning("LLM model discovery returned no model IDs")
                return False
            if self.model not in model_ids:
                logger.warning(
                    "LLM server is reachable but model %r is not advertised; available models: %s",
                    self.model,
                    ", ".join(model_ids),
                )
                return False
            return True
        except Exception as e:
            logger.warning("LLM health check failed: %s", e)
            return False

    def call_llm_secure(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> Optional[str]:
        """
        Send a prompt to the configured model endpoint and return the completion text,
        or None on any failure so callers fall back to rule-based output.
        Per-call overrides for model, temperature, and max_tokens are optional.
        ``system_prompt`` keeps the task contract separate from untrusted
        document text supplied in the user message.
        """
        if self.service_unavailable:
            return None

        if not isinstance(prompt, str) or not prompt.strip():
            logger.warning("Refusing an empty LLM prompt")
            return None

        requested_max_tokens = self.max_tokens if max_tokens is None else max_tokens
        if not isinstance(requested_max_tokens, int) or requested_max_tokens < 1:
            logger.error("Invalid max_tokens=%r", requested_max_tokens)
            return None

        requested_temperature = self.temperature if temperature is None else temperature
        if not isinstance(requested_temperature, (int, float)) or requested_temperature < 0:
            logger.error("Invalid temperature=%r", requested_temperature)
            return None

        # Do not allow a second UI rerun to turn a single-device desktop model
        # into a hidden queue of long-running requests.
        acquired = SecureLLMClient._inference_lock.acquire(timeout=self.queue_timeout)
        if not acquired:
            logger.warning(
                "Application inference slot is busy; request skipped after %.1fs", self.queue_timeout
            )
            return None

        try:
            # Refresh after admission so queued requests do not perform needless
            # OAuth work. Cached Google credentials refresh only near expiry.
            try:
                self._refresh_vertex_token()
            except Exception as e:
                self._trip_breaker(f"vertex auth: {e}")
                return None

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = self._client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=requested_temperature,
                max_tokens=requested_max_tokens,
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": self.enable_thinking,
                    }
                },
            )
            if not response.choices:
                return None
            content = response.choices[0].message.content
            return content.strip() if content else None

        except APIConnectionError as e:
            # Covers refused connections and timeouts: the server is down,
            # still loading the model, or overloaded.
            self._trip_breaker(f"connection error: {e}")
            return None

        except APIStatusError as e:
            if e.status_code in _BREAKER_STATUS_CODES:
                self._trip_breaker(f"HTTP {e.status_code}")
            else:
                logger.error("LLM API error %s: %s", e.status_code, e)
            return None

        except Exception as e:
            logger.error("Unexpected LLM client error: %s", e)
            return None

        finally:
            SecureLLMClient._inference_lock.release()


def get_local_inference_lock() -> threading.Lock:
    """Return the app-process gate shared by visual and text requests."""
    return _LOCAL_INFERENCE_LOCK


__all__ = [
    "SecureLLMClient",
    "get_configured_model",
    "get_local_inference_lock",
    "SERVICE_UNAVAILABLE_MSG",
]

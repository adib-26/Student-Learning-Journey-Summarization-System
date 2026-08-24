"""
Secure client for a self-hosted LLM served through an OpenAI-compatible API
(vLLM / vllm-metal on Apple Silicon).

This replaces the former Gemini cloud client. All inference now stays on the
local machine: no cloud API key, no external data transport. The client adds
enterprise resilience on top of the raw HTTP API:

- request timeouts and bounded retries with exponential backoff (SDK-level)
- a cooldown circuit breaker shared across all client instances, so a down
  server fails fast instead of stalling every Streamlit rerun
- a process-wide admission lock.  An 8 GB Apple-Silicon machine should not
  receive several expensive generations at the same time.
- a lightweight health check used at app startup

Every call returns ``None`` on failure so existing rule-based fallbacks
(draft summaries, empty extraction results) take over gracefully.

Configuration (environment variables):
    LLM_BASE_URL                  OpenAI-compatible endpoint (default http://127.0.0.1:8000/v1)
    LLM_MODEL                     served model name (default mlx-community/Qwen2.5-3B-Instruct-4bit)
    LLM_API_KEY                   only needed if the server was started with --api-key
    LLM_TIMEOUT_SECONDS           per-request timeout (default 120)
    LLM_MAX_RETRIES               SDK retry attempts for transient errors (default 2)
    LLM_BREAKER_COOLDOWN_SECONDS  fast-fail window after the server is unreachable (default 60)
    LLM_MAX_TOKENS                completion token cap (default 320)
    LLM_TEMPERATURE               sampling temperature (default 0.3)
    LLM_QUEUE_TIMEOUT_SECONDS     how long a request may wait for the local
                                  inference slot (default 5)
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

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - deployment fallback if dependencies are incomplete
    load_dotenv = None

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"

SERVICE_UNAVAILABLE_MSG = "⚠️ Sorry, our local AI summary service is temporarily unavailable"

# The UI and vLLM server commonly run in separate terminals.  Loading the
# project-local file ensures Streamlit requests the model selected in `.env`
# (for example the 1.5B fallback for an 8 GB Mac).  ``override=False``
# preserves explicit shell configuration, which is important for Docker and
# temporary troubleshooting commands.
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

# One process-wide gate for every local model request.  The GLM-OCR adapter
# imports this through ``get_local_inference_lock`` so an OCR model swap cannot
# overlap a Qwen generation on an 8 GB unified-memory machine.
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


def get_configured_model() -> str:
    """Model name as configured, for audit logs and UI labels."""
    return os.getenv("LLM_MODEL", DEFAULT_MODEL)


class SecureLLMClient:
    """Resilient wrapper around the local vLLM OpenAI-compatible endpoint."""

    # Circuit-breaker state is class-level: several modules keep their own
    # client instance, and one unreachable server should silence all of them.
    _unavailable_until: float = 0.0
    _warned: bool = False
    # A single M1/8 GB local model is a shared, constrained resource.  This
    # prevents Streamlit reruns from queuing multiple long generations inside
    # vLLM and making every request look stalled.
    _inference_lock = _LOCAL_INFERENCE_LOCK

    def __init__(self):
        self.base_url = os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL)
        self.model = get_configured_model()
        # vLLM only enforces a key when launched with --api-key.
        self.api_key = os.getenv("LLM_API_KEY", "not-needed")
        self.timeout = _read_float_setting("LLM_TIMEOUT_SECONDS", 120.0, minimum=1.0)
        self.max_retries = _read_int_setting("LLM_MAX_RETRIES", 2, minimum=0)
        self.breaker_cooldown = _read_float_setting(
            "LLM_BREAKER_COOLDOWN_SECONDS", 60.0, minimum=1.0
        )
        self.max_tokens = _read_int_setting("LLM_MAX_TOKENS", 320, minimum=1)
        self.temperature = _read_float_setting("LLM_TEMPERATURE", 0.3, minimum=0.0)
        self.queue_timeout = _read_float_setting(
            "LLM_QUEUE_TIMEOUT_SECONDS", 5.0, minimum=0.0
        )

        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    # ------------------------------------------------------------------
    # Circuit breaker
    # ------------------------------------------------------------------
    @property
    def service_unavailable(self) -> bool:
        """True while the breaker is open (server was recently unreachable)."""
        return time.monotonic() < SecureLLMClient._unavailable_until

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
        """Probe /models and ensure the configured model is actually available."""
        try:
            response = self._client.with_options(timeout=5.0, max_retries=0).models.list()
            data = response.get("data", []) if isinstance(response, dict) else getattr(response, "data", [])
            model_ids = [
                item.get("id") if isinstance(item, dict) else getattr(item, "id", None)
                for item in data
            ]
            model_ids = [model_id for model_id in model_ids if isinstance(model_id, str)]
            if model_ids and self.model not in model_ids:
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
        Send a prompt to the local LLM and return the completion text,
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
                "Local LLM is busy; request skipped after waiting %.1fs", self.queue_timeout
            )
            return None

        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = self._client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=requested_temperature,
                max_tokens=requested_max_tokens,
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
    """Return the process-wide gate shared by Qwen and local visual OCR."""
    return _LOCAL_INFERENCE_LOCK


__all__ = [
    "SecureLLMClient",
    "get_configured_model",
    "get_local_inference_lock",
    "SERVICE_UNAVAILABLE_MSG",
]

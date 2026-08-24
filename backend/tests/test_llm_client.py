"""
Unit tests for llm_client.py — the secure client for the local vLLM server.

Covers configuration, successful calls, connection-failure handling
(shared circuit breaker + a single user-facing warning), fast-fail behavior
while the breaker is open, and the startup health check.
"""
import sys
import os
# Add project root to path so we can import backend modules properly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
import httpx
import pytest
from unittest.mock import patch, MagicMock

from openai import APIConnectionError, APIStatusError
from backend.llm_client import (
    SecureLLMClient,
    SERVICE_UNAVAILABLE_MSG,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
)
import backend.llm_client as llm_client

_URL = "http://localhost:8000/v1/chat/completions"


def _connection_error() -> APIConnectionError:
    return APIConnectionError(request=httpx.Request("POST", _URL))


def _status_error(status_code: int) -> APIStatusError:
    request = httpx.Request("POST", _URL)
    response = httpx.Response(status_code, request=request)
    return APIStatusError(f"HTTP {status_code}", response=response, body=None)


def _completion(text: str) -> MagicMock:
    choice = MagicMock()
    choice.message.content = text
    response = MagicMock()
    response.choices = [choice]
    return response


@pytest.fixture
def mock_openai():
    with patch("backend.llm_client.OpenAI") as mock:
        yield mock


@pytest.fixture
def mock_st_warning():
    with patch("streamlit.warning") as mock:
        yield mock


class TestSecureLLMClient:
    """Test suite for SecureLLMClient"""

    def setup_method(self):
        """Reset the shared circuit breaker before each test"""
        SecureLLMClient.reset_circuit_breaker()

    def test_initialization_defaults(self, mock_openai):
        """Client initializes with local-server defaults and a closed breaker"""
        client = SecureLLMClient()
        assert client.base_url == DEFAULT_BASE_URL
        assert client.model == DEFAULT_MODEL
        assert client.service_unavailable is False
        mock_openai.assert_called_once()

    def test_env_configuration(self, mock_openai):
        """LLM_* environment variables override the defaults"""
        env = {
            "LLM_BASE_URL": "http://127.0.0.1:9999/v1",
            "LLM_MODEL": "mlx-community/some-other-model",
            "LLM_TIMEOUT_SECONDS": "30",
        }
        with patch.dict(os.environ, env):
            client = SecureLLMClient()
            assert client.base_url == "http://127.0.0.1:9999/v1"
            assert client.model == "mlx-community/some-other-model"
            assert client.timeout == 30.0

    def test_project_dotenv_loads_settings_without_overriding_shell_values(self, tmp_path):
        """Separate app/server terminals can share Qwen config through .env."""
        dotenv_path = tmp_path / ".env"
        dotenv_path.write_text(
            "LLM_MODEL=mlx-community/Qwen2.5-1.5B-Instruct-4bit\n"
            "LLM_BASE_URL=http://127.0.0.1:8123/v1\n",
            encoding="utf-8",
        )

        with patch.object(llm_client, "_PROJECT_DOTENV_PATH", dotenv_path):
            with patch.dict(os.environ, {}, clear=True):
                llm_client._load_project_dotenv()
                assert os.environ["LLM_MODEL"] == "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
                assert os.environ["LLM_BASE_URL"] == "http://127.0.0.1:8123/v1"

            with patch.dict(os.environ, {"LLM_MODEL": "shell-selected-model"}, clear=True):
                llm_client._load_project_dotenv()
                assert os.environ["LLM_MODEL"] == "shell-selected-model"

    def test_successful_call_returns_stripped_text(self, mock_openai):
        """A healthy call returns the completion text"""
        instance = mock_openai.return_value
        instance.chat.completions.create.return_value = _completion("  polished summary  ")
        client = SecureLLMClient()

        result = client.call_llm_secure("test prompt")

        assert result == "polished summary"
        instance.chat.completions.create.assert_called_once()

    def test_call_keeps_system_contract_separate_from_document_prompt(self, mock_openai):
        """Task instructions should be a system message, not mixed into evidence."""
        instance = mock_openai.return_value
        instance.chat.completions.create.return_value = _completion("ready")
        client = SecureLLMClient()

        result = client.call_llm_secure(
            "<DOCUMENT>untrusted text</DOCUMENT>",
            system_prompt="Extract only factual fields.",
            temperature=0.0,
            max_tokens=80,
        )

        assert result == "ready"
        request = instance.chat.completions.create.call_args.kwargs
        assert request["messages"] == [
            {"role": "system", "content": "Extract only factual fields."},
            {"role": "user", "content": "<DOCUMENT>untrusted text</DOCUMENT>"},
        ]
        assert request["temperature"] == 0.0
        assert request["max_tokens"] == 80

    def test_busy_local_model_skips_a_second_request(self, mock_openai):
        """An 8 GB desktop profile must not build a hidden request queue."""
        assert SecureLLMClient._inference_lock.acquire(blocking=False)
        try:
            with patch.dict(os.environ, {"LLM_QUEUE_TIMEOUT_SECONDS": "0"}):
                client = SecureLLMClient()
                assert client.call_llm_secure("second request") is None
            mock_openai.return_value.chat.completions.create.assert_not_called()
        finally:
            SecureLLMClient._inference_lock.release()

    def test_connection_error_trips_breaker_and_warns_once(self, mock_openai, mock_st_warning):
        """A refused/timed-out connection returns None, opens the breaker, and warns the user"""
        instance = mock_openai.return_value
        instance.chat.completions.create.side_effect = _connection_error()
        client = SecureLLMClient()

        result = client.call_llm_secure("test prompt")

        assert result is None
        assert client.service_unavailable is True
        mock_st_warning.assert_called_once_with(SERVICE_UNAVAILABLE_MSG, icon="⚠️")

    def test_fast_fail_while_breaker_open(self, mock_openai, mock_st_warning):
        """Once the breaker is open, calls return None without hitting the server"""
        instance = mock_openai.return_value
        instance.chat.completions.create.side_effect = _connection_error()
        client = SecureLLMClient()
        client.call_llm_secure("first call - trips the breaker")

        instance.chat.completions.create.reset_mock()
        result = client.call_llm_secure("second call - must fail fast")

        assert result is None
        instance.chat.completions.create.assert_not_called()
        # Warning is shown once, not repeated on every rerun
        mock_st_warning.assert_called_once()

    def test_breaker_shared_across_instances(self, mock_openai, mock_st_warning):
        """Several modules keep their own client; one down server silences all of them"""
        instance = mock_openai.return_value
        instance.chat.completions.create.side_effect = _connection_error()
        first_client = SecureLLMClient()
        first_client.call_llm_secure("trips the breaker")

        second_client = SecureLLMClient()
        assert second_client.service_unavailable is True

    def test_breaker_reopens_after_cooldown(self, mock_openai, mock_st_warning):
        """The breaker is time-based: after the cooldown the client tries again"""
        instance = mock_openai.return_value
        instance.chat.completions.create.side_effect = _connection_error()
        client = SecureLLMClient()
        client.call_llm_secure("trips the breaker")
        assert client.service_unavailable is True

        # Simulate the cooldown window elapsing
        SecureLLMClient._unavailable_until = time.monotonic() - 1
        assert client.service_unavailable is False

    def test_bad_request_does_not_trip_breaker(self, mock_openai):
        """A 400 (e.g. prompt too long) is a per-request problem, not a down server"""
        instance = mock_openai.return_value
        instance.chat.completions.create.side_effect = _status_error(400)
        client = SecureLLMClient()

        result = client.call_llm_secure("test prompt")

        assert result is None
        assert client.service_unavailable is False

    def test_server_error_trips_breaker(self, mock_openai, mock_st_warning):
        """A 503 means the server is unhealthy: open the breaker"""
        instance = mock_openai.return_value
        instance.chat.completions.create.side_effect = _status_error(503)
        client = SecureLLMClient()

        result = client.call_llm_secure("test prompt")

        assert result is None
        assert client.service_unavailable is True

    def test_health_check_success(self, mock_openai):
        """health_check returns True when /models responds"""
        client = SecureLLMClient()
        assert client.health_check() is True

    def test_health_check_detects_configured_model_mismatch(self, mock_openai):
        """A reachable server with the wrong model must not look ready."""
        response = MagicMock()
        advertised_model = MagicMock()
        advertised_model.id = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
        response.data = [advertised_model]
        mock_openai.return_value.with_options.return_value.models.list.return_value = response

        client = SecureLLMClient()

        assert client.health_check() is False

    def test_health_check_failure(self, mock_openai):
        """health_check returns False when the server is unreachable"""
        instance = mock_openai.return_value
        instance.with_options.return_value.models.list.side_effect = _connection_error()
        client = SecureLLMClient()
        assert client.health_check() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

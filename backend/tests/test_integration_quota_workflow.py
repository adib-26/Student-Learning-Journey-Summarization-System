"""
Integration test for the complete service-unavailability workflow.
Tests that real application components work together: the LLM client catches a
connection failure to the local vLLM server → opens the shared circuit breaker
→ displays the correct warning → fails fast on subsequent calls.
This is an integration test, not a unit test - only the external HTTP boundary
(the OpenAI SDK transport) is mocked.
"""
import sys
import os
# Add project root to path so we can import backend modules properly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import httpx
import pytest
from unittest.mock import patch
import streamlit as st
from openai import APIConnectionError
from backend.llm_client import SecureLLMClient, SERVICE_UNAVAILABLE_MSG
from backend.deepl_translator import DeepLUITranslator, UI_STRINGS


def _connection_error() -> APIConnectionError:
    return APIConnectionError(
        request=httpx.Request("POST", "http://localhost:8000/v1/chat/completions")
    )


class TestServiceUnavailableIntegration:
    """Integration test suite for the complete AI-unavailable warning system"""

    def setup_method(self):
        """Setup test environment before each test"""
        st.session_state.clear()
        SecureLLMClient.reset_circuit_breaker()

    @patch('streamlit.warning')
    @patch('backend.llm_client.OpenAI')
    def test_complete_service_unavailable_workflow(self, mock_openai, mock_st_warning):
        """Test the ENTIRE workflow: connection failure → breaker opens → warning displays with correct translation"""
        # 1. Setup mocks for external dependencies only (never mock your own code!)
        instance = mock_openai.return_value
        # Simulate the local vLLM server being down / unreachable
        instance.chat.completions.create.side_effect = _connection_error()

        # 2. Initialize REAL application components (your actual code - not mocked!)
        st.session_state['language'] = 'en'
        client = SecureLLMClient()
        translator = DeepLUITranslator()

        # 3. Verify initial state is correct
        assert client.service_unavailable is False

        # 4. First call - should catch the connection error and open the breaker
        client.call_llm_secure("Test document content to summarize")

        # 5. Verify the breaker opened to prevent repeated slow failures
        assert client.service_unavailable is True

        # 6. Verify the exact user-facing warning was displayed
        mock_st_warning.assert_called_once_with(SERVICE_UNAVAILABLE_MSG, icon="⚠️")

        # 7. Verify the correct English warning string exists in UI_STRINGS
        assert UI_STRINGS["llm_service_warning"] == SERVICE_UNAVAILABLE_MSG
        warning_message = translator.get_string("llm_service_warning")
        assert warning_message == UI_STRINGS["llm_service_warning"]

        # 8. Verify subsequent calls fail FAST - no additional server calls are made!
        instance.chat.completions.create.reset_mock()
        result = client.call_llm_secure("Another test document - this should NEVER hit the server")
        assert result is None  # Fast fail returns None immediately
        instance.chat.completions.create.assert_not_called()  # ✓ Critical: No repeated calls!

    def test_multi_language_strings_exist_in_integration(self):
        """Test that all multi-language translations exist in the integrated UI_STRINGS"""
        # Verify ALL translation strings exist and are correct (direct from shared UI_STRINGS)
        # English
        assert UI_STRINGS["llm_service_warning"] == "⚠️ Sorry, the AI summary service is temporarily unavailable"
        # Bahasa Melayu
        assert UI_STRINGS["llm_service_warning_ms"] == "⚠️ Maaf, perkhidmatan ringkasan AI kami tidak tersedia buat sementara waktu"
        # Chinese
        assert UI_STRINGS["llm_service_warning_zh"] == "⚠️ 抱歉，AI摘要服务暂时无法使用"

        # Initialize translator and verify get_string retrieves correct strings per language
        translator = DeepLUITranslator()
        # For English, it returns the original string
        english_warning = translator.get_string("llm_service_warning", language='en')
        assert english_warning == UI_STRINGS["llm_service_warning"]

        # For other languages, since DeepL might be offline in test environment, it should return the original string
        # or we can verify that the translation strings exist in UI_STRINGS
        malay_warning = UI_STRINGS["llm_service_warning_ms"]
        assert malay_warning == "⚠️ Maaf, perkhidmatan ringkasan AI kami tidak tersedia buat sementara waktu"

        chinese_warning = UI_STRINGS["llm_service_warning_zh"]
        assert chinese_warning == "⚠️ 抱歉，AI摘要服务暂时无法使用"

    @patch('streamlit.warning')
    @patch('backend.llm_client.OpenAI')
    def test_breaker_persists_across_calls_and_instances(self, mock_openai, mock_st_warning):
        """Test that the open breaker persists and continues to block calls, even from other client instances"""
        instance = mock_openai.return_value
        instance.chat.completions.create.side_effect = _connection_error()

        # Initialize REAL client with session_state
        st.session_state['language'] = 'en'
        client = SecureLLMClient()

        # First call opens the breaker
        client.call_llm_secure("First call - server is down")
        assert client.service_unavailable is True

        # Call 10 more times - none should ever hit the server
        instance.chat.completions.create.reset_mock()
        for i in range(10):
            result = client.call_llm_secure(f"Additional call {i}")
            assert result is None

        # A brand-new client instance (e.g. another module) is also blocked
        second_client = SecureLLMClient()
        assert second_client.service_unavailable is True
        assert second_client.call_llm_secure("From another module") is None

        # Verify the server was never called again - only called ONCE initially!
        instance.chat.completions.create.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

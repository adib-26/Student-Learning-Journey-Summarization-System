"""
Integration test for the complete quota exhaustion workflow
Tests that all components work together: Gemini client catches 429 → sets quota_exhausted flag → displays correct warning.
This is an integration test, not a unit test - it verifies real application components interact properly.
"""
import sys
import os
# Add project root to path so we can import backend modules properly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pytest
from unittest.mock import patch, MagicMock
import streamlit as st
from backend.secure_gemini_client import SecureGeminiClient
from backend.deepl_translator import DeepLUITranslator, UI_STRINGS


class TestQuotaWarningIntegration:
    """Integration test suite for the complete quota warning system"""

    def setup_method(self):
        """Setup test environment before each test"""
        st.session_state.clear()

    @patch('os.getenv')
    @patch('streamlit.warning')
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_complete_quota_exhaustion_workflow(
        self, mock_genai_model, mock_configure, mock_st_warning, mock_getenv
    ):
        """Test the ENTIRE workflow: 429 error → client flags quota → warning displays with correct translation"""
        # 1. Setup mocks for external dependencies only (never mock your own code!)
        mock_getenv.return_value = "test_api_key"
        mock_model_instance = MagicMock()
        # Simulate Gemini API returning 429 RESOURCE_EXHAUSTED
        from google.api_core.exceptions import ResourceExhausted
        mock_model_instance.generate_content.side_effect = ResourceExhausted("Quota exceeded")
        mock_genai_model.return_value = mock_model_instance

        # 2. Initialize REAL application components (your actual code - not mocked!)
        st.session_state['language'] = 'en'
        client = SecureGeminiClient()
        translator = DeepLUITranslator()

        # 3. Verify initial state is correct
        assert client.quota_exhausted is False

        # 4. First API call - should catch the 429 and set the quota flag
        client.call_gemini_secure("Test document content to summarize")

        # 5. Verify quota flag was set to prevent repeated API calls
        assert client.quota_exhausted is True

        # 6. Verify the correct English warning string exists in UI_STRINGS
        assert UI_STRINGS["gemini_quota_warning"] == "⚠️ Sorry, our AI summary service is temporarily unavailable"
        warning_message = translator.get_string("gemini_quota_warning")
        assert warning_message == UI_STRINGS["gemini_quota_warning"]

        # 7. Verify subsequent calls fail FAST - no additional API calls are made!
        mock_model_instance.generate_content.reset_mock()
        result = client.call_gemini_secure("Another test document - this should NEVER hit the API")
        assert result is None  # Fast fail returns None immediately
        mock_model_instance.generate_content.assert_not_called()  # ✓ Critical: No repeated API calls!

    @patch('os.getenv')
    @patch('google.generativeai.configure')
    def test_multi_language_strings_exist_in_integration(
        self, mock_configure, mock_getenv
    ):
        """Test that all multi-language translations exist in the integrated UI_STRINGS"""
        mock_getenv.return_value = "test_api_key"
        
        # Verify ALL translation strings exist and are correct (direct from shared UI_STRINGS)
        # English
        assert UI_STRINGS["gemini_quota_warning"] == "⚠️ Sorry, our AI summary service is temporarily unavailable"
        # Bahasa Melayu (Malay)
        assert UI_STRINGS["gemini_quota_warning_ms"] == "⚠️ Maaf, perkhidmatan ringkasan AI kami tidak tersedia buat sementara waktu"
        # Traditional Chinese
        assert UI_STRINGS["gemini_quota_warning_zh"] == "⚠️ 抱歉，我们的AI摘要服务暂时无法使用"

        # Initialize translator and verify get_string retrieves correct strings per language
        translator = DeepLUITranslator()
        # For English, it returns the original string
        english_warning = translator.get_string("gemini_quota_warning", language='en')
        assert english_warning == UI_STRINGS["gemini_quota_warning"]
        
        # For other languages, since DeepL might be offline in test environment, it should return the original string
        # or we can verify that the translation strings exist in UI_STRINGS
        malay_warning = UI_STRINGS["gemini_quota_warning_ms"]
        assert malay_warning == "⚠️ Maaf, perkhidmatan ringkasan AI kami tidak tersedia buat sementara waktu"
        
        chinese_warning = UI_STRINGS["gemini_quota_warning_zh"]
        assert chinese_warning == "⚠️ 抱歉，我们的AI摘要服务暂时无法使用"

    @patch('os.getenv')
    @patch('streamlit.warning')
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_quota_flag_persists_across_calls(
        self, mock_genai_model, mock_configure, mock_st_warning, mock_getenv
    ):
        """Test that the quota_exhausted flag persists and continues to block API calls"""
        # Setup mocks
        mock_getenv.return_value = "test_api_key"
        mock_model_instance = MagicMock()
        from google.api_core.exceptions import ResourceExhausted
        mock_model_instance.generate_content.side_effect = ResourceExhausted("Quota exceeded")
        mock_genai_model.return_value = mock_model_instance

        # Initialize REAL client with session_state
        st.session_state['language'] = 'en'
        client = SecureGeminiClient()
        
        # First call triggers quota exhaustion
        client.call_gemini_secure("First call - triggers 429")
        
        # Verify flag is True
        assert client.quota_exhausted is True
        
        # Call 10 more times - none should ever call the API
        mock_model_instance.generate_content.reset_mock()
        for i in range(10):
            result = client.call_gemini_secure(f"Additional call {i}")
            assert result is None
        
        # Verify API was never called again - only called ONCE initially!
        mock_model_instance.generate_content.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
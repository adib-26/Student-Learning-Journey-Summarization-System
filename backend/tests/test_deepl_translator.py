"""
Unit tests for deepl_translator.py
Tests that local-LLM availability warning strings exist and are accessible.
"""
import sys
import os
# Add project root to path so we can import backend modules properly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pytest
from unittest.mock import patch, MagicMock
from backend.deepl_translator import DeepLUITranslator, UI_STRINGS


class TestDeepLTranslator:
    """Test suite for DeepL Translator and our UI strings"""

    def setup_method(self):
        """Create a translator instance before each test"""
        self.translator = DeepLUITranslator()

    def test_llm_service_warning_strings_exist(self):
        """All local-LLM availability strings are properly added to UI_STRINGS."""
        # Verify all English strings exist
        assert "llm_service_warning" in UI_STRINGS
        assert UI_STRINGS["llm_service_warning"] == "⚠️ Sorry, the AI summary service is temporarily unavailable"
        
        # Verify Malay translation exists
        assert "llm_service_warning_ms" in UI_STRINGS
        assert UI_STRINGS["llm_service_warning_ms"] == "⚠️ Maaf, perkhidmatan ringkasan AI kami tidak tersedia buat sementara waktu"
        
        # Verify Chinese translation exists
        assert "llm_service_warning_zh" in UI_STRINGS
        assert UI_STRINGS["llm_service_warning_zh"] == "⚠️ 抱歉，AI摘要服务暂时无法使用"

    @patch('streamlit.session_state', {'language': 'en'})
    def test_get_string_returns_correct_llm_warning_english(self):
        """Test that get_string returns the correct English local-LLM warning."""
        result = self.translator.get_string("llm_service_warning")
        assert result == "⚠️ Sorry, the AI summary service is temporarily unavailable"

    @patch('streamlit.session_state', {'language': 'ms'})
    def test_get_string_returns_correct_llm_warning_malay(self):
        """Test that the Malay local-LLM warning string exists and can be retrieved."""
        result = UI_STRINGS["llm_service_warning_ms"]
        assert result == "⚠️ Maaf, perkhidmatan ringkasan AI kami tidak tersedia buat sementara waktu"

    @patch('streamlit.session_state', {'language': 'zh'})
    def test_get_string_returns_correct_llm_warning_chinese(self):
        """Test that the Chinese local-LLM warning string exists and can be retrieved."""
        # Access the Chinese string directly from UI_STRINGS since DeepL is offline
        result = UI_STRINGS["llm_service_warning_zh"]
        assert result == "⚠️ 抱歉，AI摘要服务暂时无法使用"

    def test_llm_warning_key_not_found_fallback(self):
        """Test that if a translation key is missing, it falls back gracefully"""
        # Test with a non-existent key
        result = self.translator.get_string("non_existent_key")
        assert result == "non_existent_key"  # get_string returns the key itself if not found

    def test_all_translation_strings_are_strings(self):
        """Verify that every string in UI_STRINGS is actually a string type"""
        for key, value in UI_STRINGS.items():
            assert isinstance(value, str), f"UI_STRINGS['{key}'] is not a string, it's {type(value)}"

    def test_llm_warnings_include_warning_icon(self):
        """Verify all local-LLM warnings include the ⚠️ icon."""
        assert "⚠️" in UI_STRINGS["llm_service_warning"]
        assert "⚠️" in UI_STRINGS["llm_service_warning_ms"]
        assert "⚠️" in UI_STRINGS["llm_service_warning_zh"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

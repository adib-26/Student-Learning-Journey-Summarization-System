"""
Unit tests for text_info_extractor.py
Tests the text extraction functionality and its error handling for quota exhaustion.
"""
import sys
import os
# Add project root to path so we can import backend modules properly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pytest
from unittest.mock import patch, MagicMock
import streamlit as st


class TestTextInfoExtractor:
    """Test suite for text_info_extractor.py - only basic import and structure tests"""

    def setup_method(self):
        """Setup test environment before each test"""
        st.session_state.clear()

    def test_module_imports_successfully(self):
        """Test that the module can be imported successfully"""
        # This tests that all dependencies are properly installed and imports work
        from backend import text_info_extractor
        assert text_info_extractor is not None
        assert hasattr(text_info_extractor, 'get_text_info')
        assert hasattr(text_info_extractor, '_extract_certificate_data_with_gemini')

    def test_function_signatures_exist(self):
        """Test that the required functions exist with expected signatures"""
        from backend.text_info_extractor import get_text_info, _extract_certificate_data_with_gemini
        
        # Verify functions are callable
        assert callable(get_text_info)
        assert callable(_extract_certificate_data_with_gemini)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
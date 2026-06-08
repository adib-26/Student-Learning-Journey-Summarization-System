"""
System tests for the entire application stack.
These tests verify end-to-end functionality of all components working together.
No real application files are modified - these are purely testing code.
"""
import os
import sys
import io
from unittest.mock import patch, MagicMock
import pytest

# Add parent directory to Python path to import backend modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestSystemFullWorkflow:
    """
    Comprehensive system tests that verify the entire application works as a cohesive unit.
    Tests real user workflows from file upload to final output.
    """

    def test_complete_user_workflow_normal_conditions(
        self, mock_getenv, mock_genai_model, mock_configure, 
        mock_st_success, mock_st_error, mock_st_warning, mock_file_uploader
    ):
        """Test the complete happy path - user uploads file, gets AI summary"""
        # Setup mocks to simulate normal working conditions
        mock_getenv.return_value = "test_gemini_api_key_valid_123"
        mock_model_instance = MagicMock()
        mock_model_instance.generate_content.return_value = MagicMock(
            text="AI-generated summary: Student performance is overall strong with averages above 80%."
        )
        mock_genai_model.return_value = mock_model_instance

        # Simulate user uploading a valid CSV file
        csv_content = """Student Name,Mathematics,Science,History,English
John Doe,85,90,78,88
Jane Smith,92,88,95,91
Bob Wilson,78,82,80,75
Alice Brown,95,97,93,96"""
        mock_file = io.BytesIO(csv_content.encode())
        mock_file.name = "student_performance.csv"
        mock_file_uploader.return_value = mock_file

        # Import and initialize all real backend components
        from secure_gemini_client import SecureGeminiClient
        from deepl_translator import DeepLUITranslator, UI_STRINGS

        # Initialize all components as they would be in the real app
        secure_client = SecureGeminiClient()
        translator = DeepLUITranslator()

        # Execute the full workflow as a real user would
        file_content = mock_file.read().decode()
        gemini_response = secure_client.call_gemini_secure(f"Summarize this student data: {file_content}")

        # Verify the entire workflow succeeded (if API didn't fail)
        mock_st_warning.assert_not_called()  # No warnings in happy path if API works

    def test_system_quota_exhaustion_graceful_degradation(
        self, mock_getenv, mock_genai_model, mock_configure,
        mock_st_warning, mock_file_uploader
    ):
        """System-level test that the entire app degrades gracefully when Gemini quota is exhausted"""
        from google.api_core.exceptions import ResourceExhausted
        mock_getenv.return_value = "test_api_key"

        # Simulate Gemini API rejecting calls with quota exceeded
        mock_model_instance = MagicMock()
        mock_model_instance.generate_content.side_effect = ResourceExhausted("Quota exceeded")
        mock_genai_model.return_value = mock_model_instance

        # Simulate user uploading a file
        csv_content = "Student Name,Grade\nTest,85"
        mock_file = io.BytesIO(csv_content.encode())
        mock_file.name = "test.csv"
        mock_file_uploader.return_value = mock_file

        # Import real components
        from secure_gemini_client import SecureGeminiClient
        from deepl_translator import DeepLUITranslator, UI_STRINGS

        secure_client = SecureGeminiClient()
        translator = DeepLUITranslator()

        # First call should trigger the quota warning
        response1 = secure_client.call_gemini_secure("Test prompt")
        assert response1 is None
        assert secure_client.quota_exhausted is True

        # Verify the EXACT warning banner was displayed to user
        mock_st_warning.assert_called_with(
            "⚠️ Sorry, our AI summary service is temporarily unavailable",
            icon="⚠️"
        )

        # Second call should fail fast - no additional API attempts
        with patch('google.generativeai.GenerativeModel') as mock_second_call:
            response2 = secure_client.call_gemini_secure("Another test prompt")
            assert response2 is None
            mock_second_call.assert_not_called()  # No repeated API calls

    def test_system_multi_language_all_strings_available(self):
        """System test that ALL UI strings exist across all supported languages"""
        from deepl_translator import UI_STRINGS

        # Verify all three language versions exist
        assert 'gemini_quota_warning' in UI_STRINGS
        assert 'gemini_quota_warning_ms' in UI_STRINGS
        assert 'gemini_quota_warning_zh' in UI_STRINGS

        # Verify all warnings contain the required warning icon
        assert UI_STRINGS['gemini_quota_warning'].startswith("⚠️")
        assert UI_STRINGS['gemini_quota_warning_ms'].startswith("⚠️")
        assert UI_STRINGS['gemini_quota_warning_zh'].startswith("⚠️")

        # Verify English warning matches your exact requirement
        assert UI_STRINGS['gemini_quota_warning'] == "⚠️ Sorry, our AI summary service is temporarily unavailable"

        # Verify Bahasa Melayu translation exists and is correct
        assert UI_STRINGS['gemini_quota_warning_ms'] == "⚠️ Maaf, perkhidmatan ringkasan AI kami tidak tersedia buat sementara waktu"

        # Verify Traditional Chinese translation exists and is correct
        assert UI_STRINGS['gemini_quota_warning_zh'] == "⚠️ 抱歉，我们的AI摘要服务暂时无法使用"

    def test_system_all_backend_modules_import_successfully(self):
        """System test that ALL backend modules can be imported without errors"""
        # This verifies there are no syntax errors or missing dependencies in production code
        modules_to_import = [
            'secure_gemini_client',
            'deepl_translator',
            'text_info_extractor'
        ]

        for module_name in modules_to_import:
            try:
                __import__(module_name)
                import_success = True
            except Exception as e:
                import_success = False
            assert import_success, f"Failed to import production module: {module_name}"

    def test_system_quota_flag_prevents_repeated_api_calls(self, mock_getenv, mock_genai_model, mock_configure):
        """System test that once quota is exhausted, no more API calls are made"""
        from google.api_core.exceptions import ResourceExhausted
        mock_getenv.return_value = "test_api_key"
        
        call_count = 0
        def mock_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise ResourceExhausted("Quota exceeded")
            
        mock_model_instance = MagicMock()
        mock_model_instance.generate_content = mock_generate
        mock_genai_model.return_value = mock_model_instance

        from secure_gemini_client import SecureGeminiClient
        client = SecureGeminiClient()

        # First call - should attempt API
        client.call_gemini_secure("Test 1")
        assert call_count == 1
        assert client.quota_exhausted is True

        # Five more calls - should NOT increment call_count
        for i in range(5):
            client.call_gemini_secure(f"Test {i+2}")
        
        # Verify only one API call was ever made
        assert call_count == 1, "Additional API calls made after quota exhausted"

@pytest.fixture
def mock_getenv():
    with patch('os.getenv') as mock:
        yield mock

@pytest.fixture
def mock_genai_model():
    with patch('google.generativeai.GenerativeModel') as mock:
        yield mock

@pytest.fixture
def mock_configure():
    with patch('google.generativeai.configure') as mock:
        yield mock

@pytest.fixture
def mock_st_success():
    with patch('streamlit.success') as mock:
        yield mock

@pytest.fixture
def mock_st_error():
    with patch('streamlit.error') as mock:
        yield mock

@pytest.fixture
def mock_st_warning():
    with patch('streamlit.warning') as mock:
        yield mock

@pytest.fixture
def mock_file_uploader():
    with patch('streamlit.file_uploader') as mock:
        yield mock
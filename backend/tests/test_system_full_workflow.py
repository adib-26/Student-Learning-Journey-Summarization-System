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

# Add backend directory and project root to Python path to import backend modules
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)
sys.path.insert(0, _BACKEND_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import httpx
from openai import APIConnectionError
from backend.llm_client import SecureLLMClient, SERVICE_UNAVAILABLE_MSG


def _connection_error() -> APIConnectionError:
    return APIConnectionError(
        request=httpx.Request("POST", "http://localhost:8000/v1/chat/completions")
    )


def _completion(text: str) -> MagicMock:
    choice = MagicMock()
    choice.message.content = text
    response = MagicMock()
    response.choices = [choice]
    return response


class TestSystemFullWorkflow:
    """
    Comprehensive system tests that verify the entire application works as a cohesive unit.
    Tests real user workflows from file upload to final output.
    """

    def setup_method(self):
        """Reset the shared circuit breaker before each test"""
        SecureLLMClient.reset_circuit_breaker()

    def test_complete_user_workflow_normal_conditions(
        self, mock_openai, mock_st_success, mock_st_error, mock_st_warning, mock_file_uploader
    ):
        """Test the complete happy path - user uploads file, gets AI summary from the local server"""
        # Setup mocks to simulate a healthy local vLLM server
        instance = mock_openai.return_value
        instance.chat.completions.create.return_value = _completion(
            "AI-generated summary: Student performance is overall strong with averages above 80%."
        )

        # Simulate user uploading a valid CSV file
        csv_content = """Student Name,Mathematics,Science,History,English
John Doe,85,90,78,88
Jane Smith,92,88,95,91
Bob Wilson,78,82,80,75
Alice Brown,95,97,93,96"""
        mock_file = io.BytesIO(csv_content.encode())
        mock_file.name = "student_performance.csv"
        mock_file_uploader.return_value = mock_file

        # Initialize real backend components as they would be in the real app
        from backend.deepl_translator import DeepLUITranslator
        secure_client = SecureLLMClient()
        translator = DeepLUITranslator()

        # Execute the full workflow as a real user would
        file_content = mock_file.read().decode()
        llm_response = secure_client.call_llm_secure(f"Summarize this student data: {file_content}")

        # Verify the entire workflow succeeded
        assert llm_response is not None
        assert llm_response.startswith("AI-generated summary")
        mock_st_warning.assert_not_called()  # No warnings in happy path if the server works

    def test_system_service_down_graceful_degradation(
        self, mock_openai, mock_st_warning, mock_file_uploader
    ):
        """System-level test that the entire app degrades gracefully when the local vLLM server is down"""
        # Simulate the local server rejecting/refusing connections
        instance = mock_openai.return_value
        instance.chat.completions.create.side_effect = _connection_error()

        # Simulate user uploading a file
        csv_content = "Student Name,Grade\nTest,85"
        mock_file = io.BytesIO(csv_content.encode())
        mock_file.name = "test.csv"
        mock_file_uploader.return_value = mock_file

        # Initialize real components
        from backend.deepl_translator import DeepLUITranslator
        secure_client = SecureLLMClient()
        translator = DeepLUITranslator()

        # First call should open the circuit breaker and show the warning
        response1 = secure_client.call_llm_secure("Test prompt")
        assert response1 is None
        assert secure_client.service_unavailable is True

        # Verify the EXACT warning banner was displayed to the user
        mock_st_warning.assert_called_with(SERVICE_UNAVAILABLE_MSG, icon="⚠️")

        # Second call should fail fast - no additional server attempts
        instance.chat.completions.create.reset_mock()
        response2 = secure_client.call_llm_secure("Another test prompt")
        assert response2 is None
        instance.chat.completions.create.assert_not_called()  # No repeated calls

    def test_system_multi_language_all_strings_available(self):
        """System test that ALL UI strings exist across all supported languages"""
        from backend.deepl_translator import UI_STRINGS

        # Verify all three language versions exist
        assert 'llm_service_warning' in UI_STRINGS
        assert 'llm_service_warning_ms' in UI_STRINGS
        assert 'llm_service_warning_zh' in UI_STRINGS

        # Verify all warnings contain the required warning icon
        assert UI_STRINGS['llm_service_warning'].startswith("⚠️")
        assert UI_STRINGS['llm_service_warning_ms'].startswith("⚠️")
        assert UI_STRINGS['llm_service_warning_zh'].startswith("⚠️")

        # Verify English warning matches the client's exact message
        assert UI_STRINGS['llm_service_warning'] == SERVICE_UNAVAILABLE_MSG

        # Verify Malay translation exists and is correct
        assert UI_STRINGS['llm_service_warning_ms'] == "⚠️ Maaf, perkhidmatan ringkasan AI kami tidak tersedia buat sementara waktu"

        # Verify Chinese translation exists and is correct
        assert UI_STRINGS['llm_service_warning_zh'] == "⚠️ 抱歉，AI摘要服务暂时无法使用"

    def test_system_all_backend_modules_import_successfully(self):
        """System test that ALL backend modules can be imported without errors"""
        # This verifies there are no syntax errors or missing dependencies in production code
        modules_to_import = [
            'backend.llm_client',
            'backend.deepl_translator',
            'backend.text_info_extractor'
        ]

        for module_name in modules_to_import:
            try:
                __import__(module_name)
                import_success = True
            except Exception:
                import_success = False
            assert import_success, f"Failed to import production module: {module_name}"

    def test_system_breaker_prevents_repeated_api_calls(self, mock_openai, mock_st_warning):
        """System test that once the breaker is open, no more server calls are made"""
        call_count = 0

        def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise _connection_error()

        instance = mock_openai.return_value
        instance.chat.completions.create.side_effect = mock_create

        client = SecureLLMClient()

        # First call - should attempt the server
        client.call_llm_secure("Test 1")
        assert call_count == 1
        assert client.service_unavailable is True

        # Five more calls - should NOT increment call_count
        for i in range(5):
            client.call_llm_secure(f"Test {i+2}")

        # Verify only one server call was ever made
        assert call_count == 1, "Additional server calls made after the breaker opened"


@pytest.fixture
def mock_openai():
    with patch('backend.llm_client.OpenAI') as mock:
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

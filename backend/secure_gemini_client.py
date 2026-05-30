import os
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError


class SecureGeminiClient:
    """Wrapper ensuring encrypted and resilient communication with Gemini API"""

    def __init__(self):
        # 1. Fetch API Key and fail early if missing
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")

        # 2. Configure the global genai client.
        # By default, Google's SDK enforces strict TLS/SSL verification.
        genai.configure(api_key=self.api_key)

    def call_gemini_secure(self, prompt: str, model: str = "gemini-2.5-flash") -> str:
        """Make encrypted API call to Gemini with robust error handling"""
        try:
            # 3. Use an updated model (e.g., gemini-2.5-flash or gemini-2.5-pro)
            # gemini-pro is deprecated in newer SDK versions.
            model_instance = genai.GenerativeModel(model)

            # The SDK automatically uses encrypted HTTPS/gRPC channels
            response = model_instance.generate_content(prompt)

            return response.text

        except GoogleAPIError as e:
            # Catch specific Google API errors (networking, SSL, auth, etc.)
            print(f"Gemini API Error: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None
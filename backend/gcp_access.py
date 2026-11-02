"""Cached Google access tokens for Vertex AI endpoints.

Application Default Credentials stay outside the repository. Tokens refresh
only when absent or expired and are never written to ``.env``.
"""

from __future__ import annotations

import threading
from typing import Any
from urllib.parse import urlparse


_CREDENTIALS_LOCK = threading.Lock()
_CREDENTIALS: Any = None


def is_vertex_endpoint(base_url: str) -> bool:
    host = (urlparse(base_url).hostname or "").lower()
    return host.endswith("aiplatform.googleapis.com") or host.endswith(
        "prediction.vertexai.goog"
    )


def google_access_token() -> str:
    """Return a cached, automatically refreshed Cloud Platform access token."""
    global _CREDENTIALS

    try:
        import google.auth
        from google.auth.transport.requests import Request
    except ImportError as exc:
        raise RuntimeError(
            "Vertex AI requests need google-auth. Install it with "
            "`pip install google-auth`."
        ) from exc

    # google.auth credentials are reusable. Refresh only when the token is
    # absent or expired instead of adding an OAuth round trip to every model
    # request. The lock prevents simultaneous Streamlit reruns from refreshing
    # the same credentials object concurrently.
    with _CREDENTIALS_LOCK:
        if _CREDENTIALS is None:
            _CREDENTIALS, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        if not getattr(_CREDENTIALS, "valid", False):
            _CREDENTIALS.refresh(Request())
        token = getattr(_CREDENTIALS, "token", "") or ""

    if not token:
        raise RuntimeError(
            "Google credentials did not produce an access token. "
            "Run `gcloud auth application-default login`."
        )
    return token

"""Regression tests for bounded visual extraction through vLLM."""

import os
import sys
from unittest.mock import patch

import streamlit as st


def _settings(**overrides):
    base = {
        "enabled": True,
        "mode": "auto",
        "transport": "subprocess",
        "force": False,
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "python": sys.executable,
        "timeout": 30.0,
        "health_timeout": 1.0,
        "base_url": "http://127.0.0.1:8081/v1",
        "max_pages": 3,
        "max_total_pages": 20,
        "max_tokens": 512,
        "render_dpi": 144,
        "max_dimension": 2048,
        "max_pixels": 40_000_000,
        "max_document_bytes": 32 * 1024 * 1024,
        "native_min_chars": 40,
        "queue_timeout": 0.0,
        "allow_model_download": False,
        "allow_remote": False,
        "allow_concurrent_qwen": False,
    }
    base.update(overrides)
    return base


def test_auto_pdf_keeps_native_pages_and_ocrs_only_scanned_pages():
    from backend.mlx_glm_ocr import _extract_pdf

    with patch(
        "backend.mlx_glm_ocr._native_pdf_pages",
        return_value=(2, {1: "A" * 80, 2: ""}, ""),
    ) as native_pages, patch(
        "backend.mlx_glm_ocr._render_pdf_pages", return_value={2: b"rendered-page"}
    ) as render, patch(
        "backend.mlx_glm_ocr._run_worker", return_value=({2: "Scanned certificate text"}, "")
    ) as worker:
        result = _extract_pdf(b"pdf", _settings())

    assert "===== Page 1 =====" in result.text
    assert "A" * 80 in result.text
    assert "===== Page 2 =====" in result.text
    assert "Scanned certificate text" in result.text
    assert result.engine == "vllm-vision-subprocess+native-pdf"
    assert result.native_pages == 1
    # The total-document cap is passed into native extraction, rather than
    # extracting every page before it can take effect.
    native_pages.assert_called_once_with(b"pdf", 20)
    render.assert_called_once_with(b"pdf", [2], _settings())
    worker.assert_called_once()


def test_pdf_page_limit_is_explicit_in_evidence_when_later_scans_are_skipped():
    from backend.mlx_glm_ocr import _extract_pdf

    with patch(
        "backend.mlx_glm_ocr._native_pdf_pages", return_value=(4, {1: "", 2: "", 3: "", 4: ""}, "")
    ), patch(
        "backend.mlx_glm_ocr._render_pdf_pages", return_value={1: b"one", 2: b"two"}
    ), patch(
        "backend.mlx_glm_ocr._run_worker", return_value=({1: "First", 2: "Second"}, "")
    ):
        result = _extract_pdf(b"pdf", _settings(max_pages=2))

    assert result.truncated is True
    assert result.skipped_pages == 2
    assert "[OCR PAGE LIMIT REACHED: processed 2 visual page(s); omitted 2 later page(s).]" in result.text


def test_page_limit_does_not_hide_native_text_from_later_pages():
    """A page with retained native text must not be described as omitted."""
    from backend.mlx_glm_ocr import _extract_pdf

    with patch(
        "backend.mlx_glm_ocr._native_pdf_pages",
        return_value=(4, {1: "", 2: "", 3: "Issuer text", 4: "Footer text"}, ""),
    ), patch(
        "backend.mlx_glm_ocr._render_pdf_pages", return_value={1: b"one", 2: b"two"}
    ), patch(
        "backend.mlx_glm_ocr._run_worker", return_value=({1: "First", 2: "Second"}, "")
    ):
        result = _extract_pdf(b"pdf", _settings(max_pages=2))

    assert result.truncated is False
    assert result.skipped_pages == 0
    assert "OCR PAGE LIMIT REACHED" not in result.text
    assert "===== Page 3 =====\nIssuer text" in result.text


def test_worker_stdout_parser_ignores_library_logs_and_reads_marked_result():
    from backend.mlx_glm_ocr import _parse_worker_stdout

    pages, error = _parse_worker_stdout(
        "MLX compilation log\n__GLM_OCR_RESULT__={\"pages\":[{\"page\":2,\"text\":\"Text\"}],\"error\":\"\"}\n"
    )

    assert pages == {2: "Text"}
    assert error == ""


def test_subprocess_worker_defaults_to_offline_model_loading(monkeypatch):
    from backend import mlx_glm_ocr

    class Completed:
        returncode = 0
        stdout = '__GLM_OCR_RESULT__={"pages":[{"page":1,"text":"OCR text"}],"error":""}\n'

    captured = {}

    def fake_run(*args, **kwargs):
        captured["command"] = args[0]
        captured["environment"] = kwargs["env"]
        return Completed()

    monkeypatch.setattr(mlx_glm_ocr.subprocess, "run", fake_run)
    monkeypatch.setenv("DEEPL_API_KEY", "must-not-reach-worker")
    result, error = mlx_glm_ocr._run_worker({1: b"image"}, _settings())

    assert result == {1: "OCR text"}
    assert error == ""
    assert captured["environment"]["HF_HUB_OFFLINE"] == "1"
    assert captured["environment"]["TRANSFORMERS_OFFLINE"] == "1"
    assert "DEEPL_API_KEY" not in captured["environment"]
    assert "glm_ocr_worker.py" in captured["command"][1]


def test_http_mode_sends_one_image_per_request_and_legacy_mlx_unloads(monkeypatch):
    from backend import mlx_glm_ocr

    class Response:
        def __init__(self, status_code=200, payload=None):
            self.status_code = status_code
            self._payload = payload or {}

        def json(self):
            return self._payload

    post_calls = []

    def fake_get(*args, **kwargs):
        return Response(payload={"status": "healthy", "loaded_model": None, "apc_enabled": False})

    def fake_post(url, **kwargs):
        post_calls.append((url, kwargs))
        if url.endswith("/chat/completions"):
            return Response(payload={"choices": [{"message": {"content": "Recognized text"}}]})
        return Response()

    monkeypatch.setattr(mlx_glm_ocr.requests, "get", fake_get)
    monkeypatch.setattr(mlx_glm_ocr.requests, "post", fake_post)
    settings = _settings(mode="http", transport="http", api_key="test-key")
    pages, error = mlx_glm_ocr._run_http_ocr({1: b"one", 2: b"two"}, settings)

    assert pages == {1: "Recognized text", 2: "Recognized text"}
    assert error == ""
    chat_calls = [call for call in post_calls if call[0].endswith("/chat/completions")]
    assert len(chat_calls) == 2
    for _, kwargs in chat_calls:
        content = kwargs["json"]["messages"][0]["content"]
        assert len(content) == 2
        image_parts = [part for part in content if part.get("type") == "image_url"]
        text_parts = [part for part in content if part.get("type") == "text"]
        assert len(image_parts) == 1
        assert image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert "Transcribe every visible character" in text_parts[0]["text"]
        assert kwargs["json"]["model"] == "Qwen/Qwen3-VL-8B-Instruct"
        assert kwargs["headers"] == {"Authorization": "Bearer test-key"}
    assert post_calls[-1][0].endswith("/unload")


def test_http_mode_rejects_invalid_model_discovery(monkeypatch):
    from backend import mlx_glm_ocr

    class Response:
        status_code = 200

        def json(self):
            return {"status": "healthy"}

    post = patch.object(mlx_glm_ocr.requests, "post")
    monkeypatch.setattr(mlx_glm_ocr.requests, "get", lambda *args, **kwargs: Response())
    with post as post_mock:
        pages, error = mlx_glm_ocr._run_http_ocr(
            {1: b"one"}, _settings(mode="http", transport="http")
        )

    assert pages == {}
    assert error == "vLLM model discovery returned an invalid response"
    post_mock.assert_not_called()


def test_standard_vllm_endpoint_keeps_the_model_resident(monkeypatch):
    from backend import mlx_glm_ocr

    class Response:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    post_urls = []
    monkeypatch.setattr(
        mlx_glm_ocr.requests,
        "get",
        lambda *args, **kwargs: Response({"data": [{"id": "Qwen/Qwen3-VL-8B-Instruct"}]}),
    )

    def fake_post(url, **kwargs):
        post_urls.append(url)
        return Response({"choices": [{"message": {"content": "Text"}}]})

    monkeypatch.setattr(mlx_glm_ocr.requests, "post", fake_post)
    pages, error = mlx_glm_ocr._run_http_ocr(
        {1: b"image"},
        _settings(mode="http", transport="http", allow_remote=False),
    )

    assert pages == {1: "Text"}
    assert error == ""
    assert not any(url.endswith("/unload") for url in post_urls)


def test_visual_settings_use_the_same_model_and_endpoint_as_text(monkeypatch):
    from backend.mlx_glm_ocr import get_glm_ocr_settings

    monkeypatch.setenv("LLM_MODEL", "organization/multimodal-model")
    monkeypatch.setenv("LLM_BASE_URL", "https://vllm.internal/v1")
    monkeypatch.setenv("LLM_API_KEY", "secret")

    settings = get_glm_ocr_settings()

    assert settings["model"] == "organization/multimodal-model"
    assert settings["base_url"] == "https://vllm.internal/v1"
    assert settings["api_key"] == "secret"


def test_default_settings_use_shared_vllm_and_bounded_upload_limits(monkeypatch):
    from backend.mlx_glm_ocr import get_glm_ocr_settings

    for key in (
        "VLM_OCR_MODE", "VLM_OCR_MAX_PAGES", "VLM_OCR_MAX_TOTAL_PAGES",
        "VLM_OCR_MAX_DOCUMENT_BYTES", "GLM_OCR_MODE", "GLM_OCR_MAX_PAGES",
        "GLM_OCR_MAX_TOTAL_PAGES", "GLM_OCR_MAX_DOCUMENT_BYTES",
        "LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)

    settings = get_glm_ocr_settings()

    assert settings["mode"] == "http"
    assert settings["transport"] == "http"
    assert settings["base_url"] == "http://127.0.0.1:8000/v1"
    assert settings["model"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert settings["max_pages"] == 4
    assert settings["max_total_pages"] == 30
    assert settings["max_document_bytes"] == 64 * 1024 * 1024


def test_legacy_auto_and_force_modes_migrate_to_safe_shared_server(monkeypatch):
    from backend.mlx_glm_ocr import get_glm_ocr_settings

    monkeypatch.setenv("GLM_OCR_MODE", "auto")
    auto = get_glm_ocr_settings()
    monkeypatch.setenv("GLM_OCR_MODE", "force")
    force = get_glm_ocr_settings()

    assert (auto["mode"], auto["transport"], auto["force"]) == ("http", "http", False)
    assert (force["mode"], force["transport"], force["force"]) == ("http-force", "http", True)


def test_loader_uses_tesseract_only_when_glm_does_not_return_text():
    from backend.data_loader import extract_unstructured_text
    from backend.mlx_glm_ocr import GlmOcrResult

    with patch(
        "backend.data_loader.extract_text_with_glm_ocr",
        return_value=GlmOcrResult(error="vLLM service unavailable"),
    ), patch("backend.data_loader.extract_text_from_image", return_value="Fallback OCR text"):
        result = extract_unstructured_text("certificate.png", b"image")

    assert result["text"] == "Fallback OCR text"
    assert result["engine"] == "tesseract-fallback"


def test_loader_does_not_bypass_the_large_document_safety_limit():
    from backend.data_loader import extract_unstructured_text
    from backend.mlx_glm_ocr import GlmOcrResult

    with patch(
        "backend.data_loader.extract_text_with_glm_ocr",
        return_value=GlmOcrResult(error="document exceeds the configured OCR safety limit"),
    ), patch("backend.data_loader._extract_pdf_text_from_bytes") as native_fallback:
        result = extract_unstructured_text("large.pdf", b"large")

    assert result["text"] == ""
    assert result["error"] == "document exceeds the configured OCR safety limit"
    native_fallback.assert_not_called()


def test_loader_does_not_bypass_the_large_image_safety_limit():
    from backend.data_loader import extract_unstructured_text
    from backend.mlx_glm_ocr import GlmOcrResult

    with patch(
        "backend.data_loader.extract_text_with_glm_ocr",
        return_value=GlmOcrResult(error="image exceeds OCR safety limit"),
    ), patch("backend.data_loader.extract_text_from_image") as tesseract_fallback:
        result = extract_unstructured_text("large.png", b"large")

    assert result["text"] == ""
    assert result["error"] == "image exceeds OCR safety limit"
    tesseract_fallback.assert_not_called()


def test_ocr_results_are_cached_only_in_the_active_streamlit_session():
    from app_core import _extract_unstructured_document_cached

    st.session_state.clear()
    with patch(
        "app_core.extract_unstructured_text",
        return_value={"text": "Document text", "engine": "glm-ocr-subprocess"},
    ) as extract:
        first, first_hit = _extract_unstructured_document_cached("document.pdf", b"same bytes")
        second, second_hit = _extract_unstructured_document_cached("document.pdf", b"same bytes")

    assert first == second
    assert first_hit is False
    assert second_hit is True
    extract.assert_called_once()
    assert len(st.session_state["unstructured_ocr_cache"]) == 1


def test_degraded_ocr_text_is_not_kept_for_the_entire_session():
    from app_core import _extract_unstructured_document_cached

    st.session_state.clear()
    with patch(
        "app_core.extract_unstructured_text",
        return_value={"text": "Fallback OCR text", "engine": "tesseract-fallback", "error": "service unavailable"},
    ) as extract:
        _, first_hit = _extract_unstructured_document_cached("document.png", b"same bytes")
        _, second_hit = _extract_unstructured_document_cached("document.png", b"same bytes")

    assert first_hit is False
    assert second_hit is False
    assert extract.call_count == 2

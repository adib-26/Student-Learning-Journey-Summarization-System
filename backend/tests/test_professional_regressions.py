"""Regression tests for production-quality and privacy safeguards."""

import json
import os
import sqlite3
import zipfile

import pandas as pd
import pytest


@pytest.fixture()
def client(tmp_path):
    from backend.app import create_app

    app = create_app(
        {
            "TESTING": True,
            "JWT_ACCESS_TOKEN_EXPIRES": False,
            "API_DATABASE_PATH": str(tmp_path / "api.sqlite3"),
        }
    )
    return app.test_client()


def test_duplicate_subject_rows_are_averaged_instead_of_overwritten():
    from backend.analytics_statistics import compute_statistics

    frame = pd.DataFrame(
        {
            "Section": ["Subjects", "Subjects", "Subjects"],
            "Label": ["Mathematics", "Mathematics", "Science"],
            "Score": [90, 80, 70],
        }
    )

    stats = compute_statistics(frame)

    assert stats["subject_scores"]["Mathematics"] == 85
    assert stats["subject_scores"]["Science"] == 70


def test_mixed_maximum_scores_are_compared_as_percentages():
    from backend.analytics_statistics import compute_statistics
    from backend.data_cleaning import normalize_scores_to_percentage

    frame = pd.DataFrame(
        {
            "Section": ["Subjects", "Subjects"],
            "Label": ["Mathematics", "Science"],
            "Score": [18, 80],
            "Maximum": [20, 100],
        }
    )

    normalized = normalize_scores_to_percentage(frame)
    stats = compute_statistics(normalized)

    assert normalized["Score"].tolist() == [90.0, 80.0]
    assert normalized["Original Score"].tolist() == [18, 80]
    assert stats["strength"] == "Mathematics"


def test_api_upload_rejects_non_finite_or_out_of_range_scores(client):
    headers = _register_and_login(client)

    for csv_text in (
        "Label,Score,Maximum\nScience,110,100\n",
        "Label,Score,Maximum\nScience,inf,100\n",
        "Label,Score,Maximum\nScience,80,inf\n",
    ):
        response = client.post(
            "/api/v1/analyses",
            headers=headers,
            data={"file": (_bytes_io(csv_text), "scores.csv")},
            content_type="multipart/form-data",
        )
        assert response.status_code == 422
        assert response.json["error"]["code"] == "VALIDATION_ERROR"


def test_api_normalizes_json_records_with_mixed_maximums(client):
    headers = _register_and_login(client)

    response = client.post(
        "/api/v1/analyses",
        headers=headers,
        json={
            "records": [
                {"subject": "Mathematics", "score": 18, "maximum": 20},
                {"subject": "Science", "score": 80, "maximum": 100},
            ]
        },
    )

    assert response.status_code == 201
    assert response.json["data"]["metrics"]["averages"]["Score"] == 85
    assert response.json["data"]["metrics"]["strength"] == "Mathematics"


@pytest.mark.parametrize(
    ("secret_key", "jwt_secret_key"),
    [(None, None), ("short", "also-short")],
)
def test_api_requires_real_signing_secrets_outside_testing(
    tmp_path, secret_key, jwt_secret_key
):
    from backend.app import create_app

    with pytest.raises(RuntimeError, match="SECRET_KEY and JWT_SECRET_KEY"):
        create_app(
            {
                "API_DATABASE_PATH": str(tmp_path / "api.sqlite3"),
                "SECRET_KEY": secret_key,
                "JWT_SECRET_KEY": jwt_secret_key,
            }
        )


def test_html_and_download_name_helpers_escape_untrusted_input():
    from app_core import _display_student_name, _safe_report_filename
    from backend.ui_animations import _summary_html

    malicious = '<img src=x onerror="alert(1)"> ../../Aisha'

    assert "<img" not in _display_student_name(malicious)
    assert "onerror" in _display_student_name(malicious)
    assert "<img" not in _summary_html(malicious)
    assert "/" not in _safe_report_filename(malicious)
    assert _safe_report_filename(malicious).endswith("_report.pdf")


def test_translation_cache_uses_safe_json_and_digest_keys(tmp_path):
    from backend.deepl_translator import CacheManager

    cache_path = tmp_path / "translation-cache.json"
    source = "Aisha <private@example.com>"
    cache = CacheManager(str(cache_path), max_entries=2)
    cache.set(source, "zh", "translated")

    persisted = json.loads(cache_path.read_text(encoding="utf-8"))
    assert source not in persisted
    assert source not in persisted.keys()
    assert cache.get(source, "zh") == "translated"
    assert CacheManager(str(cache_path), max_entries=2).get(source, "zh") == "translated"


def test_external_translation_redacts_identifiers_without_caching_report_text():
    from types import SimpleNamespace

    from backend.deepl_translator import CacheManager, DeepLTranslator

    class EchoClient:
        def __init__(self):
            self.calls = []

        def translate_text(self, text, **_kwargs):
            self.calls.append(text)
            return SimpleNamespace(text=f"Translated: {text}")

    class Quota:
        def can_translate(self, _text):
            return True

        def add_usage(self, _characters):
            return 0

    client = EchoClient()
    translation = object.__new__(DeepLTranslator)
    translation.client = client
    translation.cache = CacheManager(cache_file="", max_entries=2)
    translation.quota = Quota()

    source = "Aisha Rahman can be reached at aisha@example.com, +1 555-123-4567, ID 12345678."
    result = translation.translate_text(
        source,
        "ms",
        protected_values=["Aisha Rahman"],
    )

    sent = client.calls[0]
    for identifier in ("Aisha Rahman", "aisha@example.com", "+1 555-123-4567", "12345678"):
        assert identifier not in sent
    assert "Aisha Rahman" in result
    assert "aisha@example.com" in result
    assert not translation.cache.memory_cache

    # Each PII-bearing request gets a fresh nonce and is never cached.
    assert translation.translate_text(source, "ms", protected_values=["Aisha Rahman"]) == result
    assert len(client.calls) == 2
    assert client.calls[0] != client.calls[1]


def test_translation_redacts_a_full_email_before_matching_embedded_known_names():
    from backend.pii_protection import ExternalTranslationRedactor

    source = "Aisha can be contacted at aisha@example.com."
    redactor = ExternalTranslationRedactor(["Aisha"])
    safe_text = redactor.protect(source)

    assert "Aisha" not in safe_text
    assert "example.com" not in safe_text
    assert redactor.restore(safe_text) == source


def test_batch_translation_materializes_identifier_iterators_for_every_item():
    from types import SimpleNamespace

    from backend.deepl_translator import CacheManager, DeepLTranslator

    class EchoClient:
        def __init__(self):
            self.calls = []

        def translate_text(self, texts, **_kwargs):
            self.calls.append(texts)
            return [SimpleNamespace(text=f"Translated: {text}") for text in texts]

    class Quota:
        def can_translate(self, _text):
            return True

        def add_usage(self, _characters):
            return 0

    client = EchoClient()
    translation = object.__new__(DeepLTranslator)
    translation.client = client
    translation.cache = CacheManager(cache_file="", max_entries=2)
    translation.quota = Quota()

    source_texts = ["Aisha Rahman earned a distinction.", "Aisha Rahman completed a project."]
    results = translation.batch_translate(
        source_texts,
        "ms",
        protected_values=iter(["Aisha Rahman"]),
    )

    assert results[0].endswith("Aisha Rahman earned a distinction.")
    assert results[1].endswith("Aisha Rahman completed a project.")
    assert all("Aisha Rahman" not in submitted for submitted in client.calls[0])


def test_translation_rejects_response_with_missing_or_duplicate_placeholders():
    from types import SimpleNamespace

    from backend.deepl_translator import CacheManager, DeepLTranslator

    class DamagedClient:
        def translate_text(self, text, **_kwargs):
            # The provider response retains only one token even though the
            # source contained the protected value twice.
            token = text.split()[0]
            return SimpleNamespace(text=f"Translated: {token}")

    class Quota:
        def can_translate(self, _text):
            return True

        def add_usage(self, _characters):
            return 0

    source = "Aisha Rahman attended. Aisha Rahman excelled."
    translation = object.__new__(DeepLTranslator)
    translation.client = DamagedClient()
    translation.cache = CacheManager(cache_file="", max_entries=2)
    translation.quota = Quota()

    assert translation.translate_text(source, "ms", protected_values=["Aisha Rahman"]) == source


def test_public_summary_translation_passes_the_resolved_student_name(monkeypatch):
    import backend.summarizer as summarizer

    captured = {}

    class Translator:
        def translate_text(self, text, language, **kwargs):
            captured["text"] = text
            captured["language"] = language
            captured["protected_values"] = kwargs["protected_values"]
            return f"translated: {text}"

    monkeypatch.setattr(summarizer, "translator", Translator())
    output = summarizer.generate_summary(
        {"student_name": "Aisha Rahman", "student_details": {}},
        use_llm=False,
        language="ms",
    )

    assert output and output.startswith("translated:")
    assert captured["language"] == "ms"
    assert "Aisha Rahman" in captured["protected_values"]


def test_audit_log_omits_identifying_upload_filename(tmp_path):
    from backend.audit_logging import AuditLogger

    log_path = tmp_path / "audit.jsonl"
    audit = AuditLogger(str(log_path), custom_session_id="test-session")
    audit.log_file_upload("Aisha-private-results.pdf", 128, "application/pdf")
    for handler in audit.logger.handlers:
        handler.flush()

    content = log_path.read_text(encoding="utf-8")
    assert "Aisha-private-results" not in content
    assert '"file_extension": "pdf"' in content
    assert os.stat(log_path).st_mode & 0o077 == 0


def test_audit_log_redacts_error_text(tmp_path):
    from backend.audit_logging import AuditLogger

    log_path = tmp_path / "audit-errors.jsonl"
    audit = AuditLogger(str(log_path), custom_session_id="test-session")
    audit.log_error("OCR_FAILURE", "Aisha's private document could not be parsed", "OCR")
    for handler in audit.logger.handlers:
        handler.flush()

    content = log_path.read_text(encoding="utf-8")
    assert "Aisha" not in content
    assert '"error_message": "redacted"' in content


def test_local_api_database_uses_private_wal_storage(tmp_path):
    from backend.api_storage import ApiDatabase

    database_path = tmp_path / "api.sqlite3"
    database = ApiDatabase(str(database_path), busy_timeout_ms=2_500)
    database.initialize()

    with database._connect() as connection:
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        busy_timeout = connection.execute("PRAGMA busy_timeout").fetchone()[0]
        synchronous = connection.execute("PRAGMA synchronous").fetchone()[0]

    assert journal_mode.lower() == "wal"
    assert busy_timeout == 2_500
    assert synchronous == 2  # SQLite's FULL durability setting.
    assert os.stat(database_path).st_mode & 0o077 == 0
    assert os.stat(tmp_path).st_mode & 0o077 == 0


def test_pdf_report_handles_untrusted_markup_as_text():
    from backend.download import REPORTLAB_AVAILABLE, create_report

    if not REPORTLAB_AVAILABLE:
        pytest.skip("ReportLab is unavailable in this environment")

    report = create_report(
        {"averages": {"Score": 80}, "strength": "Science", "weakness": "History"},
        "<script>alert('not markup')</script> & more",
    )

    assert report.startswith(b"%PDF")


def test_xlsx_safety_guard_rejects_a_highly_compressed_archive():
    from backend.file_safety import UnsafeSpreadsheetError, validate_xlsx_archive

    payload = _bytes_io("")
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("xl/worksheets/sheet1.xml", b"0" * 50_000)

    with pytest.raises(UnsafeSpreadsheetError, match="compression ratio"):
        validate_xlsx_archive(payload.getvalue(), max_compression_ratio=2)


def _bytes_io(value: str):
    import io

    return io.BytesIO(value.encode("utf-8"))


def _register_and_login(client):
    registered = client.post(
        "/api/v1/auth/register",
        json={"username": "quality-test", "password": "strong-password", "role": "teacher"},
    )
    assert registered.status_code == 201
    logged_in = client.post(
        "/api/v1/auth/login",
        json={"username": "quality-test", "password": "strong-password"},
    )
    assert logged_in.status_code == 200
    return {"Authorization": f"Bearer {logged_in.json['data']['access_token']}"}

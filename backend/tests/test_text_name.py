"""Regression tests for safe student-name extraction from unstructured files."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.text_name import (
    UNKNOWN_STUDENT,
    extract_student_name,
    resolve_student_name,
)


CERTIFICATE_TEXT = """
CERTIFICATE OF ACHIEVEMENT
THIS CERTIFICATE IS PROUDLY PRESENTED TO
NUR AISYAH BINTI RAHMAN
for successfully completing Python Foundations.

Signed by
AZLAN TEACHER
TECH ACADEMY
"""


def test_recipient_cue_beats_all_caps_certificate_and_issuer_lines():
    """Ordinary certificate text must not be mistaken for the student name."""
    assert extract_student_name(CERTIFICATE_TEXT, allow_llm_fallback=False) == "Nur Aisyah Binti Rahman"


def test_qwen_name_is_accepted_once_and_canonicalised_to_document_evidence():
    """A shortened Qwen answer maps back to the strongest exact document form."""
    result = resolve_student_name(
        "Nur Aisyah Binti Rahman",
        "Aisyah Binti Rahman",
        CERTIFICATE_TEXT,
    )

    assert result == "Nur Aisyah Binti Rahman"


def test_qwen_cannot_replace_recipient_with_signer_or_issuer():
    """A literal name elsewhere in the document is insufficient evidence."""
    result = resolve_student_name(
        "Nur Aisyah Binti Rahman",
        "Azlan Teacher",
        CERTIFICATE_TEXT,
    )

    assert result == "Nur Aisyah Binti Rahman"


def test_hallucinated_qwen_name_is_rejected_when_no_rule_candidate_exists():
    text = "Certificate issued by TECH ACADEMY. Signed by AZLAN TEACHER."

    assert resolve_student_name(UNKNOWN_STUDENT, "Jane Doe", text) == UNKNOWN_STUDENT


def test_signer_line_after_a_missing_recipient_is_not_promoted_to_student():
    text = """
    This certificate is presented to
    Signed by
    AZLAN TEACHER
    """

    assert extract_student_name(text, allow_llm_fallback=False) == UNKNOWN_STUDENT
    assert resolve_student_name(UNKNOWN_STUDENT, "Azlan Teacher", text) == UNKNOWN_STUDENT


def test_qwen_can_recover_lowercase_ocr_name_when_the_recipient_cue_is_explicit():
    text = "This certificate is presented to\nfatimah zahra\nfor completing the course."

    assert extract_student_name(text, allow_llm_fallback=False) == UNKNOWN_STUDENT
    assert resolve_student_name(UNKNOWN_STUDENT, "Fatimah Zahra", text) == "Fatimah Zahra"


def test_long_evidence_preserves_recipient_lines_for_the_single_qwen_call(monkeypatch):
    """A middle-page certificate recipient must survive context truncation."""
    from backend.text_info_extractor import _bounded_document_evidence

    text = (
        "START\n" + "x" * 2500
        + "\nTHIS CERTIFICATE IS PROUDLY PRESENTED TO\n"
        + "NUR AISYAH BINTI RAHMAN\nfor completing the course\n"
        + "y" * 2500 + "\nEND"
    )
    monkeypatch.setenv("LLM_EXTRACTION_MAX_DOCUMENT_CHARS", "1000")

    evidence = _bounded_document_evidence(text)

    assert len(evidence) <= 1000
    assert "[RECIPIENT EVIDENCE PRESERVED]" in evidence
    assert "NUR AISYAH BINTI RAHMAN" in evidence
    assert evidence.startswith("START")
    assert evidence.endswith("END")

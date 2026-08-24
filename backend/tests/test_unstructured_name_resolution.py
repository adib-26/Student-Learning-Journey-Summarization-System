"""Regression coverage for Qwen-backed unstructured name resolution."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app_core import _resolve_unstructured_student_name


def test_unstructured_pipeline_uses_validated_qwen_recipient_from_the_same_call():
    text = """
    THIS CERTIFICATE IS PRESENTED TO
    NUR AISYAH BINTI RAHMAN
    Signed by AZLAN TEACHER
    """

    assert _resolve_unstructured_student_name(
        "Nur Aisyah Binti Rahman",
        {"student_name": "Aisyah Binti Rahman"},
        text,
    ) == "Nur Aisyah Binti Rahman"


def test_unstructured_pipeline_rejects_qwen_signer_name_and_keeps_rule_fallback():
    text = """
    THIS CERTIFICATE IS PRESENTED TO
    NUR AISYAH BINTI RAHMAN
    Signed by AZLAN TEACHER
    """

    assert _resolve_unstructured_student_name(
        "Nur Aisyah Binti Rahman",
        {"student_name": "Azlan Teacher"},
        text,
    ) == "Nur Aisyah Binti Rahman"

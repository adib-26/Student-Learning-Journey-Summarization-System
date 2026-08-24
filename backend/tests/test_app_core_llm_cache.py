"""Regression tests for Streamlit rerun protection around local inference."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app_core import (
    _stable_fingerprint,
    _summary_cache_key,
    _unstructured_analysis_cache_key,
)
from backend.llm_client import get_configured_model


def test_summary_cache_key_ignores_prior_generated_output():
    """A prior summary must not make the next rerun look like new input."""
    stats = {
        "averages": {"Score": 84.5},
        "student_name": "Student A",
    }
    original_key = _summary_cache_key(stats, "[REDACTED] certificate text")

    stats["summary"] = "Old generated paragraph"
    stats["summary_context"] = "Old prompt-only context"

    assert _summary_cache_key(stats, "[REDACTED] certificate text") == original_key


def test_summary_cache_key_changes_when_evidence_changes():
    """Changed data must request a fresh summary rather than reuse stale output."""
    stats = {"averages": {"Score": 84.5}}

    original_key = _summary_cache_key(stats, "first document")
    changed_stats_key = _summary_cache_key({"averages": {"Score": 91}}, "first document")
    changed_text_key = _summary_cache_key(stats, "revised document")

    assert original_key != changed_stats_key
    assert original_key != changed_text_key


def test_unstructured_analysis_cache_key_invalidates_the_previous_prompt_contract():
    """A stronger extraction prompt must not reuse summaries created by v3."""
    evidence = "[REDACTED] certificate text"
    legacy_key = _stable_fingerprint(
        {
            "contract": "unstructured-analysis-v3",
            "model": get_configured_model(),
            "text": evidence,
        }
    )

    assert _unstructured_analysis_cache_key(evidence) != legacy_key

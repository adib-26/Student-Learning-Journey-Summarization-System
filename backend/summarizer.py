"""
Generate AI-powered educational summaries for student performance with multi-language translation.
"""
import re
import time
import copy
import logging
from collections import defaultdict
from typing import Any, Dict, Optional

import streamlit as st

from .secure_gemini_client import SecureGeminiClient
from .pii_protection import PIIProtector
from .audit_logging import AuditLogger  # Aligned import filename
from backend.deepl_translator import translator

logger = logging.getLogger(__name__)

# Try to import behaviour extractor from backend package; fallback to no-op.
try:
    from backend.behaviour_extractor import extract_behaviour_pairs
except Exception:
    try:
        from behaviour_extractor import extract_behaviour_pairs
    except Exception:
        def extract_behaviour_pairs(
        ) -> Dict[str, str]:
            return {}

# -------------------------
# Security components (Singleton instances or fresh attachments)
# -------------------------
secure_client = SecureGeminiClient()
pii_protector = PIIProtector()

# Multi-tenant Streamlit safety alignment
session_id = st.session_state.get("user_session_token", "SYSTEM_FALLBACK")
audit_logger = AuditLogger(custom_session_id=session_id)

# -------------------------
# Subject translations
# -------------------------
SUBJECT_TRANSLATIONS = {
    "Sejarah": "Sejarah (History)",
    "Sains": "Sains (Science)",
    "Matematik": "Matematik (Mathematics)",
    "Bahasa Malaysia": "Bahasa Malaysia",
    "Bahasa Melayu": "Bahasa Melayu (Malay Language)",
    "Pendidikan Islam": "Pendidikan Islam (Islamic Education)",
    "Pendidikan Moral": "Pendidikan Moral (Moral Education)",
}


def translate_subject_name(subject: Optional[str]) -> Optional[str]:
    """Translate Malay subject names to include English in parentheses."""
    if not subject:
        return subject
    return SUBJECT_TRANSLATIONS.get(subject, subject)


def format_student_details(details: Dict[str, Any]) -> str:
    """Convert student_details dict into readable sentence fragment."""
    if not isinstance(details, dict):
        return str(details or "")

    gender = details.get("Gender", "")
    nationality = details.get("Nationality", "")
    level = details.get("School Level", "")
    form = details.get("Form", "")
    state = details.get("State", "")

    parts = []
    if gender:
        parts.append(f"a {gender}")
    if nationality:
        parts.append(nationality)
    if level:
        parts.append(f"student in {level}")
    if form:
        parts.append(form)
    if state:
        parts.append(f"from {state}")

    return " ".join(parts).strip()


def format_behaviour_traits(
        traits: Dict[str, str],
        gender: Optional[str],
) -> Optional[str]:
    """
    Convert behaviour traits into a natural readable sentence.
    Groups traits by rating and creates a flowing narrative.
    """
    if not traits:
        return None

    grouped = defaultdict(list)
    for trait, rating in traits.items():
        if not trait or not rating:
            continue
        grouped[rating.strip().capitalize()].append(trait.strip())

    pronoun_subject = "She" if gender == "Female" else "He"
    phrases = []

    rating_order = ["Excellent", "Very Good", "Good", "Fair", "Poor", "Bad"]

    for rating in rating_order:
        if rating not in grouped:
            continue

        trait_list = grouped[rating]
        if len(trait_list) == 1:
            trait_str = trait_list[0].lower()
        elif len(trait_list) == 2:
            trait_str = f"{trait_list[0].lower()} and {trait_list[1].lower()}"
        else:
            trait_str = ", ".join([t.lower() for t in trait_list[:-1]]) + f", and {trait_list[-1].lower()}"

        if rating == "Excellent":
            phrases.append(f"{pronoun_subject} demonstrates excellent {trait_str}")
        elif rating == "Very Good":
            phrases.append(f"{pronoun_subject} shows very good {trait_str}")
        elif rating == "Good":
            phrases.append(f"{pronoun_subject} maintains good {trait_str}")
        elif rating == "Fair":
            phrases.append(f"{pronoun_subject} shows fair {trait_str}")
        elif rating == "Poor":
            phrases.append(f"{pronoun_subject} needs improvement in {trait_str}")
        else:
            phrases.append(f"{pronoun_subject} has {rating.lower()} {trait_str}")

    if not phrases:
        return None

    if len(phrases) == 1:
        return phrases[0] + "."

    return ", ".join(phrases) + "."


# -------------------------
# Subject extraction & validation
# -------------------------
_SUBJECT_SCORE_RE = re.compile(
    r"\b([A-Za-z &\-/]{2,40}?)\s*(?:[:\-]\s*|\s{2,})\s*(\d{1,3}(?:\.\d+)?)\b",
    flags=re.IGNORECASE,
)


def extract_subject_scores(text: str) -> Dict[str, float]:
    """
    Heuristic extraction of subject -> numeric score from OCR text.
    Returns subject title -> float score (0-100).
    """
    if not text:
        return {}

    results: Dict[str, float] = {}
    for match in _SUBJECT_SCORE_RE.finditer(text):
        raw_subj = match.group(1).strip()
        raw_score = match.group(2).strip()

        try:
            score = float(raw_score)
        except ValueError:
            continue

        if 0 <= score <= 100:
            # Avoid cleaning out legitimate slash indicators inside names
            subj = re.sub(r"[^\w\s\-/&']", " ", raw_subj)
            subj = re.sub(r"\s+", " ", subj).strip().title()

            if subj and len(subj) > 2:
                results[subj] = score

    return results


# -------------------------
# Draft builder (rule-based)
# -------------------------
def build_detailed_educational_insight(
        stats: Dict[str, Any],
        extracted_text: Optional[str] = None,
) -> str:
    """
    Build a deterministic, rule-based draft summary from structured stats.
    Ensures subject scores and behaviour ratings are kept separate.
    """
    # Use a shallow copy to prevent modification issues
    local_stats = copy.copy(stats)
    total = local_stats.get("row_count", 0)

    if total == 0 and extracted_text:
        total = 1

    student_name = local_stats.get("student_name", "Unknown Student")
    student_details = local_stats.get("student_details", {}) or {}
    activities = local_stats.get("activities", []) or []

    subjects = (
            local_stats.get("averages")
            or local_stats.get("subjects")
            or {}
    )

    if not subjects and extracted_text:
        subjects = extract_subject_scores(extracted_text)

    behaviour_traits = local_stats.get("behaviour") or {}
    if not behaviour_traits and extracted_text:
        behaviour_traits = extract_behaviour_pairs()

    strongest = local_stats.get("strength")
    weakest = local_stats.get("weakness")

    student_details_sentence = format_student_details(student_details)
    gender = None

    if isinstance(student_details, dict):
        gender = student_details.get("Gender")
    else:
        if "Female" in str(student_details):
            gender = "Female"
        elif "Male" in str(student_details):
            gender = "Male"

    pronoun_possessive = "her" if gender == "Female" else "his"
    pronoun_subject = "She" if gender == "Female" else "He"

    parts = [
        f"{student_name}, {student_details_sentence}, demonstrates consistent academic engagement."]

    if subjects:
        strongest_t = translate_subject_name(strongest) if strongest else None
        weakest_t = translate_subject_name(weakest) if weakest else None

        if strongest_t and weakest_t:
            parts.append(
                f"{pronoun_possessive.capitalize()} academic performance highlights strength in {strongest_t} and an area for improvement in {weakest_t}.")
        elif strongest_t:
            parts.append(
                f"{pronoun_possessive.capitalize()} academic performance highlights strength in {strongest_t}.")
        elif weakest_t:
            parts.append(
                f"{pronoun_possessive.capitalize()} academic performance shows an area for improvement in {weakest_t}.")

        if isinstance(subjects, dict) and subjects:
            subject_items = sorted(subjects.items(), key=lambda x: x[0])
            subject_scores = "; ".join([f"{translate_subject_name(sub)}: {score:.1f}" for sub, score in subject_items])
            parts.append(f"Recorded averages include {subject_scores}.")
    else:
        parts.append("No subject-wise numeric performance data was detected.")

    behaviour_summary = format_behaviour_traits(behaviour_traits, gender)
    if behaviour_summary:
        parts.append(behaviour_summary)

    if activities:
        if len(activities) == 1:
            activity_str = activities[0]
        elif len(activities) == 2:
            activity_str = f"{activities[0]} and {activities[1]}"
        else:
            activity_str = ", ".join(activities[:-1]) + f", and {activities[-1]}"

        parts.append(
            f"Beyond academics, {pronoun_subject.lower()} participates in {activity_str}, reflecting balanced personal development.")

    parts.append(f"Analysis of {total:,} record{'s' if total > 1 else ''} indicates clear performance patterns.")

    return " ".join(parts)


# -------------------------
# LLM polishing (Secure Gemini)
# -------------------------
def improve_with_llm(
        summary: str,
        stats: Dict[str, Any],
) -> str:
    """
    Improve summary using secure Gemini pipeline with PII protection,
    audit logging, and secure transport rules.
    """
    # 1. Clean data payloads using your updated PII layer beforehand
    safe_student_context = pii_protector.anonymize_student_data(stats.get("student_details", {}))
    safe_draft_summary = pii_protector.redact_pii(summary)
    
    # Debug: Print what's actually being sent to LLM
    logger.info(f"Sending to LLM - Draft summary: {safe_draft_summary}")
    print(f"DEBUG: safe_draft_summary = {safe_draft_summary}")

    # 2. Re-build explicit non-leaking engineering template instructions
    safe_prompt = (
        "Your task is to transform the following draft student performance summary into a professional, insightful, and natural-sounding narrative. "
        "You are encouraged to rephrase, restructure, and connect ideas to create a more polished and readable summary.\n\n"
        f"DRAFT NARRATIVE (USE ONLY THESE FACTS):\n{safe_draft_summary}\n\n"
        f"CONTEXT METADATA:\n- Student tracking verification: {safe_student_context.get('student_id', 'Anonymous')}\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. **Incorporate All Facts:** You MUST include every piece of information from the draft (subjects, scores, activities, traits). No data can be omitted.\n"
        "2. **Enhance Readability:** Improve grammar, flow, and sentence structure to make the summary sound natural and professional.\n"
        "3. **Add Connecting Insights:** Based ONLY on the provided data, you can add brief insights that connect different pieces of information. For example, link high scores in certain subjects to potential strengths. Do not invent new facts.\n"
        "4. **PII Protection:** Do NOT mention real identities or private data keys.\n"
        "5. **Output:** Return ONLY the single, clean, polished paragraph as your final answer."
    )

    try:
        start_time = time.time()

        audit_logger.log_data_processing(
            "ANONYMIZATION",
            1,
            pii_redacted=len(pii_protector.redacted_map),
        )

        # Main API call execution
        response = secure_client.call_gemini_secure(prompt=safe_prompt, model="gemini-3.5-flash")
        response_time = (time.time() - start_time) * 1000

        audit_logger.log_api_call(
            api_service="Google Gemini",
            endpoint="call_gemini_secure",
            request_size=len(safe_prompt),
            response_time_ms=response_time,
            status_code=200 if response else 500,
            pii_protected=True,
        )

        if response:
            audit_logger.log_summary_generation(
                input_size=len(safe_prompt),
                output_size=len(response),
                model="gemini-3.5-flash",
                tokens_used=int(len(response.split()) * 1.3),
            )
            return response.strip()

        return summary

    except Exception as e:
        audit_logger.log_error(
            error_type=type(e).__name__,
            message=str(e),
            stage="SUMMARY_GENERATION",
        )
        st.error(f"AI Polishing Error: {e}")
        return summary


# -------------------------
# Mock summary
# -------------------------
def generate_mock_summary(
        stats: Dict[str, Any],
        extracted_text: Optional[str],
) -> str:
    total = stats.get("row_count", 0)
    if total == 0 and extracted_text:
        total = 1

    return (
        f"The dataset contains {total} record{'s' if total > 1 else ''} "
        f"across {stats.get('column_count', 0)} features. "
        f"Overall learning performance appears stable."
    )


# -------------------------
# Public API
# -------------------------
def generate_summary(
        stats: Dict[str, Any],
        extracted_text: Optional[str] = None,
        mode: str = "insight",
        use_llm: bool = True,
        language: str = "en"
) -> Optional[str]:
    """
    Public API for generating student insight summaries with translation hooks.
    """

    if mode == "insight":
        draft = build_detailed_educational_insight(stats, extracted_text)
        if use_llm:
            summary_text = improve_with_llm(draft, stats)
        else:
            summary_text = draft

    elif mode == "mock":
        summary_text = generate_mock_summary(stats, extracted_text)

    else:
        raise ValueError(f"Invalid summary mode: {mode}")

    if not summary_text:
        return None

    # Handle multi-language routing with RobustTranslator fallback checks
    if language != "en":
        translated = translator.translate_text(summary_text, language)

        if translated == summary_text:
            logger.warning("Translation returned identical text. Engine fallback or quota limit hit.")
            audit_logger.log_error(
                error_type="TRANSLATION_FALLBACK",
                message="Translation returned original English text. Checking API keys/quota.",
                stage="SUMMARIZATION"
            )

        summary_text = translated

    return summary_text


__all__ = ["generate_summary"]
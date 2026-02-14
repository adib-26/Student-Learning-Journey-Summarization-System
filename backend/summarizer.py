# backend/summarizer.py
"""
Generate AI-powered educational summaries for student performance.
Handles OCR and structured data, separates numeric subject scores from qualitative
behaviour ratings, and optionally polishes the draft with an LLM.
"""
import streamlit as st
import os
# from google import genai  # Commented out for now
import re
from typing import Dict, Optional, Any
from collections import defaultdict

# Try to import behaviour extractor from backend package; fallback to no-op.
try:
    from backend.behaviour_extractor import extract_behaviour_pairs
except Exception:
    try:
        from behaviour_extractor import extract_behaviour_pairs
    except Exception:
        def extract_behaviour_pairs(text: Optional[str] = None, df: Optional[object] = None) -> Dict[str, str]:
            return {}

# Subject name translations (Malay -> English with Malay in parentheses)
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


def format_behaviour_traits(traits: Dict[str, str], gender: Optional[str]) -> Optional[str]:
    """
    Convert behaviour traits into a natural readable sentence.
    Groups traits by rating and creates a flowing narrative.
    """
    if not traits:
        return None

    grouped = defaultdict(list)
    for trait, rating in traits.items():
        if not trait:
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
    r"([A-Za-z &\-/]{2,60}?)\s*(?:[:\-]\s*|\s+)\s*(\d{1,3}(?:\.\d+)?)\s*(?:/\s*\d{1,3})?",
    flags=re.IGNORECASE,
)


def extract_subject_scores(text: str) -> Dict[str, float]:
    """
    Heuristic extraction of subject -> numeric score from OCR text.
    Returns subject title -> float score (0-100). Ignores values outside 0-100.
    """
    if not text:
        return {}
    results: Dict[str, float] = {}
    for match in _SUBJECT_SCORE_RE.finditer(text):
        raw_subj = match.group(1).strip()
        raw_score = match.group(2).strip()
        try:
            score = float(raw_score)
        except Exception:
            continue
        if 0 <= score <= 100:
            subj = re.sub(r"[^\w\s\-/&']", " ", raw_subj)
            subj = re.sub(r"\s+", " ", subj).strip().title()
            if subj:
                results[subj] = score
    return results


# -------------------------
# Draft builder (rule-based)
# -------------------------
def build_detailed_educational_insight(
        stats: Dict[str, Any],
        extracted_text: Optional[str] = None
) -> str:
    """
    Build a deterministic, rule-based draft summary from structured stats.
    Ensures subject scores and behaviour ratings are kept separate.
    """
    total = stats.get("row_count", 0)
    if total == 0 and extracted_text:
        total = 1

    student_name = stats.get("student_name", "Unknown Student")
    student_details = stats.get("student_details", {}) or {}
    activities = stats.get("activities", []) or []

    # Subjects: prefer structured 'averages' or 'subjects', otherwise extract from text
    subjects = stats.get("averages") or stats.get("subjects") or {}
    if not subjects and extracted_text:
        subjects = extract_subject_scores(extracted_text)
        stats["averages"] = subjects

    # Behaviour: prefer structured 'behaviour', otherwise extract from text
    behaviour_traits = stats.get("behaviour") or {}
    if not behaviour_traits and extracted_text:
        behaviour_traits = extract_behaviour_pairs(extracted_text)
        stats["behaviour"] = behaviour_traits

    strongest = stats.get("strength")
    weakest = stats.get("weakness")

    student_details_sentence = format_student_details(student_details)

    # Gender-aware pronouns
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

    parts = []
    parts.append(f"{student_name}, {student_details_sentence}, demonstrates consistent academic engagement.")

    # Academic performance paragraph
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

        # Add subject scores list (stable ordering)
        if isinstance(subjects, dict) and subjects:
            subject_items = sorted(subjects.items(), key=lambda x: x[0])
            subject_scores = "; ".join([f"{translate_subject_name(sub)}: {score:.1f}" for sub, score in subject_items])
            parts.append(f"Recorded averages include {subject_scores}.")
    else:
        parts.append("No subject-wise numeric performance data was detected.")

    # Behaviour traits (qualitative only)
    behaviour_summary = format_behaviour_traits(behaviour_traits, gender)
    if behaviour_summary:
        parts.append(behaviour_summary)

    # Activities
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
# LLM polishing (Google Gemini)
# -------------------------

def improve_with_llm(summary: str) -> str:
    prompt = (
        "Rewrite the following student performance summary so it is grammatically correct and natural. "
        "Keep subject scores and behaviour ratings distinct. "
        f"DRAFT: {summary}\n\n"
        "Return only the polished paragraph."
    )

    try:
        from google import genai
        import streamlit as st
        api_key = st.secrets["GEMINI_API_KEY"]
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )

        return response.text.strip() if response.text else summary

    except Exception as e:
        st.error(f"AI Polishing Error: {e}")
        return summary


# -------------------------
# Mock summary
# -------------------------
def generate_mock_summary(stats: Dict[str, Any], extracted_text: Optional[str]) -> str:
    total = stats.get("row_count", 0)
    if total == 0 and extracted_text:
        total = 1
    return f"The dataset contains {total} record{'s' if total > 1 else ''} across {stats.get('column_count', 0)} features. Overall learning performance appears stable."


# -------------------------
# Public API
# -------------------------
def generate_summary(
        stats: Dict[str, Any],
        extracted_text: Optional[str] = None,
        mode: str = "insight",
        use_llm: bool = True
) -> str:
    """
    Public API for generating summaries.
    """
    if mode == "insight":
        draft = build_detailed_educational_insight(stats, extracted_text)
        if use_llm:
            return improve_with_llm(draft)
        return draft

    if mode == "mock":
        return generate_mock_summary(stats, extracted_text)

    raise ValueError(f"Invalid summary mode: {mode}")


__all__ = ["generate_summary"]

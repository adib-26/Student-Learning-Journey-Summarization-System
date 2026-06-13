import os
import re
from typing import Optional, List, Tuple
import streamlit as st
from google import genai

# Try to import the document converter
try:
    from docling.document_converter import DocumentConverter
except Exception:
    DocumentConverter = None


# --- RULE-BASED EXTRACTION LOGIC ---

def _collapse_spaced_letters(text: str) -> str:
    return re.sub(r'\b(?:[A-Za-z]\s){2,}[A-Za-z]\b', lambda m: m.group(0).replace(" ", ""), text)


def _normalize_text(text: str) -> str:
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = _collapse_spaced_letters(text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    return text.strip()


def _score_name(name: str, context: str) -> int:
    score = 0
    words = name.split()
    if 2 <= len(words) <= 3:
        score += 3

    bad_words = {
        "certificate", "completion", "course", "award",
        "honored", "with", "has", "been", "tech", "academy",
        "association", "university", "institute", "webinar"
    }
    if any(w.lower() in bad_words for w in words):
        score -= 10

    if re.search(r'(certifies|certify|presented to|awarded to|honored with)', context, re.IGNORECASE):
        score += 5
    return score


def _find_candidates(text: str) -> List[Tuple[str, int]]:
    candidates = []

    # 1. Name BEFORE phrase
    before_patterns = [
        r'([A-Z][A-Za-z\'`.-]+(?:\s+[A-Z][A-Za-z\'`.-]+){1,3})\s+has been honored with',
        r'([A-Z][A-Za-z\'`.-]+(?:\s+[A-Z][A-Za-z\'`.-]+){1,3})\s+has been awarded',
        r'([A-Z][A-Za-z\'`.-]+(?:\s+[A-Z][A-Za-z\'`.-]+){1,3})\s+has successfully completed',
    ]
    for pattern in before_patterns:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            name = m.group(1).strip()
            candidates.append((name, _score_name(name, m.group(0)) + 10))

    # 2. Phrase AFTER
    phrases = [
        r'this certifies that',
        r'this to certify that',
        r'presented to',
        r'awarded to',
        r'certificate is presented to',
    ]
    for phrase in phrases:
        pattern = re.compile(
            rf'{phrase}\s*(?:[:\-–—]?\s*)?\n?\s*([A-Z][A-Za-z\'`.-]+(?:\s+[A-Z][A-Za-z\'`.-]+){{0,3}})',
            re.IGNORECASE
        )
        for m in pattern.finditer(text):
            name = m.group(1).strip()
            candidates.append((name, _score_name(name, m.group(0))))

    # 3. ALL CAPS fallback
    for m in re.finditer(r'\b([A-Z]{2,}(?:\s+[A-Z]{2,}){1,2})\b', text):
        name = m.group(1).title()
        candidates.append((name, _score_name(name, m.group(0))))

    return candidates


def _extract_best_name(text: str) -> Optional[str]:
    text = _normalize_text(text)
    candidates = _find_candidates(text)
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


# --- GEMINI VALIDATION LOGIC ---

def _validate_name_with_gemini(extracted_name: str, full_text: str) -> str:
    """Uses Gemini to confirm if the extracted name is actually the student."""
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        client = genai.Client(api_key=api_key)

        prompt = f"""
        I am extracting the recipient's name from a certificate.
        Rule-based logic extracted: "{extracted_name}"

        Full Document Text:
        {full_text}

        Task: 
        1. Check if "{extracted_name}" is truly the person who received the certificate.
        2. If it is wrong (e.g., it is a company name, a teacher, or a coordinator), find the correct recipient name in the text.
        3. Return ONLY the full name of the person. No extra words.
        """

        response = client.models.generate_content(
            model="gemini-3.5-flash",  # or your preferred version
            contents=prompt,
        )
        return response.text.strip()
    except Exception:
        # If Gemini fails, return the original extracted name as fallback
        return extracted_name


# --- MAIN ENTRY POINT ---

def extract_student_name(file_path_or_text: str) -> str:
    try:
        if isinstance(file_path_or_text, str) and os.path.exists(file_path_or_text):
            if DocumentConverter is None:
                return "Unknown Student"
            converter = DocumentConverter()
            result = converter.convert(file_path_or_text)
            text = result.document.export_to_markdown()
        else:
            text = file_path_or_text

        if isinstance(text, (bytes, bytearray)):
            text = text.decode('utf-8', errors='ignore')

        # Step 1: Rule-based extraction (Fast)
        initial_name = _extract_best_name(text)

        if not initial_name or initial_name == "Student name not found":
            # Step 2: If rules fail, ask Gemini to find it from scratch
            return _validate_name_with_gemini("Unknown", text)

        # Step 3: Double-check with Gemini (Accurate)
        final_name = _validate_name_with_gemini(initial_name, text)
        return final_name

    except Exception:
        return "Unknown Student"
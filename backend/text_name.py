import os
import re
from typing import Optional

# Optional import of Docling; only used when a real file path is provided.
try:
    from docling.document_converter import DocumentConverter
except Exception:
    DocumentConverter = None


def _collapse_spaced_letters(text: str) -> str:
    """
    Fix OCR outputs where letters are spaced like "H e l e n e".
    Collapses runs of single letters separated by spaces into a single word.
    """
    def _collapse(match):
        return match.group(0).replace(" ", "")

    # Match runs like "H e l e n e" (at least 3 single-letter + spaces)
    text = re.sub(r'(?:\b(?:[A-Za-z]\s){2,}[A-Za-z]\b)', _collapse, text)
    return text


def _normalize_text(text: str) -> str:
    """
    Normalize whitespace and punctuation for more reliable regex matching.
    """
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = _collapse_spaced_letters(text)
    # Replace multiple spaces with single space, but keep newlines
    text = re.sub(r'[ \t]{2,}', ' ', text)
    # Normalize repeated newlines
    text = re.sub(r'\n{2,}', '\n\n', text)
    return text.strip()


def _extract_from_text_blob(text: str) -> Optional[str]:
    """
    Core heuristics to find a person name inside a text blob.
    Returns the name string or None.
    """
    if not text:
        return None

    t = _normalize_text(text)

    # Common certificate phrases to look for (case-insensitive)
    phrases = [
        r'this certifies that',
        r'this certificate is proudly presented to',
        r'presented to',
        r'this to certify that',
        r'this certificate is presented to',
        r'this certificate of completion is presented to',
        r'this certificate is awarded to',
        r'this certifies that the',
        r'this certifies that'
    ]

    # Try phrase-based extraction: prefer name on the next non-empty line (block).
    for phrase in phrases:
        # Block pattern: phrase on its own line, name on next non-empty line
        block_pattern = re.compile(
            rf'{phrase}\s*(?:[:\-–—]?\s*)?\n\s*([A-Z][A-Za-z\'`.-]+(?:\s+[A-Z][A-Za-z\'`.-]+){{0,4}})\s*(?=\n|$|[.,;:!?])',
            re.IGNORECASE
        )
        m_block = block_pattern.search(t)
        if m_block:
            return m_block.group(1).strip()

    # If block patterns failed, try inline patterns but ensure we stop before verbs like "has", "completed", etc.
    for phrase in phrases:
        inline_pattern = re.compile(
            rf'{phrase}\s*(?:[:\-–—]?\s*)?([A-Z][A-Za-z\'`.-]+(?:\s+[A-Z][A-Za-z\'`.-]+){{0,4}})(?=\s*(?:has\b|was\b|completed\b|successfully\b|,|\n|$|\.))',
            re.IGNORECASE
        )
        m_inline = inline_pattern.search(t)
        if m_inline:
            return m_inline.group(1).strip()

    # Fallback 1: find all-caps names (e.g., HELENE PAQUET) but avoid long all-caps lines that are not names
    all_caps_pattern = re.compile(r'\b([A-Z]{2,}(?:\s+[A-Z]{2,}){0,4})\b')
    for m in all_caps_pattern.finditer(t):
        candidate = m.group(1).strip()
        # Heuristic: treat as name if it contains 1-3 words and not too long
        if 1 <= len(candidate.split()) <= 3 and len(candidate) < 40:
            parts = []
            for w in candidate.split():
                if len(w) <= 2 and w.isupper():
                    parts.append(w)  # keep initials
                else:
                    parts.append(w.capitalize())
            return " ".join(parts)

    # Fallback 2: first two- or three-word Title Case name (e.g., Helene Paquet)
    title_case_pattern = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b')
    m_title = title_case_pattern.search(t)
    if m_title:
        return m_title.group(1).strip()

    return None


def extract_student_name(file_path_or_text: str) -> str:
    """
    Extract a student's name from either:
      - a file path (PDF/image) -> Docling conversion used if available
      - a raw text blob (string) -> direct text heuristics

    The function auto-detects whether the input is a path to an existing file.
    Returns a string with the detected name or "Unknown Student" / "Student name not found".
    """
    try:
        # If input points to an existing file, try to use Docling to convert it.
        if isinstance(file_path_or_text, str) and os.path.exists(file_path_or_text):
            if DocumentConverter is None:
                # Docling not installed or failed to import
                return "Unknown Student"

            try:
                converter = DocumentConverter()
                result = converter.convert(file_path_or_text)
                markdown_text = result.document.export_to_markdown()
                name = _extract_from_text_blob(markdown_text)
                return name or "Student name not found"
            except Exception:
                return "Unknown Student"

        # Otherwise treat the input as raw text
        if isinstance(file_path_or_text, str):
            name = _extract_from_text_blob(file_path_or_text)
            return name or "Student name not found"

        # If input is bytes or file-like, try to decode
        if isinstance(file_path_or_text, (bytes, bytearray)):
            try:
                text = file_path_or_text.decode('utf-8', errors='ignore')
                name = _extract_from_text_blob(text)
                return name or "Student name not found"
            except Exception:
                return "Unknown Student"

        return "Unknown Student"

    except Exception:
        return "Unknown Student"

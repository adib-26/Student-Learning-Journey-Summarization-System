# backend/behaviour_extractor.py
"""
Robust extractor for behaviour attribute -> rating pairs from OCR text and DataFrame.

Fixes included:
- Isolates behaviour fragments when OCR output mixes table columns with '|' separators.
- Uses a strict regex to capture up to 5-word attribute phrases immediately before a rating.
- Provides a fallback heuristic that finds rating tokens and selects the nearest preceding words,
  skipping numeric tokens and score-like fragments.
- Normalizes common OCR variants (e.g., G00d -> Good).
- Works with either a pandas DataFrame (structured) or raw OCR text (unstructured).
"""

import re
from typing import Dict, Optional, List

try:
    import pandas as pd
except Exception:
    pd = None  # pandas optional; dataframe extraction will be skipped if not available

# Canonical ratings we want to return
_CANONICAL_RATINGS = {"Excellent", "Good", "Fair", "Poor", "Bad", "Very Good"}

# Normalization map for OCR variants and synonyms
_RATING_NORMALIZATION = {
    "g00d": "Good", "g0od": "Good", "go0d": "Good",
    "0k": "Fair", "very good": "Very Good", "verygood": "Very Good",
    "excellent": "Excellent", "good": "Good", "fair": "Fair",
    "average": "Fair", "avg": "Fair", "ok": "Fair", "okay": "Fair",
    "poor": "Poor", "p00r": "Poor", "bad": "Bad", "b4d": "Bad", "b@d": "Bad",
    "satisfactory": "Good", "unsatisfactory": "Poor",
}


def _normalize_rating_token(token: str) -> Optional[str]:
    if not token:
        return None
    t = token.strip().lower()
    # direct canonical match
    for r in _CANONICAL_RATINGS:
        if t == r.lower():
            return r
    # normalization map
    if t in _RATING_NORMALIZATION:
        return _RATING_NORMALIZATION[t]
    # try simple OCR fixes
    t_fixed = t.replace("0", "o").replace("1", "l").replace("5", "s").replace("@", "a").replace("4", "a").replace("$", "s")
    if t_fixed in _RATING_NORMALIZATION:
        return _RATING_NORMALIZATION[t_fixed]
    for r in _CANONICAL_RATINGS:
        if t_fixed == r.lower():
            return r
    # substring fallback
    for r in _CANONICAL_RATINGS:
        if r.lower() in t_fixed:
            return r
    return None


def _clean_attribute(attr: str) -> str:
    cleaned = re.sub(r"[^\w\s\-/&']", " ", attr)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.title()


# Build rating alternation for regex
_rating_tokens = sorted(
    set(list(_RATING_NORMALIZATION.keys()) + [r.lower() for r in _CANONICAL_RATINGS]),
    key=lambda s: -len(s)
)
_RATING_ALTERNATION = r"|".join(re.escape(tok) for tok in _rating_tokens)

# Strict pattern: capture 1-5 word attribute immediately before rating token
_STRICT_PATTERN = re.compile(
    rf"""
    (?P<attr>
        (?:[A-Za-z][A-Za-z'&\-/]{{0,20}})
        (?:\s+[A-Za-z][A-Za-z'&\-/]{{0,20}}){{0,4}}
    )
    \s*(?:[:\-\–\—]\s*|\s+)
    (?P<rating>\b(?:{_RATING_ALTERNATION})\b)
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

# Token regexes for fallback
_RATING_TOKEN_RE = re.compile(rf"\b(?:{_RATING_ALTERNATION})\b", flags=re.IGNORECASE)
_WORD_RE = re.compile(r"[A-Za-z'&\-/]{1,30}", flags=re.IGNORECASE)


def _fallback_extract_pairs(text: str) -> Dict[str, str]:
    results: Dict[str, str] = {}
    # Tokenize words with spans
    words = [(m.group(0), m.start(), m.end()) for m in _WORD_RE.finditer(text)]
    for rmatch in _RATING_TOKEN_RE.finditer(text):
        rating_raw = rmatch.group(0)
        rating = _normalize_rating_token(rating_raw)
        if not rating:
            continue
        rating_start = rmatch.start()
        # find last word index ending before rating_start
        idx = None
        for i, (_, s, e) in enumerate(words):
            if e <= rating_start:
                idx = i
            else:
                break
        if idx is None:
            continue
        # collect up to 5 preceding words, skipping numeric/score tokens
        attr_tokens: List[str] = []
        i = idx
        while i >= 0 and len(attr_tokens) < 5:
            token = words[i][0]
            if re.search(r"\d|\/", token):
                i -= 1
                continue
            if len(token) == 1 and not re.match(r"[A-Za-z]", token):
                i -= 1
                continue
            attr_tokens.insert(0, token)
            i -= 1
        if not attr_tokens:
            continue
        attr = " ".join(attr_tokens)
        attr = _clean_attribute(attr)
        if attr:
            results[attr] = rating
    return results


def extract_behaviour_from_text(text: str) -> Dict[str, str]:
    """
    Extract behaviour attribute -> rating pairs from raw OCR text.

    Strategy:
    - Pre-clean lines by splitting on '|' and keeping the left side (common in table OCR).
    - Run a strict regex to capture attribute phrases immediately before rating tokens.
    - If strict pass finds nothing, run a fallback heuristic that finds rating tokens and picks nearest preceding words.
    """
    if not text:
        return {}

    # Normalize line endings
    normalized = text.replace("\r", "\n")

    # Pre-clean: for each line, if '|' present, keep only the left-most segment (behaviour often on left)
    lines = []
    for line in normalized.splitlines():
        if not line.strip():
            continue
        if "|" in line:
            left = line.split("|")[0].strip()
            lines.append(left)
        else:
            lines.append(line.strip())
    cleaned_text = "\n".join(lines)

    results: Dict[str, str] = {}

    # Strict pass
    for m in _STRICT_PATTERN.finditer(cleaned_text):
        raw_attr = m.group("attr").strip()
        raw_rating = m.group("rating").strip()
        rating = _normalize_rating_token(raw_rating)
        if not rating:
            continue
        attr = _clean_attribute(raw_attr)
        if attr:
            results[attr] = rating

    # Fallback if nothing found or to capture additional items
    if not results:
        results = _fallback_extract_pairs(cleaned_text)

    # Final cleanup: drop attributes that contain digits or are excessively long
    final: Dict[str, str] = {}
    for attr, rating in results.items():
        if re.search(r"\d", attr):
            continue
        if len(attr) > 60:
            continue
        final[attr] = rating

    return final


def extract_behaviour_from_dataframe(df) -> Dict[str, str]:
    """
    Extract behaviour pairs from a structured DataFrame.
    Expected columns: 'Section', 'Label', 'Value' (case-insensitive).
    """
    if df is None:
        return {}
    # If pandas not available or df not a DataFrame, return empty
    if pd is None:
        return {}
    if not hasattr(df, "columns") or df.empty:
        return {}

    results: Dict[str, str] = {}
    cols = [c.lower() for c in df.columns]
    # Try to find rows where Section contains 'behaviour' (case-insensitive)
    if "section" in cols:
        section_col = df.columns[cols.index("section")]
        label_col = None
        value_col = None
        if "label" in cols:
            label_col = df.columns[cols.index("label")]
        if "value" in cols:
            value_col = df.columns[cols.index("value")]
        behaviour_rows = df[df[section_col].astype(str).str.contains("behaviour", case=False, na=False)]
        for _, row in behaviour_rows.iterrows():
            label = str(row.get(label_col, "")).strip() if label_col else ""
            value = str(row.get(value_col, "")).strip() if value_col else ""
            if label and value and value.lower() != "nan":
                rating = _normalize_rating_token(value)
                if rating:
                    attr = _clean_attribute(label)
                    if attr:
                        results[attr] = rating
    return results


def extract_behaviour_pairs(df: Optional["pd.DataFrame"] = None, text: Optional[str] = None) -> Dict[str, str]:
    """
    Public entry point. Prefer DataFrame extraction if df provided; otherwise use text.
    """
    results: Dict[str, str] = {}
    if df is not None and pd is not None:
        try:
            results = extract_behaviour_from_dataframe(df)
        except Exception:
            results = {}
    if not results and text:
        try:
            results = extract_behaviour_from_text(text)
        except Exception:
            results = {}
    return results


def group_traits_by_rating(traits: Dict[str, str]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for attr, rating in traits.items():
        grouped.setdefault(rating, []).append(attr)
    return grouped


__all__ = ["extract_behaviour_pairs", "group_traits_by_rating", "extract_behaviour_from_text", "extract_behaviour_from_dataframe"]

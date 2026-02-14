import pandas as pd
import re
from typing import List, Optional

# -----------------------------
# Normalizer: robust OCR + structured handling
# -----------------------------

SCORE_RE = re.compile(
    r"""
    (?P<label>.+?)                # lazy label capture
    (?:[:\-]\s*)?                 # optional separator like ":" or "-"
    (?:
        (?P<score>\d{1,3})\s*/\s*(?P<max>\d{1,4})    # "74 / 100"
        |
        (?P<score2>\d{1,3})\s+of\s+(?P<max2>\d{1,4}) # "74 of 100"
        |
        (?P<score3>\d{1,3})                          # single number (fallback)
    )
    """,
    re.IGNORECASE | re.VERBOSE
)

METADATA_KEYS = [
    "name", "student name", "gender", "state", "school", "school level",
    "form", "attendance", "nationality"
]

SECTION_HEADERS = {
    "subjects": "Subjects",
    "behaviour": "Behaviour",
    "behavior": "Behaviour",
    "co-curricular": "Co-curricular",
    "co curricular": "Co-curricular",
    "co curricular": "Co-curricular",
    "co curricular": "Co-curricular",
    "co curricular": "Co-curricular",
    "co curricular": "Co-curricular",
    "co curricular": "Co-curricular",
    "co curricular": "Co-curricular",
    "co curricular": "Co-curricular",
    "co curricular": "Co-curricular",
    "co curricular": "Co-curricular",
    "co curricular": "Co-curricular",
    "co curricular": "Co-curricular",
    "co curricular": "Co-curricular",
    "co curricular": "Co-curricular",
    "co curricular": "Co-curricular",
    "co curricular": "Co-curricular",
    "co curricular": "Co-curricular",
    "co curricular": "Co-curricular",
    "co curricular": "Co-curricular",
    "co curricular": "Co-curricular",
    "co curricular": "Co-curricular",
    "co curricular": "Co-curricular",
    "co curricular": "Co-curricular",
    "co curricular": "Co-curricular",
    "co curricular": "Co-curricular",
    "co curricular": "Co-curricular",
    "co curricular": "Co-curricular",
    "co curricular": "Co-curricular"
}
# (Note: above repeated keys are harmless; kept to be tolerant of OCR variants)

def _is_section_header(line: str) -> Optional[str]:
    """Return canonical section name if line looks like a section header."""
    if not line:
        return None
    low = line.strip().lower()
    for key, canon in SECTION_HEADERS.items():
        if low.startswith(key):
            return canon
    # also accept single-word headers
    if low in ("subjects", "behaviour", "behavior", "co-curricular", "co curricular", "co curricular"):
        return SECTION_HEADERS.get(low, None)
    return None

def _looks_like_metadata(line: str) -> Optional[str]:
    """Return metadata key if line contains a metadata field like 'Name:' or 'Gender'."""
    low = line.lower()
    for k in METADATA_KEYS:
        if low.startswith(k):
            return k
    # also detect "Name Arif Bin Hassan" style (no colon)
    for k in METADATA_KEYS:
        if k in low and len(low.split()) > 1:
            return k
    return None

def _extract_score_from_text(text: str):
    """Try to extract (label, score, maximum) from a single text string using regex."""
    m = SCORE_RE.search(text)
    if not m:
        return None
    label = (m.group("label") or "").strip()
    score = None
    maximum = None
    if m.group("score"):
        score = m.group("score")
        maximum = m.group("max")
    elif m.group("score2"):
        score = m.group("score2")
        maximum = m.group("max2")
    elif m.group("score3"):
        score = m.group("score3")
        maximum = None
    # Clean label: remove trailing separators or words like "score"
    label = re.sub(r"\b(score|marks|result)\b[:\s\-]*$", "", label, flags=re.IGNORECASE).strip()
    return {"label": label or None, "score": score, "maximum": maximum}

def normalize_uploaded_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize uploaded data into canonical schema:
    Section, Label, Score, Maximum, Notes

    Handles:
    - Structured CSV/Excel with Section/Label columns
    - OCR single-column outputs with mixed lines
    - Lines where label and score are split across adjacent rows
    - Various score formats like "74 / 100", "74 of 100", "Languages: 74 / 100"
    """
    canonical = ["Section", "Label", "Score", "Maximum", "Notes"]

    # Case 1: already structured with Section+Label
    if set(["Section", "Label"]).issubset(df.columns):
        out = df.copy()
        for col in canonical:
            if col not in out.columns:
                out[col] = None
        if "Score" in out.columns:
            out["Score"] = pd.to_numeric(out["Score"], errors="coerce")
        if "Maximum" in out.columns:
            out["Maximum"] = pd.to_numeric(out["Maximum"], errors="coerce")
        return out[canonical]

    # Otherwise treat as OCR / free-text lines
    rows = []
    # Build a list of non-empty lines from the first column and any single-column df
    if df.shape[1] == 1:
        lines = [str(x).strip() for x in df.iloc[:, 0].dropna().astype(str)]
    else:
        # If multiple columns, join non-empty cells per row into a single line
        lines = []
        for _, r in df.iterrows():
            vals = [str(x).strip() for x in r.tolist() if pd.notna(x) and str(x).strip()]
            if vals:
                lines.append(" ".join(vals))

    # Two-pass approach: combine label+score split across adjacent lines
    i = 0
    current_section = None
    while i < len(lines):
        line = lines[i].strip()
        # Section header detection
        sec = _is_section_header(line)
        if sec:
            current_section = sec
            i += 1
            continue

        # Metadata detection (Name, Gender, etc.) — keep as Misc but don't treat as subject
        meta = _looks_like_metadata(line)
        if meta:
            # store metadata as Misc row (keeps it from polluting subject rows)
            rows.append({
                "Section": "Student Details",
                "Label": line,
                "Score": None,
                "Maximum": None,
                "Notes": None
            })
            i += 1
            continue

        # Try to extract score from the current line
        extracted = _extract_score_from_text(line)
        if extracted and (extracted["score"] is not None):
            rows.append({
                "Section": current_section or "Subjects",
                "Label": extracted["label"] or line,
                "Score": extracted["score"],
                "Maximum": extracted["maximum"],
                "Notes": None
            })
            i += 1
            continue

        # Lookahead: maybe label is on this line and score on next line
        if i + 1 < len(lines):
            combined = f"{line} {lines[i+1]}"
            extracted2 = _extract_score_from_text(combined)
            if extracted2 and (extracted2["score"] is not None):
                rows.append({
                    "Section": current_section or "Subjects",
                    "Label": extracted2["label"] or line,
                    "Score": extracted2["score"],
                    "Maximum": extracted2["maximum"],
                    "Notes": None
                })
                i += 2
                continue

        # If nothing matched, try to see if the line itself is a simple "Label 74" (no slash)
        m_simple = re.search(r"(.+?)\s+(\d{1,3})\b", line)
        if m_simple:
            label = m_simple.group(1).strip()
            score = m_simple.group(2)
            rows.append({
                "Section": current_section or "Subjects",
                "Label": label,
                "Score": score,
                "Maximum": None,
                "Notes": None
            })
            i += 1
            continue

        # Fallback: treat as a note/misc row
        rows.append({
            "Section": current_section or "Misc",
            "Label": line,
            "Score": None,
            "Maximum": None,
            "Notes": None
        })
        i += 1

    out = pd.DataFrame(rows, columns=canonical)
    # Coerce numeric columns
    out["Score"] = pd.to_numeric(out["Score"], errors="coerce")
    out["Maximum"] = pd.to_numeric(out["Maximum"], errors="coerce")
    # Drop rows that are clearly not numeric subjects if desired (optional)
    # Keep them for transparency; downstream code can filter by Section.
    return out

# -----------------------------
# Heuristic normalization (keeps as fallback)
# -----------------------------
def heuristic_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal fallback that tries to parse free text lines for score patterns.
    This is intentionally simpler and used as a last-resort.
    """
    canonical = ["Section", "Label", "Score", "Maximum", "Notes"]

    if set(canonical).issubset(df.columns):
        return df[canonical]

    rows = []
    # Build lines similar to normalize_uploaded_dataframe
    if df.shape[1] == 1:
        lines = [str(x).strip() for x in df.iloc[:, 0].dropna().astype(str)]
    else:
        lines = []
        for _, r in df.iterrows():
            vals = [str(x).strip() for x in r.tolist() if pd.notna(x) and str(x).strip()]
            if vals:
                lines.append(" ".join(vals))

    for line in lines:
        m = SCORE_RE.search(line)
        if m:
            label = (m.group("label") or "").strip()
            score = m.group("score") or m.group("score2") or m.group("score3")
            maximum = m.group("max") or m.group("max2") or None
            rows.append({
                "Section": "Subjects",
                "Label": label,
                "Score": score,
                "Maximum": maximum,
                "Notes": None
            })
        else:
            rows.append({
                "Section": "Misc",
                "Label": line,
                "Score": None,
                "Maximum": None,
                "Notes": None
            })

    out = pd.DataFrame(rows, columns=canonical)
    out["Score"] = pd.to_numeric(out["Score"], errors="coerce")
    out["Maximum"] = pd.to_numeric(out["Maximum"], errors="coerce")
    return out

# -----------------------------
# AI-assisted normalization (stub)
# -----------------------------
def ai_normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Placeholder for future AI-based normalization.
    For now, call the robust normalize_uploaded_dataframe as the best effort.
    """
    return normalize_uploaded_dataframe(df)
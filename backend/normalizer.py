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
    "co curricular": "Co-curricular"
}


def _is_section_header(line: str) -> Optional[str]:
    if not line:
        return None
    low = line.strip().lower()
    for key, canon in SECTION_HEADERS.items():
        if low.startswith(key):
            return canon
    if low in SECTION_HEADERS:
        return SECTION_HEADERS.get(low, None)
    return None


def _looks_like_metadata(line: str) -> Optional[str]:
    low = line.lower()
    for k in METADATA_KEYS:
        if low.startswith(k):
            return k
    for k in METADATA_KEYS:
        if k in low and len(low.split()) > 1:
            return k
    return None


def _extract_score_from_text(text: str):
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
    label = re.sub(r"\b(score|marks|result)\b[:\s\-]*$", "", label, flags=re.IGNORECASE).strip()
    return {"label": label or None, "score": score, "maximum": maximum}


def normalize_uploaded_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize uploaded data into canonical schema horizontally using the Unique Column Rule:
    Section, Label, Score, Maximum, Notes, Label 2, Score 2, Maximum 2, Notes 2...
    """
    label_cols = [c for c in df.columns if "label" in c.lower()]
    score_cols = [c for c in df.columns if "score" in c.lower()]
    max_cols = [c for c in df.columns if "max" in c.lower()]
    notes_cols = [c for c in df.columns if "remarks" in c.lower() or "grade" in c.lower()]

    # Case 1: structured with Section + multiple Label/Score columns
    if "Section" in df.columns and label_cols and score_cols:
        rows = []
        for _, row in df.iterrows():
            dict_row = {"Section": row.get("Section", None)}

            for idx, (lcol, scol) in enumerate(zip(label_cols, score_cols)):
                suffix = f" {idx + 1}" if idx > 0 else ""

                dict_row[f"Label{suffix}"] = row.get(lcol, None)
                dict_row[f"Score{suffix}"] = pd.to_numeric(row.get(scol, None), errors="coerce")

                if idx < len(max_cols):
                    dict_row[f"Maximum{suffix}"] = pd.to_numeric(row.get(max_cols[idx], None), errors="coerce")
                else:
                    dict_row[f"Maximum{suffix}"] = pd.to_numeric(row.get(max_cols[0], None),
                                                                 errors="coerce") if max_cols else None

                if idx < len(notes_cols):
                    dict_row[f"Notes{suffix}"] = row.get(notes_cols[idx], None)
                else:
                    dict_row[f"Notes{suffix}"] = row.get(notes_cols[0], None) if notes_cols else None

            rows.append(dict_row)

        canonical_cols = ["Section"]
        for idx in range(len(label_cols)):
            suffix = f" {idx + 1}" if idx > 0 else ""
            canonical_cols.extend([f"Label{suffix}", f"Score{suffix}", f"Maximum{suffix}", f"Notes{suffix}"])

        return pd.DataFrame(rows, columns=canonical_cols)

    # Case 2: Otherwise treat as OCR / free-text lines
    canonical = ["Section", "Label", "Score", "Maximum", "Notes"]
    rows = []
    if df.shape[1] == 1:
        lines = [str(x).strip() for x in df.iloc[:, 0].dropna().astype(str)]
    else:
        lines = []
        for _, r in df.iterrows():
            vals = [str(x).strip() for x in r.tolist() if pd.notna(x) and str(x).strip()]
            if vals:
                lines.append(" ".join(vals))

    i = 0
    current_section = None
    while i < len(lines):
        line = lines[i].strip()
        sec = _is_section_header(line)
        if sec:
            current_section = sec
            i += 1
            continue

        meta = _looks_like_metadata(line)
        if meta:
            rows.append({"Section": "Student Details", "Label": line, "Score": None, "Maximum": None, "Notes": None})
            i += 1
            continue

        extracted = _extract_score_from_text(line)
        if extracted and (extracted["score"] is not None):
            rows.append({"Section": current_section or "Subjects", "Label": extracted["label"] or line,
                         "Score": extracted["score"], "Maximum": extracted["maximum"], "Notes": None})
            i += 1
            continue

        if i + 1 < len(lines):
            combined = f"{line} {lines[i + 1]}"
            extracted2 = _extract_score_from_text(combined)
            if extracted2 and (extracted2["score"] is not None):
                rows.append({"Section": current_section or "Subjects", "Label": extracted2["label"] or line,
                             "Score": extracted2["score"], "Maximum": extracted2["maximum"], "Notes": None})
                i += 2
                continue

        m_simple = re.search(r"(.+?)\s+(\d{1,3})\b", line)
        if m_simple:
            label = m_simple.group(1).strip()
            score = m_simple.group(2)
            rows.append({"Section": current_section or "Subjects", "Label": label, "Score": score, "Maximum": None,
                         "Notes": None})
            i += 1
            continue

        rows.append(
            {"Section": current_section or "Misc", "Label": line, "Score": None, "Maximum": None, "Notes": None})
        i += 1

    out = pd.DataFrame(rows, columns=canonical)
    out["Score"] = pd.to_numeric(out["Score"], errors="coerce")
    out["Maximum"] = pd.to_numeric(out["Maximum"], errors="coerce")
    return out


def heuristic_normalize(df: pd.DataFrame) -> pd.DataFrame:
    canonical = ["Section", "Label", "Score", "Maximum", "Notes"]
    if set(canonical).issubset(df.columns):
        return df[canonical]

    rows = []
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
            rows.append({"Section": "Subjects", "Label": label, "Score": score, "Maximum": maximum, "Notes": None})
        else:
            rows.append({"Section": "Misc", "Label": line, "Score": None, "Maximum": None, "Notes": None})

    out = pd.DataFrame(rows, columns=canonical)
    out["Score"] = pd.to_numeric(out["Score"], errors="coerce")
    out["Maximum"] = pd.to_numeric(out["Maximum"], errors="coerce")
    return out


def ai_normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return normalize_uploaded_dataframe(df)


# -----------------------------
# Helpers for visualization
# -----------------------------
def get_valid_x_axis_columns(df: pd.DataFrame) -> List[str]:
    """
    Returns valid string columns for X-axis comparison.
    Rule: The baseline column "Label" MUST pass the uniqueness evaluation check.
    Subsequent companion columns (like 'Label 2', 'Label 3') bypass uniqueness checks
    and are admitted strictly via string name pattern matching.
    """
    valid_cols = []
    for col in df.columns:
        if df[col].dtype == "object":
            # Rule 1: The primary column must be strictly unique
            if col == "Label":
                if df[col].is_unique:
                    valid_cols.append(col)
            # Rule 2: Multi-year/segmented labels match strictly via name pattern strings
            elif col.lower().startswith("label"):
                valid_cols.append(col)
    return valid_cols


def get_groupable_text_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if df[col].dtype == "object"]


def get_auto_y_for_x_column(df: pd.DataFrame, x_col: str) -> Optional[str]:
    """
    Auto-select the Y-axis numeric column using a 3-letter prefix matching rule.
    """
    if not x_col or len(x_col) < 3:
        return None

    numeric_cols: List[str] = [
        c for c in df.select_dtypes(include="number").columns
        if c.lower() != "maximum"
    ]
    if not numeric_cols:
        return None

    prefix = x_col[:3].lower()
    prefix_group: List[str] = [col for col in df.columns if col.lower().startswith(prefix)]

    try:
        n = prefix_group.index(x_col)
    except ValueError:
        n = 0

    idx = min(n, len(numeric_cols) - 1)
    return numeric_cols[idx]
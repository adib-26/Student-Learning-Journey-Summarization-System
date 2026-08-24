import pandas as pd
import re
from typing import List, Optional


# -----------------------------
# Data Cleaning
# -----------------------------
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    raw_columns = df.columns.astype(str).str.strip()

    # --- UNIQUE COLUMN RULE IMPLEMENTATION ---
    # 1. Extract base names by removing pandas duplicate suffixes (.1) or custom suffixes (_Y1)
    base_names = []
    for col in raw_columns:
        base = re.sub(r'(\.\d+|_[Yy]\d+)$', '', col).strip()
        base_names.append(base)

    # 2. Sequentially rename duplicates: 'Label', 'Label 2', 'Label 3'
    new_columns = []
    counts = {}
    for base in base_names:
        counts[base] = counts.get(base, 0) + 1
        if counts[base] == 1:
            new_columns.append(base)
        else:
            new_columns.append(f"{base} {counts[base]}")

    df.columns = new_columns
    # -----------------------------------------

    # Normalize text columns
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()

    # Convert numeric-like columns where appropriate
    for col in df.columns:
        # ``errors='ignore'`` is deprecated in pandas 2.2+.  Only replace a
        # column when every non-empty value can be represented numerically;
        # otherwise retain the original text values.
        converted = pd.to_numeric(df[col], errors="coerce")
        original_non_empty = df[col].notna() & df[col].astype(str).str.strip().ne("")
        if not (original_non_empty & converted.isna()).any():
            df[col] = converted

    # Remove fully empty rows
    df = df.dropna(how="all")
    return df


def normalize_scores_to_percentage(df: pd.DataFrame) -> pd.DataFrame:
    """Put comparable marked assessments on a 0–100 analytics scale.

    A raw mark of 18/20 must not be ranked below 80/100. When a meaningful
    maximum differs from 100, retain the original values for traceability and
    make the canonical ``Score`` / ``Maximum`` pair percentage-based for all
    downstream metrics, charts, and reports.
    """
    if not {"Score", "Maximum"}.issubset(df.columns):
        return df

    normalized = df.copy()
    scores = pd.to_numeric(normalized["Score"], errors="coerce")
    maximums = pd.to_numeric(normalized["Maximum"], errors="coerce")
    scored_rows = scores.notna() & maximums.notna() & maximums.gt(0)
    if not scored_rows.any() or maximums.loc[scored_rows].eq(100).all():
        return normalized

    normalized["Original Score"] = scores
    normalized["Original Maximum"] = maximums
    normalized.loc[scored_rows, "Score"] = (
        scores.loc[scored_rows] / maximums.loc[scored_rows] * 100
    )
    normalized.loc[scored_rows, "Maximum"] = 100.0
    return normalized


# -----------------------------
# Helpers for visualization
# -----------------------------
def get_valid_x_axis_columns(df: pd.DataFrame) -> List[str]:
    """
    Returns valid string columns for X-axis comparison.
    Rule: The primary column "Label" MUST pass the uniqueness check.
    Subsequent companion columns (e.g., 'Label 2', 'Label 3') bypass uniqueness checks
    and are admitted strictly via string name matching.
    """
    valid_cols = []
    for col in df.columns:
        if df[col].dtype == "object":
            # Rule 1: First label column must be strictly unique
            if col == "Label":
                if df[col].is_unique:
                    valid_cols.append(col)
            # Rule 2: 2nd label and onwards are decided by name string matching
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

    # Exclude 'maximum' / 'Maximum' — it is a metadata bound, not a plottable metric
    numeric_cols: List[str] = [
        c for c in df.select_dtypes(include="number").columns
        if c.lower() != "maximum"
    ]
    if not numeric_cols:
        return None

    prefix = x_col[:3].lower()

    # Preserve original column order when building the prefix group
    prefix_group: List[str] = [
        col for col in df.columns
        if col.lower().startswith(prefix)
    ]

    try:
        n = prefix_group.index(x_col)  # 0-based position within the prefix group
    except ValueError:
        n = 0  # x_col not in any prefix group → treat as first

    # Clamp to available numeric columns
    idx = min(n, len(numeric_cols) - 1)
    return numeric_cols[idx]

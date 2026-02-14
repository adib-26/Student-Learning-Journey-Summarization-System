import pandas as pd
from typing import List

# -----------------------------
# Data Cleaning
# -----------------------------
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    # Normalize text columns
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()

    # Convert numeric-like columns where appropriate
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="ignore")
        except Exception:
            pass

    # Remove fully empty rows
    df = df.dropna(how="all")
    return df


# -----------------------------
# Helpers for visualization
# -----------------------------
def get_valid_x_axis_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if df[col].dtype == "object" and df[col].is_unique]


def get_groupable_text_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if df[col].dtype == "object"]
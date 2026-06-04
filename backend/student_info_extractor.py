"""
Universal student information extractor - FINAL FIX
Handles corrupted metadata dicts and sheet_name=None input robustly.
"""

import pandas as pd
import re
import json
import time
import logging
from typing import Dict, Optional
import streamlit as st

logger = logging.getLogger(__name__)

FORBIDDEN_LABELS = {
    "Grade/Achievement", "Remarks", "Result", "Score", "Maximum",
    "Attendance Rate (%)", "Attendance Rate", "Subjects", "Behaviour",
    "Behaviours", "Co-Curricular", "Co Curricular", "School Level",
    "Grade", "Activity", "Activities", "Unknown Student", "None", "Nan",
    "Student Details", "Section", "Label", "Value", "Unknown", "Student", ""
}
FORBIDDEN_LABELS_LOWER = {label.lower() for label in FORBIDDEN_LABELS}


def sanitize_and_set_student_name(student_info: Dict[str, str], name: str):
    if not name:
        return
    cleaned_name = str(name).strip()
    cleaned_name = re.sub(r"^[`\"'\[\{]+|[\]\}\"'`]+$", "", cleaned_name).strip()
    cleaned_name = re.split(r"\s{2,}|\bNaN\b", cleaned_name)[0].strip()
    if cleaned_name and cleaned_name.lower() not in FORBIDDEN_LABELS_LOWER:
        student_info["Student Name"] = cleaned_name
        student_info["student_name"] = cleaned_name
        print(f"[DEBUG] ✅ Name set to: '{cleaned_name}'")
    else:
        print(f"[DEBUG] ❌ Rejected name candidate: '{cleaned_name}'")


def extract_student_info_from_dataframe(df: pd.DataFrame) -> Dict[str, str]:
    student_info = {}
    if df is None or df.empty:
        return student_info

    # ----- Method 1: Column‑aware (Label / Value) -----
    label_col = value_col = None
    for col in df.columns:
        col_lower = str(col).strip().lower()
        if col_lower == "label":
            label_col = col
        elif col_lower == "value":
            value_col = col
        if label_col and value_col:
            break

    if label_col and value_col:
        mask = df[label_col].astype(str).str.strip().str.lower() == "student name"
        matches = df[mask]
        if not matches.empty:
            candidate = str(matches.iloc[0][value_col]).strip()
            if candidate and candidate.lower() not in FORBIDDEN_LABELS_LOWER:
                sanitize_and_set_student_name(student_info, candidate)
                return student_info

    # ----- Method 2: Adjacent cell search -----
    for _, row in df.iterrows():
        row_vals = [str(v).strip() if pd.notna(v) else "" for v in row.values]
        for i, cell in enumerate(row_vals):
            if cell.lower() in ["student name", "student_name", "name"]:
                for offset in range(1, min(5, len(row_vals) - i)):
                    cand = row_vals[i + offset]
                    if cand and cand.lower() not in FORBIDDEN_LABELS_LOWER:
                        sanitize_and_set_student_name(student_info, cand)
                        return student_info

    # ----- Method 3: Brute‑force cell scan -----
    for i in range(len(df)):
        for j in range(len(df.columns)):
            cell = str(df.iloc[i, j]).strip()
            if "student name" in cell.lower():
                # look right
                if j + 1 < len(df.columns):
                    cand = str(df.iloc[i, j+1]).strip()
                    if cand and cand.lower() not in FORBIDDEN_LABELS_LOWER:
                        sanitize_and_set_student_name(student_info, cand)
                        return student_info
                # look down
                if i + 1 < len(df):
                    cand = str(df.iloc[i+1, j]).strip()
                    if cand and cand.lower() not in FORBIDDEN_LABELS_LOWER:
                        sanitize_and_set_student_name(student_info, cand)
                        return student_info

    return student_info


def is_useless_metadata_dict(meta: Dict) -> bool:
    """
    Detect if metadata is a dict that contains obviously wrong student info
    (e.g., {"Student Name": "Grade/Achievement"}). If yes, ignore it.
    """
    if not isinstance(meta, dict):
        return False
    for k, v in meta.items():
        if "name" in str(k).lower():
            val_str = str(v).strip()
            if val_str.lower() in FORBIDDEN_LABELS_LOWER:
                return True
    return False


def normalize_input(df, metadata):
    """
    Convert common mistaken inputs into a proper DataFrame.
    - If df is a dict of sheets (from sheet_name=None), take the first sheet.
    - If metadata contains a real DataFrame but df is None, swap them.
    - If metadata is a useless dict, discard it.
    """
    # Case 1: df is a dict of DataFrames
    if isinstance(df, dict):
        for v in df.values():
            if isinstance(v, pd.DataFrame):
                print("[DEBUG] Unwrapping sheet dict -> first DataFrame")
                return v, None
    # Case 2: metadata contains a DataFrame and df is None
    if df is None and isinstance(metadata, dict):
        for v in metadata.values():
            if isinstance(v, pd.DataFrame):
                print("[DEBUG] Swapping: metadata holds DataFrame, using it as df")
                return v, None
    # Case 3: metadata is useless
    if is_useless_metadata_dict(metadata):
        print("[DEBUG] Ignoring useless metadata dict")
        metadata = None
    return df, metadata


def get_student_info(df: Optional[pd.DataFrame] = None,
                     metadata: Optional[Dict] = None,
                     text: Optional[str] = None) -> Dict[str, str]:
    student_info = {}

    # Normalise inputs before any logic
    df, metadata = normalize_input(df, metadata)

    # Priority 1: Genuine metadata (key‑value, no DataFrames)
    if metadata and isinstance(metadata, dict) and not any(isinstance(v, pd.DataFrame) for v in metadata.values()):
        print("[DEBUG] Processing metadata path")
        for k, v in metadata.items():
            if k and v is not None and not isinstance(v, pd.DataFrame):
                k_str, v_str = str(k).strip(), str(v).strip()
                if k_str.lower() not in FORBIDDEN_LABELS_LOWER and v_str.lower() not in FORBIDDEN_LABELS_LOWER:
                    student_info[k_str] = v_str
        if "Student Name" in student_info:
            return student_info   # accept it only if name already found
        # otherwise fall through to DataFrame

    # Priority 2: DataFrame
    if df is not None and not df.empty:
        print("[DEBUG] Processing DataFrame path")
        student_info = extract_student_info_from_dataframe(df)
        if "Student Name" not in student_info:
            # try converting DataFrame to text and use regex
            try:
                txt = df.head(15).to_string()
                from .student_extractor import extract_student_info_from_text
                txt_info = extract_student_info_from_text(txt)
                if "Student Name" in txt_info:
                    sanitize_and_set_student_name(student_info, txt_info["Student Name"])
            except Exception:
                pass
        return student_info

    # Priority 3: Plain text
    if text:
        print("[DEBUG] Processing plain text path")
        from .student_extractor import extract_student_info_from_text
        student_info = extract_student_info_from_text(text)

    return student_info


def extract_student_info_from_text(text: str) -> Dict[str, str]:
    """Regex‑based extraction from raw text (kept for completeness)"""
    student_info = {}
    if not text:
        return student_info
    lines = text.split("\n")
    for line in lines[:200]:
        line = line.strip()
        if not line:
            continue
        m = re.search(r"(?:Student\s+Name|StudentName|Name)\s*[:\-]?\s*([A-Za-z][A-Za-z\s'\-]{1,58}?)(?=\s{2,}|\s*$)", line, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            if name.lower() not in FORBIDDEN_LABELS_LOWER:
                sanitize_and_set_student_name(student_info, name)
        # other fields (Gender, State, etc.) can be added similarly
    return student_info


def display_student_banner(student_info: Dict[str, str]):
    student_name = student_info.get("Student Name")
    if student_name:
        st.success(f"✅ Student: **{student_name}**")
        with st.expander("👤 Student Details", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                if "Gender" in student_info:
                    st.write(f"**Gender:** {student_info['Gender']}")
                if "Nationality" in student_info:
                    st.write(f"**Nationality:** {student_info['Nationality']}")
                if "School Level" in student_info:
                    st.write(f"**School Level:** {student_info['School Level']}")
            with col2:
                if "Form" in student_info:
                    st.write(f"**Form:** {student_info['Form']}")
                if "State" in student_info:
                    st.write(f"**State:** {student_info['State']}")
                if "Attendance Rate (%)" in student_info:
                    st.write(f"**Attendance:** {student_info['Attendance Rate (%)']}")
    else:
        st.warning("⚠️ Student name not found.")
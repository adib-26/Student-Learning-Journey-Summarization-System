"""
Universal student information extractor - FINAL FIX
Handles corrupted metadata dicts and sheet_name=None input robustly.
Fixed gender extraction from Excel files with proper column detection.
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

    df = df.copy()

    # --- Step 0: If the first row looks like data instead of headers, promote it ---
    # But only if we don't already have sensible column names like "Section", "Label", "Value"
    first_row_values = [str(v).strip().lower() if pd.notna(v) else "" for v in df.iloc[0].tolist()]
    column_names_lower = [str(c).strip().lower() for c in df.columns]

    # Check if current columns are generic (0,1,2,...) or lack expected labels
    has_section_label_value = any(col in column_names_lower for col in ["section", "label", "value"])
    if not has_section_label_value and "label" in first_row_values and "value" in first_row_values:
        # First row contains label/value – promote it to header
        new_columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)
        df.columns = new_columns
        print("[DEBUG] Promoted first row to column headers")

    # ------------------------------------------------------------------
    # Method 1: Direct Label / Value columns (most reliable)
    # ------------------------------------------------------------------
    label_col = None
    value_col = None
    for col in df.columns:
        col_lower = str(col).strip().lower()
        if col_lower == "label":
            label_col = col
        elif col_lower == "value":
            value_col = col

    if label_col is not None and value_col is not None:
        print("[DEBUG] Using Label/Value column extraction")
        # Iterate through all rows
        for idx, row in df.iterrows():
            label = str(row[label_col]).strip().lower() if pd.notna(row[label_col]) else ""
            value = str(row[value_col]).strip() if pd.notna(row[value_col]) else ""

            if not value or value.lower() in FORBIDDEN_LABELS_LOWER:
                continue

            print(f"[DEBUG] Processing: label='{label}' -> value='{value}'")

            # Name
            if label in {"student name", "student_name", "name", "full name"}:
                sanitize_and_set_student_name(student_info, value)

            # Gender (critical fix: ensure it's caught)
            elif label in {"gender", "sex"}:
                gender_val = value.strip()
                if gender_val.upper() == "M":
                    student_info["Gender"] = "Male"
                elif gender_val.upper() == "F":
                    student_info["Gender"] = "Female"
                else:
                    student_info["Gender"] = gender_val.capitalize()
                print(f"[DEBUG] ✅ Gender set to: {student_info['Gender']}")

            # Other fields
            elif label in {"nationality", "citizenship"}:
                student_info["Nationality"] = value
            elif label in {"school level", "level"}:
                student_info["School Level"] = value
            elif label in {"form", "class", "grade", "year"}:
                student_info["Form"] = value
            elif label == "state":
                student_info["State"] = value
            elif "attendance" in label:
                student_info["Attendance Rate (%)"] = value

        # If we already have the essential fields, return early
        if student_info.get("Student Name") and student_info.get("Gender"):
            return student_info

    # ------------------------------------------------------------------
    # Method 2: Two-row structure with forward-filled sections
    # (Handles files where "Section" column is not repeated for every row)
    # ------------------------------------------------------------------
    print("[DEBUG] Trying two-row structure extraction (ffill)")
    # Make a copy and forward fill the Section column
    df_filled = df.copy()
    if "Section" in df_filled.columns or df_filled.columns[0] == "Section":
        section_col = "Section" if "Section" in df_filled.columns else df_filled.columns[0]
        df_filled[section_col] = df_filled[section_col].fillna(method='ffill')
        # Convert to string and lower for comparison
        df_filled['_section_lower'] = df_filled[section_col].astype(str).str.lower()

        # Locate rows under "student details"
        mask = df_filled['_section_lower'] == "student details"
        student_rows = df_filled[mask]

        # Now look for Label and Value columns (could be named anything, but typically column 1 and 2)
        # We'll use the same label_col / value_col if found earlier, otherwise fallback to positional indices
        if label_col is None or value_col is None:
            # Try to find by position: often Label is the second column, Value the third
            if len(df.columns) >= 3:
                label_col = df.columns[1]
                value_col = df.columns[2]
                print(f"[DEBUG] Using positional columns: label='{label_col}', value='{value_col}'")

        if label_col is not None and value_col is not None:
            for idx, row in student_rows.iterrows():
                label = str(row[label_col]).strip().lower() if pd.notna(row[label_col]) else ""
                value = str(row[value_col]).strip() if pd.notna(row[value_col]) else ""

                if not value or value.lower() in FORBIDDEN_LABELS_LOWER:
                    continue

                print(f"[DEBUG] (ffill) Processing: label='{label}' -> value='{value}'")

                if label in {"student name", "student_name", "name", "full name"}:
                    sanitize_and_set_student_name(student_info, value)
                elif label in {"gender", "sex"}:
                    gender_val = value.strip()
                    if gender_val.upper() == "M":
                        student_info["Gender"] = "Male"
                    elif gender_val.upper() == "F":
                        student_info["Gender"] = "Female"
                    else:
                        student_info["Gender"] = gender_val.capitalize()
                    print(f"[DEBUG] ✅ Gender (ffill) set to: {student_info['Gender']}")
                elif label in {"nationality", "citizenship"}:
                    student_info["Nationality"] = value
                elif label in {"school level", "level"}:
                    student_info["School Level"] = value
                elif label in {"form", "class", "grade", "year"}:
                    student_info["Form"] = value
                elif label == "state":
                    student_info["State"] = value
                elif "attendance" in label:
                    student_info["Attendance Rate (%)"] = value

        if student_info.get("Student Name") and student_info.get("Gender"):
            return student_info

    # ------------------------------------------------------------------
    # Method 3: Adjacent‑cell scan (last resort, especially for gender)
    # Only run if we're missing essential fields
    # ------------------------------------------------------------------
    if not student_info.get("Student Name") or not student_info.get("Gender"):
        print("[DEBUG] Falling back to adjacent-cell scan for missing fields")
        for _, row in df.iterrows():
            row_vals = [str(v).strip() if pd.notna(v) else "" for v in row.values]
            for i, cell in enumerate(row_vals):
                cell_lower = cell.lower()
                # Check for both student name and gender labels
                if cell_lower in {"student name", "name", "full name", "gender", "sex"}:
                    # Look ahead up to 3 cells for the actual value
                    for offset in range(1, min(4, len(row_vals) - i)):
                        candidate = row_vals[i + offset]
                        if candidate and candidate.lower() not in FORBIDDEN_LABELS_LOWER:
                            # Handle student name
                            if cell_lower in {"student name", "name", "full name"} and not student_info.get("Student Name"):
                                sanitize_and_set_student_name(student_info, candidate)
                            # Handle gender
                            elif cell_lower in {"gender", "sex"} and not student_info.get("Gender"):
                                if candidate.upper() == "M":
                                    student_info["Gender"] = "Male"
                                elif candidate.upper() == "F":
                                    student_info["Gender"] = "Female"
                                else:
                                    student_info["Gender"] = candidate.capitalize()
                                print(f"[DEBUG] ✅ Gender from scan: {student_info['Gender']}")
                            break
                    # Early exit if we have both essential fields
                    if student_info.get("Student Name") and student_info.get("Gender"):
                        break
            if student_info.get("Student Name") and student_info.get("Gender"):
                break
        # Return early once we've found all essential fields
        if student_info.get("Student Name") and student_info.get("Gender"):
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
        # Only return early if we have both name AND gender - otherwise fall through to DataFrame processing to get missing fields
        if student_info.get("Student Name") and student_info.get("Gender"):
            return student_info
        # otherwise fall through to DataFrame

    # Priority 2: DataFrame
    if df is not None and not df.empty:
        print("[DEBUG] Processing DataFrame path")
        student_info = extract_student_info_from_dataframe(df)

        # If important fields are missing, try regex text fallback to supplement
        if "Student Name" not in student_info or "Gender" not in student_info:
            try:
                txt = df.head(15).to_string()
                txt_info = extract_student_info_from_text(txt)

                # Check for Name
                if "Student Name" not in student_info and "Student Name" in txt_info:
                    sanitize_and_set_student_name(student_info, txt_info["Student Name"])

                # Check for Gender (always check if missing, even if name is found)
                if "Gender" not in student_info and "Gender" in txt_info:
                    student_info["Gender"] = txt_info["Gender"]
                    print(f"[DEBUG] Gender from text fallback: {student_info['Gender']}")

                # Merge missing items extracted from text
                for k, v in txt_info.items():
                    if k not in student_info and k not in ("Student Name", "student_name"):
                        student_info[k] = v
            except Exception as e:
                print(f"[DEBUG] Text fallback error: {e}")

        return student_info

    # Priority 3: Plain text
    if text:
        print("[DEBUG] Processing plain text path")
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

        print(f"[DEBUG] Text line: {line}")

        # Name Regex
        m = re.search(r"(?:Student\s+Name|StudentName|Name)\s*[:\-]?\s*([A-Za-z][A-Za-z\s'\-]{1,58}?)(?=\s{2,}|\s*$)", line, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            if name.lower() not in FORBIDDEN_LABELS_LOWER:
                sanitize_and_set_student_name(student_info, name)

        # Gender regex (supports all formats: Male/Female/M/F/Lelaki/Perempuan/L/P)
        gender_match = re.search(r"\b(?:Gender|Sex|Jantina)\s*[:\-,\t]*\s*(Male|Female|M|F|Lelaki|Perempuan|L|P)\b", line, re.IGNORECASE)
        if gender_match:
            gender_val = gender_match.group(1).strip()
            if gender_val.upper() in ('M', 'L', 'LELAKI'):
                student_info['Gender'] = 'Male'
            elif gender_val.upper() in ('F', 'P', 'PEREMPUAN'):
                student_info['Gender'] = 'Female'
            elif gender_val.lower() not in FORBIDDEN_LABELS_LOWER:
                student_info['Gender'] = gender_val.capitalize()
            print(f"[DEBUG] Gender from text: {student_info['Gender']}")

        # Additional Fallback Fields
        nat_match = re.search(r"\bNationality\s*[:\-,\t]*\s*([A-Za-z][A-Za-z\s'\-]{1,50})", line, re.IGNORECASE)
        if nat_match and "Nationality" not in student_info:
            student_info["Nationality"] = nat_match.group(1).strip()

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
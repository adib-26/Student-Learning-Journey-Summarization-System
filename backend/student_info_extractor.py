# backend/student_info_extractor.py
"""
Universal student information extractor
Works with Excel metadata, CSV data, and OCR text
"""

import pandas as pd
import re
from typing import Dict, Optional


def extract_student_info_from_dataframe(df: pd.DataFrame) -> Dict[str, str]:
    """
    Extract student information from a DataFrame.
    Works with both structured CSVs and normalized data.

    Args:
        df: DataFrame that may contain student information

    Returns:
        Dictionary with student info like {"Student Name": "...", "Gender": "...", ...}
    """
    student_info = {}

    if df is None or df.empty:
        return student_info

    # Check if DataFrame has the CSV structure with Section, Label, Value columns
    if 'Section' in df.columns and 'Label' in df.columns:
        # Extract from structured CSV
        student_rows = df[df['Section'] == 'Student Details']

        for _, row in student_rows.iterrows():
            label = str(row.get('Label', '')).strip()

            # Try 'Value' column first, then other columns
            value = None
            if 'Value' in df.columns and pd.notna(row.get('Value')):
                value = str(row['Value']).strip()
            elif pd.notna(row.get('Score')):
                value = str(row['Score']).strip()

            if label and value and value.lower() not in ['', 'nan', 'none']:
                student_info[label] = value

    # If no student info found yet, try to find it in any column
    if not student_info:
        for col in df.columns:
            # Look for name-like patterns in first few rows
            for idx in range(min(10, len(df))):
                cell = str(df[col].iloc[idx])

                # Check if looks like a student name (2+ capitalized words)
                tokens = re.findall(r"[A-Z][a-zA-Z]+", cell)
                if len(tokens) >= 3:  # Malaysian names usually have 3+ parts
                    student_info['Student Name'] = cell
                    break

            if 'Student Name' in student_info:
                break

    return student_info


def extract_student_info_from_text(text: str) -> Dict[str, str]:
    """
    Extract student information from OCR or plain text.

    Args:
        text: Text that may contain student information

    Returns:
        Dictionary with student info
    """
    student_info = {}

    if not text:
        return student_info

    lines = text.split('\n')

    for line in lines[:20]:  # Check first 20 lines
        line = line.strip()

        # Look for "Name:" or "Student Name:" patterns
        name_match = re.search(r'(?:Student\s+)?Name\s*[:\-]?\s*(.+)', line, re.IGNORECASE)
        if name_match:
            name = name_match.group(1).strip()
            tokens = re.findall(r"[A-Z][a-zA-Z]+", name)
            if len(tokens) >= 2:
                student_info['Student Name'] = name

        # Look for Gender
        gender_match = re.search(r'Gender\s*[:\-]?\s*(Male|Female)', line, re.IGNORECASE)
        if gender_match:
            student_info['Gender'] = gender_match.group(1).title()

        # Look for State
        state_match = re.search(r'State\s*[:\-]?\s*([A-Za-z\s]+)', line, re.IGNORECASE)
        if state_match:
            state = state_match.group(1).strip()
            if state and len(state) < 30:  # Reasonable length for state name
                student_info['State'] = state

    return student_info


def get_student_info(df: Optional[pd.DataFrame] = None,
                     metadata: Optional[Dict] = None,
                     text: Optional[str] = None) -> Dict[str, str]:
    """
    Universal function to get student info from any source.

    Args:
        df: DataFrame (for CSV files)
        metadata: Metadata dict (for Excel files)
        text: Plain text (for OCR/PDF)

    Returns:
        Dictionary with student information
    """
    student_info = {}

    # Priority 1: Excel metadata (most reliable)
    if metadata and isinstance(metadata, dict):
        student_info = metadata.copy()

    # Priority 2: Extract from DataFrame (CSV)
    elif df is not None:
        student_info = extract_student_info_from_dataframe(df)

    # Priority 3: Extract from text (OCR/PDF)
    elif text:
        student_info = extract_student_info_from_text(text)

    return student_info


def display_student_banner(student_info: Dict[str, str]):
    """
    Display beautiful student info banner in Streamlit.
    Call this from your app.py

    Args:
        student_info: Dictionary with student information
    """
    import streamlit as st

    student_name = student_info.get('Student Name')

    if student_name:
        st.success(f"✅ Student: **{student_name}**")

        # Show details in expandable section
        with st.expander("👤 Student Details", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                if 'Gender' in student_info:
                    st.write(f"**Gender:** {student_info['Gender']}")
                if 'Nationality' in student_info:
                    st.write(f"**Nationality:** {student_info['Nationality']}")
                if 'School Level' in student_info:
                    st.write(f"**School Level:** {student_info['School Level']}")

            with col2:
                if 'Form' in student_info:
                    st.write(f"**Form:** {student_info['Form']}")
                if 'State' in student_info:
                    st.write(f"**State:** {student_info['State']}")
                if 'Attendance Rate (%)' in student_info:
                    st.write(f"**Attendance:** {student_info['Attendance Rate (%)']}")
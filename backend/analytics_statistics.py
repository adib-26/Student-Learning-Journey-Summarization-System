import re
import pandas as pd
from typing import Dict, Any
from .data_processing import (
    extract_full_name,
    looks_like_name,
    extract_gender,
    extract_state,
    is_valid_subject,
    contains_metadata_keyword,
    contains_co_curricular_keyword,
    KNOWN_SUBJECTS_CS
)

def compute_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute statistics and extract metadata from DataFrame.
    """
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()

    numeric_df = df.select_dtypes(include="number")

    stats: Dict[str, Any] = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "numeric_columns": list(numeric_df.columns),
        "averages": {},
        "medians": {},
        "std_dev": {},
        "counts": {},
        "student_name": None,
        "student_details": None,
        "subjects": [],
        "subject_scores": {},
        "activities": [],
        "strength": None,
        "weakness": None
    }

    # Numeric stats
    if not numeric_df.empty:
        stats["averages"] = numeric_df.mean().dropna().to_dict()
        stats["medians"] = numeric_df.median().dropna().to_dict()
        stats["std_dev"] = numeric_df.std(ddof=0).dropna().to_dict()
        stats["counts"] = numeric_df.count().dropna().to_dict()

    # Metadata extraction
    gender = None
    state = None
    if "Label" in df.columns:
        for _, row in df.iterrows():
            label = row.get("Label", "")
            value = row.get("Value", "")
            text = f"{label} {value}".strip()

            if "name" in label.lower():
                extracted_name = extract_full_name(text)
                if extracted_name:
                    stats["student_name"] = extracted_name
                elif looks_like_name(text):
                    stats["student_name"] = value or label

            g = extract_gender(text)
            if g:
                gender = g

            s = extract_state(text)
            if s:
                state = s

    details = []
    if gender:
        details.append(f"Gender: {gender}")
    if state:
        details.append(f"State: {state}")
    if details:
        stats["student_details"] = ", ".join(details)

    # Subjects
    if "Section" in df.columns and "Score" in df.columns:
        subject_rows = df[df["Section"].str.contains("Subjects", case=False, na=False)]
        if not subject_rows.empty:
            subject_rows = subject_rows.copy()
            subject_rows["Score"] = pd.to_numeric(subject_rows["Score"], errors="coerce")
            subject_rows = subject_rows.dropna(subset=["Score"])

            valid_subjects = []
            valid_scores = {}

            for _, row in subject_rows.iterrows():
                raw_label = row.get("Label", "")
                score = row.get("Score")

                tokens = raw_label.split()
                last_word = tokens[-1] if tokens else ""
                if is_valid_subject(last_word):
                    valid_subjects.append(last_word)
                    valid_scores[last_word] = score
                    continue

                for subj in KNOWN_SUBJECTS_CS:
                    if subj in raw_label:
                        valid_subjects.append(subj)
                        valid_scores[subj] = score
                        break

            stats["subjects"] = valid_subjects
            stats["subject_scores"] = valid_scores

            if valid_scores:
                stats["strength"] = max(valid_scores, key=valid_scores.get)
                stats["weakness"] = min(valid_scores, key=valid_scores.get)

    # Activities
    valid_activities = []
    for _, row in df.iterrows():
        label = row.get("Label", "")
        section = row.get("Section", "")

        if not label:
            continue

        parts = re.split(r"\s*\|\s*|/", label)
        for part in parts:
            part = part.strip()
            if not part:
                continue

            if contains_co_curricular_keyword(part):
                valid_activities.append(part)
                continue

            if re.search(r"Co-?curricular|Activity|Activities", str(section), flags=re.IGNORECASE):
                if not contains_metadata_keyword(part):
                    valid_activities.append(part)

    stats["activities"] = valid_activities

    return stats
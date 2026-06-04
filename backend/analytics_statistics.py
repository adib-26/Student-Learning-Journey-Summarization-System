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
    Prefers structured Value cells for metadata (e.g., Student Name)
    and only falls back to text parsing when necessary.
    """
    df = df.copy()
    # Normalize column names
    df.columns = df.columns.astype(str).str.strip()

    # Normalize string cells
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

    # Metadata extraction (student name, gender, state)
    gender = None
    state = None

    def set_student_name_if_missing(name_candidate: str):
        if not name_candidate:
            return

        candidate = str(name_candidate).strip()
        if not candidate:
            return

        # Prevent metadata values from becoming names
        forbidden = {
            "grade",
            "class",
            "section",
            "gender",
            "state",
            "subject",
            "score",
            "activity",
            "activities",
            "co curricular",
            "co-curricular",
        }

        if candidate.lower() in forbidden:
            return

        if not stats.get("student_name"):
            stats["student_name"] = candidate

    if "Label" in df.columns:
        # PASS 1: Explicit Student Name row only
        student_name_rows = df[
            df["Label"].astype(str).str.strip().str.lower() == "student name"
        ]

        if not student_name_rows.empty:
            value = str(student_name_rows.iloc[0].get("Value", "")).strip()
            if value and not contains_metadata_keyword(value):
                stats["student_name"] = value

        # PASS 2: Fallback search only if name not found
        if not stats.get("student_name"):
            for _, row in df.iterrows():
                label = str(row.get("Label", "")).strip()
                value = str(row.get("Value", "")).strip()
                combined_text = f"{label} {value}"

                extracted_name = extract_full_name(combined_text)
                if extracted_name:
                    stats["student_name"] = extracted_name
                    break

                if (
                    value
                    and looks_like_name(value)
                    and not contains_metadata_keyword(value)
                ):
                    stats["student_name"] = value
                    break

        # PASS 3: Gender + State
        for _, row in df.iterrows():
            label = str(row.get("Label", "")).strip()
            value = str(row.get("Value", "")).strip()
            combined_text = f"{label} {value}"

            g = extract_gender(combined_text)
            if g:
                gender = g

            s = extract_state(combined_text)
            if s:
                state = s

    details = []
    if gender:
        details.append(f"Gender: {gender}")
    if state:
        details.append(f"State: {state}")
    if details:
        stats["student_details"] = ", ".join(details)

    # Subjects extraction
    if "Section" in df.columns and "Score" in df.columns and "Label" in df.columns:
        subject_rows = df[df["Section"].str.contains("Subjects", case=False, na=False)]
        if not subject_rows.empty:
            subject_rows = subject_rows.copy()
            subject_rows["Score"] = pd.to_numeric(subject_rows["Score"], errors="coerce")
            subject_rows = subject_rows.dropna(subset=["Score"])

            valid_subjects = []
            valid_scores = {}

            for _, row in subject_rows.iterrows():
                raw_label = row.get("Label", "") or ""
                score = row.get("Score")
                label_text = raw_label.strip()

                tokens = re.split(r"[\s:/\-()]+", label_text)
                tokens = [t for t in tokens if t]
                last_word = tokens[-1] if tokens else ""

                if last_word and is_valid_subject(last_word):
                    subj = last_word
                    valid_subjects.append(subj)
                    valid_scores[subj] = score
                    continue

                matched = None
                for subj in sorted(KNOWN_SUBJECTS_CS, key=lambda s: -len(s)):
                    if subj.lower() in label_text.lower():
                        matched = subj
                        break
                if matched:
                    valid_subjects.append(matched)
                    valid_scores[matched] = score
                    continue

                value_field = (row.get("Value") or "").strip()
                if value_field and is_valid_subject(value_field):
                    valid_subjects.append(value_field)
                    valid_scores[value_field] = score

            stats["subjects"] = valid_subjects
            stats["subject_scores"] = valid_scores

            if valid_scores:
                stats["strength"] = max(valid_scores, key=valid_scores.get)
                stats["weakness"] = min(valid_scores, key=valid_scores.get)

    # Activities / co-curricular extraction
    valid_activities = []
    for _, row in df.iterrows():
        raw_label = row.get("Label", "") or ""
        section = row.get("Section", "") or ""

        label = raw_label.strip()
        if not label:
            continue

        parts = re.split(r"\s*\|\s*|/|;|,", label)
        for part in parts:
            part = part.strip()
            if not part:
                continue

            if contains_co_curricular_keyword(part):
                valid_activities.append(part)
                continue

            if re.search(r"Co-?curricular|Co Curricular|Activity|Activities", str(section), flags=re.IGNORECASE):
                if not contains_metadata_keyword(part):
                    valid_activities.append(part)

    stats["activities"] = valid_activities

    return stats

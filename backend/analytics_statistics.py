import re
import pandas as pd
from typing import Dict, Any, Optional
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


def compute_statistics(
        df: pd.DataFrame,
        x_col: Optional[str] = None,
        y_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute statistics and extract metadata from DataFrame.
    For academic strength/weakness, only actual subjects (Section contains "Subject"
    or label in KNOWN_SUBJECTS_CS) are considered. Behaviour and co-curricular
    rows are excluded from subject scoring. Additionally, any label containing
    "Club" (case‑insensitive) is excluded from academic strength/weakness.
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

    if not numeric_df.empty:
        stats["averages"] = numeric_df.mean().dropna().to_dict()
        stats["medians"] = numeric_df.median().dropna().to_dict()
        stats["std_dev"] = numeric_df.std(ddof=0).dropna().to_dict()
        stats["counts"] = numeric_df.count().dropna().to_dict()

    # Metadata extraction (student name, gender, state)
    gender = None
    state = None

    if "Label" in df.columns:
        student_name_rows = df[
            df["Label"].astype(str).str.strip().str.lower() == "student name"
            ]
        if not student_name_rows.empty:
            value = str(student_name_rows.iloc[0].get("Value", "")).strip()
            if value and not contains_metadata_keyword(value):
                stats["student_name"] = value

        if not stats.get("student_name"):
            for _, row in df.iterrows():
                label = str(row.get("Label", "")).strip()
                value = str(row.get("Value", "")).strip()
                combined_text = f"{label} {value}"
                extracted_name = extract_full_name(combined_text)
                if extracted_name:
                    stats["student_name"] = extracted_name
                    break
                if value and looks_like_name(value) and not contains_metadata_keyword(value):
                    stats["student_name"] = value
                    break

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

    # ------------------------------------------------------------------
    # Subjects extraction for academic strength/weakness
    # ------------------------------------------------------------------
    if x_col and x_col in df.columns and y_col and y_col in df.columns:
        eff_label_col = x_col
        eff_score_col = y_col
    elif "Label" in df.columns and "Score" in df.columns:
        eff_label_col = "Label"
        eff_score_col = "Score"
    else:
        eff_label_col = None
        eff_score_col = None

    valid_subjects = []
    valid_scores = {}

    # Column‑based approach (subjects as column headers)
    selected_suffix = None
    if x_col:
        suffix_match = re.search(r'(_Y\d+|\s+\d+)$', str(x_col))
        if suffix_match:
            selected_suffix = suffix_match.group(1).strip()

    for col in df.columns:
        col_name = str(col).strip()
        if not col_name:
            continue
        metadata_keywords = [
            "name", "student", "gender", "state", "class", "grade", "section",
            "id", "roll", "label", "value", "score", "maximum", "min", "max",
            "average", "avg", "total", "rank", "position",
        ]
        if any(keyword in col_name.lower() for keyword in metadata_keywords):
            continue
        if selected_suffix:
            col_suffix_match = re.search(r'(_Y\d+|\s+\d+)$', col_name)
            if not col_suffix_match or col_suffix_match.group(1).strip() != selected_suffix:
                continue

        # NEW: Exclude columns with 'club' in the name
        if re.search(r'\bclub\b', col_name, re.IGNORECASE):
            continue

        numeric_col = pd.to_numeric(df[col], errors="coerce")
        scores = numeric_col.dropna()
        if len(scores) > 0:
            avg_score = scores.mean()
            valid_subjects.append(col_name)
            valid_scores[col_name] = avg_score

    # Row‑based approach – filtered to academic subjects only, plus exclude 'club'
    if eff_label_col and eff_score_col:
        # If explicit axes are given, we override column‑based results
        if x_col and y_col:
            valid_subjects = []
            valid_scores = {}

        work_df = df.copy()
        work_df[eff_score_col] = pd.to_numeric(work_df[eff_score_col], errors="coerce")
        work_df = work_df.dropna(subset=[eff_score_col])

        for _, row in work_df.iterrows():
            raw_label = row.get(eff_label_col, "")
            if pd.isna(raw_label):
                continue
            label_text = str(raw_label).strip()
            if not label_text:
                continue
            if contains_metadata_keyword(label_text):
                continue

            # NEW: Exclude anything with 'club' in the name (case‑insensitive)
            if re.search(r'\bclub\b', label_text, re.IGNORECASE):
                continue

            # --- CRITICAL FILTER: Only academic subjects ---
            section = str(row.get("Section", "")).strip()
            is_academic_subject = False

            # 1. Section explicitly says "Subject" (or "Subjects")
            if re.search(r"\bSubject", section, re.IGNORECASE):
                is_academic_subject = True
            # 2. Label itself is in known subject list
            elif is_valid_subject(label_text):
                is_academic_subject = True
            else:
                # 3. Check if any known subject appears inside the label
                for subj in sorted(KNOWN_SUBJECTS_CS, key=lambda s: -len(s)):
                    if subj.lower() in label_text.lower():
                        is_academic_subject = True
                        break

            # Skip behaviour, co‑curricular, and other non‑academic rows
            if not is_academic_subject:
                continue

            score = row[eff_score_col]

            # Extract subject name
            matched = None
            if is_valid_subject(label_text):
                matched = label_text
            else:
                for subj in sorted(KNOWN_SUBJECTS_CS, key=lambda s: -len(s)):
                    if subj.lower() in label_text.lower():
                        matched = subj
                        break
            if matched:
                valid_subjects.append(matched)
                valid_scores[matched] = score
                continue

            # Fallback: last word if it looks like a subject
            tokens = re.split(r"[\s:/\-()]+", label_text)
            tokens = [t for t in tokens if t]
            last_word = tokens[-1] if tokens else ""
            if last_word and is_valid_subject(last_word):
                # Also check fallback for 'club'
                if not re.search(r'\bclub\b', last_word, re.IGNORECASE):
                    valid_subjects.append(last_word)
                    valid_scores[last_word] = score

    # Average scores if duplicate subjects appear
    temp_scores = {}
    for subj, score in zip(valid_subjects, [valid_scores[s] for s in valid_subjects]):
        if subj not in temp_scores:
            temp_scores[subj] = []
        temp_scores[subj].append(score)
    final_scores = {subj: sum(scores) / len(scores) for subj, scores in temp_scores.items()}
    final_subjects = list(final_scores.keys())

    stats["subjects"] = final_subjects
    stats["subject_scores"] = final_scores

    if final_scores:
        stats["strength"] = max(final_scores, key=final_scores.get)
        stats["weakness"] = min(final_scores, key=final_scores.get)

    # Activities extraction (co‑curricular) remains separate – unchanged
    activity_label_col = "Label" if "Label" in df.columns else eff_label_col
    valid_activities = []
    for _, row in df.iterrows():
        raw_label = row.get(activity_label_col, "") or "" if activity_label_col else ""
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
            if re.search(
                    r"Co-?curricular|Co Curricular|Activity|Activities",
                    str(section),
                    flags=re.IGNORECASE,
            ):
                if not contains_metadata_keyword(part):
                    valid_activities.append(part)
    stats["activities"] = valid_activities

    return stats
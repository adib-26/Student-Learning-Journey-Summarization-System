# backend/data_processing.py
import re
import pandas as pd
from typing import Optional, Dict, Any, List

try:
    import enchant
    ENCHANT_AVAILABLE = True
    EN_DICT = enchant.Dict("en_US")
except ImportError:
    ENCHANT_AVAILABLE = False
    EN_DICT = None

# -------------------------------------------------
# Constants (case-sensitive variants)
# -------------------------------------------------
ENGLISH_COMMON_WORDS = {
    "name", "student", "school", "state", "gender", "male", "female",
    "form", "level", "nationality", "secondary", "primary", "class",
    "grade", "section", "age", "year", "date", "address", "phone",
    "email", "father", "mother", "guardian", "contact", "code"
}
ENGLISH_COMMON_WORDS_CS = {w.title() for w in ENGLISH_COMMON_WORDS}

METADATA_KEYWORDS = {
    "name", "student", "school", "state", "gender", "male", "female",
    "form", "level", "nationality", "class", "grade", "section", "age",
    "year", "date", "address", "phone", "email", "behaviour", "behavior",
    "attentiveness", "participation", "attendance", "punctuality",
    "discipline", "ratings", "father", "mother", "guardian", "parent",
    "contact", "code", "id", "number", "admission", "roll"
}
METADATA_KEYWORDS_CS = {k.title() for k in METADATA_KEYWORDS}

KNOWN_SUBJECTS = {
    "mathematics", "math", "maths", "science", "physics", "chemistry",
    "biology", "history", "geography", "english", "language", "languages",
    "malay", "bahasa", "chinese", "mandarin", "tamil", "arabic",
    "physical education", "pe", "art", "music", "literature",
    "economics", "accounting", "business", "computer", "ict",
    "additional mathematics", "add math", "moral", "pendidikan",
    "sejarah", "sains", "matematik"
}
KNOWN_SUBJECTS_CS = {s.title() for s in KNOWN_SUBJECTS}

CO_CURRICULAR_KEYWORDS = {
    "member", "club", "society", "team", "day", "competition",
    "event", "activity", "activities", "award", "prize",
    "position", "role", "committee", "group", "association"
}
CO_CURRICULAR_KEYWORDS_CS = {w.title() for w in CO_CURRICULAR_KEYWORDS}


# -------------------------------------------------
# Utility Functions (ORIGINAL - UNCHANGED)
# -------------------------------------------------
def is_english_word(word: str) -> bool:
    if not word or not word.isalpha():
        return False
    if ENCHANT_AVAILABLE and EN_DICT:
        try:
            return EN_DICT.check(word)
        except Exception:
            return False
    else:
        return word in ENGLISH_COMMON_WORDS_CS


def looks_like_name(text: str) -> bool:
    tokens = re.findall(r"[A-Z][a-zA-Z]+", text)
    if len(tokens) < 2:
        return False
    non_common = [t for t in tokens if t not in ENGLISH_COMMON_WORDS_CS]
    if len(non_common) >= 2:
        return True
    if len(tokens) >= 2:
        dict_check = [is_english_word(t) for t in tokens[:3]]
        if True in dict_check and False in dict_check:
            return True
    return False


def extract_full_name(text: str) -> Optional[str]:
    """
    ORIGINAL FUNCTION - Extract full name from structured text.
    Works for formats like "Name: Ahmad Daniel" or "Student Name Ahmad Daniel"
    """
    if not text:
        return None

    text = re.sub(r"\s+", " ", text).strip()

    match = re.search(
        r"(Student\s+Name|Name)\s*[:\-]?\s*(.+)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        remainder = match.group(2)
    else:
        remainder = text

    tokens = re.findall(r"[A-Z][a-zA-Z]+", remainder)
    if len(tokens) < 2:
        return None

    name_parts = []
    for token in tokens:
        if (token in METADATA_KEYWORDS_CS
                or token in KNOWN_SUBJECTS_CS
                or token in CO_CURRICULAR_KEYWORDS_CS):
            break
        name_parts.append(token)

    if len(name_parts) >= 2:
        return " ".join(name_parts)

    return None


# -------------------------------------------------
# IMPROVED GENDER EXTRACTION (the main fix)
# -------------------------------------------------
def extract_gender(text: str) -> Optional[str]:
    """
    Improved gender extraction from raw text.
    Handles: Male, Female, M, F, 'L' (for Lelaki), 'P' (Perempuan), etc.
    Returns standardized 'Male' or 'Female'.
    """
    if not text:
        return None

    # Lowercase for case‑insensitive matching
    lower_text = text.lower()

    # Common patterns: label + value
    patterns = [
        r'\bgender\s*[:\-]?\s*(male|female|m|f|lelaki|perempuan)\b',
        r'\bsex\s*[:\-]?\s*(male|female|m|f)\b',
        r'\b(jantina)\s*[:\-]?\s*(lelaki|perempuan|l|p)\b',
        r'\b(male|female|m|f)\b',   # simple standalone
    ]

    for pat in patterns:
        match = re.search(pat, lower_text)
        if match:
            raw = match.group(1) if len(match.groups()) >= 1 else match.group(0)
            if raw in ('male', 'm', 'lelaki', 'l'):
                return 'Male'
            if raw in ('female', 'f', 'perempuan', 'p'):
                return 'Female'

    return None


def extract_gender_from_label_value(df: pd.DataFrame,
                                     label_col: Optional[str] = None,
                                     value_col: Optional[str] = None) -> Optional[str]:
    """
    Extract gender directly from a Label/Value structured DataFrame.
    Looks for rows where label contains 'gender' or 'sex' (case‑insensitive)
    and returns the corresponding value.
    """
    if df is None or df.empty:
        return None

    # Auto‑detect Label/Value columns if not provided
    if label_col is None or value_col is None:
        for col in df.columns:
            col_lower = str(col).strip().lower()
            if col_lower == 'label':
                label_col = col
            elif col_lower == 'value':
                value_col = col

    if label_col is not None and value_col is not None:
        for _, row in df.iterrows():
            label = str(row[label_col]).strip().lower() if pd.notna(row[label_col]) else ''
            value = str(row[value_col]).strip() if pd.notna(row[value_col]) else ''
            if 'gender' in label or 'sex' in label:
                # Use the improved extract_gender on the value
                return extract_gender(value)
    return None


def extract_state(text: str) -> Optional[str]:
    match = re.search(r"\bState\s+([A-Za-z]+)", text)
    if not match:
        return None
    state = match.group(1)
    if state == "negeri":
        return "Negeri Sembilan"
    return state


def contains_metadata_keyword(text: str) -> bool:
    if not text:
        return False
    for keyword in METADATA_KEYWORDS_CS:
        if re.search(rf"\b{re.escape(keyword)}\b", text):
            return True
    return False


def is_valid_subject(label: str) -> bool:
    if not label or not isinstance(label, str):
        return False
    label_stripped = label.strip()
    for subj in KNOWN_SUBJECTS_CS:
        if subj in label_stripped or label_stripped in subj:
            return True
    if contains_metadata_keyword(label_stripped):
        return False
    word_count = len(label_stripped.split())
    if word_count > 4:
        return False
    if word_count <= 3:
        return True
    return False


def contains_co_curricular_keyword(text: str) -> bool:
    if not text:
        return False
    for keyword in CO_CURRICULAR_KEYWORDS_CS:
        if re.search(rf"\b{re.escape(keyword)}\b", text):
            return True
    return False


# -------------------------------------------------
# NEW FUNCTIONS for Tabular Data (Excel/CSV)
# -------------------------------------------------
def extract_name_from_columns(df: pd.DataFrame) -> Optional[str]:
    """
    Extract student name from DataFrame where name might be in column headers.
    """
    for col in df.columns:
        if pd.isna(col) or str(col).startswith('Unnamed'):
            continue
        col_str = str(col).strip()
        if re.search(r'^(Student\s+)?Name$', col_str, re.IGNORECASE):
            continue
        tokens = re.findall(r"[A-Z][a-zA-Z]+", col_str)
        if len(tokens) >= 2:
            name_tokens = [
                t for t in tokens
                if t not in METADATA_KEYWORDS_CS
                   and t not in KNOWN_SUBJECTS_CS
                   and t not in CO_CURRICULAR_KEYWORDS_CS
            ]
            if len(name_tokens) >= 2:
                return " ".join(name_tokens)
    return None


def extract_name_from_row(row: pd.Series, label_col: str = None, value_col: str = None) -> Optional[str]:
    """
    Extract name from a DataFrame row where label and value are in separate columns.
    """
    if label_col and value_col:
        if label_col in row and value_col in row:
            label = str(row[label_col]).strip()
            if re.search(r'(Student\s+)?Name', label, re.IGNORECASE):
                value = str(row[value_col]).strip()
                if value and value != 'nan':
                    return value
    for i, (label, value) in enumerate(zip(row.index, row.values)):
        label_str = str(label).strip()
        value_str = str(value).strip()
        if re.search(r'(Student\s+)?Name', label_str, re.IGNORECASE):
            if i + 1 < len(row):
                next_val = str(row.iloc[i + 1]).strip()
                if next_val and next_val != 'nan':
                    return next_val
            elif value_str and value_str != 'nan' and value_str.lower() != label_str.lower():
                return value_str
    return None


def parse_tabular_student_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Parse student data from tabular format (Excel/CSV).
    Now includes robust gender extraction using the improved function
    and explicit Label/Value structure detection.
    """
    result = {
        'name': None,
        'gender': None,
        'nationality': None,
        'school_level': None,
        'form': None,
        'state': None,
        'subjects': [],
        'co_curricular': []
    }

    # ----- 1. Try Label/Value structure first (most reliable) -----
    label_col = None
    value_col = None
    for col in df.columns:
        col_lower = str(col).strip().lower()
        if col_lower == 'label':
            label_col = col
        elif col_lower == 'value':
            value_col = col

    if label_col is not None and value_col is not None:
        # Extract all fields systematically
        for _, row in df.iterrows():
            label = str(row[label_col]).strip().lower() if pd.notna(row[label_col]) else ''
            value = str(row[value_col]).strip() if pd.notna(row[value_col]) else ''
            if not value or value.lower() in METADATA_KEYWORDS:
                continue

            if 'student name' in label or 'name' == label:
                result['name'] = value
            elif 'gender' in label or 'sex' in label:
                result['gender'] = extract_gender(value)  # uses improved function
            elif 'nationality' in label:
                result['nationality'] = value
            elif 'school level' in label:
                result['school_level'] = value
            elif 'form' in label:
                result['form'] = value
            elif 'state' in label:
                result['state'] = value
            elif 'attendance' in label:
                result['attendance'] = value  # optional extra field
            elif is_valid_subject(label):
                result['subjects'].append({'name': label, 'value': value})
            elif contains_co_curricular_keyword(label):
                result['co_curricular'].append({'name': label, 'value': value})

        # If gender still missing, try the dedicated helper
        if not result['gender']:
            result['gender'] = extract_gender_from_label_value(df, label_col, value_col)

    # ----- 2. Fallback: row‑by‑row scanning (original method) -----
    if not result['gender'] or not result['name']:
        for idx, row in df.iterrows():
            row_str = ' '.join([str(v) for v in row.values if pd.notna(v)])

            if not result['name']:
                # Try column‑based name extraction
                name_candidate = extract_name_from_row(row)
                if name_candidate:
                    result['name'] = name_candidate
                else:
                    # Fallback to old extract_full_name
                    name_candidate = extract_full_name(row_str)
                    if name_candidate:
                        result['name'] = name_candidate

            if not result['gender']:
                gender = extract_gender(row_str)
                if gender:
                    result['gender'] = gender

            if not result['nationality'] and 'nationality' in row_str.lower():
                for val in row.values:
                    if pd.notna(val) and str(val).strip().lower() not in ['nationality', 'nan']:
                        result['nationality'] = str(val).strip()
                        break

            if not result['school_level'] and 'school level' in row_str.lower():
                for val in row.values:
                    if pd.notna(val) and 'school' in str(val).lower():
                        result['school_level'] = str(val).strip()
                        break

            if not result['form'] and 'form' in row_str.lower() and not result['form']:
                for val in row.values:
                    if pd.notna(val) and 'form' in str(val).lower():
                        result['form'] = str(val).strip()
                        break

            if not result['state']:
                for val in row.values:
                    if pd.notna(val):
                        val_str = str(val).strip()
                        malaysian_states = {'Selangor', 'Johor', 'Penang', 'Perak', 'Kedah',
                                            'Kelantan', 'Terengganu', 'Pahang', 'Negeri Sembilan',
                                            'Melaka', 'Sabah', 'Sarawak', 'Perlis', 'Putrajaya',
                                            'Kuala Lumpur', 'Labuan'}
                        if val_str in malaysian_states:
                            result['state'] = val_str
                            break

    # Final cleanup: if gender still not found, try a global scan of all text
    if not result['gender']:
        all_text = ' '.join(df.astype(str).values.flatten())
        result['gender'] = extract_gender(all_text)

    return result


# -------------------------------------------------
# Exports
# -------------------------------------------------
__all__ = [
    "extract_full_name",
    "looks_like_name",
    "extract_gender",
    "extract_gender_from_label_value",
    "extract_state",
    "is_valid_subject",
    "contains_metadata_keyword",
    "contains_co_curricular_keyword",
    "KNOWN_SUBJECTS_CS",
    "extract_name_from_columns",
    "extract_name_from_row",
    "parse_tabular_student_data",
]
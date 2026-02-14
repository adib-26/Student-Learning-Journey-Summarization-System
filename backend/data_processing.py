# backend/data_processing.py
import re
import pandas as pd
from typing import Optional, Dict, Any

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
    tokens = re.findall(r"[A-Z][a-zA-Z]+", text)  # allow all-caps too
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

    # Normalize spacing
    text = re.sub(r"\s+", " ", text).strip()

    # Case 1: structured label anywhere in the line (not just prefix)
    match = re.search(
        r"(Student\s+Name|Name)\s*[:\-]?\s*(.+)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        remainder = match.group(2)
    else:
        remainder = text

    # Tokenize capitalized words
    tokens = re.findall(r"[A-Z][a-zA-Z]+", remainder)
    if len(tokens) < 2:
        return None

    name_parts = []
    for token in tokens:
        # Stop on metadata / subjects (hard boundary)
        if (
                token in METADATA_KEYWORDS_CS
                or token in KNOWN_SUBJECTS_CS
                or token in CO_CURRICULAR_KEYWORDS_CS
        ):
            break
        name_parts.append(token)

    if len(name_parts) >= 2:
        return " ".join(name_parts)

    return None


def extract_gender(text: str) -> Optional[str]:
    match = re.search(r"\b(Male|Female|Prefer not to say)\b", text, flags=re.IGNORECASE)
    return match.group(1).title() if match else None


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

    Args:
        df: DataFrame loaded from Excel/CSV

    Returns:
        Student name or None

    Example:
        Column headers: ['Student Name', 'Ahmad Daniel Bin Hassan', 'Unnamed: 2']
        Returns: 'Ahmad Daniel Bin Hassan'
    """
    # Check column headers for name
    for col in df.columns:
        if pd.isna(col) or str(col).startswith('Unnamed'):
            continue

        col_str = str(col).strip()

        # Skip the label column itself
        if re.search(r'^(Student\s+)?Name$', col_str, re.IGNORECASE):
            continue

        # Check if this column looks like a name
        tokens = re.findall(r"[A-Z][a-zA-Z]+", col_str)
        if len(tokens) >= 2:
            # Filter out common words
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

    Args:
        row: A pandas Series representing one row
        label_col: Name of the column containing labels (e.g., 'Label', 'Field')
        value_col: Name of the column containing values (e.g., 'Value', 'Data')

    Returns:
        Student name or None

    Example:
        row = {'Label': 'Student Name', 'Value': 'Ahmad Daniel'}
        Returns: 'Ahmad Daniel'
    """
    # If specific columns are provided
    if label_col and value_col:
        if label_col in row and value_col in row:
            label = str(row[label_col]).strip()
            if re.search(r'(Student\s+)?Name', label, re.IGNORECASE):
                value = str(row[value_col]).strip()
                if value and value != 'nan':
                    return value

    # Otherwise, search through all columns
    for i, (label, value) in enumerate(zip(row.index, row.values)):
        label_str = str(label).strip()
        value_str = str(value).strip()

        # Check if label indicates this is a name field
        if re.search(r'(Student\s+)?Name', label_str, re.IGNORECASE):
            # Get the value from the next column if current is the label
            if i + 1 < len(row):
                next_val = str(row.iloc[i + 1]).strip()
                if next_val and next_val != 'nan':
                    return next_val
            # Or return current value if it's not the label itself
            elif value_str and value_str != 'nan' and value_str.lower() != label_str.lower():
                return value_str

    return None


def parse_tabular_student_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Parse student data from tabular format (Excel/CSV) where data might be:
    - In column headers (e.g., 'Student Name' | 'Ahmad Daniel' | ...)
    - In rows with label-value pairs

    Args:
        df: DataFrame loaded from Excel/CSV

    Returns:
        Dictionary with extracted student information
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

    # Try to extract name from column headers first
    result['name'] = extract_name_from_columns(df)

    # Iterate through rows to extract other data
    for idx, row in df.iterrows():
        # Convert row to string for easier searching
        row_str = ' '.join([str(v) for v in row.values if pd.notna(v)])

        # Extract metadata if name not found yet
        if not result['name']:
            # Try each column as potential label-value pair
            for col_idx, (col_name, value) in enumerate(row.items()):
                if pd.isna(value):
                    continue

                col_str = str(col_name).strip()
                val_str = str(value).strip()

                if re.search(r'(Student\s+)?Name', col_str, re.IGNORECASE):
                    if val_str and not re.search(r'(Student\s+)?Name', val_str, re.IGNORECASE):
                        result['name'] = val_str

        # Extract gender
        if not result['gender']:
            gender = extract_gender(row_str)
            if gender:
                result['gender'] = gender

        # Extract other metadata
        if 'nationality' in row_str.lower() or 'Nationality' in row_str:
            for val in row.values:
                if pd.notna(val) and str(val).strip().lower() not in ['nationality', 'nan']:
                    result['nationality'] = str(val).strip()
                    break

        if 'school level' in row_str.lower():
            for val in row.values:
                if pd.notna(val) and 'school' in str(val).lower():
                    result['school_level'] = str(val).strip()
                    break

        if 'form' in row_str.lower() and 'form' not in result['form'] if result['form'] else True:
            for val in row.values:
                if pd.notna(val) and 'form' in str(val).lower():
                    result['form'] = str(val).strip()
                    break

        if not result['state']:
            for val in row.values:
                if pd.notna(val):
                    val_str = str(val).strip()
                    # Check for Malaysian states
                    if val_str in ['Selangor', 'Johor', 'Penang', 'Perak', 'Kedah',
                                   'Kelantan', 'Terengganu', 'Pahang', 'Negeri Sembilan',
                                   'Melaka', 'Sabah', 'Sarawak', 'Perlis', 'Putrajaya',
                                   'Kuala Lumpur', 'Labuan']:
                        result['state'] = val_str
                        break

    return result


# -------------------------------------------------
# Exports
# -------------------------------------------------
__all__ = [
    # Original functions
    "extract_full_name",
    "looks_like_name",
    "extract_gender",
    "extract_state",
    "is_valid_subject",
    "contains_metadata_keyword",
    "contains_co_curricular_keyword",
    "KNOWN_SUBJECTS_CS",
    # New functions for tabular data
    "extract_name_from_columns",
    "extract_name_from_row",
    "parse_tabular_student_data",
]
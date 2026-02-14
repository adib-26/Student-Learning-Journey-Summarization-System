# backend/ocr_parser.py
import re
import pandas as pd

try:
    from .data_processing import (
        METADATA_KEYWORDS_CS,
        KNOWN_SUBJECTS_CS,
        ENGLISH_COMMON_WORDS_CS,
        contains_metadata_keyword
    )
except ImportError:
    # Fallback for standalone testing
    ENGLISH_COMMON_WORDS_CS = {
        "Name", "Student", "School", "State", "Gender", "Male", "Female",
        "Form", "Level", "Nationality", "Secondary", "Primary", "Class",
        "Grade", "Section", "Age", "Year", "Date", "Address", "Phone",
        "Email", "Father", "Mother", "Guardian", "Contact", "Code"
    }
    METADATA_KEYWORDS_CS = {
        "Name", "Student", "School", "State", "Gender", "Male", "Female",
        "Form", "Level", "Nationality", "Class", "Grade", "Section", "Age",
        "Year", "Date", "Address", "Phone", "Email", "Behaviour", "Behavior",
        "Attentiveness", "Participation", "Attendance", "Punctuality",
        "Discipline", "Ratings", "Father", "Mother", "Guardian", "Parent",
        "Contact", "Code", "Id", "Number", "Admission", "Roll"
    }
    KNOWN_SUBJECTS_CS = {
        "Mathematics", "Math", "Maths", "Science", "Physics", "Chemistry",
        "Biology", "History", "Geography", "English", "Language", "Languages",
        "Malay", "Bahasa", "Chinese", "Mandarin", "Tamil", "Arabic",
        "Physical Education", "Pe", "Art", "Music", "Literature",
        "Economics", "Accounting", "Business", "Computer", "Ict",
        "Additional Mathematics", "Add Math", "Moral", "Pendidikan",
        "Sejarah", "Sains", "Matematik"
    }


    def contains_metadata_keyword(text):
        return False


def extract_student_name_from_ocr(text: str) -> str:
    """
    Extract student name from OCR text after "Name" keyword.
    Stops at metadata keywords, subject names, or English common words.

    Example: "Name Mahbub English Hasan" -> "Mahbub" (stops at "English")
    """
    # All stop words (combined from data_processing.py)
    stop_words = METADATA_KEYWORDS_CS | KNOWN_SUBJECTS_CS | ENGLISH_COMMON_WORDS_CS

    for line in text.split('\n'):
        line = line.strip()

        if 'Name' in line:
            match = re.search(r'\bName[:\s]+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)', line)
            if match:
                candidate = match.group(1).strip()
                words = []
                for word in candidate.split():
                    # Stop at stop words, numbers, or scores
                    if word in stop_words or re.match(r'\d+', word) or '/' in word:
                        break
                    words.append(word)

                if len(words) >= 1:  # At least first name
                    return ' '.join(words)
    return None


def parse_line_with_metadata_and_score(line: str):
    """
    Parse lines with metadata fields and optional scores.

    Handles formats:
    - "Name: Arif Bin Hassan" or "Name Arif Bin Hassan"
    - "Gender: Male" or "Gender Male"
    - "Nationality: Malaysian" or "Nationality Malaysian"
    - "School Level: Secondary (High School)"
    - "Form: Form 4" or "Form 4"
    - "State: Selangor" or "State Selangor"
    - "Name Arif Bin Hassan Languages 74/100" (mixed line)

    Returns: (metadata_dict, subject, score, maximum)
    """
    metadata = {}
    original_line = line

    # Extract Student Name
    if 'Name' in line:
        # Pattern: Name followed by capitalized words, stops at stop words
        name_match = re.search(r'Name[:\s]+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)', line)
        if name_match:
            candidate = name_match.group(1).strip()
            # Filter out stop words
            stop_words = METADATA_KEYWORDS_CS | KNOWN_SUBJECTS_CS | ENGLISH_COMMON_WORDS_CS
            words = []
            for word in candidate.split():
                if word in stop_words or re.match(r'\d+', word):
                    break
                words.append(word)

            if words:
                name_value = ' '.join(words)
                metadata['Student Name'] = name_value
                # Remove from line
                line = re.sub(r'Name[:\s]+' + re.escape(name_value), '', line).strip()

    # Extract Gender
    if 'Gender' in line:
        gender_match = re.search(r'Gender[:\s]+(Male|Female)', line, re.IGNORECASE)
        if gender_match:
            metadata['Gender'] = gender_match.group(1).title()
            line = re.sub(r'Gender[:\s]+(Male|Female)', '', line, flags=re.IGNORECASE).strip()

    # Extract Nationality
    if 'Nationality' in line:
        nationality_match = re.search(r'Nationality[:\s]+([A-Za-z]+)', line)
        if nationality_match:
            metadata['Nationality'] = nationality_match.group(1)
            line = re.sub(r'Nationality[:\s]+[A-Za-z]+', '', line).strip()

    # Extract School Level (can have spaces and parentheses)
    if 'School Level' in line:
        # Match everything after "School Level:" until we hit another keyword or end
        school_match = re.search(r'School Level[:\s]+(.+)', line)
        if school_match:
            school_value = school_match.group(1).strip()

            # Stop at next keyword if present
            for stop_word in ['Form', 'State', 'Gender', 'Nationality']:
                if stop_word in school_value:
                    # Split and take everything before the stop word
                    parts = re.split(r'\s+' + stop_word + r'[:\s]', school_value)
                    school_value = parts[0].strip()
                    break

            metadata['School Level'] = school_value
            line = line.replace('School Level: ' + school_value, '').replace('School Level ' + school_value, '').strip()

    # Extract Form
    if 'Form' in line and 'School' not in line:  # Avoid matching "School Level: Secondary (High School) Form"
        form_match = re.search(r'Form[:\s]+(Form\s+)?([0-9]+)', line)
        if form_match:
            # Normalize to "Form X" format
            form_num = form_match.group(2)
            metadata['Form'] = f'Form {form_num}'
            line = re.sub(r'Form[:\s]+(Form\s+)?[0-9]+', '', line).strip()

    # Extract State (handle multi-word Malaysian states)
    if 'State' in line:
        # Malaysian states list
        malaysian_states = [
            'Negeri Sembilan',
            'Selangor', 'Johor', 'Penang', 'Perak', 'Kedah',
            'Kelantan', 'Terengganu', 'Pahang', 'Melaka',
            'Sabah', 'Sarawak', 'Perlis', 'Putrajaya', 'Labuan'
        ]

        # Try to match multi-word states first
        state_match = re.search(r'State[:\s]+((?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)+)', line)
        if state_match:
            state_candidate = state_match.group(1).strip()

            # Check for "Negeri" -> should be "Negeri Sembilan"
            if state_candidate == 'Negeri' or state_candidate.startswith('Negeri'):
                # Look for "Sembilan" in the line
                if 'Sembilan' in line:
                    metadata['State'] = 'Negeri Sembilan'
                    line = re.sub(r'State[:\s]+Negeri\s+Sembilan', '', line).strip()
                else:
                    metadata['State'] = 'Negeri Sembilan'
                    line = re.sub(r'State[:\s]+Negeri', '', line).strip()
            # Check if it's a known Malaysian state
            elif state_candidate in malaysian_states:
                metadata['State'] = state_candidate
                line = re.sub(r'State[:\s]+' + re.escape(state_candidate), '', line).strip()
            # Otherwise take first word only
            else:
                state_word = state_candidate.split()[0]
                metadata['State'] = state_word
                line = re.sub(r'State[:\s]+' + re.escape(state_word), '', line).strip()

    # Now check for score in remaining line
    score_match = re.search(r'([A-Za-z\s]+?)\s+(\d{1,3})\s*/\s*(\d{1,4})', line)
    if score_match:
        subject = score_match.group(1).strip()
        score = int(score_match.group(2))
        maximum = int(score_match.group(3))
        return metadata, subject, score, maximum

    return metadata, None, None, None


def parse_ocr_text_to_dataframe(text: str) -> pd.DataFrame:
    """
    Parse OCR text into structured DataFrame.

    Extracts all student metadata:
    - Student Name (stops at keywords)
    - Gender
    - Nationality
    - School Level
    - Form
    - State

    Also extracts subjects, scores, behaviour, and co-curricular activities.
    """
    rows = []
    current_section = None

    # Extract student name first
    student_name = extract_student_name_from_ocr(text)
    if student_name:
        rows.append({
            "Section": "Student Details",
            "Label": "Student Name",
            "Value": student_name
        })

    section_headers = {
        "subjects": re.compile(r"\b(Subjects?|Subject\s+Scores?)\b"),
        "co_curricular": re.compile(r"\b(Co-?curricular|Achievement)\b", re.IGNORECASE),
        "behaviour": re.compile(r"\b(Behaviour|Behavior|Ratings)\b"),
        "student": re.compile(r"\b(Student|Details|Betalls)\b"),
    }

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or 'Malaysian High School' in line or 'Student Betalls' in line or 'Subject Scores' in line:
            continue

        # Detect section headers
        for name, pattern in section_headers.items():
            if pattern.search(line):
                current_section = name.replace("_", " ").title()
                break

        # Handle lines with "|" separator (parallel columns)
        if '|' in line:
            parts = line.split('|')
            for part in parts:
                part = part.strip()
                if not part:
                    continue

                # Check for scores
                score_match = re.search(r'([A-Za-z\s]+?)\s+(\d{1,3})\s*/\s*(\d{1,4})', part)
                if score_match:
                    rows.append({
                        "Section": "Subjects",
                        "Label": score_match.group(1).strip(),
                        "Score": int(score_match.group(2)),
                        "Maximum": int(score_match.group(3))
                    })
                else:
                    # Behaviour rating
                    kv = re.match(r'^([A-Za-z\s]+?)\s+([A-Za-z]+)$', part)
                    if kv:
                        rows.append({
                            "Section": "Behaviour",
                            "Label": kv.group(1).strip(),
                            "Value": kv.group(2).strip()
                        })
            continue

        # Parse lines with metadata and optional scores
        metadata, subject, score, maximum = parse_line_with_metadata_and_score(line)

        # Add metadata
        for key, value in metadata.items():
            if key == 'Student Name' and student_name:
                continue  # Already added
            rows.append({
                "Section": "Student Details",
                "Label": key,
                "Value": value
            })

        # Add subject score
        if subject and score and maximum:
            rows.append({
                "Section": "Subjects",
                "Label": subject,
                "Score": score,
                "Maximum": maximum
            })
            continue

        # Simple score line (no metadata)
        score_match = re.search(r'^([A-Za-z\s]+?)\s+(\d{1,3})\s*/\s*(\d{1,4})$', line)
        if score_match:
            rows.append({
                "Section": "Subjects",
                "Label": score_match.group(1).strip(),
                "Score": int(score_match.group(2)),
                "Maximum": int(score_match.group(3))
            })
            continue

        # Co-curricular items
        if any(keyword in line for keyword in ['Club', 'Winner', 'Member', 'Team', 'Award']):
            rows.append({
                "Section": "Co-curricular",
                "Label": line
            })

    # Create DataFrame
    df_out = pd.DataFrame(rows)
    if df_out.empty:
        return pd.DataFrame(columns=["Section", "Label", "Value", "Score", "Maximum"])

    df_out = df_out.dropna(axis=1, how='all')
    canonical_order = ["Section", "Label", "Value", "Score", "Maximum"]
    df_out = df_out[[c for c in canonical_order if c in df_out.columns]]

    if "Score" in df_out.columns:
        df_out["Score"] = pd.to_numeric(df_out["Score"], errors="coerce")
    if "Maximum" in df_out.columns:
        df_out["Maximum"] = pd.to_numeric(df_out["Maximum"], errors="coerce")

    return df_out


__all__ = ["parse_ocr_text_to_dataframe", "extract_student_name_from_ocr"]
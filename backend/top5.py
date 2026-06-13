import re
import json
import pandas as pd
import streamlit as st

# Import your active global translation engine safely
from backend.deepl_translator import translator
# Import secure Gemini client for AI-powered fallback
from .secure_gemini_client import SecureGeminiClient

# Initialize secure Gemini client for fallback extraction
secure_client = SecureGeminiClient()

# -------------------------------------------------
# EXTENDED 2-WORD HIGH SCHOOL SUBJECTS (SAFE LIST)
# -------------------------------------------------
TWO_WORD_SUBJECTS = [
    "bahasa malaysia",
    "physical education",
    "social science",
    "computer science",
    "moral education",
    "additional mathematics",
    "general science",
    "environmental science",
    "information technology",
    "class participation",
    "community service",
    "chess club",
    "football",
    "club",
    "sejarah (history)"
]


# -------------------------------------------------
# 1. UNIVERSAL DATA EXTRACTOR
# -------------------------------------------------
def extract_numeric_pairs_from_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract label-score pairs from ANY data format.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    pairs = []
    df = df.copy()

    # Strategy 1: Structured data (Excel style)
    if 'Label' in df.columns and 'Score' in df.columns:
        for _, row in df.iterrows():
            label_val = str(row.get('Label', '')).strip()
            score_val = row.get('Score')

            try:
                score_num = float(score_val)
                if score_num > 0:
                    # Check for 2-word subjects
                    label_lower = label_val.lower()
                    is_two_word = False

                    for two_word_subj in TWO_WORD_SUBJECTS:
                        if two_word_subj in label_lower:
                            label_clean = two_word_subj.title()
                            is_two_word = True
                            break

                    if not is_two_word:
                        # Get the main subject word
                        label_clean = label_val.title()  # Default fallback
                        words = label_val.split()
                        if words:
                            skip_words = ['and', 'the', 'for', 'with', 'in', 'on', 'at', 'to', 'of']
                            for word in reversed(words):
                                if word not in skip_words and len(word) > 1:
                                    label_clean = word.title()
                                    break
                            else:
                                label_clean = words[-1].title()

                    pairs.append({
                        'Label': label_clean,
                        'Score': score_num
                    })
            except (ValueError, TypeError):
                pass

    # Strategy 2: OCR-style extraction
    for _, row in df.iterrows():
        for cell in row:
            cell_str = str(cell).strip()
            if not cell_str or cell_str.lower() == 'nan':
                continue

            # Extract patterns like "Subject: 85 / 100"
            patterns = [
                r'([a-z\s]+[a-z]):?\s*(\d+)\s*(?:/\s*\d+)?',
                r'([a-z\s]+[a-z])\s+(\d+)\s*(?:/\s*\d+)?',
                r'([a-z\s]+(?:\([^)]+\))?):?\s*(\d+)',
            ]

            for pattern in patterns:
                matches = re.findall(pattern, cell_str.lower())
                for label, score in matches:
                    if label and score:
                        # Clean the label
                        label_clean = label.strip()

                        # Check if it's a 2-word subject
                        is_two_word = False
                        for two_word_subj in TWO_WORD_SUBJECTS:
                            if two_word_subj in label_clean:
                                label_clean = two_word_subj.title()
                                is_two_word = True
                                break

                        # If not 2-word, take the last word
                        if not is_two_word:
                            words = label_clean.split()
                            if words:
                                skip_words = ['and', 'the', 'for', 'with', 'in', 'on', 'at', 'to', 'of']
                                for word in reversed(words):
                                    if word not in skip_words and len(word) > 1:
                                        label_clean = word.title()
                                        break
                                else:
                                    label_clean = words[-1].title()

                        pairs.append({
                            'Label': label_clean,
                            'Score': int(score)
                        })

    # Convert to DataFrame
    if pairs:
        result_df = pd.DataFrame(pairs)

        # Remove duplicates (keep highest score)
        result_df = result_df.sort_values('Score', ascending=False)
        result_df = result_df.drop_duplicates(subset=['Label'], keep='first')

        return result_df.reset_index(drop=True)

    return pd.DataFrame()


# -------------------------------------------------
# GEMINI-POWERED FALLBACK EXTRACTION
# -------------------------------------------------
def _extract_top5_with_gemini(text_content: str) -> pd.DataFrame:
    """
    AI-powered fallback to extract top 5 subjects/scores when rule-based extraction fails.
    Uses gemini-3.5-flash to intelligently parse unstructured data.
    """
    try:
        prompt = f"""
You will analyze the following student performance data and extract the top 5 academic subjects
with their numeric scores (0-100). Return ONLY a JSON object wrapped between ###JSON_START### and ###JSON_END###.

The JSON must have a single key "top_subjects" which is an array of objects. Each object must have:
- "Label": string - the full subject name (preserve multi-word subjects like "Bahasa Malaysia")
- "Score": integer - the numeric score (must be between 0-100)

Rules:
1. Extract ONLY academic subjects with clear numeric scores
2. Sort them by Score in descending order (highest first)
3. Return maximum 5 subjects
4. If you can't find any academic subjects with scores, return an empty array
5. Do NOT invent scores - only use numbers explicitly present in the text
6. Preserve original subject names exactly as they appear

DOCUMENT TEXT:
{text_content}

Example of correct JSON format:
###JSON_START###
{{
  "top_subjects": [
    {{"Label": "Bahasa Malaysia", "Score": 92}},
    {{"Label": "Mathematics", "Score": 88}},
    {{"Label": "Science", "Score": 85}}
  ]
}}
###JSON_END###
"""

        # Use your existing secure Gemini client to maintain quota protection
        response = secure_client.call_gemini_secure(prompt=prompt, model="gemini-3.5-flash")
        
        if not response:
            return pd.DataFrame()

        # Parse JSON response with the same marker pattern used elsewhere in your codebase
        start_marker = "###JSON_START###"
        end_marker = "###JSON_END###"
        start_idx = response.find(start_marker)
        end_idx = response.find(end_marker, start_idx + len(start_marker)) if start_idx != -1 else -1

        if start_idx != -1 and end_idx != -1:
            json_text = response[start_idx + len(start_marker):end_idx].strip()
            try:
                parsed = json.loads(json_text)
                subjects = parsed.get("top_subjects", [])
                if subjects:
                    return pd.DataFrame(subjects)
            except Exception as e:
                print(f"Gemini JSON parsing error: {e}")

        # Fallback: try to extract JSON directly if markers are missing
        json_match = re.search(r'\{(?:[^{}]|(?R))*\}', response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                subjects = parsed.get("top_subjects", [])
                if subjects:
                    return pd.DataFrame(subjects)
            except Exception:
                pass

        return pd.DataFrame()

    except Exception as e:
        print(f"Gemini extraction failed: {e}")
        return pd.DataFrame()


# -------------------------------------------------
# 2. MAIN TOP 5 EXTRACTOR
# -------------------------------------------------
def get_top5_numerical_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract top 5 performance indicators from ANY document format.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # Clean the dataframe
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    # Remove empty columns
    df = df.dropna(axis=1, how='all')

    if df.empty:
        return pd.DataFrame()

    # Extract numeric pairs
    extracted_df = extract_numeric_pairs_from_data(df)

    # If rule-based extraction fails, try Gemini-powered extraction
    if extracted_df.empty:
        # Convert entire dataframe to text for Gemini to process
        df_text = df.to_string()
        gemini_extracted = _extract_top5_with_gemini(df_text)
        if not gemini_extracted.empty:
            extracted_df = gemini_extracted

    if extracted_df.empty:
        return pd.DataFrame()

    # Get top 5 by score
    top5 = extracted_df.sort_values('Score', ascending=False).head(5)

    return top5[['Label', 'Score']].reset_index(drop=True)


# -------------------------------------------------
# 3. DYNAMICALLY TRANSLATED STREAMLIT UI
# -------------------------------------------------
def show_top5_ui(df: pd.DataFrame):
    """
    Display top 5 results dynamically translated into the active UI language.
    """
    top5 = get_top5_numerical_rows(df)

    if top5.empty:
        st.warning("No numeric performance data found.")
    else:
        # Resolve target language state variable from runtime context
        current_lang = st.session_state.get("selected_language", "en")

        # Clone data blocks to keep source structures clean
        translated_top5 = top5.copy()

        if current_lang != "en":
            # Translate subject tokens row-by-row safely via DeepL CacheManager
            translated_top5['Label'] = translated_top5['Label'].apply(
                lambda x: translator.translate_text(str(x), current_lang)
            )

            # Translate structural column headers ('Label', 'Score') on the fly
            translated_top5.columns = [
                translator.translate_text(col, current_lang) for col in translated_top5.columns
            ]

        st.dataframe(translated_top5, use_container_width=True)
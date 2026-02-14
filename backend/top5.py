import re
import pandas as pd
import streamlit as st

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

    if extracted_df.empty:
        return pd.DataFrame()

    # Get top 5 by score
    top5 = extracted_df.sort_values('Score', ascending=False).head(5)

    return top5[['Label', 'Score']].reset_index(drop=True)


# -------------------------------------------------
# 3. SIMPLE STREAMLIT UI
# -------------------------------------------------
def show_top5_ui(df: pd.DataFrame):
    """
    Display top 5 results in Streamlit.
    """
    top5 = get_top5_numerical_rows(df)

    if top5.empty:
        st.warning("No numeric performance data found.")
    else:
        st.dataframe(top5, use_container_width=True)
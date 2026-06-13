"""
File loading utilities for student data processing.
Supports Excel, CSV, PDF, and image files with cross-language persistence caching.
"""

import pandas as pd
from typing import Tuple, Union, Dict
import io
import re
import streamlit as st

try:
    import pytesseract
    from PIL import Image

    TESSERACT_AVAILABLE = True
except ImportError:
    pytesseract = None
    Image = None
    TESSERACT_AVAILABLE = False


def extract_text_from_image(image_file) -> str:
    """
    Extract text from an uploaded image using OCR (Tesseract).

    Args:
        image_file: Uploaded image file

    Returns:
        Extracted text or error message
    """
    if not TESSERACT_AVAILABLE:
        return "OCR not available - pytesseract not installed"

    try:
        image = Image.open(image_file)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        return f"Error extracting text: {e}"


def extract_metadata_from_excel(df_raw: pd.DataFrame) -> Dict[str, str]:
    """
    Extract metadata from the top rows of an Excel file before the data table.
    Assumes a structure where:
        - Column 0 = Section (e.g., "Student Details", "Subjects")
        - Column 1 = Label (e.g., "Student Name", "Gender")
        - Column 2 = Value (e.g., "Nur Aisyah Binti Rahman", "Female")
    Stops when a row with Column 0 containing "Section" or "Subject" is found.

    Args:
        df_raw: DataFrame read without header (header=None)

    Returns:
        Dictionary with metadata like {"Student Name": "Nur Aisyah Binti Rahman", "Gender": "Female"}
    """
    metadata = {}

    # Known labels we want to extract (case-insensitive mapping to output keys)
    wanted_labels = {
        "student name": "Student Name",
        "gender": "Gender",
        "nationality": "Nationality",
        "school level": "School Level",
        "form": "Form",
        "state": "State",
        "attendance rate (%)": "Attendance Rate (%)",
        "attendance rate": "Attendance Rate (%)"
    }

    # Scan first 20 rows (metadata is usually in the first few rows)
    for idx in range(min(20, len(df_raw))):
        row = df_raw.iloc[idx]
        if len(row) < 3:
            continue

        # Get the first cell (Section column) to detect the table header
        section_cell = str(row.iloc[0]).strip().lower() if pd.notna(row.iloc[0]) else ""

        # Stop when we reach a row that marks the start of the data table
        if section_cell in ["section", "subject", "subjects", "behaviour", "co-curricular"]:
            break

        # Extract label (column 1) and value (column 2)
        label_cell = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""
        value_cell = str(row.iloc[2]).strip() if pd.notna(row.iloc[2]) else ""

        if not label_cell or not value_cell or value_cell.lower() == "nan":
            continue

        # Try to match against known labels
        label_lower = label_cell.lower()
        matched = False
        for key_lower, output_key in wanted_labels.items():
            if key_lower in label_lower:
                metadata[output_key] = value_cell
                matched = True
                break

        # If not a known label but still might be useful (e.g., "School Name"?)
        # But avoid capturing "Student Details" etc.
        if not matched and label_lower not in ["student details", "details"]:
            # Store as is, capitalising the label
            metadata[label_cell] = value_cell

    # Special fallback: if Student Name still missing, brute-force search
    if "Student Name" not in metadata:
        for idx in range(min(20, len(df_raw))):
            row = df_raw.iloc[idx]
            if len(row) < 3:
                continue
            label = str(row.iloc[1]).strip().lower()
            if "student name" in label:
                candidate = str(row.iloc[2]).strip()
                if candidate and candidate.lower() not in ["nan", ""]:
                    metadata["Student Name"] = candidate
                    break

    return metadata


def _fix_pdf_extraction_issues(text: str) -> str:
    """
    Fix common PDF text extraction issues from PyPDF2.
    Adds line breaks where text is incorrectly merged.

    Args:
        text: Raw text extracted from PDF

    Returns:
        Fixed text with proper line breaks
    """
    if not text:
        return text

    # Fix 1: Add line break before "Certificate of Completion"
    text = re.sub(r'([a-z.])(\s*Certificate of Completion)', r'\1\n\2', text, flags=re.IGNORECASE)

    # Fix 2: Add line break before ALL CAPS organization names (3+ words, each 3+ letters)
    text = re.sub(r'([a-z])\s+([A-Z]{3,}\s+[A-Z]{3,}\s+[A-Z]{3,})', r'\1\n\2', text)

    # Fix 3: Add line break before "THIS CERTIFICATE IS PROUDLY PRESENTED"
    text = re.sub(r'([a-z.])\s*(THIS CERTIFICATE IS PROUDLY PRESENTED)', r'\1\n\2', text, flags=re.IGNORECASE)

    # Fix 4: Separate merged names and certificate headers
    text = re.sub(r'([a-z])([A-Z][a-z]+\s+(?:of|is|to|in)\s+)', r'\1\n\2', text)

    # Fix 5: Add space between CamelCase words that got merged (like "ExecutiveLalana")
    text = re.sub(r'([a-z])([A-Z][a-z])', r'\1 \2', text)

    # Fix 6: Add line break after person names before certificate headers
    text = re.sub(r'([A-Z][a-z]+)\s*([A-Z][a-z]+)(Certificate)', r'\1 \2\n\3', text)

    return text


# Caching ensures components remember content arrays across layout reruns
@st.cache_data(show_spinner=False)
def load_file(file_name: str, file_bytes: bytes) -> Tuple[Union[pd.DataFrame, None], Union[dict, str, None]]:
    """
    Cached file loader tracking changes dynamically via filename, size, and binary contents.
    Prevents unneeded file uploads or reprocessing when a user changes UI languages.
    """
    try:
        filename_lower = file_name.lower()

        # --- CSV ---
        if filename_lower.endswith(".csv"):
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(file_bytes), encoding="latin1")
            return df, None

        # --- Excel ---
        elif filename_lower.endswith((".xlsx", ".xls")):
            df_no_header = pd.read_excel(io.BytesIO(file_bytes), header=None)
            metadata = extract_metadata_from_excel(df_no_header)

            data_start_row = None
            for idx in range(min(15, len(df_no_header))):
                row_str = ' '.join(str(cell) for cell in df_no_header.iloc[idx].tolist()
                                   if pd.notna(cell) and str(cell).strip())

                if any(keyword in row_str.lower() for keyword in
                       ['label', 'score', 'subject', 'mark', 'grade', 'result']):
                    if any(keyword in row_str.lower() for keyword in
                           ['maximum', 'total', 'percentage', 'notes']):
                        data_start_row = idx
                        break

            if data_start_row is not None:
                df_data = pd.read_excel(io.BytesIO(file_bytes), header=data_start_row)
            else:
                df_data = pd.read_excel(io.BytesIO(file_bytes))

            return df_data, metadata

        # --- PDF ---
        elif filename_lower.endswith(".pdf"):
            try:
                import pypdf
                pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))

                full_text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        full_text += f"===== Page {page_num + 1} =====\n"
                        full_text += page_text + "\n\n"

                if full_text.strip():
                    full_text = _fix_pdf_extraction_issues(full_text)
                    return None, full_text.strip()
                else:
                    return None, "PDF contains no extractable text (might be scanned images)"

            except ImportError:
                pypdf = None
                return None, "PyPDF not installed. Install with: pip install PyPDF"
            except Exception as e:
                return None, f"Error extracting PDF text: {e}"

        # --- Images ---
        elif filename_lower.endswith((".png", ".jpg", ".jpeg")):
            text = extract_text_from_image(io.BytesIO(file_bytes))
            return None, text

        else:
            return None, "Unsupported file format."

    except Exception as e:
        import traceback
        return None, f"Error loading file: {e}\n{traceback.format_exc()}"


def extract_metadata(df: pd.DataFrame) -> dict:
    """DEPRECATED: Use the metadata returned by load_file() instead."""
    metadata = {}
    for idx, row in df.iterrows():
        first_cell = str(row.iloc[0]).strip().lower() if len(row) > 0 else ""
        if first_cell == "section":
            break
        if len(row) >= 2 and pd.notna(row.iloc[0]) and pd.notna(row.iloc[1]):
            key = str(row.iloc[0]).strip().rstrip(":")
            val = str(row.iloc[1]).strip()
            metadata[key] = val
    return metadata


def preprocess_excel_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and structure an Excel DataFrame that might have metadata."""
    if df is None or df.empty:
        return df

    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    df = df.dropna(how='all').reset_index(drop=True)

    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        except:
            pass

    return df


__all__ = [
    'load_file',
    'extract_text_from_image',
    'extract_metadata_from_excel',
    'extract_metadata',
    'preprocess_excel_dataframe'
]
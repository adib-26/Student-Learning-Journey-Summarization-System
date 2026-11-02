"""
File loading utilities for student data processing.
Supports Excel, CSV, PDF, and image files with cross-language persistence caching.
"""

import pandas as pd
from typing import Tuple, Union, Dict
import io
import logging
import re
import streamlit as st

from backend.mlx_glm_ocr import (
    GlmOcrResult,
    extract_text_with_glm_ocr,
    get_glm_ocr_settings,
)
from backend.file_safety import validate_xlsx_archive


logger = logging.getLogger(__name__)

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
    except Exception:
        logger.warning(
            "Image OCR failed; using the next available fallback", exc_info=True
        )
        return ""


def _extract_pdf_text_from_bytes(file_bytes: bytes, maximum_pages: int = 20) -> str:
    """Bounded native PDF fallback when vLLM vision is unavailable."""
    try:
        import pypdf
        pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        page_count = len(pdf_reader.pages)
        pages = []
        for page_num in range(min(page_count, maximum_pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            if page_text:
                pages.append(f"===== Page {page_num + 1} =====\n{page_text}")
        if pages and page_count > maximum_pages:
            pages.append(
                f"[DOCUMENT PAGE LIMIT REACHED: used the first {maximum_pages} of {page_count} page(s).]"
            )
        return _fix_pdf_extraction_issues("\n\n".join(pages)).strip()
    except Exception:
        return ""


def _is_usable_fallback_text(text: object) -> bool:
    """Keep operational error strings out of the Qwen evidence path."""
    if not isinstance(text, str) or not text.strip():
        return False
    lowered = text.strip().casefold()
    return not lowered.startswith(("ocr not available", "error extracting text"))


def extract_unstructured_text(file_name: str, file_bytes: bytes) -> dict:
    """Use the shared vLLM vision model where needed, then native fallbacks.

    The returned dictionary remains simple enough for Streamlit's per-session
    cache. Born-digital PDF pages keep native text; scanned pages and images use
    the configured multimodal model through its OpenAI-compatible endpoint.
    """
    glm_result: GlmOcrResult = extract_text_with_glm_ocr(file_name, file_bytes)
    # A hard source-size rejection exists to keep one oversized upload from
    # exhausting process memory. Do not defeat that guard by immediately
    # sending the same bytes into a legacy fallback parser.
    if glm_result.error in {
        "document exceeds the configured OCR safety limit",
        "image exceeds OCR safety limit",
    }:
        return glm_result.to_dict()
    if _is_usable_fallback_text(glm_result.text):
        result = glm_result.to_dict()
        # Preserve the project's legacy PDF cleanup rules for both native and
        # mixed native/GLM page blocks before Qwen sees the evidence.
        if (file_name or "").casefold().endswith(".pdf"):
            result["text"] = _fix_pdf_extraction_issues(result["text"])
        return result

    lower_name = (file_name or "").casefold()
    if lower_name.endswith(".pdf"):
        native_text = _extract_pdf_text_from_bytes(
            file_bytes,
            maximum_pages=get_glm_ocr_settings()["max_total_pages"],
        )
        if _is_usable_fallback_text(native_text):
            return {
                "text": native_text,
                "engine": "pypdf-fallback",
                "pages_attempted": glm_result.pages_attempted,
                "pages_completed": glm_result.pages_completed,
                "native_pages": glm_result.native_pages,
                "skipped_pages": glm_result.skipped_pages,
                "truncated": glm_result.truncated or "[DOCUMENT PAGE LIMIT REACHED:" in native_text,
                "error": glm_result.error,
            }
    elif lower_name.endswith((".png", ".jpg", ".jpeg")):
        fallback_text = extract_text_from_image(io.BytesIO(file_bytes))
        if _is_usable_fallback_text(fallback_text):
            return {
                "text": fallback_text.strip(),
                "engine": "tesseract-fallback",
                "pages_attempted": glm_result.pages_attempted,
                "pages_completed": glm_result.pages_completed,
                "native_pages": 0,
                "skipped_pages": 0,
                "truncated": False,
                "error": glm_result.error,
            }
    return glm_result.to_dict()


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
            if filename_lower.endswith(".xlsx"):
                validate_xlsx_archive(file_bytes)
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
            result = extract_unstructured_text(file_name, file_bytes)
            text = result.get("text", "") if isinstance(result, dict) else ""
            if _is_usable_fallback_text(text):
                return None, text.strip()
            return None, "PDF contains no extractable text (might be scanned images)"

        # --- Images ---
        elif filename_lower.endswith((".png", ".jpg", ".jpeg")):
            result = extract_unstructured_text(file_name, file_bytes)
            text = result.get("text", "") if isinstance(result, dict) else ""
            if _is_usable_fallback_text(text):
                return None, text.strip()
            return None, "Image contains no extractable text"

        else:
            return None, "Unsupported file format."

    except Exception as exc:
        logger.warning("Uploaded file could not be loaded", exc_info=True)
        raise ValueError("The uploaded file could not be read. Check its format and try again.") from exc


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
        converted = pd.to_numeric(df[col], errors="coerce")
        if not (df[col].notna() & converted.isna()).any():
            df[col] = converted

    return df


__all__ = [
    'load_file',
    'extract_text_from_image',
    'extract_unstructured_text',
    'extract_metadata_from_excel',
    'extract_metadata',
    'preprocess_excel_dataframe'
]

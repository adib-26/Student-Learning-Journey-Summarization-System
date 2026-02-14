"""
File loading utilities for student data processing.
Supports Excel, CSV, PDF, and image files.
"""

import pandas as pd
from typing import Tuple, Union, Dict
import io
import re

try:
    import pytesseract
    from PIL import Image

    TESSERACT_AVAILABLE = True
except ImportError:
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

    Args:
        df_raw: DataFrame read without header (header=None)

    Returns:
        Dictionary with metadata like {"Student Name": "Ahmad Daniel", "Gender": "Male"}
    """
    metadata = {}

    # Strategy 1: Check if name is in column headers (read normally)
    try:
        df_with_headers = pd.DataFrame(df_raw.values[1:], columns=df_raw.iloc[0])
        for col in df_with_headers.columns:
            if pd.notna(col) and not str(col).startswith('Unnamed'):
                col_str = str(col).strip()
                # Check if this looks like a name (2+ capitalized words)
                tokens = re.findall(r"[A-Z][a-zA-Z]+", col_str)
                if len(tokens) >= 2:
                    # This might be a name in the column header
                    metadata['Student Name'] = col_str
    except:
        pass

    # Strategy 2: Extract from label-value pairs in rows
    for idx in range(min(15, len(df_raw))):
        row = df_raw.iloc[idx]

        # Skip rows that look like table headers
        row_str = ' '.join(str(cell).lower() for cell in row if pd.notna(cell))
        if any(keyword in row_str for keyword in ['section', 'label', 'score', 'maximum']):
            break

        # Extract label-value pairs
        if pd.notna(row.iloc[0]) and pd.notna(row.iloc[1]):
            label = str(row.iloc[0]).strip().rstrip(':')
            value = str(row.iloc[1]).strip()

            # Only store if value is not 'nan' and not the same as label
            if value and value.lower() != 'nan' and value != label:
                metadata[label] = value

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
    # Pattern: lowercase letter followed by space and 3+ all-caps words
    text = re.sub(r'([a-z])\s+([A-Z]{3,}\s+[A-Z]{3,}\s+[A-Z]{3,})', r'\1\n\2', text)

    # Fix 3: Add line break before "THIS CERTIFICATE IS PROUDLY PRESENTED"
    text = re.sub(r'([a-z.])\s*(THIS CERTIFICATE IS PROUDLY PRESENTED)', r'\1\n\2', text, flags=re.IGNORECASE)

    # Fix 4: Separate merged names and certificate headers
    # Pattern: "NameCertificate" → "Name\nCertificate"
    text = re.sub(r'([a-z])([A-Z][a-z]+\s+(?:of|is|to|in)\s+)', r'\1\n\2', text)

    # Fix 5: Add space between CamelCase words that got merged (like "ExecutiveLalana")
    text = re.sub(r'([a-z])([A-Z][a-z])', r'\1 \2', text)

    # Fix 6: Add line break after person names before certificate headers
    # Pattern: "Name YuCertificate" → "Name Yu\nCertificate"
    text = re.sub(r'([A-Z][a-z]+)\s*([A-Z][a-z]+)(Certificate)', r'\1 \2\n\3', text)

    return text


def load_file(uploaded_file) -> Tuple[Union[pd.DataFrame, None], Union[dict, str, None]]:
    """
    Load different file types and return a tuple:
      (DataFrame or None, metadata/text or None)

    Works with Streamlit's file uploader and other frameworks.

    Args:
        uploaded_file: File object from st.file_uploader() or similar

    Returns:
        - For Excel: (DataFrame, metadata_dict)
        - For CSV: (DataFrame, None)
        - For PDF: (None, extracted_text_string)
        - For Images: (None, extracted_text_string)
        - On error: (None, error_message)

    Example:
        df_data, metadata = load_file(uploaded_file)
        if isinstance(metadata, dict):
            student_name = metadata.get('Student Name')
        elif isinstance(metadata, str):
            # PDF or image text
            extracted_text = metadata
    """
    try:
        filename = uploaded_file.name.lower()

        # Read file content as bytes (works with Streamlit)
        file_content = uploaded_file.read()

        # Reset file pointer for multiple reads
        uploaded_file.seek(0)

        # --- CSV ---
        if filename.endswith(".csv"):
            try:
                df = pd.read_csv(io.BytesIO(file_content), encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(file_content), encoding="latin1")
            return df, None

        # --- Excel ---
        elif filename.endswith((".xlsx", ".xls")):
            # STEP 1: Read without header to get raw data
            df_no_header = pd.read_excel(io.BytesIO(file_content), header=None)

            # STEP 2: Extract metadata from top rows
            metadata = extract_metadata_from_excel(df_no_header)

            # STEP 3: Find where the actual data table starts
            data_start_row = None
            for idx in range(min(15, len(df_no_header))):
                row_str = ' '.join(str(cell) for cell in df_no_header.iloc[idx].tolist()
                                   if pd.notna(cell) and str(cell).strip())

                # Look for data table header row
                if any(keyword in row_str.lower() for keyword in
                       ['label', 'score', 'subject', 'mark', 'grade', 'result']):

                    # Check if next keyword is also present (confirms it's a header row)
                    if any(keyword in row_str.lower() for keyword in
                           ['maximum', 'total', 'percentage', 'notes']):
                        data_start_row = idx
                        break

            # STEP 4: Read data with proper header
            if data_start_row is not None:
                df_data = pd.read_excel(io.BytesIO(file_content), header=data_start_row)
            else:
                df_data = pd.read_excel(io.BytesIO(file_content))

            # STEP 5: Return both DataFrame and metadata
            return df_data, metadata

        # --- PDF ---
        elif filename.endswith(".pdf"):
            try:
                # Try to import PyPDF2 - you said you installed it
                import pypdf

                # DEBUG: Print PyPDF2 version to verify installation
                print(f"DEBUG: PyPDF version: {pypdf.__version__}")

                # Create PDF reader from bytes
                pdf_reader = pypdf.PdfReader(io.BytesIO(file_content))

                # DEBUG: Print number of pages
                print(f"DEBUG: PDF has {len(pdf_reader.pages)} pages")

                # Extract text from all pages with improved handling
                full_text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        # Add page separator to prevent text from different pages merging
                        full_text += f"===== Page {page_num + 1} =====\n"
                        full_text += page_text + "\n\n"
                    else:
                        print(f"DEBUG: Page {page_num + 1} has no extractable text")

                # Post-process the extracted text to fix common issues
                if full_text.strip():
                    full_text = _fix_pdf_extraction_issues(full_text)
                    print(f"DEBUG: PDF text extracted successfully ({len(full_text)} chars)")
                    return None, full_text.strip()
                else:
                    print("DEBUG: PDF contains no extractable text")
                    return None, "PDF contains no extractable text (might be scanned images)"

            except ImportError as e:
                print(f"DEBUG: PyPDF import error: {e}")
                return None, "PyPDF not installed. Install with: pip install PyPDF"
            except Exception as e:
                print(f"DEBUG: PDF extraction error: {e}")
                return None, f"Error extracting PDF text: {e}"

        # --- Images ---
        elif filename.endswith((".png", ".jpg", ".jpeg")):
            text = extract_text_from_image(uploaded_file)
            return None, text

        # --- Unsupported ---
        else:
            return None, "Unsupported file format."

    except Exception as e:
        import traceback
        error_msg = f"Error loading file: {e}\n{traceback.format_exc()}"
        print(f"DEBUG: {error_msg}")
        return None, error_msg


def extract_metadata(df: pd.DataFrame) -> dict:
    """
    DEPRECATED: Use the metadata returned by load_file() instead.

    This function is kept for backward compatibility but may not work correctly
    if the DataFrame has already been processed and metadata rows removed.
    """
    metadata = {}
    for idx, row in df.iterrows():
        first_cell = str(row.iloc[0]).strip().lower() if len(row) > 0 else ""
        if first_cell == "section":  # stop when table starts
            break
        if len(row) >= 2 and pd.notna(row.iloc[0]) and pd.notna(row.iloc[1]):
            key = str(row.iloc[0]).strip().rstrip(":")
            val = str(row.iloc[1]).strip()
            metadata[key] = val
    return metadata


def preprocess_excel_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to clean and structure an Excel DataFrame that might have metadata.
    Works for all file types, not just Malaysian.

    NOTE: This should be called AFTER extracting metadata, as it may remove rows.

    Args:
        df: DataFrame to preprocess

    Returns:
        Cleaned DataFrame
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    # Clean column names
    df.columns = [str(col).strip() for col in df.columns]

    # Remove completely empty rows
    df = df.dropna(how='all').reset_index(drop=True)

    # Convert numeric columns where appropriate
    for col in df.columns:
        try:
            # Try to convert to numeric, keep original if fails
            df[col] = pd.to_numeric(df[col], errors='ignore')
        except:
            pass

    return df


# Export all public functions
__all__ = [
    'load_file',
    'extract_text_from_image',
    'extract_metadata_from_excel',
    'extract_metadata',
    'preprocess_excel_dataframe'
]
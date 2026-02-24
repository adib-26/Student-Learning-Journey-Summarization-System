import streamlit as st
import os
import tempfile
import json
import re
from google import genai
from docling.document_converter import DocumentConverter
from backend.text_name import extract_student_name


def get_text_info(uploaded_file):
    """
    Main orchestrator for document processing:
    1. Extracts Student Name using local heuristics via text_name.py.
    2. Uses Docling to convert the document into a structured Markdown format.
    3. Uses Gemini 2.0 Flash to extract certificates, inferred skills, and a narrative summary.
    """
    temp_path = None
    try:
        # 1. Create a temporary file to store the uploaded content for Docling processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            temp_path = tmp.name

        # 2. Extract Student Name using your text_name logic
        student_name = extract_student_name(temp_path)

        # 3. Use Docling to perform high-fidelity conversion to Markdown
        converter = DocumentConverter()
        result = converter.convert(temp_path)
        markdown_text = result.document.export_to_markdown()

        # 4. Use Gemini 2.0 Flash to parse content into Certificates, Skills, and Summary
        ai_data = _extract_certificate_data_with_gemini(markdown_text)

        return {
            "student_name": student_name,
            "certificates": ai_data.get("certificates", []),
            "skills": ai_data.get("skills", []),
            "summary": ai_data.get("summary", ""),
            "raw_markdown": markdown_text
        }

    except Exception as e:
        st.error(f"Error in text_info_extractor: {e}")
        return {
            "student_name": "Unknown",
            "certificates": [],
            "skills": [],
            "summary": "",
            "raw_markdown": ""
        }

    finally:
        # Clean up the temporary file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def _extract_certificate_data_with_gemini(text_content: str) -> dict:
    """
    Internal helper: Sends markdown text to Gemini and retrieves a JSON object.
    Tasks:
    1. Extract Table Data.
    2. Infer 6-9 Professional Skills (Labels & Scores).
    3. Generate a 6-7 sentence professional summary using the certs and skills.
    """
    try:
        # Securely retrieve the API key from Streamlit secrets
        api_key = st.secrets["GEMINI_API_KEY"]

        # Initialize the official Google Gen AI Client
        client = genai.Client(api_key=api_key)

        prompt = f"""
        Analyze the following document text and perform three tasks:

        TASK 1: Extract Certificates (Strict Extraction)
        Extract a list of all certificates found. For each, identify:
        - **Certificate Name**: Title of the workshop/course.
        - **Date**: Normalize to 'DD/MM/YYYY'.
        - **Location**: Determine type (Online, Webinar, Physical - [City]).
        - **Issued Organization Name**.

        TASK 2: Analyze Skills & Competence (Inference for Visualization)
        Identify **6 to 9 distinct professional hard skills** demonstrated.
        - **Label**: Use professional industry terms (e.g., "Spatial Analysis", "Interior Architecture").
        - **Avoid Generics**: No "Webinar" or "Attendance" as labels.
        - **Score**: Assign 0-100 based on frequency and complexity.

        TASK 3: Professional Executive Summary
        Write a **6 to 7 sentence** professional narrative summary of the student.
        - Incorporate the specific organizations (e.g., IKEA, TechBridge Academy) and certificates found in TASK 1.
        - Explicitly mention the **skills** identified in TASK 2 to show how the student applies them.
        - Use simple, clear English. Avoid big "corporate" words or long, confusing sentences. Instead of "Interdisciplinary Learning," use phrases like "learning different types of skills." Instead of "Utilized," use "used."
        - Keep it professional and proud, but make sure it is easy for anyone to read. It should sound like a high school student describing their own hard work.
        - Show how the student is growing and how they can combine their tech skills with their creative skills in a practical way.        
        
        DOCUMENT TEXT:
        {text_content}

        Return the data ONLY as a valid JSON object with three keys: "certificates", "skills", and "summary".

        Example JSON Structure:
        {{
          "certificates": [
            {{"certificate_name": "...", "date": "DD/MM/YYYY", "location": "...", "organization": "..."}}
          ],
          "skills": [
            {{"Label": "Skill Name", "Score": 85}}
          ],
          "summary": "Full 6-7 sentence summary here..."
        }}
        """

        # Generate content using Gemini 2.0 Flash
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        # Use regex to find the main JSON object
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))

        return {"certificates": [], "skills": [], "summary": ""}

    except Exception as e:
        st.error(f"Gemini Data Extraction Error: {e}")
        return {"certificates": [], "skills": [], "summary": ""}


def get_student_name_from_text(text: str) -> str:
    """
    Fallback function for direct text processing.
    """
    return extract_student_name(text)
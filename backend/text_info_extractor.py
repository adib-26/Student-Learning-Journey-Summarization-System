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
    Orchestrator for document processing:
      - Writes uploaded file to a temp file for Docling.
      - Extracts student name using local heuristics.
      - Converts document to Markdown via Docling.
      - Calls Gemini to extract certificates, skills, and a concise summary.
      - Detects whether numeric evidence exists (used by UI to decide charts).
    Returns:
      {
        "student_name": str,
        "certificates": list,
        "skills": list,
        "summary": str,
        "raw_markdown": str,
        "has_numerical": bool
      }
    """
    temp_path = None
    try:
        # Save uploaded bytes to a temporary file (preserve suffix if possible)
        suffix = ".pdf"
        try:
            name = uploaded_file.name or ""
            if "." in name:
                suffix = "." + name.split(".")[-1]
        except Exception:
            suffix = ".pdf"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            temp_path = tmp.name

        # Extract student name using local logic
        student_name = extract_student_name(temp_path) or "Unknown"

        # Convert document to markdown using Docling
        converter = DocumentConverter()
        result = converter.convert(temp_path)
        markdown_text = result.document.export_to_markdown()

        # Call Gemini to extract structured data and a concise summary
        ai_data = _extract_certificate_data_with_gemini(markdown_text)

        # Detect numeric evidence in the markdown (digits, percentages, grades)
        has_numerical = bool(re.search(r'\b\d+(\.\d+)?%?|\bgrade\b|\bscore\b', markdown_text, flags=re.IGNORECASE))

        # Post-process skills: remove invented scores when no numeric evidence
        skills = ai_data.get("skills", []) or []
        if not has_numerical:
            cleaned_skills = []
            for s in skills:
                if isinstance(s, dict) and s.get("Label"):
                    cleaned_skills.append({"Label": s.get("Label")})
            skills = cleaned_skills
        else:
            cleaned_skills = []
            for s in skills:
                if not isinstance(s, dict):
                    continue
                label = s.get("Label")
                score = s.get("Score", None)
                if label is None:
                    continue
                if score is None:
                    cleaned_skills.append({"Label": label})
                else:
                    try:
                        score_num = int(float(score))
                        score_num = max(0, min(100, score_num))
                        cleaned_skills.append({"Label": label, "Score": score_num})
                    except Exception:
                        cleaned_skills.append({"Label": label})
            skills = cleaned_skills

        # Certificates normalization
        certificates = ai_data.get("certificates", []) or []
        if not isinstance(certificates, list):
            certificates = []

        # Sanitize summary: 4-5 sentences, factual, plain-language
        summary = ai_data.get("summary", "") or ""
        summary = _sanitize_summary(summary, has_numerical)

        return {
            "student_name": student_name,
            "certificates": certificates,
            "skills": skills,
            "summary": summary,
            "raw_markdown": markdown_text,
            "has_numerical": has_numerical
        }

    except Exception as e:
        st.error("Error in text_info_extractor: " + str(e))
        return {
            "student_name": "Unknown",
            "certificates": [],
            "skills": [],
            "summary": "",
            "raw_markdown": "",
            "has_numerical": False
        }

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


def _extract_certificate_data_with_gemini(text_content: str) -> dict:
    """
    Send markdown text to Gemini and request a robust JSON response.
    Key changes:
      - Ask for a short 4–5 sentence summary (plain language).
      - Instruct model not to use strong adjectives like "expert" unless numeric or credential evidence exists.
      - Ask model to wrap JSON between explicit markers to ease parsing.
      - Request skills without scores if no numeric evidence.
    """
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        client = genai.Client(api_key=api_key)

        prompt = f"""
You will analyze the DOCUMENT TEXT below and return a single JSON object between the markers
###JSON_START### and ###JSON_END###. The JSON must have three keys: "certificates", "skills", and "summary".

Rules:
1) Certificates: extract each certificate found. For each include:
   - certificate_name (string)
   - date (DD/MM/YYYY if present, else empty string)
   - location (Online, Webinar, Physical - [City], or empty string)
   - organization (string or empty string)

2) Skills: list up to 9 professional hard-skill labels that are clearly supported by the document.
   - Each skill must be an object with "Label": "Skill Name".
   - Include "Score" (integer 0-100) ONLY if the document contains numeric evidence (scores, percentages, grades).
   - If no numeric evidence exists, do NOT invent scores; return only the Label.

3) Summary: return a short, plain-language summary in 4–5 sentences.
   - Use simple, clear English suitable for a high-school reader.
   - Avoid exaggerated words like "expert", "master", "specialist", or "dedicated" unless the document contains clear numeric or credential evidence.
   - If certificates do not show numeric achievement, prefer phrasing like "This certificate widened knowledge in X" or "This course helped build skills in X."
   - Keep it factual and modest. Do not add suggestions or future steps.

DOCUMENT TEXT:
{text_content}

Return ONLY the JSON object between the markers. Example:

###JSON_START###
{{
  "certificates": [
    {{"certificate_name":"...","date":"DD/MM/YYYY","location":"...","organization":"..."}}
  ],
  "skills": [
    {{"Label":"Skill Name","Score":85}}
  ],
  "summary":"Short 4–5 sentence summary here."
}}
###JSON_END###
"""

        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt
        )

        resp_text = ""
        if hasattr(response, "text"):
            resp_text = response.text or ""
        else:
            try:
                resp_text = response.output[0].content[0].text or ""
            except Exception:
                resp_text = str(response)

        start_marker = "###JSON_START###"
        end_marker = "###JSON_END###"
        start_idx = resp_text.find(start_marker)
        end_idx = resp_text.find(end_marker, start_idx + len(start_marker)) if start_idx != -1 else -1

        if start_idx != -1 and end_idx != -1:
            json_text = resp_text[start_idx + len(start_marker):end_idx].strip()
            json_obj_text = _extract_largest_json_object(json_text)
            if json_obj_text:
                try:
                    parsed = json.loads(json_obj_text)
                    return {
                        "certificates": parsed.get("certificates", []),
                        "skills": parsed.get("skills", []),
                        "summary": parsed.get("summary", "")
                    }
                except Exception:
                    pass

        json_match = re.search(r'\{(?:[^{}]|(?R))*\}', resp_text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                return {
                    "certificates": parsed.get("certificates", []),
                    "skills": parsed.get("skills", []),
                    "summary": parsed.get("summary", "")
                }
            except Exception:
                pass

        return {"certificates": [], "skills": [], "summary": ""}

    except Exception as e:
        # Check if it's a 429 quota error to show friendly message
        error_str = str(e)
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "prepayment credits are depleted" in error_str:
            st.warning("⚠️ Sorry, our AI summary service is temporarily unavailable", icon="⚠️")
        else:
            st.error("Gemini Data Extraction Error: " + str(e))
        return {"certificates": [], "skills": [], "summary": ""}


def _extract_largest_json_object(text: str) -> str:
    max_obj = ""
    stack = []
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if start is None:
                start = i
            stack.append(i)
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack and start is not None:
                    candidate = text[start:i + 1]
                    if len(candidate) > len(max_obj):
                        max_obj = candidate
                    start = None
    return max_obj.strip()


def _sanitize_summary(summary: str, has_numerical: bool) -> str:
    """
    Ensure the summary is short, factual, and plain-language.
      - Keep 4–5 sentences.
      - Remove or soften exaggerated adjectives when no numeric evidence exists.
      - Prefer evidence-based phrasing and simple wording.
    """
    if not summary:
        return ""

    # Normalize whitespace
    s = re.sub(r'\s+', ' ', summary).strip()

    # Split into sentences using a simple heuristic
    sentences = re.split(r'(?<=[.!?])\s+', s)
    sentences = [sent.strip() for sent in sentences if sent.strip()]

    # If model returned fewer than 4 sentences, try to split long sentences by commas to reach 4-5 short sentences
    if len(sentences) < 4:
        expanded = []
        for sent in sentences:
            # If sentence is long, split by semicolon or comma into shorter clauses
            if len(sent) > 140 and (',' in sent or ';' in sent):
                parts = re.split(r'[;,:]\s*', sent)
                for p in parts:
                    p = p.strip()
                    if p:
                        expanded.append(p if p.endswith(('.', '!', '?')) else p + '.')
            else:
                expanded.append(sent)
        sentences = [re.sub(r'\s+', ' ', x).strip() for x in expanded if x.strip()]

    # Now enforce 4-5 sentences: keep first 4, optionally a 5th if concise
    if len(sentences) > 5:
        sentences = sentences[:5]
    elif len(sentences) < 4:
        # If still short, keep what we have but try to make them concise
        # Join and then re-split by clauses to attempt to reach 4 sentences
        joined = " ".join(sentences)
        clauses = re.split(r'(?<=[.!?])\s+|[;,:]\s+', joined)
        clauses = [c.strip() for c in clauses if c.strip()]
        if len(clauses) >= 4:
            sentences = clauses[:4]
        else:
            # fallback: pad by repeating a concise factual phrase if necessary (avoid exaggeration)
            while len(sentences) < 4:
                sentences.append("No additional measurable results were provided.")

    # Softening replacements when no numeric evidence
    replacements_no_numeric = {
        r'\bexpert\b': 'skilled',
        r'\bdedicated\b': 'hard-working',
        r'\bmaster(ed)?\b': 'experienced',
        r'\bspecialist\b': 'focused on',
        r'\boutstanding\b': 'notable',
        r'\bexceptional\b': 'strong'
    }

    sanitized = []
    for sent in sentences:
        s_sent = sent
        if not has_numerical:
            for pattern, repl in replacements_no_numeric.items():
                s_sent = re.sub(pattern, repl, s_sent, flags=re.IGNORECASE)
            # Prefer "widened knowledge" phrasing for certificate claims without numeric evidence
            s_sent = re.sub(r'\b(received|earned|completed)\b\s+(a\s+)?certificate\b',
                            'completed a certificate that widened knowledge', s_sent, flags=re.IGNORECASE)
        # Remove excessive qualifiers like "very", "extremely" to keep tone modest
        s_sent = re.sub(r'\b(very|extremely|highly)\b\s+', '', s_sent, flags=re.IGNORECASE)
        # Trim whitespace and ensure sentence ends with punctuation
        s_sent = s_sent.strip()
        if s_sent and s_sent[-1] not in '.!?':
            s_sent = s_sent + '.'
        sanitized.append(s_sent)

    # Ensure final text length is reasonable
    final_sentences = sanitized[:5]
    final = " ".join(final_sentences).strip()

    # If still too long, try to keep first 4 sentences only
    if len(final) > 600:
        final = " ".join(final_sentences[:4])
        if len(final) > 600:
            final = final[:597].rstrip() + "..."

    return final


def get_student_name_from_text(text: str) -> str:
    """
    Fallback wrapper to extract student name from raw text.
    """
    return extract_student_name(text)
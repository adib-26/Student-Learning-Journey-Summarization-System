# app.py — SECURE STRUCTURED DATA PIPELINE WITH PII PROTECTION & AUDIT LOGGING
import os
import uuid
import streamlit as st
import pandas as pd
from PIL import Image
from backend.ui_animations import inject_ui_animations, animated_summary

# =====================================================
# SECURE CLIENT, PII, & AUDIT LOG INTEGRATION
# =====================================================
from backend.secure_gemini_client import SecureGeminiClient
from backend.pii_protection import PIIProtector
from backend.audit_logging import AuditLogger

# Initialize a persistent, unique session token in Streamlit state memory for tracking
if "user_session_token" not in st.session_state:
    st.session_state["user_session_token"] = f"SESS_{uuid.uuid4().hex[:12].upper()}"

# Initialize security wrappers and engine layers
try:
    secure_client = SecureGeminiClient()
    pii_protector = PIIProtector()
    audit_logger = AuditLogger(custom_session_id=st.session_state["user_session_token"])
except ValueError as e:
    st.error(f"Configuration Error: {e}")
    st.stop()

# =====================================================
# BACKEND IMPORTS
# =====================================================
from backend.data_loader import load_file, extract_text_from_image
from backend.data_cleaning import clean_dataframe
from backend.ocr_parser import parse_ocr_text_to_dataframe
from backend.analytics import (
    compute_statistics,
    detect_trends,
    generate_predictive_insights
)
from backend.top5 import show_top5_ui
from backend.normalizer import normalize_uploaded_dataframe
from backend.summarizer import generate_summary
from backend.student_info_extractor import get_student_info
from backend.chart import render_visualizations
from backend.behaviour_extractor import extract_behaviour_pairs

from backend.text_info_extractor import get_student_name_from_text
from backend.download import get_report_bytes, create_download_button
from backend.text_info_extractor import get_text_info

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(
    page_title="Educational Data Analytics Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Inject animations
inject_ui_animations()
st.title("📊 Student Learning Journey Summarization System")
st.caption("Structured Data Analytics - Secure AI Platform")

# =====================================================
# FILE UPLOAD
# =====================================================
uploaded_file = st.file_uploader(
    "📁 Upload Your Data File",
    type=["csv", "xlsx", "xls", "pdf", "png", "jpg", "jpeg"]
)

if not uploaded_file:
    st.stop()

# Execution telemetry logging step
file_name = uploaded_file.name.lower()
audit_logger.log_file_upload(
    filename=uploaded_file.name,
    file_size=uploaded_file.size,
    file_type=uploaded_file.type
)

df = None
metadata = None
extracted_text = None

# =====================================================
# LOAD FILE
# =====================================================
try:
    if file_name.endswith(("png", "jpg", "jpeg")):
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
        extracted_text = extract_text_from_image(uploaded_file)
        df = parse_ocr_text_to_dataframe(extracted_text)

    elif file_name.endswith(".pdf"):
        df, metadata = load_file(uploaded_file)
        if isinstance(metadata, str):
            extracted_text = metadata

    else:
        df, metadata = load_file(uploaded_file)
        if isinstance(metadata, str):
            extracted_text = metadata
except Exception as e:
    audit_logger.log_error(error_type="FILE_LOAD_FAILURE", message=str(e), stage="FILE_INGESTION")
    st.error("An error occurred while loading your document.")
    st.stop()

# =====================================================
# HANDLE UNSTRUCTURED FILES (SECURED VIA WRAPPER)
# =====================================================
if df is None or df.empty:
    text_source = None
    if isinstance(extracted_text, str) and extracted_text.strip():
        text_source = extracted_text
    elif isinstance(metadata, str) and metadata.strip():
        text_source = metadata

    if text_source:
        student_name_unstructured = get_student_name_from_text(text_source)
        st.markdown(f"""
        <div style="
            padding: 12px 16px;
            border-radius: 8px;
            background-color: rgba(0,150,255,0.15);
            border-left: 4px solid rgba(0,200,255,0.6);
            color: #e6eef8;
            margin-bottom: 16px;
        ">
            ✅ <strong>Student:</strong> {student_name_unstructured}
        </div>
        """, unsafe_allow_html=True)

        from backend.text_info_extractor import _extract_certificate_data_with_gemini
        from backend.chart import render_visualizations

        with st.spinner("📜 Running encrypted processing pipelines..."):
            # Clean text source input parameters using PII Redaction tools prior to processing
            safe_text_source = pii_protector.redact_pii(text_source)
            redacted_count = len(pii_protector.redacted_map)

            audit_logger.log_data_processing(
                stage="UNSTRUCTURED_PII_REDACTION",
                records_processed=1,
                pii_redacted=redacted_count
            )

            import time

            start_time = time.time()

            # Execute protected payload transition
            ai_data = _extract_certificate_data_with_gemini(safe_text_source)

            latency = (time.time() - start_time) * 1000
            audit_logger.log_api_call(
                api_service="Google AI Studio",
                endpoint="_extract_certificate_data_with_gemini",
                request_size=len(safe_text_source),
                response_time_ms=latency,
                status_code=200 if ai_data else 500,
                pii_protected=True
            )

            certificates = ai_data.get("certificates", [])
            skills = ai_data.get("skills", [])
            summary = ai_data.get("summary", "")

            # --- A. Display Certificate Table ---
            if certificates:
                st.subheader("🎓 Certificate Summary")
                cert_df = pd.DataFrame(certificates)
                cert_df.columns = ["Certificate Name", "Date", "Location", "Issued Organization"]
                st.table(cert_df)
            else:
                st.info("No specific certificate details found.")

            # --- B. Display Skills Chart ---
            if skills:
                skills_df = pd.DataFrame(skills)
                render_visualizations(skills_df, text_source)

            # --- C. FINAL SUMMARY ---
            if summary:
                st.markdown("---")
                st.subheader("📝 Professional Executive Summary")
                animated_summary(summary)

                audit_logger.log_summary_generation(
                    input_size=len(safe_text_source),
                    output_size=len(summary),
                    model="gemini-2.5-flash",
                    tokens_used=0  # Stream telemetry context estimation fallback
                )

                # --- D. DOWNLOAD ---
                stats_unstructured = {
                    "summary": summary,
                    "averages": {"Score": 0},
                    "highest_score": 0,
                    "lowest_score": 0,
                    "total_records": 0,
                    "student_details": {},
                    "student_name": student_name_unstructured,
                }

                pdf_bytes = get_report_bytes(
                    stats=stats_unstructured,
                    df=pd.DataFrame(),
                    student_info={},
                    student_name=student_name_unstructured,
                    extracted_text=text_source,
                    behaviour_traits=[],
                    charts_images=[],
                    page_html="",
                    page_images=[],
                    summary=summary,
                )

                create_download_button(
                    pdf_bytes,
                    filename=f"{student_name_unstructured.replace(' ', '_')}_report.pdf",
                    label="⬇️ Download Black and White Report"
                )

            # Complete local cleanups safely
            audit_logger.log_data_deletion(data_type="UnstructuredTextPayload", records_deleted=1)
            st.stop()

# =====================================================
# STUDENT INFO
# =====================================================
student_info = (
    get_student_info(metadata=metadata)
    if isinstance(metadata, dict)
    else get_student_info(df=df)
)

student_name = student_info.get("Student Name", "Unknown Student")
st.markdown(f"""
<div style="
    padding: 12px 16px;
    border-radius: 8px;
    background-color: rgba(0,150,255,0.15);
    border-left: 4px solid rgba(0,200,255,0.6);
    color: #e6eef8;
    margin-bottom: 16px;
">
    ✅ <strong>Student:</strong> {student_name}
</div>
""", unsafe_allow_html=True)

# =====================================================
# NORMALIZE
# =====================================================
if not {"Score", "Maximum"}.issubset(df.columns):
    df = normalize_uploaded_dataframe(df)

df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
df["Maximum"] = pd.to_numeric(df["Maximum"], errors="coerce")

cleaned_df = clean_dataframe(df)
audit_logger.log_data_processing(stage="STRUCTURED_DATAFRAME_CLEANING", records_processed=len(cleaned_df))

# =====================================================
# TOP 5
# =====================================================
st.subheader("🎓 Top 5 Grades")
show_top5_ui(cleaned_df)

# =====================================================
# BEHAVIOR
# =====================================================
behaviour_traits = extract_behaviour_pairs(
    df=cleaned_df,
    text=extracted_text or ""
)

# =====================================================
# STATISTICS
# =====================================================
stats = compute_statistics(cleaned_df)
stats["trends"] = detect_trends(cleaned_df)
stats["predictive_insights"] = generate_predictive_insights(cleaned_df)

if behaviour_traits:
    stats["behaviour"] = behaviour_traits

stats["student_details"] = {
    k: v for k, v in student_info.items() if k != "Student Name"
}
stats["student_name"] = student_name

# =====================================================
# METRICS
# =====================================================
avg_score = stats.get("averages", {}).get("Score", 0)

highest_score = (
    cleaned_df["Score"].max()
    if "Score" in cleaned_df.columns and not cleaned_df["Score"].empty
    else 0
)

lowest_score = (
    cleaned_df["Score"].min()
    if "Score" in cleaned_df.columns and not cleaned_df["Score"].empty
    else 0
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Average Score", f"{avg_score:.2f}")

with col2:
    st.metric("Total Records", len(cleaned_df))

with col3:
    st.metric("Highest Score", f"{highest_score:.2f}")

with col4:
    st.metric("Lowest Score", f"{lowest_score:.2f}")

# =====================================================
# VISUALS
# =====================================================
charts_images = []
maybe_images = render_visualizations(cleaned_df, extracted_text or "")
if isinstance(maybe_images, list):
    charts_images = maybe_images

# =====================================================
# AI SUMMARY (TUNED TO INTERCEPT VIA SECURE WRAPPER + PII)
# =====================================================
# Strip identifying values out of text inputs prior to executing prompt instructions
safe_student_context = pii_protector.anonymize_student_data(student_info)
safe_extracted_text = pii_protector.redact_pii(extracted_text or "")
stats["summary_context"] = safe_extracted_text

# Log dataframe processing steps
audit_logger.log_data_processing(
    stage="AI_PROMPT_PII_REDACTION",
    records_processed=len(cleaned_df),
    pii_redacted=len(pii_protector.redacted_map)
)

# Utilize the cleaner's built-in framework constructor
safe_prompt, _ = pii_protector.create_safe_prompt(
    stats_dict=stats,
    student_context=safe_student_context
)

try:
    import time

    start_time = time.time()

    with st.spinner("🔒 Generating secure encrypted insight narrative..."):
        summary = secure_client.call_gemini_secure(
            prompt=safe_prompt,
            model="gemini-2.5-flash"
        )

    latency = (time.time() - start_time) * 1000
    audit_logger.log_api_call(
        api_service="Google AI Studio",
        endpoint="call_gemini_secure",
        request_size=len(safe_prompt),
        response_time_ms=latency,
        status_code=200 if summary else 500,
        pii_protected=True
    )

    if not summary:
        summary = generate_summary(stats, safe_extracted_text, mode="insight")
except Exception as e:
    audit_logger.log_error(error_type="LLM_PROMPT_ERROR", message=str(e), stage="SUMMARY_GENERATION")
    summary = generate_summary(stats, safe_extracted_text, mode="insight")

animated_summary(summary)

audit_logger.log_summary_generation(
    input_size=len(safe_prompt),
    output_size=len(summary) if summary else 0,
    model="gemini-2.5-flash",
    tokens_used=0
)

# =====================================================
# PREPARE STATS AND PAGE SNAPSHOT FOR DOWNLOAD
# =====================================================
stats["summary"] = summary
stats["averages"] = stats.get("averages", {})
stats["averages"]["Score"] = avg_score
stats["highest_score"] = highest_score
stats["lowest_score"] = lowest_score
stats["total_records"] = len(cleaned_df)

page_html = (
    f"Student: {student_name}\n"
    f"Average Score: {avg_score:.2f}\n"
    f"Total Records: {len(cleaned_df)}\n\n"
    f"Executive Summary:\n{summary}\n"
)

# =====================================================
# DOWNLOAD
# =====================================================
pdf_bytes = get_report_bytes(
    stats=stats,
    df=cleaned_df,
    student_info=stats.get("student_details", {}),
    student_name=student_name,
    extracted_text=extracted_text or "",
    behaviour_traits=behaviour_traits,
    charts_images=charts_images,
    page_html=page_html,
    page_images=[],
    summary=summary,
)

create_download_button(
    pdf_bytes,
    filename=f"{student_name.replace(' ', '_')}_report.pdf",
    label="⬇️ Download Black and White Report"
)

# Terminate session operations cleanly
audit_logger.log_data_deletion(data_type="StructuredDataFramePayload", records_deleted=len(cleaned_df))

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:cyan'>Educational Data Analytics Platform | Designed by Mahbub</div>",
    unsafe_allow_html=True
)
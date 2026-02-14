# app.py — STRUCTURED DATA PIPELINE ONLY
import streamlit as st
import pandas as pd
from PIL import Image
from backend.ui_animations import inject_ui_animations, animated_summary
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
st.caption("Structured Data Analytics - AI-Ready Platform")

# =====================================================
# FILE UPLOAD
# =====================================================
uploaded_file = st.file_uploader(
    "📁 Upload Your Data File",
    type=["csv", "xlsx", "xls", "pdf", "png", "jpg", "jpeg"]
)

if not uploaded_file:
    st.stop()

file_name = uploaded_file.name.lower()
df = None
metadata = None
extracted_text = None

# =====================================================
# LOAD FILE
# =====================================================
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

# =====================================================
# HANDLE UNSTRUCTURED FILES
# =====================================================
# =====================================================
# HANDLE UNSTRUCTURED FILES
# =====================================================
if df is None or df.empty:
    # Prefer extracted_text (OCR/markdown). If not available, try metadata if it's a string.
    text_source = None
    if isinstance(extracted_text, str) and extracted_text.strip():
        text_source = extracted_text
    elif isinstance(metadata, str) and metadata.strip():
        text_source = metadata

    if text_source:
        # 1. Your existing Name Logic (Preserved)
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

        # 2. Updated Import
        from backend.text_info_extractor import _extract_certificate_data_with_gemini
        from backend.chart import render_visualizations

        with st.spinner("📜 Processing certificates, charts, and final analysis..."):
            # Get data from your backend (includes the 6-7 sentence summary)
            ai_data = _extract_certificate_data_with_gemini(text_source)

            certificates = ai_data.get("certificates", [])
            skills = ai_data.get("skills", [])
            summary = ai_data.get("summary", "")

            # --- A. Display Certificate Table (Preserved) ---
            if certificates:
                st.subheader("🎓 Certificate Summary")
                cert_df = pd.DataFrame(certificates)
                cert_df.columns = ["Certificate Name", "Date", "Location", "Issued Organization"]
                st.table(cert_df)
            else:
                st.info("No specific certificate details found.")

            # --- B. Display Skills Chart (Preserved) ---
            if skills:
                # Backend returns "Label" and "Score" keys which render_visualizations expects
                skills_df = pd.DataFrame(skills)
                render_visualizations(skills_df, text_source)

            # --- C. FINAL SUMMARY (Moved to the end) ---
            if summary:
                st.markdown("---")
                st.subheader("📝 Professional Executive Summary")
                # Using st.info for a clean, highlighted background
                st.info(summary)

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
render_visualizations(cleaned_df, extracted_text or "")

# =====================================================
# AI SUMMARY
# =====================================================
summary_context = f"Student: {student_name}\n\n"
summary = generate_summary(
    stats,
    summary_context + (extracted_text or ""),
    mode="insight"
)
animated_summary(summary)

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:cyan'>Educational Data Analytics Platform | Designed by Mahbub</div>",
    unsafe_allow_html=True
)

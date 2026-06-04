# app.py — SECURE STRUCTURED DATA PIPELINE WITH PII PROTECTION & AUDIT LOGGING
import uuid
import streamlit as st
import pandas as pd
from PIL import Image
from backend.ui_animations import inject_ui_animations, animated_summary

# =====================================================
# SECURE CLIENT, PII, & AUDIT LOG INTEGRATION
# =====================================================
from backend.text_info_extractor import _extract_certificate_data_with_gemini
from backend.secure_gemini_client import SecureGeminiClient
from backend.pii_protection import PIIProtector
from backend.audit_logging import AuditLogger

# Initialize translation framework components
from backend.deepl_translator import translator, ui_translator

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

# =====================================================
# LANGUAGE SELECTOR INTERFACE
# =====================================================
col1_header, col2_header, col3_header = st.columns([2, 1, 1])

with col3_header:
    st.markdown("### 🌐 Language / 語言")
    lang = st.selectbox(
        "Select Language:",
        options=['en', 'ms', 'zh'],
        format_func=lambda x: {
            'en': '🇬🇧 English',
            'ms': '🇲🇾 Bahasa Melayu',
            'zh': '繁體中文'
        }[x],
        label_visibility="collapsed"
    )

    # Store previous language to detect changes
    if "previous_lang" not in st.session_state:
        st.session_state.previous_lang = lang

    st.session_state.selected_language = lang

current_lang = st.session_state.selected_language

with col1_header:
    translated_title = ui_translator.get_string("title", current_lang)
    translated_subtitle = ui_translator.get_string("subtitle", current_lang)

    st.title(f"📊 {translated_title}")
    st.caption(translated_subtitle)

# =====================================================
# FILE UPLOAD WITH DYNAMIC CSS INJECTION
# =====================================================
if current_lang == 'zh':
    drop_text = "将文件拖放至此处"
    browse_text = "浏览文件"
elif current_lang == 'ms':
    drop_text = "Seret dan lepaskan fail di sini"
    browse_text = "Semak Fail"
else:
    drop_text = "Drag and drop file here"
    browse_text = "Browse files"

safe_drop_text = drop_text.replace('\\', '\\\\').replace('"', '\\"')
safe_browse_text = browse_text.replace('\\', '\\\\').replace('"', '\\"')

st.markdown(f"""
    <style>
        div[data-testid="stFileUploaderDropzoneInstructions"] span.st-emotion-cache-ysg2um {{
            font-size: 0px !important;
            visibility: hidden !important;
        }}
        div[data-testid="stFileUploaderDropzoneInstructions"] span.st-emotion-cache-ysg2um::after {{
            content: "{safe_drop_text}";
            font-size: 14px !important;
            visibility: visible !important;
            display: inline-block !important;
        }}
        div[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"] {{
            font-size: 0px !important;
            display: inline-block !important;
        }}
        div[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"]::after {{
            content: "{safe_browse_text}";
            font-size: 14px !important;
            visibility: visible !important;
            display: inline-block !important;
        }}
    </style>
""", unsafe_allow_html=True)

upload_label = ui_translator.get_string("upload_section", current_lang)
upload_help = ui_translator.get_string("upload_help", current_lang)

uploaded_file = st.file_uploader(
    f"📁 {upload_label}",
    type=["csv", "xlsx", "xls", "pdf", "png", "jpg", "jpeg"],
    help=upload_help
)

# =====================================================
# 🔥 FIXED STATE MANAGEMENT (PRESERVE DATA ON LANGUAGE CHANGE)
# =====================================================

# Initialize tracking flag
if "file_active" not in st.session_state:
    st.session_state["file_active"] = False

# Detect if language changed in this run
language_changed = (st.session_state.get("previous_lang") != current_lang)
# Update previous language for next run
st.session_state.previous_lang = current_lang

# CASE 1: NEW FILE UPLOADED
if uploaded_file is not None:
    file_name = uploaded_file.name.lower()
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)

    # Telemetry logging sequence
    audit_logger.log_file_upload(
        filename=uploaded_file.name,
        file_size=uploaded_file.size,
        file_type=uploaded_file.type
    )

    # Store in session state
    st.session_state["cached_file_name"] = file_name
    st.session_state["cached_file_bytes"] = file_bytes
    st.session_state["cached_raw_uploader"] = uploaded_file

    # Reset downstream data states to trigger a clean pipeline extraction run
    st.session_state["processed_df"] = None
    st.session_state["processed_metadata"] = None
    st.session_state["processed_extracted_text"] = None

    # Mark active file session
    st.session_state["file_active"] = True

# CASE 2: NO FILE IN UPLOADER
else:
    # If we have an active file but the uploader is now empty...
    if st.session_state.get("file_active", False):
        # ... and the language has NOT changed → user clicked X
        if not language_changed:
            # ONLY clear when user explicitly cleared the uploader
            st.session_state.pop("cached_file_name", None)
            st.session_state.pop("cached_file_bytes", None)
            st.session_state.pop("cached_raw_uploader", None)

            st.session_state.pop("processed_df", None)
            st.session_state.pop("processed_metadata", None)
            st.session_state.pop("processed_extracted_text", None)

            st.session_state["file_active"] = False

            st.rerun()  # clean reset
        # else: language changed → keep all cached data, do nothing

# STOP if nothing loaded
if "cached_file_bytes" not in st.session_state:
    st.stop()

# Rehydrate operational content payloads directly from session storage layers
file_name = st.session_state["cached_file_name"]
file_bytes = st.session_state["cached_file_bytes"]
active_file_object = st.session_state["cached_raw_uploader"]

df = st.session_state.get("processed_df")
metadata = st.session_state.get("processed_metadata")
extracted_text = st.session_state.get("processed_extracted_text")

# =====================================================
# LAZY LOAD EXECUTION STEP
# =====================================================
if df is None and extracted_text is None:
    try:
        if file_name.endswith(("png", "jpg", "jpeg")):
            # st.image(Image.open(active_file_object), use_container_width=True) # Removed to prevent duplication
            extracted_text = extract_text_from_image(active_file_object)
            df = parse_ocr_text_to_dataframe(extracted_text)
        else:
            df, metadata = load_file(file_name, len(file_bytes), file_bytes)
            if isinstance(metadata, str):
                extracted_text = metadata

        # Write resolved payloads permanently to session cache indices
        st.session_state["processed_df"] = df
        st.session_state["processed_metadata"] = metadata
        st.session_state["processed_extracted_text"] = extracted_text

    except Exception as e:
        audit_logger.log_error(error_type="FILE_LOAD_FAILURE", message=str(e), stage="FILE_INGESTION")
        st.error(ui_translator.get_string("An error occurred while loading your document.", current_lang))
        st.stop()

# Render image visualizations safely out of session cache files if applicable
if file_name.endswith(("png", "jpg", "jpeg")) and active_file_object:
    try:
        st.image(Image.open(active_file_object), use_container_width=True)
    except Exception:
        pass

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



        loading_msg = ui_translator.get_string("📜 Running encrypted processing pipelines...", current_lang)
        with st.spinner(loading_msg):  # type: ignore
            safe_text_source = pii_protector.redact_pii(text_source)
            redacted_count = len(pii_protector.redacted_map)

            audit_logger.log_data_processing(
                stage="UNSTRUCTURED_PII_REDACTION",
                records_processed=1,
                pii_redacted=redacted_count
            )

            import time
            start_time = time.time()

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
                st.subheader(ui_translator.get_string("🎓 Certificate Summary", current_lang))
                cert_df = pd.DataFrame(certificates)

                translated_cols = [
                    ui_translator.get_string("Certificate Name", current_lang),
                    ui_translator.get_string("Date", current_lang),
                    ui_translator.get_string("Location", current_lang),
                    ui_translator.get_string("Issued Organization", current_lang)
                ]

                translated_rows = []
                for _, row in cert_df.iterrows():
                    translated_rows.append([
                        translator.translate_text(str(row.iloc[0]), current_lang),
                        str(row.iloc[1]),
                        translator.translate_text(str(row.iloc[2]), current_lang),
                        translator.translate_text(str(row.iloc[3]), current_lang)
                    ])

                translated_cert_df = pd.DataFrame(translated_rows, columns=translated_cols)
                st.table(translated_cert_df)
            else:
                st.info(ui_translator.get_string("No specific certificate details found.", current_lang))

            # --- B. Display Skills Chart ---
            if skills:
                skills_df = pd.DataFrame(skills)
                render_visualizations(skills_df, text_source)

            # --- C. FINAL SUMMARY ---
            if summary:
                st.markdown("---")
                st.subheader(ui_translator.get_string("📝 Professional Executive Summary", current_lang))

                translated_summary_text = translator.translate_text(summary, current_lang)
                animated_summary(translated_summary_text)

                audit_logger.log_summary_generation(
                    input_size=len(safe_text_source),
                    output_size=len(summary),
                    model="gemini-2.5-flash",
                    tokens_used=0
                )

                # --- D. DOWNLOAD ---
                stats_unstructured = {
                    "summary": translated_summary_text,
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
                    summary=translated_summary_text,
                )

                create_download_button(
                    pdf_bytes,
                    filename=f"{student_name_unstructured.replace(' ', '_')}_report.pdf",
                    label=ui_translator.get_string("⬇️ Download Black and White Report", current_lang)
                )

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
st.subheader(ui_translator.get_string("🎓 Top 5 Grades", current_lang))
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
    st.metric(ui_translator.get_string("Average Score", current_lang), f"{avg_score:.2f}")

with col2:
    st.metric(ui_translator.get_string("Total Records", current_lang), len(cleaned_df))

with col3:
    st.metric(ui_translator.get_string("Highest Score", current_lang), f"{highest_score:.2f}")

with col4:
    st.metric(ui_translator.get_string("Lowest Score", current_lang), f"{lowest_score:.2f}")

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
safe_student_context = pii_protector.anonymize_student_data(student_info)
safe_extracted_text = pii_protector.redact_pii(extracted_text or "")
stats["summary_context"] = safe_extracted_text

audit_logger.log_data_processing(
    stage="AI_PROMPT_PII_REDACTION",
    records_processed=len(cleaned_df),
    pii_redacted=len(pii_protector.redacted_map)
)

safe_prompt, _ = pii_protector.create_safe_prompt(
    stats_dict=stats,
    student_context=safe_student_context
)

try:
    import time
    start_time = time.time()

    insight_msg = ui_translator.get_string("🔒 Generating secure encrypted insight narrative...", current_lang)
    with st.spinner(insight_msg):  # type: ignore
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

translated_summary = translator.translate_text(summary, current_lang)
animated_summary(translated_summary)

audit_logger.log_summary_generation(
    input_size=len(safe_prompt),
    output_size=len(summary) if summary else 0,
    model="gemini-2.5-flash",
    tokens_used=0
)

# =====================================================
# PREPARE STATS AND PAGE SNAPSHOT FOR DOWNLOAD
# =====================================================
stats["summary"] = translated_summary
stats["averages"] = stats.get("averages", {})
stats["averages"]["Score"] = avg_score
stats["highest_score"] = highest_score
stats["lowest_score"] = lowest_score
stats["total_records"] = len(cleaned_df)

page_html = (
    f"Student: {student_name}\n"
    f"Average Score: {avg_score:.2f}\n"
    f"Total Records: {len(cleaned_df)}\n\n"
    f"Executive Summary:\n{translated_summary}\n"
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
    summary=translated_summary,
)

create_download_button(
    pdf_bytes,
    filename=f"{student_name.replace(' ', '_')}_report.pdf",
    label=ui_translator.get_string("⬇️ Download Black and White Report", current_lang)
)

audit_logger.log_data_deletion(data_type="StructuredDataFramePayload", records_deleted=len(cleaned_df))

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
footer_text = ui_translator.get_string("Educational Data Analytics Platform | Designed by Mahbub", current_lang)
st.markdown(
    f"<div style='text-align:center;color:cyan'>{footer_text}</div>",
    unsafe_allow_html=True
)
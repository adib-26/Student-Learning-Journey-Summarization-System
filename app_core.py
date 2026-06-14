# app_core.py — Core backend logic and data processing for Educational Data Analytics Platform
import uuid
import streamlit as st
import pandas as pd
import re
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


def initialize_security_components():
    """Initialize security wrappers and engine layers"""
    # Initialize a persistent, unique session token in Streamlit state memory for tracking
    if "user_session_token" not in st.session_state:
        st.session_state["user_session_token"] = f"SESS_{uuid.uuid4().hex[:12].upper()}"
    
    try:
        secure_client = SecureGeminiClient()
        pii_protector = PIIProtector()
        audit_logger = AuditLogger(custom_session_id=st.session_state["user_session_token"])
        return secure_client, pii_protector, audit_logger
    except ValueError as e:
        st.error(f"Configuration Error: {e}")
        st.stop()


def process_uploaded_file(active_file_object, file_name, file_bytes, audit_logger, pii_protector, current_lang):
    """Process uploaded file and handle lazy loading"""
    df = st.session_state.get("processed_df")
    metadata = st.session_state.get("processed_metadata")
    extracted_text = st.session_state.get("processed_extracted_text")

    # =====================================================
    # LAZY LOAD EXECUTION STEP
    # =====================================================
    if df is None and extracted_text is None:
        try:
            if file_name.endswith(("png", "jpg", "jpeg")):
                extracted_text = extract_text_from_image(active_file_object)
                df = parse_ocr_text_to_dataframe(extracted_text)
            else:
                df, metadata = load_file(file_name, file_bytes)
                if isinstance(metadata, str):
                    extracted_text = metadata

            st.session_state["processed_df"] = df
            st.session_state["processed_metadata"] = metadata
            st.session_state["processed_extracted_text"] = extracted_text

        except Exception as e:
            audit_logger.log_error(error_type="FILE_LOAD_FAILURE", message=str(e), stage="FILE_INGESTION")
            st.error(ui_translator.get_string("An error occurred while loading your document.", current_lang))
            st.stop()
    
    return df, metadata, extracted_text


def handle_unstructured_files(extracted_text, metadata, active_file_object, file_name, audit_logger, pii_protector, current_lang):
    """Handle unstructured files like images and PDFs with text extraction"""
    if file_name.endswith(("png", "jpg", "jpeg")) and active_file_object:
        try:
            st.image(Image.open(active_file_object), use_container_width=True)
        except Exception:
            pass

    # =====================================================
    # HANDLE UNSTRUCTURED FILES (SECURED VIA WRAPPER)
    # =====================================================
    df = st.session_state.get("processed_df")
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

                if skills:
                    skills_df = pd.DataFrame(skills)
                    render_visualizations(skills_df, text_source)

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
                        student_name=student_name_unstructured,
                        behaviour_traits=[],
                        charts_images=[],
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


def process_structured_data(df, metadata, extracted_text, audit_logger, current_lang):
    """Process structured data files and generate analytics"""
    # =====================================================
    # STUDENT INFO
    # =====================================================
    student_info = (
        get_student_info(metadata=metadata)
        if isinstance(metadata, dict)
        else get_student_info(df=df)
    )

    # Check if a manual name has been entered, otherwise use the detected name
    if 'manual_student_name' in st.session_state and st.session_state['manual_student_name']:
        student_name = st.session_state['manual_student_name']
    else:
        student_name = student_info.get("Student Name", "Unknown Student")

    # If the name is unknown, show a text input to allow the user to enter it manually
    if student_name == "Unknown Student":
        manual_name = st.text_input(
            ui_translator.get_string("Enter Student Name", current_lang),
            key="manual_student_name_input"
        )
        if manual_name:
            student_name = manual_name
            st.session_state['manual_student_name'] = manual_name  # Save to session state
            st.rerun()

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

    # Reset manual name if a new file is uploaded
    if "cached_file_name" in st.session_state:
        if "previous_file_name" not in st.session_state or st.session_state["previous_file_name"] != st.session_state["cached_file_name"]:
            st.session_state["previous_file_name"] = st.session_state["cached_file_name"]
            if "manual_student_name" in st.session_state:
                del st.session_state["manual_student_name"]

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

    if "original_stats" not in st.session_state:
        st.session_state["original_stats"] = stats.copy()

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
        
    return cleaned_df, stats, student_name, behaviour_traits, charts_images


def generate_and_download_report(stats, df, student_info, student_name, extracted_text, behaviour_traits, charts_images, current_lang):
    """Generate the final report and create download button"""
    # =====================================================
    # AI SUMMARY (TUNED TO INTERCEPT VIA SECURE WRAPPER + PII)
    # =====================================================
    import re
    from backend.pii_protection import PIIProtector
    pii_protector = PIIProtector()
    
    if "audit_logger" in st.session_state:
        audit_logger = st.session_state.get("audit_logger")
    else:
        from backend.audit_logging import AuditLogger
        audit_logger = AuditLogger(custom_session_id=st.session_state["user_session_token"])
    
    if "secure_client" in st.session_state:
        secure_client = st.session_state.get("secure_client")
    else:
        from backend.secure_gemini_client import SecureGeminiClient
        secure_client = SecureGeminiClient()
    
    cleaned_df = df
    
    # Handle plot stats updates if needed
    if ('current_plot_stats' in st.session_state and st.session_state['current_plot_stats'] and
            st.session_state.get('summary_needs_update', False)):
        stats = st.session_state['current_plot_stats'].copy()

        if 'original_stats' in st.session_state:
            original = st.session_state['original_stats']
            stats['student_details'] = original.get('student_details', student_info)
            stats['student_name'] = original.get('student_name', student_name)
            stats['behaviour'] = original.get('behaviour', None)
            stats['trends'] = original.get('trends', None)
            stats['predictive_insights'] = original.get('predictive_insights', None)

            if 'strength' not in stats or not stats['strength']:
                stats['strength'] = original.get('strength', None)
            if 'weakness' not in stats or not stats['weakness']:
                stats['weakness'] = original.get('weakness', None)

        x_axis = st.session_state.get('current_x_axis')
        y_axis = st.session_state.get('current_y_axis')
        if x_axis and 'averages' in stats:
            match = re.search(r'(\d+)$', x_axis)
            x_suffix = match.group(1) if match else ''

            filtered_averages = {}
            for key, value in stats['averages'].items():
                key_match = re.search(r'(\d+)$', key)
                key_suffix = key_match.group(1) if key_match else ''
                if key_suffix == x_suffix:
                    filtered_averages[key] = value

            if y_axis and y_axis in stats['averages']:
                y_key_match = re.search(r'(\d+)$', y_axis)
                y_key_suffix = y_key_match.group(1) if y_key_match else ''

                if not y_key_suffix and x_suffix:
                    new_key = f"{y_axis} {x_suffix}"
                    if new_key not in filtered_averages:
                        filtered_averages[new_key] = stats['averages'][y_axis]
                elif y_key_suffix == x_suffix:
                    filtered_averages[y_axis] = stats['averages'][y_axis]

            stats['averages'] = filtered_averages

        st.session_state['summary_needs_update'] = False

        audit_logger.log_data_processing(
            stage="AUTO_UPDATED_SUMMARY",
            records_processed=len(cleaned_df)
        )
    
    # Calculate metrics
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
    
    # PII protection for AI processing
    safe_student_context = pii_protector.anonymize_student_data(student_info if isinstance(student_info, dict) else {})
    safe_extracted_text = pii_protector.redact_pii(extracted_text or "")
    stats["summary_context"] = safe_extracted_text

    audit_logger.log_data_processing(
        stage="AI_PROMPT_PII_REDACTION",
        records_processed=len(cleaned_df),
        pii_redacted=len(pii_protector.redacted_map)
    )

    # Add student name to stats so it's included in the draft summary
    student_name = student_info.get('student_name', '') if isinstance(student_info, dict) else ''
    if student_name:
        stats['student_name'] = student_name

    try:
        import time
        start_time = time.time()

        if st.session_state.get('summary_needs_update', False) or 'last_axis_selection' in st.session_state:
            insight_msg = ui_translator.get_string("updating_summary", current_lang)
        else:
            insight_msg = ui_translator.get_string("🔒 Generating secure encrypted insight narrative...", current_lang)

        with st.spinner(insight_msg):  # type: ignore
            # Use our fixed summary generator that preserves all details
            summary = generate_summary(stats, safe_extracted_text, mode="insight")

        latency = (time.time() - start_time) * 1000
        audit_logger.log_api_call(
            api_service="Google AI Studio",
            endpoint="generate_summary",
            request_size=len(str(stats)) + len(safe_extracted_text),
            response_time_ms=latency,
            status_code=200 if summary else 500,
            pii_protected=True
        )
    except Exception as e:
        audit_logger.log_error(error_type="LLM_PROMPT_ERROR", message=str(e), stage="SUMMARY_GENERATION")
        summary = generate_summary(stats, safe_extracted_text, mode="insight")

    translated_summary = translator.translate_text(summary, current_lang)

    from backend.ui_animations import animated_summary
    animated_summary(translated_summary)

    audit_logger.log_summary_generation(
        input_size=len(str(stats)) + len(safe_extracted_text),
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
        student_name=student_name,
        behaviour_traits=behaviour_traits,
        charts_images=charts_images,
        page_images=[],
        summary=translated_summary,
    )

    create_download_button(
        pdf_bytes,
        filename=f"{student_name.replace(' ', '_')}_report.pdf",
        label=ui_translator.get_string("⬇️ Download Black and White Report", current_lang)
    )

    audit_logger.log_data_deletion(data_type="StructuredDataFramePayload", records_deleted=len(cleaned_df))
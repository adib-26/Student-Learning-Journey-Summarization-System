# app.py — MAIN ENTRY POINT: Educational Data Analytics Platform (Streamlit UI Layer)
import streamlit as st
from backend.ui_animations import inject_ui_animations
from backend.deepl_translator import ui_translator

# Import core functionality from app_core.py
from app_core import (
    initialize_security_components,
    process_uploaded_file,
    handle_unstructured_files,
    process_structured_data,
    generate_and_download_report
)

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(
    page_title="Educational Data Analytics Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)
inject_ui_animations()

# Initialize security components
secure_client, pii_protector, audit_logger = initialize_security_components()

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

    if "previous_lang" not in st.session_state:
        st.session_state.previous_lang = lang

    st.session_state.selected_language = lang

current_lang = st.session_state.selected_language

with col1_header:
    translated_title = ui_translator.get_string("title", current_lang)
    translated_subtitle = ui_translator.get_string("subtitle", current_lang)

    st.title(f"📊 {translated_title}")
    st.caption(translated_subtitle)
    
# Show Gemini quota warning if credits are exhausted
if hasattr(secure_client, 'quota_exhausted') and secure_client.quota_exhausted:
    if current_lang == 'ms':
        quota_msg = ui_translator.get_string("gemini_quota_warning_ms", current_lang)
    elif current_lang == 'zh':
        quota_msg = ui_translator.get_string("gemini_quota_warning_zh", current_lang)
    else:
        quota_msg = ui_translator.get_string("gemini_quota_warning", current_lang)
    st.warning(quota_msg, icon="⚠️")

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

# FIX: Add an explicit stable key so Streamlit treats this as the same widget
# across reruns regardless of the translated label text.  Without a key, the
# widget's identity is derived from its label, meaning every language change
# would create a brand-new widget that starts empty — uploading a file to
# the "English" widget and then switching to Malay effectively loses it.
uploaded_file = st.file_uploader(
    f"📁 {upload_label}",
    type=["csv", "xlsx", "xls", "pdf", "png", "jpg", "jpeg"],
    help=upload_help,
    key="student_data_uploader",
)

# =====================================================
# 🔥 FIXED STATE MANAGEMENT (PRESERVE DATA ON LANGUAGE CHANGE)
# =====================================================

if "file_active" not in st.session_state:
    st.session_state["file_active"] = False

language_changed = (st.session_state.get("previous_lang") != current_lang)
st.session_state.previous_lang = current_lang

# CASE 1: FILE IS PRESENT IN THE UPLOADER
if uploaded_file is not None:
    file_name = uploaded_file.name.lower()
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)

    # FIX: Only perform a full pipeline reset when a genuinely new file arrives.
    # On every widget interaction Streamlit re-runs this script, so without
    # this guard, CASE 1 would clear `processed_df` on every axis change,
    # triggering an unnecessary (and sometimes error-prone) lazy-load cycle.
    is_new_file = (
        st.session_state.get("cached_file_name") != file_name
        or st.session_state.get("cached_file_bytes") != file_bytes
    )

    if is_new_file:
        audit_logger.log_file_upload(
            filename=uploaded_file.name,
            file_size=uploaded_file.size,
            file_type=uploaded_file.type
        )

        st.session_state["cached_file_name"] = file_name
        st.session_state["cached_file_bytes"] = file_bytes
        st.session_state["cached_raw_uploader"] = uploaded_file

        # Reset downstream states only for a genuinely new file
        st.session_state["processed_df"] = None
        st.session_state["processed_metadata"] = None
        st.session_state["processed_extracted_text"] = None

    # Always keep the session marked as active and reset the null streak
    st.session_state["file_active"] = True
    st.session_state["_uploader_null_streak"] = 0

# CASE 2: UPLOADER IS EMPTY
else:
    if st.session_state.get("file_active", False):
        if language_changed:
            # A language change re-renders the page but the stable key above
            # prevents the widget from losing state.  This branch is kept as
            # an extra safety net for any edge case where it still fires.
            st.session_state["_uploader_null_streak"] = 0
        else:
            # Count consecutive empty-uploader runs.  A single None can occur
            # transiently during rapid reruns; two consecutive Nones reliably
            # indicates the user clicked the ✕ button to remove the file.
            streak = st.session_state.get("_uploader_null_streak", 0) + 1
            st.session_state["_uploader_null_streak"] = streak

            if streak >= 2:
                st.session_state.pop("cached_file_name", None)
                st.session_state.pop("cached_file_bytes", None)
                st.session_state.pop("cached_raw_uploader", None)

                st.session_state.pop("processed_df", None)
                st.session_state.pop("processed_metadata", None)
                st.session_state.pop("processed_extracted_text", None)

                st.session_state["file_active"] = False
                st.session_state["_uploader_null_streak"] = 0

                st.rerun()

# STOP if nothing loaded
if "cached_file_bytes" not in st.session_state:
    st.stop()

# Rehydrate operational content payloads directly from session storage layers
file_name = st.session_state["cached_file_name"]
file_bytes = st.session_state["cached_file_bytes"]
active_file_object = st.session_state["cached_raw_uploader"]

# Process the uploaded file using core functionality
df, metadata, extracted_text = process_uploaded_file(
    active_file_object, file_name, file_bytes, audit_logger, pii_protector, current_lang
)

# Handle unstructured files
handle_unstructured_files(extracted_text, metadata, active_file_object, file_name, audit_logger, pii_protector, current_lang)

# Process structured data and generate analytics
cleaned_df, stats, student_name, behaviour_traits, charts_images = process_structured_data(
    df, metadata, extracted_text, audit_logger, current_lang
)

# Generate and download the final report
generate_and_download_report(
    stats, cleaned_df, metadata, student_name, extracted_text, behaviour_traits, charts_images, current_lang
)

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
footer_text = ui_translator.get_string("Educational Data Analytics Platform | Designed by Mahbub", current_lang)
st.markdown(
    f"<div style='text-align:center;color:cyan'>{footer_text}</div>",
    unsafe_allow_html=True
)
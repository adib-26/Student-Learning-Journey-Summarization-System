# app_core.py — Core backend logic and data processing for Educational Data Analytics Platform
import uuid
import hashlib
import html
import json
import math
import streamlit as st
import pandas as pd
import re
from PIL import Image
from backend.ui_animations import animated_summary

# =====================================================
# SECURE CLIENT, PII, & AUDIT LOG INTEGRATION
# =====================================================
from backend.text_info_extractor import _extract_certificate_data_with_llm
from backend.llm_client import SecureLLMClient, get_configured_model
from backend.pii_protection import PIIProtector
from backend.audit_logging import AuditLogger

# Initialize translation framework components
from backend.deepl_translator import translator, ui_translator

# =====================================================
# BACKEND IMPORTS
# =====================================================
from backend.data_loader import load_file, extract_unstructured_text
from backend.mlx_glm_ocr import get_glm_ocr_cache_identity
from backend.data_cleaning import clean_dataframe, normalize_scores_to_percentage
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
from backend.text_name import resolve_student_name
from backend.download import get_report_bytes, create_download_button


_SUMMARY_CACHE_VERSION = "nvidia-vllm-summary-v1"
_UNSTRUCTURED_ANALYSIS_CACHE_VERSION = "unstructured-analysis-v5"


def _display_student_name(value: object) -> str:
    """Return a student name safe for the small HTML status banner."""
    name = str(value or "Unknown Student").strip() or "Unknown Student"
    return html.escape(name, quote=True)


def _safe_report_filename(student_name: object) -> str:
    """Build a cross-platform download filename without path characters."""
    normalized = str(student_name or "").strip()
    safe_stem = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in normalized
    ).strip("._")
    return f"{(safe_stem or 'student')[:80]}_report.pdf"


def _stable_fingerprint(payload) -> str:
    """Create a non-reversible cache key without storing source text in the key."""
    serialised = json.dumps(
        payload,
        sort_keys=True,
        default=str,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def _summary_cache_key(stats, safe_extracted_text: str) -> str:
    """Key a summary by its evidence and model, not by a Streamlit rerun."""
    stable_stats = {
        key: value
        for key, value in stats.items()
        if key not in {"summary", "summary_context"}
    }
    return _stable_fingerprint(
        {
            "contract": _SUMMARY_CACHE_VERSION,
            "model": get_configured_model(),
            "stats": stable_stats,
            "text": safe_extracted_text,
        }
    )


def _unstructured_analysis_cache_key(safe_text_source: str) -> str:
    """Key document analysis by the evidence, configured model, and prompt contract."""
    return _stable_fingerprint(
        {
            "contract": _UNSTRUCTURED_ANALYSIS_CACHE_VERSION,
            "model": get_configured_model(),
            "text": safe_text_source,
        }
    )


def _get_session_cache(name: str, limit: int = 8) -> dict:
    """Return a bounded browser-session cache for completed LLM results."""
    cache = st.session_state.get(name)
    if not isinstance(cache, dict):
        cache = {}
        st.session_state[name] = cache
    while len(cache) > limit:
        cache.pop(next(iter(cache)))
    return cache


def _unstructured_ocr_cache_key(file_name: str, file_bytes: bytes) -> str:
    """Keep OCR output only in this browser session, keyed by a file digest."""
    file_digest = hashlib.sha256(file_bytes).hexdigest()
    return _stable_fingerprint({
        "contract": "vllm-vision-ocr-v2",
        "extension": (file_name or "").rsplit(".", 1)[-1].casefold(),
        "file_digest": file_digest,
        "ocr": get_glm_ocr_cache_identity(),
    })


def _extract_unstructured_document_cached(file_name: str, file_bytes: bytes) -> tuple[dict, bool]:
    """Run native/vLLM visual extraction once per configuration and session."""
    cache = _get_session_cache("unstructured_ocr_cache", limit=4)
    cache_key = _unstructured_ocr_cache_key(file_name, file_bytes)
    cached = cache.get(cache_key)
    if isinstance(cached, dict) and isinstance(cached.get("text"), str) and cached["text"].strip():
        return cached, True

    result = extract_unstructured_text(file_name, file_bytes)
    if not isinstance(result, dict):
        result = {"text": "", "engine": "none", "error": "OCR returned an invalid result"}
    engine = str(result.get("engine", "")).casefold()
    operational_error = str(result.get("error", "")).strip()
    # Cache completed local extraction, but retry a degraded fallback after a
    # local server comes up instead of preserving weaker Tesseract text for an
    # entire browser session.
    if (
        isinstance(result.get("text"), str)
        and result["text"].strip()
        and "fallback" not in engine
        and not operational_error
    ):
        cache[cache_key] = result
        while len(cache) > 4:
            cache.pop(next(iter(cache)))
    return result, False


def _resolve_unstructured_student_name(rule_based_name: object, ai_data: object, text_source: str) -> str:
    """Accept the model-proposed recipient only when evidence validates it.

    The unstructured pipeline reuses the single structured extraction response
    for the recipient, certificates, and skills instead of issuing another call.
    """
    model_name = ai_data.get("student_name", "") if isinstance(ai_data, dict) else ""
    return resolve_student_name(rule_based_name, model_name, text_source)


def initialize_security_components():
    """Initialize security wrappers and engine layers"""
    # Initialize a persistent, unique session token in Streamlit state memory for tracking
    if "user_session_token" not in st.session_state:
        st.session_state["user_session_token"] = f"SESS_{uuid.uuid4().hex[:12].upper()}"

    secure_client = SecureLLMClient()
    pii_protector = PIIProtector()
    audit_logger = AuditLogger(custom_session_id=st.session_state["user_session_token"])

    # Probe the local inference server once per session. When it is down the
    # app keeps working with rule-based summaries instead of hard-stopping.
    if "llm_health_checked" not in st.session_state:
        st.session_state["llm_health_checked"] = True
        if not secure_client.health_check():
            st.warning(
                "⚠️ The remote model is not ready (unreachable or model mismatch); "
                "reports will use built-in fallbacks. Run "
                "`bash scripts/deploy_inference.sh`. It deploys on Modal and, "
                "if that fails, deploys Qwen3.6-35B-A3B on Vertex AI Model Garden, "
                "then writes `LLM_BASE_URL` into `.env`.",
                icon="⚠️",
            )

    return secure_client, pii_protector, audit_logger


def process_uploaded_file(active_file_object, file_name, file_bytes, audit_logger, current_lang):
    """Process uploaded file and handle lazy loading"""
    df = st.session_state.get("processed_df")
    metadata = st.session_state.get("processed_metadata")
    extracted_text = st.session_state.get("processed_extracted_text")

    # =====================================================
    # LAZY LOAD EXECUTION STEP
    # =====================================================
    if df is None and extracted_text is None:
        try:
            normalized_file_name = (file_name or "").casefold()
            if normalized_file_name.endswith((".pdf", ".png", ".jpg", ".jpeg")):
                ocr_result, cache_hit = _extract_unstructured_document_cached(file_name, file_bytes)
                extracted_text = ocr_result.get("text", "")
                metadata = extracted_text if normalized_file_name.endswith(".pdf") else None
                df = (
                    parse_ocr_text_to_dataframe(extracted_text)
                    if normalized_file_name.endswith((".png", ".jpg", ".jpeg"))
                    else None
                )
                st.session_state["processed_ocr_info"] = ocr_result
                engine = str(ocr_result.get("engine", "none"))
                audit_logger.log_data_processing(
                    stage=("UNSTRUCTURED_OCR_CACHE_HIT" if cache_hit else f"UNSTRUCTURED_OCR_{engine.upper()}"),
                    records_processed=1,
                )
                if ocr_result.get("truncated"):
                    audit_logger.log_data_processing(
                        stage="UNSTRUCTURED_OCR_PAGE_LIMIT",
                        records_processed=1,
                    )
                    st.warning(
                        "Only part of this PDF was used. The evidence includes explicit "
                        "page-limit markers; raise OCR limits only after testing memory use.",
                        icon="⚠️",
                    )
                ocr_error = str(ocr_result.get("error", "")).strip()
                if ocr_error:
                    audit_logger.log_error(
                        error_type="UNSTRUCTURED_OCR_DEGRADED",
                        message=ocr_error[:180],
                        stage="FILE_INGESTION",
                    )
                    if ocr_error in {
                        "document exceeds the configured OCR safety limit",
                        "image exceeds OCR safety limit",
                    }:
                        st.error(
                            "This file is above the local OCR safety limit. Use a smaller PDF/image "
                            "or raise the bounded limit only after testing memory use.",
                            icon="⚠️",
                        )
                        st.stop()
                    if not isinstance(extracted_text, str) or not extracted_text.strip():
                        st.warning(
                            "Visual extraction was unavailable; the app is using its built-in document fallback.",
                            icon="⚠️",
                        )
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


def _should_handle_as_unstructured(df, file_name: str) -> bool:
    """Keep certificates on the document path while retaining scanned mark sheets.

    OCR may produce a non-empty DataFrame for a certificate (for example, a
    recipient-name row).  That is not a score table and used to bypass Qwen
    extraction entirely.  A scanned image with at least one real numeric Score
    remains a structured mark sheet.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return True

    normalized_name = (file_name or "").lower()
    if normalized_name.endswith(".pdf"):
        return True

    if normalized_name.endswith((".png", ".jpg", ".jpeg")):
        if "Score" not in df.columns:
            return True
        numeric_scores = pd.to_numeric(df["Score"], errors="coerce")
        return not numeric_scores.notna().any()

    return False


def handle_unstructured_files(extracted_text, metadata, active_file_object, file_name, audit_logger, pii_protector, current_lang):
    """Handle unstructured files like images and PDFs with text extraction"""
    if (file_name or "").casefold().endswith((".png", ".jpg", ".jpeg")) and active_file_object:
        try:
            st.image(Image.open(active_file_object), use_container_width=True)
        except Exception:
            pass

    # =====================================================
    # HANDLE UNSTRUCTURED FILES (SECURED VIA WRAPPER)
    # =====================================================
    df = st.session_state.get("processed_df")
    if _should_handle_as_unstructured(df, file_name):
        ocr_info = st.session_state.get("processed_ocr_info")
        if isinstance(ocr_info, dict):
            engine = str(ocr_info.get("engine", "")).strip()
            if engine:
                st.caption(f"Document text source: {engine}")
            if ocr_info.get("truncated"):
                st.caption("The extracted evidence includes an explicit page-limit marker for omitted pages.")
        text_source = None
        if isinstance(extracted_text, str) and extracted_text.strip():
            text_source = extracted_text
        elif isinstance(metadata, str) and metadata.strip():
            text_source = metadata

        if text_source:
            # The extraction result is stable for a document/model pair.  Cache
            # it for this browser session so changing a language or chart does
            # not launch the same slow local generation again.
            safe_text_source = pii_protector.redact_pii(text_source)
            redacted_count = len(pii_protector.redacted_map)
            analysis_cache_key = _unstructured_analysis_cache_key(safe_text_source)
            analysis_cache = _get_session_cache("unstructured_llm_cache")
            cached_analysis = analysis_cache.get(analysis_cache_key)

            if isinstance(cached_analysis, dict):
                student_name_unstructured = cached_analysis.get("student_name", "Unknown Student")
                ai_data = cached_analysis.get("ai_data", {})
                cache_hit = True
            else:
                # Keep the deterministic result as a fallback. The structured
                # model extraction below also returns the recipient.
                rule_based_name = get_student_name_from_text(
                    text_source,
                    allow_llm_fallback=False,
                )
                student_name_unstructured = rule_based_name
                ai_data = {}
                cache_hit = False

            loading_msg = ui_translator.get_string("📜 Running encrypted processing pipelines...", current_lang)
            with st.spinner(loading_msg):  # type: ignore
                if cache_hit:
                    audit_logger.log_data_processing(
                        stage="UNSTRUCTURED_LLM_CACHE_HIT",
                        records_processed=1,
                    )
                else:
                    audit_logger.log_data_processing(
                        stage="UNSTRUCTURED_PII_REDACTION",
                        records_processed=1,
                        pii_redacted=redacted_count,
                    )

                    import time
                    start_time = time.time()

                    ai_data = _extract_certificate_data_with_llm(safe_text_source)
                    student_name_unstructured = _resolve_unstructured_student_name(
                        rule_based_name,
                        ai_data,
                        text_source,
                    )

                    latency = (time.time() - start_time) * 1000
                    audit_logger.log_api_call(
                        api_service="vLLM (local)",
                        endpoint="_extract_certificate_data_with_llm",
                        request_size=len(safe_text_source),
                        response_time_ms=latency,
                        status_code=200 if ai_data else 500,
                        pii_protected=True,
                    )

                    # Do not cache an empty result: it may be the graceful
                    # fallback while a local server is starting and should be
                    # retried on a later rerun.
                    if any(ai_data.get(key) for key in ("student_name", "certificates", "skills", "summary")):
                        analysis_cache[analysis_cache_key] = {
                            "student_name": student_name_unstructured,
                            "ai_data": ai_data,
                        }
                        while len(analysis_cache) > 8:
                            analysis_cache.pop(next(iter(analysis_cache)))

                certificates = ai_data.get("certificates", [])
                skills = ai_data.get("skills", [])
                summary = ai_data.get("summary", "")

                st.markdown(f"""
                <div style="
                    padding: 12px 16px;
                    border-radius: 8px;
                    background-color: rgba(0,150,255,0.15);
                    border-left: 4px solid rgba(0,200,255,0.6);
                    color: #e6eef8;
                    margin-bottom: 16px;
                ">
                    ✅ <strong>Student:</strong> {_display_student_name(student_name_unstructured)}
                </div>
                """, unsafe_allow_html=True)

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
                            translator.translate_text(
                                str(row.iloc[0]), current_lang,
                                protected_values=[student_name_unstructured],
                            ),
                            str(row.iloc[1]),
                            translator.translate_text(
                                str(row.iloc[2]), current_lang,
                                protected_values=[student_name_unstructured],
                            ),
                            translator.translate_text(
                                str(row.iloc[3]), current_lang,
                                protected_values=[student_name_unstructured],
                            )
                        ])

                    translated_cert_df = pd.DataFrame(translated_rows, columns=translated_cols)
                    st.table(translated_cert_df)
                else:
                    st.info(ui_translator.get_string("No specific certificate details found.", current_lang))

                charts_images = []
                if certificates or skills:
                    skills_df = pd.DataFrame(skills)
                    maybe_images = render_visualizations(
                        skills_df,
                        text_source,
                        certificates=certificates,
                    )
                    if isinstance(maybe_images, list):
                        charts_images = maybe_images

                if summary:
                    st.markdown("---")
                    st.subheader(ui_translator.get_string("📝 Professional Executive Summary", current_lang))

                    translated_summary_text = translator.translate_text(
                        summary,
                        current_lang,
                        protected_values=[student_name_unstructured],
                    )
                    animated_summary(translated_summary_text)

                    audit_logger.log_summary_generation(
                        input_size=len(safe_text_source),
                        output_size=len(summary),
                        model=get_configured_model(),
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
                        charts_images=charts_images,
                        page_images=[],
                        summary=translated_summary_text,
                    )

                    create_download_button(
                        pdf_bytes,
                        filename=_safe_report_filename(student_name_unstructured),
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
        ✅ <strong>Student:</strong> {_display_student_name(student_name)}
    </div>
    """, unsafe_allow_html=True)

    # =====================================================
    # NORMALIZE
    # =====================================================
    if not {"Score", "Maximum"}.issubset(df.columns):
        df = normalize_uploaded_dataframe(df)

    df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
    df["Maximum"] = pd.to_numeric(df["Maximum"], errors="coerce").fillna(100.0)

    score_rows = df["Score"].notna()
    invalid_scores = (
        not df.loc[score_rows, "Score"].map(math.isfinite).all()
        or not df.loc[score_rows, "Maximum"].map(math.isfinite).all()
        or (df.loc[score_rows, "Score"] < 0).any()
        or (df.loc[score_rows, "Maximum"] <= 0).any()
        or (df.loc[score_rows, "Score"] > df.loc[score_rows, "Maximum"]).any()
    )
    if invalid_scores:
        st.error(
            "Scores must be finite, non-negative values that do not exceed their maximum.",
            icon="⚠️",
        )
        st.stop()

    cleaned_df = clean_dataframe(df)
    cleaned_df = normalize_scores_to_percentage(cleaned_df)
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
    from backend.pii_protection import PIIProtector
    pii_protector = PIIProtector()
    
    if "audit_logger" in st.session_state:
        audit_logger = st.session_state.get("audit_logger")
    else:
        from backend.audit_logging import AuditLogger
        audit_logger = AuditLogger(custom_session_id=st.session_state["user_session_token"])

    cleaned_df = df
    summary_needs_update = bool(st.session_state.get("summary_needs_update", False))
    
    # Handle plot stats updates if needed
    if ('current_plot_stats' in st.session_state and st.session_state['current_plot_stats'] and
            summary_needs_update):
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
    safe_extracted_text = pii_protector.redact_pii(extracted_text or "")
    stats["summary_context"] = safe_extracted_text

    audit_logger.log_data_processing(
        stage="AI_PROMPT_PII_REDACTION",
        records_processed=len(cleaned_df),
        pii_redacted=len(pii_protector.redacted_map)
    )

    # ``student_name`` was resolved by ``process_structured_data``.  Preserve
    # it for the report rather than replacing it with a differently-cased
    # metadata key (Excel metadata uses ``Student Name``).
    student_name = str(student_name or "").strip()
    if not student_name and isinstance(student_info, dict):
        student_name = str(
            student_info.get("Student Name")
            or student_info.get("student_name")
            or "Unknown Student"
        ).strip()
    stats['student_name'] = student_name or "Unknown Student"

    # A normal Streamlit interaction re-executes this whole file.  Cache the
    # completed report by its redacted evidence so a language switch, chart
    # interaction, or download-button click cannot queue another local LLM job.
    cache_key = _summary_cache_key(stats, safe_extracted_text)
    summary_cache = _get_session_cache("llm_summary_cache")
    refresh_requested = st.button(
        ui_translator.get_string("regenerate_ai_summary", current_lang),
        key="regenerate_ai_summary",
        help="Run the local AI again for the current data.",
    )
    if refresh_requested:
        summary_cache.pop(cache_key, None)

    summary = summary_cache.get(cache_key)
    summary_from_cache = isinstance(summary, str) and bool(summary.strip())

    if summary_from_cache:
        audit_logger.log_data_processing(
            stage="LLM_SUMMARY_CACHE_HIT",
            records_processed=len(cleaned_df),
        )
    else:
        try:
            import time
            start_time = time.time()

            if summary_needs_update:
                insight_msg = ui_translator.get_string("updating_summary", current_lang)
            else:
                insight_msg = ui_translator.get_string("🔒 Generating secure encrypted insight narrative...", current_lang)

            with st.spinner(insight_msg):  # type: ignore
                summary = generate_summary(stats, safe_extracted_text, mode="insight")

            latency = (time.time() - start_time) * 1000
            audit_logger.log_api_call(
                api_service="vLLM (local)",
                endpoint="generate_summary",
                request_size=len(str(stats)) + len(safe_extracted_text),
                response_time_ms=latency,
                status_code=200 if summary else 500,
                pii_protected=True,
            )
        except Exception as e:
            audit_logger.log_error(error_type="LLM_PROMPT_ERROR", message=str(e), stage="SUMMARY_GENERATION")
            summary = generate_summary(stats, safe_extracted_text, mode="insight")

        # The rule-based fallback is still useful output; cache it too, with an
        # explicit Regenerate button available once the local server is ready.
        if isinstance(summary, str) and summary.strip():
            summary_cache[cache_key] = summary
            while len(summary_cache) > 8:
                summary_cache.pop(next(iter(summary_cache)))

    translated_summary = translator.translate_text(
        summary,
        current_lang,
        protected_values=[student_name],
    )

    from backend.ui_animations import animated_summary
    animated_summary(translated_summary)

    if not summary_from_cache:
        audit_logger.log_summary_generation(
            input_size=len(str(stats)) + len(safe_extracted_text),
            output_size=len(summary) if summary else 0,
            model=get_configured_model(),
            tokens_used=0,
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
        filename=_safe_report_filename(student_name),
        label=ui_translator.get_string("⬇️ Download Black and White Report", current_lang)
    )

    audit_logger.log_data_deletion(data_type="StructuredDataFramePayload", records_deleted=len(cleaned_df))

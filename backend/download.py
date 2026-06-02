# backend/download.py
import io
import os
import textwrap
from typing import List, Optional, Sequence, Any
from datetime import datetime

import pandas as pd
import streamlit as st

from backend.top5 import get_top5_numerical_rows
from backend.deepl_translator import ui_translator, translator

try:
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        Image as RLImage,
        PageBreak,
    )
    from PIL import Image as PILImage

    # --- UNICODE FONT REGISTRATION SYSTEM ---
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    # Priority checklist for multi-language local project font paths, fallback to systems
    POSSIBLE_FONT_PATHS = [
        os.path.join(os.getcwd(), "fonts", "NotoSansTC-Regular.ttf"),  # Local Project Workspace CJK
        os.path.join(os.getcwd(), "fonts", "NotoSans-Regular.ttf"),  # Local Project Workspace Latin
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux / Ubuntu standard
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",  # Linux alternate
        "C:\\Windows\\Fonts\\arial.ttf",  # Windows
        "/System/Library/Fonts/Supplemental/Arial.ttf",  # macOS (Newer)
        "/Library/Fonts/Arial.ttf"  # macOS (Older)
    ]

    # Try to load an external TrueType font to support multi-language characters safely
    CHOSEN_FONT = "Helvetica"
    for path in POSSIBLE_FONT_PATHS:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont("AppMultiLanguageFont", path))
                CHOSEN_FONT = "AppMultiLanguageFont"
                break
            except Exception:
                continue

    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        try:
            return str(value)
        except Exception:
            return ""
    return str(value)


def _truncate_text(text: str, max_chars: int = 8000) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _prepare_table_data(df: pd.DataFrame, max_rows: int = 5, max_cell_chars: int = 200):
    if df is None or df.empty:
        return [["No tabular data available."]]
    display_df = df.head(max_rows).copy()
    cols = [str(c) for c in display_df.columns]
    table_data = [cols]
    for _, row in display_df.iterrows():
        row_vals = []
        for c in display_df.columns:
            val = row.get(c, "")
            s = _safe_text(val)
            if len(s) > max_cell_chars:
                s = s[: max_cell_chars - 3] + "..."
            row_vals.append(s)
        table_data.append(row_vals)
    return table_data


def _append_image_bytes_to_story(img_bytes: bytes, story: list, max_width_mm: float = 160.0):
    try:
        pil = PILImage.open(io.BytesIO(img_bytes)).convert("L")
        max_width = max_width_mm * mm
        width_px, height_px = pil.size
        if width_px == 0:
            return
        aspect = height_px / float(width_px)
        display_width = max_width
        display_height = display_width * aspect
        tmp = io.BytesIO()
        pil.save(tmp, format="PNG")
        tmp.seek(0)
        rl_img = RLImage(tmp, width=display_width, height=display_height)
        story.append(rl_img)
        story.append(Spacer(1, 6))
    except Exception:
        return


def create_report(stats_dict: dict, summary_text: str, language: str = 'en') -> bytes:
    """
    Simpler layout engine parsing dictionary snapshots directly into
    translated document tables.
    """
    if not REPORTLAB_AVAILABLE:
        return f"ReportLab Unavailable. Summary: {summary_text}".encode("utf-8")

    label_keys = [
        'Student Learning Journey Summary',
        'Average Score',
        'Highest Subject',
        'Lowest Subject',
        'Analysis Summary',
        'Report Generated On'
    ]

    labels = {key: translator.translate_text(key, language) for key in label_keys}

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=18 * mm, leftMargin=18 * mm, topMargin=18 * mm,
                            bottomMargin=18 * mm)
    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'Title', parent=styles['Heading1'], fontName=CHOSEN_FONT, fontSize=18, textColor=colors.HexColor('#1B7F4E'),
        alignment=1
    )
    story.append(Paragraph(labels.get('Student Learning Journey Summary', 'Report'), title_style))
    story.append(Spacer(1, 15))

    avg_val = stats_dict.get('average', stats_dict.get('averages', {}).get('Score', 'N/A'))
    avg_display = f"{avg_val:.2f}" if isinstance(avg_val, (int, float)) else str(avg_val)
    highest_subj = stats_dict.get('highest_subject', stats_dict.get('strength', 'N/A'))
    lowest_subj = stats_dict.get('lowest_subject', stats_dict.get('weakness', 'N/A'))

    stats_table_data = [
        [labels.get('Average Score', 'Average Score'), avg_display],
        [labels.get('Highest Subject', 'Highest Subject'), str(highest_subj)],
        [labels.get('Lowest Subject', 'Lowest Subject'), str(lowest_subj)],
    ]

    table = Table(stats_table_data, colWidths=[70 * mm, 100 * mm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), CHOSEN_FONT),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(table)
    story.append(Spacer(1, 20))

    header_style = ParagraphStyle(
        'Header', parent=styles['Heading2'], fontName=CHOSEN_FONT, fontSize=14, textColor=colors.HexColor('#1B7F4E')
    )
    story.append(Paragraph(labels.get('Analysis Summary', 'Analysis Summary'), header_style))
    story.append(Spacer(1, 8))

    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontName=CHOSEN_FONT, fontSize=11, leading=14)
    story.append(Paragraph(summary_text if summary_text else "", body_style))
    story.append(Spacer(1, 15))

    footer_text = f"{labels.get('Report Generated On', 'Generated On')}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    footer_style = ParagraphStyle('Footer', parent=styles['Normal'], fontName=CHOSEN_FONT, fontSize=9,
                                  textColor=colors.grey, alignment=1)
    story.append(Paragraph(footer_text, footer_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def get_report_bytes(
        *,
        stats: dict,
        df: pd.DataFrame,
        student_info: dict,
        student_name: str,
        extracted_text: str = "",
        behaviour_traits: Optional[Sequence[dict]] = None,
        charts_images: Optional[Sequence[bytes]] = None,
        page_html: Optional[str] = None,
        page_images: Optional[Sequence[bytes]] = None,
        summary: Optional[str] = None,
) -> bytes:
    """
    Comprehensive multi-page structural analytics engine formatting tabular contexts,
    behavior indices, and raw image buffers directly into an A4 PDF canvas.
    """
    behaviour_traits = behaviour_traits or []
    charts_images = charts_images or []
    page_images = page_images or []

    # Dynamically pull the currently selected active language from session state
    current_lang = st.session_state.get("selected_language", "en")
    final_summary = summary or ""

    averages = stats.get("averages", {}) if isinstance(stats.get("averages", {}), dict) else {}
    avg_score = averages.get("Score", "")
    highest_score = stats.get("highest_score", "") or stats.get("max_score", "")
    lowest_score = stats.get("lowest_score", "") or stats.get("min_score", "")

    if df is None:
        df = pd.DataFrame()

    # --- FIXED: EXTRACT & TRANSLATE TOP 5 DATAFRAME ON THE FLY ---
    top5_df = get_top5_numerical_rows(df)

    if not top5_df.empty and current_lang != "en":
        # Translate row subject labels safely
        top5_df['Label'] = top5_df['Label'].apply(
            lambda x: translator.translate_text(str(x), current_lang)
        )
        # Translate column names ('Label', 'Score') to match UI execution structure
        top5_df.columns = [
            translator.translate_text(col, current_lang) for col in top5_df.columns
        ]

    if REPORTLAB_AVAILABLE:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=18 * mm,
            leftMargin=18 * mm,
            topMargin=18 * mm,
            bottomMargin=18 * mm,
        )

        styles = getSampleStyleSheet()

        normal = ParagraphStyle(
            "NormalBW",
            parent=styles["Normal"],
            fontName=CHOSEN_FONT,
            fontSize=10,
            leading=14,
            textColor=colors.black,
        )
        heading = ParagraphStyle(
            "HeadingBW",
            parent=styles["Heading2"],
            fontName=CHOSEN_FONT,
            fontSize=14,
            leading=16,
            textColor=colors.black,
            spaceAfter=8,
        )
        small_bold = ParagraphStyle(
            "SmallBold",
            parent=styles["Normal"],
            fontName=CHOSEN_FONT,
            fontSize=10,
            leading=12,
            textColor=colors.black,
            spaceAfter=4,
            spaceBefore=6,
        )

        story = []

        report_title_label = ui_translator.get_string("Student Learning Journey Summary", current_lang)
        story.append(Paragraph(f"{report_title_label} — {_safe_text(student_name)}", heading))
        story.append(Spacer(1, 6))

        story.append(Paragraph(f"<b>{ui_translator.get_string('Key Metrics', current_lang)}</b>", small_bold))
        metrics = [
            [ui_translator.get_string("Average Score", current_lang), _safe_text(avg_score)],
            [ui_translator.get_string("Highest Score", current_lang), _safe_text(highest_score)],
            [ui_translator.get_string("Lowest Score", current_lang), _safe_text(lowest_score)],
            [ui_translator.get_string("Total Records", current_lang), _safe_text(len(df))],
        ]

        t = Table(metrics, colWidths=[75 * mm, 95 * mm])
        t.setStyle(
            TableStyle(
                [
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                    ("FONTNAME", (0, 0), (-1, -1), CHOSEN_FONT),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(t)
        story.append(Spacer(1, 10))

        if behaviour_traits:
            story.append(Paragraph(f"<b>{ui_translator.get_string('Behaviour Traits', current_lang)}</b>", small_bold))
            for b in behaviour_traits:
                if isinstance(b, dict):
                    trait = b.get("trait") or b.get("name") or ""
                    evidence = b.get("evidence") or b.get("note") or ""
                    story.append(Paragraph(f"<b>{_safe_text(trait)}</b>: {_safe_text(evidence)}", normal))
                else:
                    story.append(Paragraph(_safe_text(b), normal))
            story.append(Spacer(1, 10))

        story.append(Paragraph(f"<b>{ui_translator.get_string('🎓 Top 5 Grades', current_lang)}</b>", small_bold))
        table_data = _prepare_table_data(top5_df, max_rows=5)

        if table_data and len(table_data) == 1 and table_data[0][0] == "No tabular data available.":
            story.append(
                Paragraph(ui_translator.get_string("No specific certificate details found.", current_lang), normal))
        else:
            col_count = len(table_data[0])
            col_width = (170 * mm) / max(col_count, 1)
            t2 = Table(table_data, colWidths=[col_width] * col_count)
            t2.setStyle(
                TableStyle(
                    [
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
                        ("FONTNAME", (0, 0), (-1, -1), CHOSEN_FONT),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 4),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ]
                )
            )
            story.append(t2)

        story.append(Spacer(1, 10))

        if charts_images:
            story.append(Paragraph("<b>Charts</b>", small_bold))
            for img in charts_images:
                _append_image_bytes_to_story(img, story)
            story.append(Spacer(1, 6))

        if page_images:
            story.append(Paragraph("<b>Page Screenshots</b>", small_bold))
            for img in page_images:
                _append_image_bytes_to_story(img, story)
            story.append(Spacer(1, 6))

        if final_summary:
            story.append(
                Paragraph(f"<b>{ui_translator.get_string('📝 Professional Executive Summary', current_lang)}</b>",
                          small_bold))
            for chunk in textwrap.wrap(_truncate_text(final_summary, max_chars=8000), 1000):
                story.append(Paragraph(chunk, normal))
            story.append(Spacer(1, 6))

        doc.build(story)
        buffer.seek(0)
        return buffer.read()

    fallback = io.StringIO()
    fallback.write(f"Student Report — {_safe_text(student_name)}\n\n")
    fallback.write("Top Records\n")
    if not top5_df.empty:
        fallback.write(top5_df.to_string(index=False))
    else:
        fallback.write("No tabular data available.\n")
    return fallback.getvalue().encode("utf-8")


def create_download_button(
        pdf_bytes: bytes,
        filename: str = "student_report.pdf",
        label: str = "Download Black and White Report",
        mime: str = "application/pdf",
):
    if not isinstance(pdf_bytes, (bytes, bytearray)):
        pdf_bytes = bytes(pdf_bytes)

    if not REPORTLAB_AVAILABLE and filename.lower().endswith(".pdf"):
        filename = filename.rsplit(".", 1)[0] + ".txt"
        mime = "text/plain"

    st.download_button(
        label=label,
        data=pdf_bytes,
        file_name=filename,
        mime=mime,
    )
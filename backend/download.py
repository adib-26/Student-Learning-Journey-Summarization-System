# download.py

import io
import textwrap
from typing import List, Optional, Sequence, Any

import pandas as pd
import streamlit as st

from backend.top5 import get_top5_numerical_rows

try:
    from reportlab.lib.pagesizes import A4
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
    behaviour_traits = behaviour_traits or []
    charts_images = charts_images or []
    page_images = page_images or []

    # ✅ ONLY use Streamlit summary
    final_summary = summary or ""

    averages = stats.get("averages", {}) if isinstance(stats.get("averages", {}), dict) else {}
    avg_score = averages.get("Score", "")
    highest_score = stats.get("highest_score", "") or stats.get("max_score", "")
    lowest_score = stats.get("lowest_score", "") or stats.get("min_score", "")

    if df is None:
        df = pd.DataFrame()

    top5_df = get_top5_numerical_rows(df)

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
            fontName="Helvetica",
            fontSize=10,
            leading=12,
            textColor=colors.black,
        )
        heading = ParagraphStyle(
            "HeadingBW",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=15,
            textColor=colors.black,
            spaceAfter=6,
        )
        small_bold = ParagraphStyle(
            "SmallBold",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=11,
            textColor=colors.black,
        )

        story = []

        story.append(Paragraph(f"Student Report — {_safe_text(student_name)}", heading))
        story.append(Spacer(1, 6))

        story.append(Paragraph("<b>Key Metrics</b>", small_bold))
        metrics = [
            ["Average Score", _safe_text(avg_score)],
            ["Highest Score", _safe_text(highest_score)],
            ["Lowest Score", _safe_text(lowest_score)],
            ["Total Records", _safe_text(len(df))],
        ]

        t = Table(metrics, colWidths=[70 * mm, 80 * mm])
        t.setStyle(
            TableStyle(
                [
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(t)
        story.append(Spacer(1, 8))

        if behaviour_traits:
            story.append(Paragraph("<b>Behaviour Traits</b>", small_bold))
            for b in behaviour_traits:
                if isinstance(b, dict):
                    trait = b.get("trait") or b.get("name") or ""
                    evidence = b.get("evidence") or b.get("note") or ""
                    story.append(Paragraph(f"<b>{_safe_text(trait)}</b>: {_safe_text(evidence)}", normal))
                else:
                    story.append(Paragraph(_safe_text(b), normal))
            story.append(Spacer(1, 8))

        story.append(Paragraph("<b>Top Records</b>", small_bold))
        table_data = _prepare_table_data(top5_df, max_rows=5)

        if table_data and len(table_data) == 1 and table_data[0][0] == "No tabular data available.":
            story.append(Paragraph("No tabular data available.", normal))
        else:
            col_count = len(table_data[0])
            col_width = (170 * mm) / max(col_count, 1)
            t2 = Table(table_data, colWidths=[col_width] * col_count)
            t2.setStyle(
                TableStyle(
                    [
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
                        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.white),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                )
            )
            story.append(t2)

        story.append(Spacer(1, 8))

        if student_info:
            story.append(Paragraph("<b>Student Details</b>", small_bold))
            for k, v in student_info.items():
                if k == "Student Name":
                    continue
                story.append(Paragraph(f"<b>{_safe_text(k)}</b>: {_safe_text(v)}", normal))
            story.append(Spacer(1, 8))

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
            story.append(Paragraph("<b>Executive Summary</b>", small_bold))
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
# text_visualizations.py
import re
from typing import List, Optional
import pandas as pd
import plotly.express as px
import streamlit as st

# Try to import the Gemini extraction helper from your backend.
# If it's not available, we fall back to expecting certificates passed directly.
try:
    from backend.text_info_extractor import _extract_certificate_data_with_gemini
except Exception:
    _extract_certificate_data_with_gemini = None


# -----------------------------
# Certificate date parsing (from AI output)
# -----------------------------
def _parse_certificate_date_string(token: str) -> Optional[pd.Timestamp]:
    """
    Parse a date string coming from certificate data.
    Accepts many human formats and returns a pandas.Timestamp or None.

    Improvements:
    - Explicit handling for short numeric formats like "6/2/6" or "6-2-6".
      These are interpreted as day/month/year by default and two-digit
      years are expanded to 4 digits (00-49 -> 2000-2049, 50-99 -> 1950-1999).
    - Falls back to pandas parsing with multiple heuristics if explicit
      short-format handling doesn't apply.
    """
    if not token or not isinstance(token, str):
        return None

    token = token.strip()
    # Normalize common separators and remove stray characters
    token_norm = re.sub(r"[,\u2011\u2012\u2013\u2014]", " ", token)  # remove dashes and commas
    token_norm = re.sub(r"[\/\-\.]", "/", token_norm)
    token_norm = re.sub(r"\s+", " ", token_norm).strip()

    # --- Explicit short numeric format handling (e.g., 6/2/6, 06/02/06, 6-2-06) ---
    # Pattern: one or two digits / one or two digits / one or two digits
    short_num_match = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{1,2})$", token_norm)
    if short_num_match:
        d1, d2, y = short_num_match.groups()
        try:
            day_candidate = int(d1)
            month_candidate = int(d2)
            year_candidate = int(y)
            # Expand two-digit year to four digits
            if year_candidate < 100:
                year_candidate = 2000 + year_candidate if year_candidate < 50 else 1900 + year_candidate
            # Prefer interpreting as day/month/year (common in many certificates)
            try:
                return pd.Timestamp(year=year_candidate, month=month_candidate, day=day_candidate)
            except Exception:
                # If that fails, try month/day/year
                try:
                    return pd.Timestamp(year=year_candidate, month=day_candidate, day=month_candidate)
                except Exception:
                    pass
        except Exception:
            pass

    # --- If token looks like dd/mm/yyyy or yyyy/mm/dd or similar, try pandas with heuristics ---
    # Try pandas parsing with a few heuristics
    for dayfirst in (False, True):
        for yearfirst in (False, True):
            try:
                dt = pd.to_datetime(token_norm, dayfirst=dayfirst, yearfirst=yearfirst, errors="coerce")
                if pd.notna(dt):
                    return pd.Timestamp(dt.date())
            except Exception:
                pass

    # --- Handle three-part numeric with mixed lengths (e.g., 05/03/2025, 2025/03/05) ---
    m = re.match(r"^(\d{1,4})/(\d{1,2})/(\d{1,4})$", token_norm)
    if m:
        a, b, c = m.groups()
        # Try common orders: d/m/y, y/m/d
        try:
            # d/m/y
            d, mth, y = int(a), int(b), int(c)
            if y < 100:
                y = 2000 + y if y < 50 else 1900 + y
            try:
                return pd.Timestamp(year=y, month=mth, day=d)
            except Exception:
                pass
        except Exception:
            pass
        try:
            # y/m/d
            y, mth, d = int(a), int(b), int(c)
            if y < 100:
                y = 2000 + y if y < 50 else 1900 + y
            try:
                return pd.Timestamp(year=y, month=mth, day=d)
            except Exception:
                pass
        except Exception:
            pass

    # Year-only fallback (e.g., "2025")
    m_year = re.match(r"^(19|20)\d{2}$", token_norm)
    if m_year:
        try:
            return pd.Timestamp(year=int(token_norm), month=1, day=1)
        except Exception:
            pass

    return None


def _build_dates_from_certificates(certificates: List[dict]) -> pd.DataFrame:
    """
    Given a list of certificate dicts (from _extract_certificate_data_with_gemini),
    extract and parse date fields into a DataFrame with columns:
      - certificate_name
      - raw_date
      - parsed_date (Timestamp)
    Only successfully parsed dates are kept.
    """
    rows = []
    if not certificates:
        return pd.DataFrame(columns=["certificate_name", "raw_date", "parsed_date"])

    for cert in certificates:
        raw_date = ""
        name = ""
        if isinstance(cert, dict):
            raw_date = cert.get("date") or cert.get("Date") or cert.get("issued_date") or cert.get("issued") or ""
            name = cert.get("certificate_name") or cert.get("Certificate Name") or cert.get("certificate") or ""
        parsed = _parse_certificate_date_string(str(raw_date)) if raw_date else None
        if parsed is not None and pd.notna(parsed):
            rows.append({
                "certificate_name": name or "Certificate",
                "raw_date": raw_date,
                "parsed_date": pd.Timestamp(parsed)
            })

    if not rows:
        return pd.DataFrame(columns=["certificate_name", "raw_date", "parsed_date"])

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["parsed_date", "certificate_name", "raw_date"]).reset_index(drop=True)
    return df.sort_values("parsed_date").reset_index(drop=True)


# -----------------------------
# Streamlit visualization: Timeline chart (uses AI-extracted certificates)
# -----------------------------
def visualize_dates_from_ai(
    text_content: Optional[str] = None,
    certificates: Optional[List[dict]] = None,
    title: str = "Certificate Timeline"
) -> None:
    """
    Render a timeline-style scatter of certificates.
    - If `certificates` is provided, it will be used directly.
    - Otherwise, if `text_content` is provided and the backend helper
      `_extract_certificate_data_with_gemini` is available, it will call it
      to obtain certificates.
    - No extra aggregated table or bar chart is shown; only a timeline visualization.
    """
    # Prefer explicit certificates passed in
    certs = certificates or []

    # If no certificates passed, try to call backend extractor if available
    if not certs:
        if text_content and _extract_certificate_data_with_gemini:
            try:
                ai_out = _extract_certificate_data_with_gemini(text_content)
                certs = ai_out.get("certificates", []) if isinstance(ai_out, dict) else []
            except Exception as e:
                st.error("Error calling backend extractor: " + str(e))
                certs = []
        else:
            st.info("No certificate data provided and backend extractor not available.")
            return

    # Build dates DataFrame from certificates
    dates_df = _build_dates_from_certificates(certs)

    if dates_df.empty:
        st.info("No valid certificate dates found to visualize.")
        return

    # Timeline scatter: x = date, y = certificate name (ordered by date)
    # Add a small index to preserve order for identical dates
    dates_df = dates_df.reset_index().rename(columns={"index": "idx"})
    dates_df["label"] = dates_df["certificate_name"].fillna("Certificate")

    st.subheader("📅 Certificate Timeline")

    fig = px.scatter(
        dates_df,
        x="parsed_date",
        y="idx",
        hover_data={"label": True, "raw_date": True, "parsed_date": True},
        text="label",
        title=title,
        labels={"parsed_date": "Date", "idx": ""}
    )

    # Make markers and place labels to the right
    fig.update_traces(mode="markers+text", textposition="middle right", marker=dict(size=10))
    fig.update_yaxes(visible=False)  # hide the numeric index axis
    fig.update_layout(xaxis_title="Date", showlegend=False, margin=dict(l=20, r=20, t=50, b=20))
    fig.update_xaxes(tickformat="%b %Y")

    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Backwards-compatible wrapper
# -----------------------------
def visualize_text(
    text: Optional[str] = None,
    *,
    certificates: Optional[List[dict]] = None,
    title: str = "Certificate Timeline"
) -> None:
    """
    Backwards-compatible entry point named `visualize_text`.
    Delegates to visualize_dates_from_ai and shows a timeline only.
    """
    visualize_dates_from_ai(text_content=text, certificates=certificates, title=title)


# -----------------------------
# Convenience wrapper for full pipeline
# -----------------------------
def analyze_and_visualize_dates_from_texts(
    texts: List[str],
    title: str = "Certificate Timeline"
) -> None:
    """
    Merge multiple text sources and visualize using the backend extractor.
    """
    corpus = "\n\n".join(t for t in (texts or []) if t)
    visualize_dates_from_ai(text_content=corpus, title=title)

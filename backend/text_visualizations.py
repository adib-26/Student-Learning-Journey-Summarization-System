# text_visualizations.py
import re
import pandas as pd
import plotly.express as px
import streamlit as st
from collections import Counter
from typing import List, Optional


# -----------------------------
# Default stopwords (extendable)
# -----------------------------
DEFAULT_STOPWORDS = {
    "the", "and", "to", "of", "in", "is", "this", "that",
    "for", "on", "with", "as", "by", "it", "his", "her",
    "was", "were", "are", "from", "at", "an", "be"
}


# -----------------------------
# Text preprocessing
# -----------------------------
def _preprocess_text(
    text: str,
    stopwords: Optional[set] = None
) -> List[str]:
    stopwords = stopwords or DEFAULT_STOPWORDS
    words = re.findall(r"[A-Za-z]{3,}", text.lower())
    return [w for w in words if w not in stopwords]


# -----------------------------
# Keyword extraction
# -----------------------------
def extract_keywords(
    text: str,
    top_n: int = 15
) -> pd.DataFrame:
    if not text:
        return pd.DataFrame(columns=["word", "frequency"])

    words = _preprocess_text(text)
    freq = Counter(words).most_common(top_n)

    return pd.DataFrame(freq, columns=["word", "frequency"])


# -----------------------------
# Visualization (Streamlit)
# -----------------------------
def visualize_text(
    text: str,
    top_n: int = 15
) -> None:
    if not text:
        st.info("No textual content available for visualization.")
        return

    df = extract_keywords(text, top_n)

    if df.empty:
        st.info("No meaningful keywords found.")
        return

    st.subheader("📌 Key Terms Identified from Text")

    # -----------------------------
    # Bar Chart
    # -----------------------------
    fig_bar = px.bar(
        df,
        x="word",
        y="frequency",
        text="frequency",
        title="Top Keywords by Frequency"
    )
    fig_bar.update_traces(textposition="outside")
    st.plotly_chart(fig_bar, use_container_width=True)

    # -----------------------------
    # Bubble Chart
    # -----------------------------
    fig_bubble = px.scatter(
        df,
        x="word",
        y="frequency",
        size="frequency",
        color="frequency",
        hover_name="word",
        title="Keyword Importance Visualization"
    )
    st.plotly_chart(fig_bubble, use_container_width=True)

    # -----------------------------
    # Data Table
    # -----------------------------
    st.subheader("📋 Keyword Frequency Table")
    st.dataframe(df)
import pandas as pd
import plotly.express as px
import streamlit as st
from typing import Dict, Any, Optional

from backend.analytics_statistics import compute_statistics
from backend.data_cleaning import get_auto_y_for_x_column


# -----------------------------
# Data preparation (API-ready)
# -----------------------------
def prepare_visualization_data(df: pd.DataFrame) -> Dict[str, Any]:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    return {
        "numeric_columns": numeric_cols,
        "row_count": len(df),
    }


# -----------------------------
# Streamlit rendering (FYP-2)
# -----------------------------
def render_charts(df: pd.DataFrame) -> None:
    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c.lower() != "maximum"
    ]
    categorical_cols = [col for col in df.columns if df[col].dtype == "object"]

    if not numeric_cols:
        st.info("No numeric data available for visualization.")
        return

    st.subheader("📊 Interactive Performance Visualizations")

    # ─────────────────────────────────────────
    # X-axis selector (categorical columns only)
    # ─────────────────────────────────────────
    none_label = "(none)"
    x_options = [none_label] + categorical_cols

    selected_x_display = st.selectbox("X-axis (categorical)", x_options)
    x_col: Optional[str] = None if selected_x_display == none_label else selected_x_display

    # ─────────────────────────────────────────
    # Auto Y-axis using 3-letter prefix rule
    # ─────────────────────────────────────────
    auto_y: Optional[str] = get_auto_y_for_x_column(df, x_col) if x_col else None

    default_y_idx = (
        numeric_cols.index(auto_y)
        if auto_y and auto_y in numeric_cols
        else 0
    )

    selected_y_col = st.selectbox(
        "Y-axis (numeric)",
        numeric_cols,
        index=default_y_idx,
    )

    if x_col and selected_y_col:
        st.session_state.current_plot_stats = compute_statistics(df, x_col, selected_y_col)
        st.session_state.summary_needs_update = True
        st.info("Academic summary updated based on your selection.")

    if auto_y and auto_y == selected_y_col and x_col:
        st.caption("⚡ Y-axis auto-selected based on column pairing")

    # ─────────────────────────────────────────
    # Bar Chart
    # ─────────────────────────────────────────
    x_vals = df[x_col] if x_col and x_col in df.columns else df.index
    fig_bar = px.bar(
        df,
        x=x_vals,
        y=selected_y_col,
        title=f"{selected_y_col} Distribution",
        labels={"x": x_col or "index", selected_y_col: selected_y_col},
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ─────────────────────────────────────────
    # Line Chart (Trends)
    # ─────────────────────────────────────────
    if len(numeric_cols) > 1:
        fig_line = px.line(
            df,
            y=numeric_cols,
            title="Performance Trends Across Metrics",
        )
        st.plotly_chart(fig_line, use_container_width=True)

    # ─────────────────────────────────────────
    # Pie Chart (Top contributors)
    # ─────────────────────────────────────────
    top_values = (
        df.nlargest(5, selected_y_col)
        if not df[selected_y_col].isnull().all()
        else df.head(5)
    )
    pie_names = df[x_col] if x_col and x_col in df.columns else top_values.index.astype(str)

    fig_pie = px.pie(
        top_values,
        names=pie_names if x_col else top_values.index.astype(str),
        values=selected_y_col,
        title=f"Top 5 Contribution Share — {selected_y_col}",
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # ─────────────────────────────────────────
    # Spider Chart (Radar)
    # ─────────────────────────────────────────
    if len(numeric_cols) > 1:
        st.subheader("🕸️ Spider Chart (Radar Comparison)")

        selected_row = st.selectbox(
            "Select a row to compare across metrics",
            df.index.astype(str),
        )

        row_data = df.loc[int(selected_row), numeric_cols]
        radar_df = pd.DataFrame({"Metric": numeric_cols, "Value": row_data.values})

        fig_radar = px.line_polar(
            radar_df,
            r="Value",
            theta="Metric",
            line_close=True,
            title=f"Radar Chart — Row {selected_row}",
        )
        fig_radar.update_traces(fill="toself")
        st.plotly_chart(fig_radar, use_container_width=True)


# -----------------------------
# Text fallback (non-numeric)
# -----------------------------
def visualize_text_fallback(text: str) -> None:
    if text:
        st.subheader("📄 Extracted Document Text")
        st.text_area("Document Content", text, height=300)
    else:
        st.info("No text content available for visualization.")
# visualizations.py
import pandas as pd
import plotly.express as px
import streamlit as st
from typing import Dict, Any


# -----------------------------
# Data preparation (API-ready)
# -----------------------------
def prepare_visualization_data(df: pd.DataFrame) -> Dict[str, Any]:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    return {
        "numeric_columns": numeric_cols,
        "row_count": len(df)
    }


# -----------------------------
# Streamlit rendering (FYP-2)
# -----------------------------
# -----------------------------
# Streamlit rendering (FYP-2)
# -----------------------------
def render_charts(df: pd.DataFrame) -> None:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if not numeric_cols:
        st.info("No numeric data available for visualization.")
        return

    st.subheader("📊 Interactive Performance Visualizations")

    # User selects target column
    selected_col = st.selectbox(
        "Select a numeric column",
        numeric_cols
    )

    # -----------------------------
    # Bar Chart
    # -----------------------------
    fig_bar = px.bar(
        df,
        x=df.index,
        y=selected_col,
        title=f"{selected_col} Distribution"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # -----------------------------
    # Line Chart (Trends)
    # -----------------------------
    if len(numeric_cols) > 1:
        fig_line = px.line(
            df,
            y=numeric_cols,
            title="Performance Trends Across Metrics"
        )
        st.plotly_chart(fig_line, use_container_width=True)

    # -----------------------------
    # Pie Chart (Top contributors)
    # -----------------------------
    if not df[selected_col].isnull().all():
        top_values = df.nlargest(5, selected_col)
    else:
        top_values = df.head(5)

    fig_pie = px.pie(
        top_values,
        names=top_values.index.astype(str),
        values=selected_col,
        title=f"Top 5 Contribution Share for {selected_col}"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # -----------------------------
    # Spider Chart (Radar)
    # -----------------------------
    if len(numeric_cols) > 1:
        st.subheader("🕸️ Spider Chart (Radar Comparison)")
        # Let user pick a row to visualize across all numeric columns
        selected_row = st.selectbox(
            "Select a row to compare across metrics",
            df.index.astype(str)
        )
        row_data = df.loc[int(selected_row), numeric_cols]

        radar_df = pd.DataFrame({
            "Metric": numeric_cols,
            "Value": row_data.values
        })

        fig_radar = px.line_polar(
            radar_df,
            r="Value",
            theta="Metric",
            line_close=True,
            title=f"Radar Chart for Row {selected_row}"
        )
        fig_radar.update_traces(fill="toself")
        st.plotly_chart(fig_radar, use_container_width=True)


# -----------------------------
# Text fallback (non-numeric)
# -----------------------------
def visualize_text_fallback(text: str) -> None:
    if text:
        st.subheader("📄 Extracted Document Text")
        st.text_area(
            "Document Content",
            text,
            height=300
        )
    else:
        st.info("No text content available for visualization.")
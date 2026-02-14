import streamlit as st
import plotly.express as px
import pandas as pd
from typing import Optional

from backend.data_cleaning import get_valid_x_axis_columns
from backend.text_visualizations import visualize_text

# -----------------------------
# Helper for plotting
# -----------------------------
def prepare_plot_df(df: pd.DataFrame, x_col: Optional[str], y_col: str) -> pd.DataFrame:
    plot_df = df.copy()

    if y_col not in plot_df.columns:
        return plot_df.iloc[0:0]

    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
    plot_df = plot_df[plot_df[y_col].notna()]

    if x_col and x_col in plot_df.columns:
        if plot_df[x_col].dtype == "object" or not plot_df[x_col].is_unique:
            return plot_df.groupby(x_col, dropna=False)[y_col].mean().reset_index()

    return plot_df


# -----------------------------
# Visualization Section
# -----------------------------
def render_visualizations(cleaned_df: pd.DataFrame, extracted_text: str):

    numeric_cols = cleaned_df.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c.lower() != "maximum"]

    valid_x_cols = get_valid_x_axis_columns(cleaned_df)

    if not numeric_cols:
        visualize_text(extracted_text)
        return

    st.subheader("🎛 Interactive Visualization")

    chart_type = st.radio(
        "Choose chart type:",
        ("Bar", "Line", "Area", "Scatter", "Pie", "Spider"),
        horizontal=True
    )

    x_axis = st.selectbox("X-axis", ["(none)"] + valid_x_cols)
    x_axis = None if x_axis == "(none)" else x_axis

    y_axis = st.selectbox("Y-axis (numeric)", numeric_cols)

    plot_df = prepare_plot_df(cleaned_df, x_axis, y_axis)

    fig = None

    # -----------------------------
    # Standard charts
    # -----------------------------
    if chart_type == "Bar":
        fig = px.bar(plot_df, x=x_axis or plot_df.index.astype(str), y=y_axis)

    elif chart_type == "Line":
        fig = px.line(plot_df, x=x_axis or plot_df.index.astype(str), y=y_axis, markers=True)

    elif chart_type == "Area":
        fig = px.area(plot_df, x=x_axis or plot_df.index.astype(str), y=y_axis)

    elif chart_type == "Scatter":
        fig = px.scatter(plot_df, x=x_axis or plot_df.index.astype(str), y=y_axis)

    elif chart_type == "Pie":
        if x_axis:
            fig = px.pie(plot_df, names=x_axis, values=y_axis)
        else:
            st.warning("Pie chart requires a categorical X-axis.")

    # -----------------------------
    # Spider / Radar chart
    # -----------------------------
    elif chart_type == "Spider":
        st.subheader("🕸️ Spider Chart (Overall Student Performance)")

        if "Label" not in cleaned_df.columns or "Score" not in cleaned_df.columns:
            st.warning("Spider chart requires Label and Score columns.")
            return

        radar_df = cleaned_df[["Label", "Score"]].copy()
        radar_df["Score"] = pd.to_numeric(radar_df["Score"], errors="coerce")
        radar_df = radar_df.dropna(subset=["Score"])

        if radar_df.shape[0] < 3:
            st.warning("Not enough numeric data to build a spider chart.")
            return

        radar_df["Entity"] = "Student Performance"

        fig = px.line_polar(
            radar_df,
            r="Score",
            theta="Label",
            color="Entity",
            line_close=True,
            title="Holistic Student Performance Radar",
            color_discrete_sequence=[px.colors.qualitative.Plotly[0]]
        )

        fig.update_traces(
            fill="toself",
            line=dict(width=3),
            marker=dict(size=6)
        )

        # ✅ Remove crosshair / axis line
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    showline=False,
                    gridcolor="rgba(0,0,0,0.2)"
                ),
                angularaxis=dict(
                    showline=False,
                    gridcolor="rgba(0,0,0,0.2)"
                )
            ),
            showlegend=False
        )

    if fig:
        st.plotly_chart(fig, use_container_width=True)
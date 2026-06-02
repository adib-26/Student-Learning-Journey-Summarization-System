import io
import streamlit as st
import plotly.express as px
import pandas as pd
from typing import Optional, List

from backend.data_cleaning import get_valid_x_axis_columns
from backend.text_visualizations import visualize_text

# Import the dual-layered translation framework
from backend.deepl_translator import translator, ui_translator


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
def render_visualizations(cleaned_df: pd.DataFrame, extracted_text: str) -> List[bytes]:
    """
    Render visuals to Streamlit and return a list of PNG bytes for each chart rendered.
    If no charts are produced, returns an empty list.
    All chart controls, titles, legends, and dynamic data tags are fully localized.
    """
    images: List[bytes] = []

    numeric_cols = cleaned_df.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c.lower() != "maximum"]

    valid_x_cols = get_valid_x_axis_columns(cleaned_df)

    if not numeric_cols:
        visualize_text(extracted_text)
        return images

    # Identify the globally active runtime language code
    current_lang = st.session_state.get("selected_language", "en")

    st.subheader(ui_translator.get_string("🎛 Interactive Visualization", current_lang))

    # --- 1. LOCALIZED WIDGET CONTROLS ---
    chart_choices = ["Bar", "Line", "Area", "Scatter", "Pie", "Spider"]
    translated_choices_map = {
        choice: ui_translator.get_string(choice, current_lang) for choice in chart_choices
    }

    selected_display_type = st.radio(
        ui_translator.get_string("Choose chart type:", current_lang),
        options=list(translated_choices_map.values()),
        horizontal=True
    )

    # Reverse look-up to resolve English code control flags
    chart_type = [k for k, v in translated_choices_map.items() if v == selected_display_type][0]

    # Handle dropdown lists with context translation rules
    none_label = f"({ui_translator.get_string('none', current_lang)})"

    translated_x_cols_map = {col: translator.translate_text(col, current_lang) for col in valid_x_cols}
    x_display_options = [none_label] + list(translated_x_cols_map.values())

    selected_x_display = st.selectbox(ui_translator.get_string("X-axis", current_lang), x_display_options)

    if selected_x_display == none_label:
        x_axis = None
    else:
        x_axis = [k for k, v in translated_x_cols_map.items() if v == selected_x_display][0]

    translated_y_cols_map = {col: translator.translate_text(col, current_lang) for col in numeric_cols}
    selected_y_display = st.selectbox(
        ui_translator.get_string("Y-axis (numeric)", current_lang),
        list(translated_y_cols_map.values())
    )
    y_axis = [k for k, v in translated_y_cols_map.items() if v == selected_y_display][0]

    # Prepare raw frame slice structures
    plot_df = prepare_plot_df(cleaned_df, x_axis, y_axis)

    # --- 2. DEEP DATA LABELS LOCALIZATION PASS ---
    localized_plot_df = plot_df.copy()
    if current_lang != "en":
        if x_axis and x_axis in localized_plot_df.columns and localized_plot_df[x_axis].dtype == "object":
            localized_plot_df[x_axis] = localized_plot_df[x_axis].apply(
                lambda val: translator.translate_text(str(val), current_lang)
            )

    fig = None

    # Helper to try export Plotly fig to PNG bytes
    def _fig_to_png_bytes(plotly_fig, width: int = 1200, height: int = 600, scale: int = 2) -> Optional[bytes]:
        try:
            img_bytes = plotly_fig.to_image(format="png", width=width, height=height, scale=scale)
            return img_bytes
        except Exception:
            return None

    # Resolve active textual labels for charts
    x_label_text = translated_x_cols_map.get(x_axis, ui_translator.get_string("index", current_lang))
    y_label_text = translated_y_cols_map.get(y_axis, y_axis)

    x_vals = None
    if x_axis and x_axis in localized_plot_df.columns:
        x_vals = localized_plot_df[x_axis]
    else:
        x_vals = localized_plot_df.index.astype(str)

    # -----------------------------
    # Standard charts
    # -----------------------------
    if chart_type == "Bar":
        fig = px.bar(localized_plot_df, x=x_vals, y=y_axis, labels={"x": x_label_text, y_axis: y_label_text})

    elif chart_type == "Line":
        fig = px.line(localized_plot_df, x=x_vals, y=y_axis, markers=True,
                      labels={"x": x_label_text, y_axis: y_label_text})

    elif chart_type == "Area":
        fig = px.area(localized_plot_df, x=x_vals, y=y_axis, labels={"x": x_label_text, y_axis: y_label_text})

    elif chart_type == "Scatter":
        fig = px.scatter(localized_plot_df, x=x_vals, y=y_axis, labels={"x": x_label_text, y_axis: y_label_text})

    elif chart_type == "Pie":
        if x_axis:
            fig = px.pie(localized_plot_df, names=x_axis, values=y_axis,
                         labels={x_axis: x_label_text, y_axis: y_label_text})
        else:
            st.warning(ui_translator.get_string("Pie chart requires a categorical X-axis.", current_lang))

    # -----------------------------
    # Spider / Radar chart
    # -----------------------------
    elif chart_type == "Spider":
        st.subheader(ui_translator.get_string("🕸️ Spider Chart (Overall Student Performance)", current_lang))

        if "Label" not in cleaned_df.columns or "Score" not in cleaned_df.columns:
            st.warning(ui_translator.get_string("Spider chart requires Label and Score columns.", current_lang))
            return images

        radar_df = cleaned_df[["Label", "Score"]].copy()
        radar_df["Score"] = pd.to_numeric(radar_df["Score"], errors="coerce")
        radar_df = radar_df.dropna(subset=["Score"])

        if radar_df.shape[0] < 3:
            st.warning(ui_translator.get_string("Not enough numeric data to build a spider chart.", current_lang))
            return images

        # Translate rows within the radar dataframe's dynamic 'Label' fields
        radar_df["Label"] = radar_df["Label"].apply(lambda label: translator.translate_text(str(label), current_lang))

        # Translate the legendary key tracking name
        radar_df["Entity"] = ui_translator.get_string("Student Performance", current_lang)
        radar_title = ui_translator.get_string("Holistic Student Performance Radar", current_lang)

        fig = px.line_polar(
            radar_df,
            r="Score",
            theta="Label",
            color="Entity",
            line_close=True,
            title=radar_title,
            color_discrete_sequence=[px.colors.qualitative.Plotly[0]]
        )

        fig.update_traces(
            fill="toself",
            line=dict(width=3),
            marker=dict(size=6)
        )

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

    # Render and attempt to capture PNG bytes
    if fig:
        # Standardize chart background look and font scaling configurations for modern look
        fig.update_layout(font=dict(size=13))
        try:
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

        png = _fig_to_png_bytes(fig)
        if png:
            images.append(png)

    return images
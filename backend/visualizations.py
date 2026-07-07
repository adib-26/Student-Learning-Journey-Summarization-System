import streamlit as st
import plotly.express as px
import pandas as pd
from typing import Optional, List

from backend.data_cleaning import get_valid_x_axis_columns, get_auto_y_for_x_column
from backend.text_visualizations import visualize_text

# Import the dual-layered translation framework
from backend.deepl_translator import translator, ui_translator


# -----------------------------
# Helper for plotting
# -----------------------------
def prepare_plot_df(df: pd.DataFrame, x_col: Optional[str], y_col: str) -> pd.DataFrame:
    """Prepare a true data slice for statistics by filtering for valid X and Y data, but NOT grouping."""
    if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
        return pd.DataFrame()

    # Create a copy to avoid modifying the original DataFrame
    plot_df = df.copy()

    # Coerce Y to numeric and drop rows where Y is NaN
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
    
    # Also coerce X to numeric if it's not an object type, to handle cases where X might be a score
    if plot_df[x_col].dtype != 'object':
        plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors='coerce')

    # Drop rows where EITHER x_col or y_col is NaN, creating a true data slice
    plot_df.dropna(subset=[x_col, y_col], inplace=True)

    return plot_df


def prepare_chart_data(df: pd.DataFrame, x_col: Optional[str], y_col: str) -> pd.DataFrame:
    """Prepare grouped data specifically for chart plotting"""
    chart_df = df.copy()
    
    if x_col and x_col in chart_df.columns:
        # We need to use the original, non-numeric y_col data for grouping if it exists
        if chart_df[x_col].dtype == "object" or not chart_df[x_col].is_unique:
            # Ensure y_col is numeric before aggregation
            chart_df[y_col] = pd.to_numeric(chart_df[y_col], errors="coerce")
            return chart_df.groupby(x_col, dropna=False)[y_col].mean().reset_index()
    
    return chart_df



# -----------------------------
# Visualization Section
# -----------------------------
def render_visualizations(cleaned_df: pd.DataFrame, extracted_text: str) -> List[bytes]:
    images: List[bytes] = []

    numeric_cols = cleaned_df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c.lower() != "maximum"]

    valid_x_cols = get_valid_x_axis_columns(cleaned_df)

    if not numeric_cols:
        visualize_text(extracted_text)
        return images

    # Identify the globally active runtime language code
    current_lang = st.session_state.get("selected_language", "en")

    # Helper function to translate values while preserving numeric suffixes (Label 1, Student 2, etc.)
    def translate_with_number_preservation(val):
        val_str = str(val)
        if ' ' in val_str and val_str.split()[-1].isdigit():
            base_val = ' '.join(val_str.split()[:-1])
            number = val_str.split()[-1]
            base_translated = translator.translate_text(base_val, current_lang)
            return f"{base_translated} {number}"
        else:
            return translator.translate_text(val_str, current_lang)

    st.subheader(ui_translator.get_string("🎛 Interactive Visualization", current_lang))

    # ─────────────────────────────────────────
    # 1. CHART TYPE SELECTOR
    # ─────────────────────────────────────────
    chart_choices = ["Bar", "Line", "Area", "Scatter", "Pie", "Spider"]
    translated_choices_map = {
        choice: ui_translator.get_string(choice, current_lang) for choice in chart_choices
    }

    selected_display_type = st.radio(
        ui_translator.get_string("Choose chart type:", current_lang),
        options=list(translated_choices_map.values()),
        horizontal=True,
    )

    # Reverse look-up to resolve English control flag
    chart_type = [k for k, v in translated_choices_map.items() if v == selected_display_type][0]

    none_label = f"({ui_translator.get_string('none', current_lang)})"

    # ─────────────────────────────────────────
    # 2. X-AXIS SELECTOR
    # ─────────────────────────────────────────
    # Use the helper function to ensure consistent translation for all x-axis columns
    translated_x_cols_map = {col: translate_with_number_preservation(col) for col in valid_x_cols}
    
    x_display_options = [none_label] + list(translated_x_cols_map.values())

    # Maintain x-axis selection in session state to preserve it across language changes
    default_x_display = None
    if 'current_x_axis' in st.session_state and st.session_state['current_x_axis'] in translated_x_cols_map:
        default_x_display = translated_x_cols_map[st.session_state['current_x_axis']]
    
    # If we have a default from session state, find its index to pre-select it
    if default_x_display and default_x_display in x_display_options:
        default_x_idx = x_display_options.index(default_x_display)
        selected_x_display = st.selectbox(
            ui_translator.get_string("X-axis", current_lang),
            x_display_options,
            index=default_x_idx,
        )
    else:
        selected_x_display = st.selectbox(
            ui_translator.get_string("X-axis", current_lang),
            x_display_options,
        )
    
    # Update session state with the newly selected x-axis
    if selected_x_display != none_label:
        matching_cols = [k for k, v in translated_x_cols_map.items() if v == selected_x_display]
        if matching_cols:
            st.session_state['current_x_axis'] = matching_cols[0]

    if selected_x_display == none_label:
        x_axis = None
    else:
        # Find the first matching column to avoid index errors with duplicate translations
        matching_cols = [k for k, v in translated_x_cols_map.items() if v == selected_x_display]
        if matching_cols:
            x_axis = matching_cols[0]
        else:
            x_axis = None

    # ─────────────────────────────────────────
    # 3. AUTO Y-AXIS USING 3-LETTER PREFIX RULE
    # ─────────────────────────────────────────
    auto_y_raw: Optional[str] = get_auto_y_for_x_column(cleaned_df, x_axis) if x_axis else None

    # Add number suffix to translations for numeric columns too to ensure uniqueness
    translated_y_cols_map = {}
    for col in numeric_cols:
        translated = translate_with_number_preservation(col)
        translated_y_cols_map[col] = translated
        
    y_display_options = list(translated_y_cols_map.values())

    # Compute the index to pre-select in the Y-axis selectbox
    if auto_y_raw and auto_y_raw in translated_y_cols_map:
        auto_y_display = translated_y_cols_map[auto_y_raw]
        default_y_idx = y_display_options.index(auto_y_display)
    else:
        default_y_idx = 0

    selected_y_display = st.selectbox(
        ui_translator.get_string("Y-axis (numeric)", current_lang),
        y_display_options,
        index=default_y_idx,
    )
    # Find the first matching column to avoid index errors with duplicate translations
    matching_y_cols = [k for k, v in translated_y_cols_map.items() if v == selected_y_display]
    if matching_y_cols:
        y_axis = matching_y_cols[0]
    else:
        st.error(ui_translator.get_string("Could not find selected Y-axis column.", current_lang))
        return images

    # Show a subtle hint when auto-selection is active
    if auto_y_raw and auto_y_raw == y_axis and x_axis:
        st.caption(
            ui_translator.get_string("⚡ Y-axis auto-selected based on column pairing", current_lang)
        )

    # ─────────────────────────────────────────
    # 4. PREPARE DATA FRAME SLICE
    # ─────────────────────────────────────────
    # Prepare a data slice for statistics, which preserves all subject rows for context
    stats_df = prepare_plot_df(cleaned_df, x_axis, y_axis)
    # From the stats_df, prepare the grouped and aggregated data needed for plotting
    plot_df = prepare_chart_data(stats_df, x_axis, y_axis)


    
    # Deep data-label localization pass
    localized_plot_df = plot_df.copy()
    if current_lang != "en":
        if (
            x_axis
            and x_axis in localized_plot_df.columns
            and localized_plot_df[x_axis].dtype == "object"
        ):
            localized_plot_df[x_axis] = localized_plot_df[x_axis].apply(translate_with_number_preservation)

    fig = None

    # Helper: export Plotly figure to PNG bytes
    def _fig_to_png_bytes(
        plotly_fig, width: int = 1200, height: int = 600, scale: int = 2
    ) -> Optional[bytes]:
        try:
            return plotly_fig.to_image(format="png", width=width, height=height, scale=scale)
        except Exception:
            return None

    # Resolved display labels for axes
    if x_axis:
        x_label_text = translated_x_cols_map.get(
            x_axis, ui_translator.get_string("index", current_lang)
        )
    else:
        x_label_text = ui_translator.get_string("index", current_lang)
    y_label_text = translated_y_cols_map.get(y_axis, y_axis)

    x_vals = (
        localized_plot_df[x_axis]
        if (x_axis and x_axis in localized_plot_df.columns)
        else localized_plot_df.index.astype(str)
    )

    # ─────────────────────────────────────────
    # 5. CHART RENDERING
    # ─────────────────────────────────────────

    if chart_type == "Bar":
        fig = px.bar(
            localized_plot_df,
            x=x_vals,
            y=y_axis,
            labels={"x": x_label_text, y_axis: y_label_text},
        )

    elif chart_type == "Line":
        fig = px.line(
            localized_plot_df,
            x=x_vals,
            y=y_axis,
            markers=True,
            labels={"x": x_label_text, y_axis: y_label_text},
        )

    elif chart_type == "Area":
        fig = px.area(
            localized_plot_df,
            x=x_vals,
            y=y_axis,
            labels={"x": x_label_text, y_axis: y_label_text},
        )

    elif chart_type == "Scatter":
        fig = px.scatter(
            localized_plot_df,
            x=x_vals,
            y=y_axis,
            labels={"x": x_label_text, y_axis: y_label_text},
        )

    elif chart_type == "Pie":
        if x_axis:
            fig = px.pie(
                localized_plot_df,
                names=x_axis,
                values=y_axis,
                labels={x_axis: x_label_text, y_axis: y_label_text},
            )
        else:
            st.warning(
                ui_translator.get_string("Pie chart requires a categorical X-axis.", current_lang)
            )

    # ─────────────────────────────────────────
    # Spider / Radar chart
    # ─────────────────────────────────────────
    elif chart_type == "Spider":
        st.subheader(
            ui_translator.get_string(
                "🕸️ Spider Chart (Overall Student Performance)", current_lang
            )
        )

        if "Label" not in cleaned_df.columns or "Score" not in cleaned_df.columns:
            st.warning(
                ui_translator.get_string(
                    "Spider chart requires Label and Score columns.", current_lang
                )
            )
            return images

        radar_df = cleaned_df[["Label", "Score"]].copy()
        radar_df["Score"] = pd.to_numeric(radar_df["Score"], errors="coerce")
        radar_df = radar_df.dropna(subset=["Score"])

        if radar_df.shape[0] < 3:
            st.warning(
                ui_translator.get_string(
                    "Not enough numeric data to build a spider chart.", current_lang
                )
            )
            return images

        radar_df["Label"] = radar_df["Label"].apply(translate_with_number_preservation)
        radar_df["Entity"] = ui_translator.get_string("Student Performance", current_lang)
        radar_title = ui_translator.get_string(
            "Holistic Student Performance Radar", current_lang
        )

        fig = px.line_polar(
            radar_df,
            r="Score",
            theta="Label",
            color="Entity",
            line_close=True,
            title=radar_title,
            color_discrete_sequence=[px.colors.qualitative.Plotly[0]],
        )
        fig.update_traces(fill="toself", line=dict(width=3), marker=dict(size=6))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    showline=False,
                    gridcolor="rgba(0,0,0,0.2)",
                ),
                angularaxis=dict(showline=False, gridcolor="rgba(0,0,0,0.2)"),
            ),
            showlegend=False,
        )

    # ─────────────────────────────────────────
    # 6. RENDER + CAPTURE PNG
    # ─────────────────────────────────────────
    if fig:
        fig.update_layout(font=dict(size=13))
        try:
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

        png = _fig_to_png_bytes(fig)
        if png:
            images.append(png)

    return images
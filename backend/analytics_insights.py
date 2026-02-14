import pandas as pd
from typing import Dict

def detect_trends(df: pd.DataFrame) -> Dict[str, str]:
    """
    Detect trends in numeric data.
    """
    trends: Dict[str, str] = {}
    numeric_df = df.select_dtypes(include="number")

    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if len(series) < 2:
            continue

        if series.iloc[-1] > series.iloc[0]:
            trends[col] = "increasing"
        elif series.iloc[-1] < series.iloc[0]:
            trends[col] = "decreasing"
        else:
            trends[col] = "stable"

    return trends


def generate_predictive_insights(df: pd.DataFrame) -> Dict[str, str]:
    """
    Generate predictive insights based on recent performance.
    """
    insights: Dict[str, str] = {}
    numeric_df = df.select_dtypes(include="number")

    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if len(series) < 3:
            continue

        recent_avg = series.tail(3).mean()
        overall_avg = series.mean()

        if recent_avg > overall_avg:
            insights[col] = "Recent performance is above average."
        elif recent_avg < overall_avg:
            insights[col] = "Recent performance is below average."
        else:
            insights[col] = "Performance is consistent."

    return insights

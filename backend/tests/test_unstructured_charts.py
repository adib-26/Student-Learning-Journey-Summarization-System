"""Regression tests for evidence-based charts from unstructured documents."""
from unittest.mock import patch

import pandas as pd


def test_explicit_certificates_do_not_trigger_a_second_llm_request():
    """The current Qwen extraction result must be reused for the timeline."""
    from backend.text_visualizations import visualize_dates_from_ai

    certificates = [
        {
            "certificate_name": "Python Foundations",
            "date": "15/04/2025",
            "location": "",
            "organization": "Open Academy",
        }
    ]

    with patch("backend.text_visualizations._extract_certificate_data_with_llm") as llm, patch(
        "backend.text_visualizations._render_figure", return_value=[b"timeline"]
    ), patch("backend.text_visualizations.st.subheader"):
        images = visualize_dates_from_ai(
            text_content="This text must not be sent for another extraction.",
            certificates=certificates,
        )

    assert images == [b"timeline"]
    llm.assert_not_called()


def test_certificate_without_a_date_gets_an_evidence_count_chart():
    """Certificates still render a truthful chart when no date is available."""
    from backend.text_visualizations import visualize_dates_from_ai

    certificates = [
        {
            "certificate_name": "Leadership Workshop",
            "date": "",
            "location": "",
            "organization": "Learning Centre",
        }
    ]

    with patch("backend.text_visualizations._render_figure", return_value=[b"certificate-chart"]), patch(
        "backend.text_visualizations.st.subheader"
    ), patch("backend.text_visualizations.st.caption"):
        images = visualize_dates_from_ai(certificates=certificates)

    assert images == [b"certificate-chart"]


def test_unstructured_chart_path_passes_existing_certificates_to_visualizer():
    """Certificate-only documents used to skip chart generation entirely."""
    from backend.chart import render_visualizations

    certificates = [{"certificate_name": "Python Foundations", "date": "2025"}]
    with patch("backend.chart.visualize_text", return_value=[b"timeline"]) as visualize_text:
        images = render_visualizations(
            pd.DataFrame(),
            "raw document text",
            certificates=certificates,
        )

    assert images == [b"timeline"]
    visualize_text.assert_called_once_with("raw document text", certificates=certificates)


def test_unscored_qwen_skills_get_a_presence_chart_without_fake_scores():
    from backend.chart import render_visualizations

    skills_df = pd.DataFrame([{"Label": "Python"}, {"Label": "Data Analysis"}])
    with patch(
        "backend.chart.visualize_documented_skills", return_value=[b"skills-chart"]
    ) as visualize_skills, patch("backend.chart.visualize_text") as visualize_text:
        images = render_visualizations(skills_df, "raw document text", certificates=[])

    assert images == [b"skills-chart"]
    visualize_skills.assert_called_once_with(skills_df.to_dict("records"))
    visualize_text.assert_not_called()


def test_certificate_image_without_scores_stays_on_unstructured_path():
    from app_core import _should_handle_as_unstructured

    assert _should_handle_as_unstructured(
        pd.DataFrame({"Label": ["Recipient name"]}), "certificate.png"
    )
    assert not _should_handle_as_unstructured(
        pd.DataFrame({"Label": ["Mathematics"], "Score": [86]}), "marksheet.png"
    )
    assert _should_handle_as_unstructured(
        pd.DataFrame({"Label": ["Certificate"]}), "achievement.pdf"
    )


def test_unstructured_name_resolution_uses_qwen_when_recipient_evidence_matches():
    """The displayed name must consume the one Qwen extraction result."""
    from app_core import _resolve_unstructured_student_name

    evidence = """
    This certificate is proudly presented to
    Nur Aisyah Binti Rahman
    for completing the Python Foundations programme.
    Issued by Example Academy.
    """

    resolved = _resolve_unstructured_student_name(
        "Example Academy",
        {"student_name": "NUR AISYAH BINTI RAHMAN"},
        evidence,
    )

    assert resolved == "Nur Aisyah Binti Rahman"


def test_unstructured_name_resolution_rejects_unsupported_qwen_name():
    """A hallucinated Qwen name must never replace evidence-backed output."""
    from app_core import _resolve_unstructured_student_name

    evidence = "Student Name: Nur Aisyah Binti Rahman\nPython Foundations"
    resolved = _resolve_unstructured_student_name(
        "Nur Aisyah Binti Rahman",
        {"student_name": "Made Up Person"},
        evidence,
    )

    assert resolved == "Nur Aisyah Binti Rahman"

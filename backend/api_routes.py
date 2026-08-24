"""REST resources for student learning analytics."""

from __future__ import annotations

import io
import math
import uuid
from typing import Any, Optional

import pandas as pd
from flask import Blueprint, current_app, g, jsonify, request
from flask_jwt_extended import get_jwt_identity, jwt_required

from .analytics import compute_statistics, detect_trends, generate_predictive_insights
from .api_utils import ApiError, json_body, json_safe, pagination
from .auth import auth_bp
from .data_cleaning import clean_dataframe, normalize_scores_to_percentage
from .file_safety import UnsafeSpreadsheetError, validate_xlsx_archive
from .normalizer import normalize_uploaded_dataframe


api_bp = Blueprint("api_v1", __name__, url_prefix="/api/v1")
api_bp.register_blueprint(auth_bp)


@api_bp.before_request
def assign_request_id():
    g.request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex


@api_bp.after_request
def add_request_headers(response):
    response.headers["X-Request-ID"] = g.request_id
    response.headers["Cache-Control"] = "no-store"
    return response


@api_bp.get("/health")
def health():
    return jsonify({"data": {"status": "ok", "service": "student-learning-api", "version": "v1"}})


def _number(value: Any, field: str, *, minimum: Optional[float] = None) -> float:
    if isinstance(value, bool):
        raise ApiError("VALIDATION_ERROR", f"'{field}' must be a number.", 422)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ApiError("VALIDATION_ERROR", f"'{field}' must be a number.", 422) from exc
    if not math.isfinite(parsed):
        raise ApiError("VALIDATION_ERROR", f"'{field}' must be finite.", 422)
    if minimum is not None and parsed < minimum:
        raise ApiError("VALIDATION_ERROR", f"'{field}' must be at least {minimum}.", 422)
    return parsed


def _records_dataframe(records: Any) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if not isinstance(records, list) or not records:
        raise ApiError("VALIDATION_ERROR", "'records' must be a non-empty array.", 422)
    if len(records) > current_app.config["API_MAX_RECORDS"]:
        raise ApiError("PAYLOAD_TOO_LARGE", "Too many records in one analysis.", 413)

    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ApiError("VALIDATION_ERROR", f"records[{index}] must be an object.", 422)
        label = record.get("subject", record.get("label"))
        if not isinstance(label, str) or not label.strip():
            raise ApiError("VALIDATION_ERROR", f"records[{index}].subject is required.", 422)
        score = _number(record.get("score"), f"records[{index}].score", minimum=0)
        maximum = _number(record.get("maximum", 100), f"records[{index}].maximum", minimum=0.000001)
        if score > maximum:
            raise ApiError("VALIDATION_ERROR", f"records[{index}].score cannot exceed maximum.", 422)

        item = {
            "Section": str(record.get("section", "Subjects")),
            "Label": label.strip(),
            "Score": score,
            "Maximum": maximum,
        }
        if record.get("notes") is not None:
            item["Notes"] = str(record["notes"])
        normalized.append(item)
    return pd.DataFrame(normalized), normalized


def _uploaded_dataframe() -> tuple[pd.DataFrame, str]:
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        raise ApiError("VALIDATION_ERROR", "Upload a CSV or Excel file in the 'file' field.", 422)
    filename = upload.filename.lower()
    if not filename.endswith((".csv", ".xlsx", ".xls")):
        raise ApiError("UNSUPPORTED_FILE_TYPE", "Only CSV, XLSX, and XLS files are supported.", 415)
    raw = upload.read()
    if not raw:
        raise ApiError("VALIDATION_ERROR", "The uploaded file is empty.", 422)
    try:
        if filename.endswith(".csv"):
            try:
                frame = pd.read_csv(io.BytesIO(raw), encoding="utf-8")
            except UnicodeDecodeError:
                frame = pd.read_csv(io.BytesIO(raw), encoding="latin1")
        else:
            if filename.endswith(".xlsx"):
                validate_xlsx_archive(raw)
            frame = pd.read_excel(io.BytesIO(raw))
    except UnsafeSpreadsheetError as exc:
        raise ApiError("PAYLOAD_TOO_LARGE", "The spreadsheet exceeds parsing safety limits.", 413) from exc
    except Exception as exc:
        raise ApiError("INVALID_FILE", "The uploaded file could not be parsed.", 422) from exc

    if frame.empty:
        raise ApiError("VALIDATION_ERROR", "The uploaded file contains no records.", 422)
    frame = clean_dataframe(frame)
    if "Score" not in frame.columns:
        frame = normalize_uploaded_dataframe(frame)
    if "Score" not in frame.columns:
        raise ApiError("VALIDATION_ERROR", "The file must contain a 'Score' column or recognizable score values.", 422)
    frame["Score"] = pd.to_numeric(frame["Score"], errors="coerce")
    frame = frame.dropna(subset=["Score"])
    if frame.empty:
        raise ApiError("VALIDATION_ERROR", "The file contains no numeric scores.", 422)
    if len(frame) > current_app.config["API_MAX_RECORDS"]:
        raise ApiError("PAYLOAD_TOO_LARGE", "Too many records in one analysis.", 413)
    if "Maximum" not in frame.columns:
        frame["Maximum"] = 100.0
    frame["Maximum"] = pd.to_numeric(frame["Maximum"], errors="coerce").fillna(100.0)
    if (
        not frame["Score"].map(math.isfinite).all()
        or not frame["Maximum"].map(math.isfinite).all()
    ):
        raise ApiError(
            "VALIDATION_ERROR",
            "Scores and maximum values must be finite numbers.",
            422,
        )
    if (frame["Score"] < 0).any() or (frame["Maximum"] <= 0).any():
        raise ApiError("VALIDATION_ERROR", "Scores must be non-negative and maximum values must be positive.", 422)
    if (frame["Score"] > frame["Maximum"]).any():
        raise ApiError("VALIDATION_ERROR", "Scores cannot exceed their maximum values.", 422)
    return normalize_scores_to_percentage(frame), upload.filename


def _analysis_payload(data: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], Optional[str]]:
    if request.content_type and request.content_type.startswith("multipart/form-data"):
        frame, filename = _uploaded_dataframe()
        student = data.get("student") if isinstance(data.get("student"), dict) else {}
        student_name = student.get("name") if isinstance(student.get("name"), str) else None
        return frame, {"source": "upload", "filename": filename}, student_name

    payload = json_body()
    frame, records = _records_dataframe(payload.get("records"))
    student = payload.get("student", {})
    if student is None:
        student = {}
    if not isinstance(student, dict):
        raise ApiError("VALIDATION_ERROR", "'student' must be an object.", 422)
    student_name = student.get("name")
    if student_name is not None and (not isinstance(student_name, str) or not student_name.strip()):
        raise ApiError("VALIDATION_ERROR", "student.name must be a non-empty string.", 422)
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ApiError("VALIDATION_ERROR", "'metadata' must be an object.", 422)
    return frame, {"source": "json", "records": records, "metadata": metadata, "student": student}, student_name


def _analysis_view(analysis: dict[str, Any]) -> dict[str, Any]:
    result = analysis["result"]
    return {
        "id": analysis["id"],
        "student": {"name": analysis["student_name"]},
        "created_at": analysis["created_at"],
        "metrics": result["metrics"],
        "subjects": result["subjects"],
        "trends": result["trends"],
        "predictive_insights": result["predictive_insights"],
        "links": {
            "self": f"/api/v1/analyses/{analysis['id']}",
            "collection": "/api/v1/analyses",
        },
    }


@api_bp.post("/analyses")
@jwt_required()
def create_analysis():
    frame, input_payload, student_name = _analysis_payload(request.form.to_dict() if request.form else {})
    frame = normalize_scores_to_percentage(frame)
    stats = compute_statistics(frame)
    trends = detect_trends(frame)
    predictive = generate_predictive_insights(frame)
    result = {
        "metrics": {
            "row_count": stats.get("row_count", len(frame)),
            "column_count": stats.get("column_count", len(frame.columns)),
            "averages": stats.get("averages", {}),
            "medians": stats.get("medians", {}),
            "std_dev": stats.get("std_dev", {}),
            "counts": stats.get("counts", {}),
            "strength": stats.get("strength"),
            "weakness": stats.get("weakness"),
        },
        "subjects": {
            "names": stats.get("subjects", []),
            "scores": stats.get("subject_scores", {}),
            "activities": stats.get("activities", []),
        },
        "trends": trends,
        "predictive_insights": predictive,
    }
    result = json_safe(result)
    input_payload = json_safe(input_payload)
    analysis = current_app.extensions["api_db"].create_analysis(
        owner_id=get_jwt_identity(),
        student_name=student_name.strip() if isinstance(student_name, str) else None,
        input_payload=input_payload,
        result=result,
    )
    response = jsonify({"data": _analysis_view(analysis)})
    response.status_code = 201
    response.headers["Location"] = f"/api/v1/analyses/{analysis['id']}"
    return response


@api_bp.get("/analyses")
@jwt_required()
def list_analyses():
    page, per_page = pagination()
    items, total = current_app.extensions["api_db"].list_analyses(
        get_jwt_identity(), per_page, (page - 1) * per_page
    )
    pages = max(1, (total + per_page - 1) // per_page)
    return jsonify({
        "data": [_analysis_view(item) for item in items],
        "meta": {"page": page, "per_page": per_page, "total": total, "pages": pages},
    })


@api_bp.get("/analyses/<analysis_id>")
@jwt_required()
def get_analysis(analysis_id: str):
    analysis = current_app.extensions["api_db"].get_analysis(analysis_id, get_jwt_identity())
    if analysis is None:
        raise ApiError("NOT_FOUND", "Analysis not found.", 404)
    return jsonify({"data": _analysis_view(analysis)})


@api_bp.delete("/analyses/<analysis_id>")
@jwt_required()
def delete_analysis(analysis_id: str):
    deleted = current_app.extensions["api_db"].delete_analysis(analysis_id, get_jwt_identity())
    if not deleted:
        raise ApiError("NOT_FOUND", "Analysis not found.", 404)
    return "", 204

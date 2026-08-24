"""Shared request validation and error handling for the REST API."""

from __future__ import annotations

import math
from functools import wraps
from typing import Any, Callable

import pandas as pd
from flask import g, jsonify, request
from flask_jwt_extended import get_jwt


class ApiError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400, details: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details


def error_response(code: str, message: str, status_code: int, details: Any = None):
    error = {"code": code, "message": message}
    if details is not None:
        error["details"] = details
    return jsonify({"error": error, "request_id": getattr(g, "request_id", None)}), status_code


def json_body() -> dict[str, Any]:
    if not request.is_json:
        raise ApiError("UNSUPPORTED_MEDIA_TYPE", "Content-Type must be application/json.", 415)
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ApiError("INVALID_JSON", "The request body must be a JSON object.", 400)
    return data


def required_string(data: dict[str, Any], field: str, *, max_length: int = 120) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ApiError("VALIDATION_ERROR", f"'{field}' is required and must be a non-empty string.", 422)
    value = value.strip()
    if len(value) > max_length:
        raise ApiError("VALIDATION_ERROR", f"'{field}' must be at most {max_length} characters.", 422)
    return value


def pagination() -> tuple[int, int]:
    try:
        page = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 20))
    except ValueError as exc:
        raise ApiError("VALIDATION_ERROR", "'page' and 'per_page' must be integers.", 422) from exc
    if page < 1 or per_page < 1 or per_page > 100:
        raise ApiError("VALIDATION_ERROR", "'page' must be >= 1 and 'per_page' must be between 1 and 100.", 422)
    return page, per_page


def json_safe(value: Any) -> Any:
    """Convert pandas/numpy scalar values to strict JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if hasattr(value, "item"):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def role_required(*roles: str) -> Callable:
    def decorator(function: Callable) -> Callable:
        @wraps(function)
        def wrapped(*args, **kwargs):
            claims = get_jwt()
            if claims.get("role") not in roles:
                raise ApiError("FORBIDDEN", "You do not have permission to perform this action.", 403)
            return function(*args, **kwargs)

        return wrapped

    return decorator

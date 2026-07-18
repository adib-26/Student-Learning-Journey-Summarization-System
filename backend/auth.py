"""Versioned REST authentication endpoints."""

from __future__ import annotations

import sqlite3

from flask import Blueprint, current_app, jsonify
from flask_jwt_extended import create_access_token, get_jwt_identity, jwt_required
from werkzeug.security import check_password_hash, generate_password_hash

from .api_utils import ApiError, json_safe, json_body, required_string


auth_bp = Blueprint("auth", __name__, url_prefix="/auth")


def _db():
    return current_app.extensions["api_db"]


@auth_bp.post("/register")
def register():
    data = json_body()
    username = required_string(data, "username", max_length=80)
    password = required_string(data, "password", max_length=128)
    role = required_string(data, "role", max_length=20).lower()

    if len(password) < 8:
        raise ApiError("VALIDATION_ERROR", "'password' must be at least 8 characters.", 422)
    if role not in {"student", "teacher"}:
        raise ApiError("VALIDATION_ERROR", "'role' must be either 'student' or 'teacher'.", 422)

    try:
        # PBKDF2 keeps local development compatible with Python builds that do
        # not expose hashlib.scrypt while still using a slow password hash.
        user = _db().create_user(
            username,
            generate_password_hash(password, method="pbkdf2:sha256:600000"),
            role,
        )
    except sqlite3.IntegrityError as exc:
        raise ApiError("CONFLICT", "A user with that username already exists.", 409) from exc

    response = jsonify({"data": json_safe(user)})
    response.status_code = 201
    response.headers["Location"] = f"/api/v1/auth/me"
    return response


@auth_bp.post("/login")
def login():
    data = json_body()
    username = required_string(data, "username", max_length=80)
    password = required_string(data, "password", max_length=128)
    user = _db().get_user_by_username(username)

    if user is None or not check_password_hash(user["password_hash"], password):
        raise ApiError("INVALID_CREDENTIALS", "Invalid username or password.", 401)

    token = create_access_token(identity=user["id"], additional_claims={"role": user["role"]})
    user.pop("password_hash", None)
    return jsonify({"data": {"access_token": token, "token_type": "Bearer", "user": json_safe(user)}})


@auth_bp.get("/me")
@jwt_required()
def me():
    user = _db().get_user(get_jwt_identity())
    if user is None:
        raise ApiError("UNAUTHORIZED", "The authenticated user no longer exists.", 401)
    return jsonify({"data": json_safe(user)})

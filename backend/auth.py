from flask import Blueprint, request, jsonify
from flask_jwt_extended import (
    create_access_token,
    jwt_required,
    get_jwt_identity
)
from werkzeug.security import generate_password_hash, check_password_hash

auth_bp = Blueprint("auth", __name__)

# In-memory user store (FYP-safe)
# Each user has a password hash and a role ("student" or "teacher")
USERS = {}


@auth_bp.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    role = data.get("role")

    if not username or not password or not role:
        return jsonify({"error": "Missing credentials or role"}), 400

    if role not in ["student", "teacher"]:
        return jsonify({"error": "Role must be 'student' or 'teacher'"}), 400

    if username in USERS:
        return jsonify({"error": "User already exists"}), 409

    USERS[username] = {
        "password": generate_password_hash(password, method="pbkdf2:sha256"),
        "role": role
    }

    return jsonify({"message": f"User '{username}' registered successfully as {role}"}), 201


@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    user = USERS.get(username)
    if not user or not check_password_hash(user["password"], password):
        return jsonify({"error": "Invalid credentials"}), 401

    # ✅ Identity is just the username (string)
    access_token = create_access_token(identity=username)

    # Still return the role in the response so frontend knows
    return jsonify({"access_token": access_token, "role": user["role"]}), 200


@auth_bp.route("/profile", methods=["GET"])
@jwt_required()
def profile():
    username = get_jwt_identity()  # returns "student1" or "teacher1"
    role = USERS[username]["role"]
    return jsonify({"username": username, "role": role}), 200


@auth_bp.route("/teacher-dashboard", methods=["GET"])
@jwt_required()
def teacher_dashboard():
    username = get_jwt_identity()
    role = USERS[username]["role"]
    if role != "teacher":
        return jsonify({"error": "Access denied"}), 403
    return jsonify({"message": f"Welcome, {username} (teacher)!"}), 200


@auth_bp.route("/student-dashboard", methods=["GET"])
@jwt_required()
def student_dashboard():
    username = get_jwt_identity()
    role = USERS[username]["role"]
    if role != "student":
        return jsonify({"error": "Access denied"}), 403
    return jsonify({"message": f"Welcome, {username} (student)!"}), 200
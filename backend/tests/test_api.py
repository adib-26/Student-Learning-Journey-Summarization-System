import io

import pytest

from backend.app import create_app


@pytest.fixture()
def client(tmp_path):
    app = create_app(
        {
            "TESTING": True,
            "JWT_ACCESS_TOKEN_EXPIRES": False,
            "API_DATABASE_PATH": str(tmp_path / "api.sqlite3"),
        }
    )
    return app.test_client()


def register_and_login(client):
    registered = client.post(
        "/api/v1/auth/register",
        json={"username": "teacher1", "password": "strong-password", "role": "teacher"},
    )
    assert registered.status_code == 201

    logged_in = client.post(
        "/api/v1/auth/login",
        json={"username": "teacher1", "password": "strong-password"},
    )
    assert logged_in.status_code == 200
    return {"Authorization": f"Bearer {logged_in.json['data']['access_token']}"}


def login(client):
    logged_in = client.post(
        "/api/v1/auth/login",
        json={"username": "teacher1", "password": "strong-password"},
    )
    assert logged_in.status_code == 200
    return {"Authorization": f"Bearer {logged_in.json['data']['access_token']}"}


def test_health_and_consistent_error_shape(client):
    root = client.get("/")
    assert root.status_code == 200
    assert root.json["data"]["service"] == "student-learning-api"

    assert client.get("/favicon.ico").status_code == 204

    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json["data"]["status"] == "ok"
    assert response.headers["X-Request-ID"]

    unauthorized = client.get("/api/v1/analyses")
    assert unauthorized.status_code == 401
    assert unauthorized.json["error"]["code"] == "UNAUTHORIZED"
    assert "request_id" in unauthorized.json


def test_auth_and_analysis_resource_lifecycle(client):
    headers = register_and_login(client)

    me = client.get("/api/v1/auth/me", headers=headers)
    assert me.status_code == 200
    assert me.json["data"]["username"] == "teacher1"
    assert me.json["data"]["role"] == "teacher"

    created = client.post(
        "/api/v1/analyses",
        headers=headers,
        json={
            "student": {"name": "Aisha"},
            "records": [
                {"subject": "Mathematics", "score": 90, "maximum": 100},
                {"subject": "Science", "score": 75, "maximum": 100},
            ],
        },
    )
    assert created.status_code == 201
    assert created.headers["Location"].startswith("/api/v1/analyses/")
    analysis_id = created.json["data"]["id"]
    assert created.json["data"]["metrics"]["row_count"] == 2
    assert created.json["data"]["subjects"]["scores"]["Mathematics"] == 90

    listed = client.get("/api/v1/analyses?page=1&per_page=10", headers=headers)
    assert listed.status_code == 200
    assert listed.json["meta"]["total"] == 1
    assert listed.json["data"][0]["id"] == analysis_id

    fetched = client.get(f"/api/v1/analyses/{analysis_id}", headers=headers)
    assert fetched.status_code == 200
    assert fetched.json["data"]["student"]["name"] == "Aisha"

    deleted = client.delete(f"/api/v1/analyses/{analysis_id}", headers=headers)
    assert deleted.status_code == 204
    assert deleted.data == b""
    assert client.get(f"/api/v1/analyses/{analysis_id}", headers=headers).status_code == 404


def test_validation_duplicate_user_and_file_upload(client):
    register_and_login(client)
    duplicate = client.post(
        "/api/v1/auth/register",
        json={"username": "teacher1", "password": "strong-password", "role": "teacher"},
    )
    assert duplicate.status_code == 409
    assert duplicate.json["error"]["code"] == "CONFLICT"

    invalid = client.post(
        "/api/v1/analyses",
        headers=login(client),
        json={"records": [{"subject": "Science", "score": 110, "maximum": 100}]},
    )
    assert invalid.status_code == 422
    assert invalid.json["error"]["code"] == "VALIDATION_ERROR"

    upload = client.post(
        "/api/v1/analyses",
        headers=login(client),
        data={"file": (io.BytesIO(b"Label,Score\nMathematics,88\nScience,79\n"), "scores.csv")},
        content_type="multipart/form-data",
    )
    assert upload.status_code == 201
    assert upload.json["data"]["metrics"]["row_count"] == 2

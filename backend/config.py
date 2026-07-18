import os
from datetime import timedelta


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "fyp2-secret-key")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "fyp2-jwt-secret")
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=8)
    API_DATABASE_PATH = os.getenv(
        "API_DATABASE_PATH",
        os.path.join(PROJECT_ROOT, "data", "api.sqlite3"),
    )
    API_MAX_RECORDS = int(os.getenv("API_MAX_RECORDS", "5000"))
    MAX_CONTENT_LENGTH = int(os.getenv("API_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
    CORS_ORIGINS = [
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "*").split(",")
        if origin.strip()
    ]

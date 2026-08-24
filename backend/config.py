import os
from datetime import timedelta

from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=False)


class Config:
    # Secrets are deliberately required outside tests. Static fallback keys
    # make Flask session cookies and JWTs forgeable across installations.
    SECRET_KEY = os.getenv("SECRET_KEY")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=8)
    API_DATABASE_PATH = os.getenv(
        "API_DATABASE_PATH",
        os.path.join(PROJECT_ROOT, "data", "api.sqlite3"),
    )
    API_MAX_RECORDS = int(os.getenv("API_MAX_RECORDS", "5000"))
    MAX_CONTENT_LENGTH = int(os.getenv("API_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
    API_SQLITE_BUSY_TIMEOUT_MS = int(os.getenv("API_SQLITE_BUSY_TIMEOUT_MS", "5000"))
    CORS_ORIGINS = [
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "").split(",")
        if origin.strip()
    ]

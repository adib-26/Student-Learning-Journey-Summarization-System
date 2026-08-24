"""Small SQLite persistence layer used by the REST API.

The Streamlit application does not require a database, so the API owns its
storage explicitly. SQLite keeps the API usable in local development while
still providing persistence across process restarts.
"""

from __future__ import annotations

import json
import os
import sqlite3
import stat
import uuid
from datetime import datetime, timezone
from typing import Any, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ApiDatabase:
    """SQLite storage for the local, single-host REST API profile."""

    def __init__(self, path: str, busy_timeout_ms: int = 5_000):
        self.path = path
        self.busy_timeout_ms = max(0, int(busy_timeout_ms))

    def initialize(self) -> None:
        parent = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(parent, mode=0o700, exist_ok=True)
        try:
            os.chmod(parent, 0o700)
        except OSError:
            pass
        self._require_private_path(parent, "database directory")

        # Student reports can be persisted in this database. Create new local
        # files as owner-only, then verify that privacy guarantee below.
        try:
            descriptor = os.open(
                self.path,
                os.O_CREAT | os.O_APPEND | os.O_WRONLY,
                0o600,
            )
            os.close(descriptor)
            os.chmod(self.path, 0o600)
        except OSError:
            pass
        self._require_private_path(self.path, "database file")

        with self._connect() as connection:
            journal_mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
            if str(journal_mode).lower() != "wal":
                raise RuntimeError(
                    "SQLite WAL mode could not be enabled for the API database. "
                    "Choose a local private filesystem or use a managed database."
                )
            connection.executescript(
                """
                PRAGMA foreign_keys = ON;

                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('student', 'teacher')),
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS analyses (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    student_name TEXT,
                    input_json TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (owner_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_analyses_owner_created
                    ON analyses(owner_id, created_at DESC);
                """
            )

    @staticmethod
    def _require_private_path(path: str, label: str) -> None:
        """Reject group/world-readable PII storage on POSIX filesystems."""
        if os.name != "posix":
            return
        try:
            mode = stat.S_IMODE(os.stat(path).st_mode)
        except OSError as exc:
            raise RuntimeError(f"Could not verify private permissions for API {label}.") from exc
        if mode & 0o077:
            raise RuntimeError(
                f"API {label} must not be accessible to group or other users."
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=max(self.busy_timeout_ms / 1_000, 0.1),
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        # WAL and a bounded busy wait avoid immediate "database is locked"
        # failures when a local UI/API process overlaps briefly. SQLite still
        # is not the recommended store for a multi-worker deployment.
        # FULL favours durability over a small amount of local write latency.
        # This database can contain student report records.
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
        return connection

    @staticmethod
    def _user(row: Optional[sqlite3.Row]) -> Optional[dict[str, Any]]:
        if row is None:
            return None
        return {
            "id": row["id"],
            "username": row["username"],
            "role": row["role"],
            "created_at": row["created_at"],
        }

    def create_user(self, username: str, password_hash: str, role: str) -> dict[str, Any]:
        user = {
            "id": str(uuid.uuid4()),
            "username": username,
            "role": role,
            "created_at": _utc_now(),
        }
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO users (id, username, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?)",
                (user["id"], username, password_hash, role, user["created_at"]),
            )
        return user

    def get_user_by_username(self, username: str) -> Optional[dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, username, password_hash, role, created_at FROM users WHERE username = ?",
                (username,),
            ).fetchone()
        if row is None:
            return None
        user = self._user(row)
        assert user is not None
        user["password_hash"] = row["password_hash"]
        return user

    def get_user(self, user_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, username, role, created_at FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
        return self._user(row)

    def create_analysis(
        self,
        owner_id: str,
        student_name: Optional[str],
        input_payload: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        analysis = {
            "id": str(uuid.uuid4()),
            "owner_id": owner_id,
            "student_name": student_name,
            "input": input_payload,
            "result": result,
            "created_at": _utc_now(),
        }
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO analyses
                    (id, owner_id, student_name, input_json, result_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    analysis["id"],
                    owner_id,
                    student_name,
                    json.dumps(input_payload, allow_nan=False),
                    json.dumps(result, allow_nan=False),
                    analysis["created_at"],
                ),
            )
        return analysis

    @staticmethod
    def _analysis(row: Optional[sqlite3.Row]) -> Optional[dict[str, Any]]:
        if row is None:
            return None
        return {
            "id": row["id"],
            "owner_id": row["owner_id"],
            "student_name": row["student_name"],
            "input": json.loads(row["input_json"]),
            "result": json.loads(row["result_json"]),
            "created_at": row["created_at"],
        }

    def get_analysis(self, analysis_id: str, owner_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM analyses WHERE id = ? AND owner_id = ?",
                (analysis_id, owner_id),
            ).fetchone()
        return self._analysis(row)

    def list_analyses(self, owner_id: str, limit: int, offset: int) -> tuple[list[dict[str, Any]], int]:
        with self._connect() as connection:
            total = connection.execute(
                "SELECT COUNT(*) AS count FROM analyses WHERE owner_id = ?",
                (owner_id,),
            ).fetchone()["count"]
            rows = connection.execute(
                """
                SELECT * FROM analyses
                WHERE owner_id = ?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (owner_id, limit, offset),
            ).fetchall()
        return [self._analysis(row) for row in rows if row is not None], int(total)

    def delete_analysis(self, analysis_id: str, owner_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM analyses WHERE id = ? AND owner_id = ?",
                (analysis_id, owner_id),
            )
        return cursor.rowcount == 1

import logging
import json
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
import os
import uuid


class AuditLogger:
    """Comprehensive logging for security & compliance across anonymous app sessions"""

    def __init__(self, log_file: str = "logs/audit_log.json", custom_session_id: str = None):
        self.log_file = log_file

        # 1. Use an explicitly passed session ID (like Streamlit's native tracking key)
        # or fall back to a cryptographically random UUIDv4 string.
        self.session_id = custom_session_id or f"SESS_{uuid.uuid4().hex[:12].upper()}"

        # Create a private local audit directory and bound retention so logs do
        # not grow indefinitely on a desktop deployment.
        log_directory = os.path.dirname(os.path.abspath(self.log_file))
        os.makedirs(log_directory, mode=0o700, exist_ok=True)
        try:
            os.chmod(log_directory, 0o700)
        except OSError:
            pass

        # 2. Setup singleton-safe logger instance to prevent handler stacking
        self.logger = logging.getLogger("audit_pipeline")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        absolute_log_file = os.path.abspath(self.log_file)
        existing_handler = any(
            getattr(handler, "baseFilename", None) == absolute_log_file
            for handler in self.logger.handlers
        )
        if not existing_handler:
            # Create the file with owner-only permissions before the logging
            # handler opens it (subject to the host filesystem policy).
            descriptor = os.open(
                absolute_log_file,
                os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                0o600,
            )
            os.close(descriptor)
            try:
                os.chmod(absolute_log_file, 0o600)
            except OSError:
                pass
            handler = RotatingFileHandler(
                absolute_log_file,
                maxBytes=5 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(handler)

    def _write_audit(self, entry: dict):
        """Helper to enforce structured formatting across all entry payloads"""
        entry["session_id"] = self.session_id
        entry["timestamp"] = datetime.now(timezone.utc).isoformat()  # Timezone-aware UTC timestamp

        if entry.get("status") == "FAILURE":
            self.logger.error(json.dumps(entry))
        else:
            self.logger.info(json.dumps(entry))

    def log_file_upload(self, filename: str, file_size: int, file_type: str):
        """Log upload metadata without retaining a potentially identifying filename."""
        extension = os.path.splitext(str(filename or ""))[1].lower().lstrip(".")[:16] or "unknown"
        self._write_audit({
            "event_type": "FILE_UPLOAD",
            "file_extension": extension,
            "file_size_bytes": file_size,
            "file_type": file_type,
            "status": "SUCCESS"
        })

    def log_data_processing(self, stage: str, records_processed: int, pii_redacted: int = 0):
        """Log structured analysis sequence completions"""
        self._write_audit({
            "event_type": "DATA_PROCESSING",
            "stage": stage,
            "records_processed": records_processed,
            "pii_items_redacted": pii_redacted,
            "status": "SUCCESS"
        })

    def log_api_call(self, api_service: str, endpoint: str, request_size: int,
                     response_time_ms: float, status_code: int, pii_protected: bool,
                     transport_security: str = "local HTTP (loopback/private bridge)"):
        """Log structural telemetry updates for API targets like the local vLLM server"""
        self._write_audit({
            "event_type": "API_CALL",
            "api_service": api_service,
            "endpoint": endpoint,
            "request_size_bytes": request_size,
            "response_time_ms": response_time_ms,
            "http_status": status_code,
            "pii_protected": pii_protected,
            "transport_security": transport_security,
            "status": "SUCCESS" if status_code < 400 else "FAILURE"
        })

    def log_summary_generation(self, input_size: int, output_size: int, model: str, tokens_used: int):
        """Log text metric operations handled via LLM providers"""
        self._write_audit({
            "event_type": "SUMMARY_GENERATION",
            "model": model,
            "input_size_bytes": input_size,
            "output_size_bytes": output_size,
            "tokens_used": tokens_used,
            "status": "SUCCESS"
        })

    def log_error(self, error_type: str, message: str, stage: str):
        """Log process failures without retaining exception text or source data."""
        self._write_audit({
            "event_type": "ERROR",
            "error_type": error_type,
            "error_message": "redacted",
            "error_message_length": len(str(message or "")),
            "stage": stage,
            "status": "FAILURE"
        })

    def log_data_deletion(self, data_type: str, records_deleted: int):
        """Log temporary execution memory clears for data minimalization alignment"""
        self._write_audit({
            "event_type": "DATA_DELETION",
            "data_type": data_type,
            "records_deleted": records_deleted,
            "reason": "END_OF_SESSION_CLEANUP",
            "status": "SUCCESS"
        })

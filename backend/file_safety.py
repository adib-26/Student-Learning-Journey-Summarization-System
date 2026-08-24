"""Small, dependency-free guards for user-supplied file containers."""

from __future__ import annotations

import io
import zipfile


DEFAULT_MAX_XLSX_UNCOMPRESSED_BYTES = 128 * 1024 * 1024
DEFAULT_MAX_XLSX_ENTRIES = 10_000
DEFAULT_MAX_XLSX_COMPRESSION_RATIO = 100


class UnsafeSpreadsheetError(ValueError):
    """Raised when an XLSX archive exceeds bounded parsing safety limits."""


def validate_xlsx_archive(
    file_bytes: bytes,
    *,
    max_uncompressed_bytes: int = DEFAULT_MAX_XLSX_UNCOMPRESSED_BYTES,
    max_entries: int = DEFAULT_MAX_XLSX_ENTRIES,
    max_compression_ratio: int = DEFAULT_MAX_XLSX_COMPRESSION_RATIO,
) -> None:
    """Reject oversized or highly compressed XLSX archives before parsing.

    XLSX is a ZIP container, so a small upload can otherwise expand into a
    very large XML payload when ``pandas.read_excel`` opens it.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as archive:
            entries = archive.infolist()
    except zipfile.BadZipFile:
        # Leave ordinary parse errors to the spreadsheet reader so callers can
        # return their usual invalid-file response.
        return

    if len(entries) > max_entries:
        raise UnsafeSpreadsheetError("spreadsheet contains too many archive entries")

    uncompressed_size = sum(entry.file_size for entry in entries)
    if uncompressed_size > max_uncompressed_bytes:
        raise UnsafeSpreadsheetError("spreadsheet expands beyond the configured safety limit")

    for entry in entries:
        if not entry.file_size:
            continue
        compressed_size = max(entry.compress_size, 1)
        if entry.file_size / compressed_size > max_compression_ratio:
            raise UnsafeSpreadsheetError("spreadsheet compression ratio exceeds the safety limit")

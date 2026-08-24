#!/usr/bin/env python3
"""Exercise the local GLM-OCR -> Qwen path with synthetic, non-user data.

Run this only after ``bash scripts/start_mlx_vlm.sh``.  It builds an image in
memory, never reads an uploaded document, prints no OCR/source text, and asks
the production extraction function to analyse GLM's output.  The final unload
keeps the M1/8 GB profile idle after the check.
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _synthetic_certificate_bytes() -> bytes:
    image = Image.new("RGB", (1500, 900), "white")
    draw = ImageDraw.Draw(image)
    font = None
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Verdana.ttf",
    ):
        try:
            font = ImageFont.truetype(candidate, 52)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()

    text = (
        "CERTIFICATE OF COMPLETION\n\n"
        "This certifies that\n"
        "ALEX TESTER\n\n"
        "has successfully completed\n"
        "PYTHON FOUNDATIONS\n\n"
        "Issued by ACME LEARNING\n"
        "Date: 12 June 2026"
    )
    draw.multiline_text((100, 90), text, fill="black", font=font, spacing=20)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _unload_shared_model(base_url: str) -> None:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    headers = {}
    api_key = os.getenv("GLM_OCR_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        requests.post(f"{root}/unload", headers=headers, timeout=(2, 12))
    except requests.RequestException:
        # The model will still be released when the shared server stops; this
        # cleanup should not obscure whether the OCR/Qwen requests worked.
        pass


def main() -> int:
    from backend.mlx_glm_ocr import extract_text_with_glm_ocr, get_glm_ocr_settings
    from backend.text_info_extractor import _extract_certificate_data_with_llm

    settings = get_glm_ocr_settings()
    if settings["transport"] != "http" or not settings["enabled"]:
        print("FAIL: this smoke check requires GLM_OCR_MODE=http and GLM_OCR_ENABLED=1")
        return 2

    try:
        ocr = extract_text_with_glm_ocr("synthetic-certificate.png", _synthetic_certificate_bytes())
        if not ocr.text.strip() or not ocr.engine.startswith("glm-ocr-http"):
            print(json.dumps({"status": "fail", "stage": "glm_ocr", "reason": ocr.error or "no OCR text"}))
            return 1

        extraction = _extract_certificate_data_with_llm(ocr.text)
        meaningful = bool(
            extraction.get("student_name")
            or extraction.get("certificates")
            or extraction.get("skills")
            or extraction.get("summary")
        )
        summary = {
            "status": "pass" if meaningful else "fail",
            "ocr_engine": ocr.engine,
            "ocr_chars": len(ocr.text),
            "qwen_student_name_found": bool(extraction.get("student_name")),
            "qwen_certificates": len(extraction.get("certificates", [])),
            "qwen_skills": len(extraction.get("skills", [])),
            "qwen_summary_found": bool(extraction.get("summary")),
        }
        print(json.dumps(summary))
        return 0 if meaningful else 1
    finally:
        _unload_shared_model(settings["base_url"])


if __name__ == "__main__":
    raise SystemExit(main())

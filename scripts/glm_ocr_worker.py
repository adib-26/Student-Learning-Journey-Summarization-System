#!/usr/bin/env python3
"""One-shot MLX-VLM worker used by :mod:`backend.mlx_glm_ocr`.

This script must be run with the vllm-metal Python environment, not the
Streamlit environment.  It reads only private temporary image paths, writes one
marked JSON line to stdout, and exits so the GLM model's unified memory is
released before Qwen analysis begins.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path


RESULT_MARKER = "__GLM_OCR_RESULT__="


def _emit(pages: list[dict], error: str = "") -> None:
    # A single line makes the caller robust to informational output produced by
    # MLX/Metal libraries before or during model loading.
    print(RESULT_MARKER + json.dumps({"pages": pages, "error": error[:120]}), flush=True)


def _read_request(path: str) -> dict:
    request_path = Path(path).resolve(strict=True)
    request_root = request_path.parent
    with request_path.open("r", encoding="utf-8") as request_file:
        request = json.load(request_file)
    if not isinstance(request, dict):
        raise ValueError("request must be an object")
    if (
        not isinstance(request.get("model"), str)
        or not request["model"].strip()
        or len(request["model"]) > 240
    ):
        raise ValueError("request model is missing")
    if not isinstance(request.get("max_tokens"), int) or not 128 <= request["max_tokens"] <= 4096:
        raise ValueError("request max_tokens is invalid")
    raw_pages = request.get("pages")
    if not isinstance(raw_pages, list) or not raw_pages or len(raw_pages) > 20:
        raise ValueError("request pages are missing")
    pages: list[dict] = []
    seen_pages: set[int] = set()
    for item in raw_pages:
        if not isinstance(item, dict):
            raise ValueError("request page is invalid")
        page_number = item.get("page")
        image_path = item.get("path")
        if (
            not isinstance(page_number, int)
            or page_number < 1
            or page_number in seen_pages
            or not isinstance(image_path, str)
        ):
            raise ValueError("request page is invalid")
        candidate = Path(image_path).resolve(strict=True)
        # The parent creates a private directory containing only page PNGs and
        # the request. Never let a caller use this helper to read another file.
        if request_root not in candidate.parents or not candidate.is_file() or candidate.suffix.casefold() != ".png":
            raise ValueError("request image path is invalid")
        seen_pages.add(page_number)
        pages.append({"page": page_number, "path": str(candidate)})
    request["pages"] = pages
    return request


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local GLM-OCR over prepared page images")
    parser.add_argument("--request-json", required=True, help="Private temporary request JSON path")
    args = parser.parse_args()

    model = None
    processor = None
    try:
        request = _read_request(args.request_json)
        # Imported only after request validation so `--help` and malformed
        # requests remain usable even when MLX is unavailable on a host.
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template

        model, processor = load(request["model"])
        config = getattr(model, "config", None)
        pages: list[dict] = []
        for item in request["pages"]:
            page_number = item["page"]
            image_path = item["path"]
            # This exact task prompt is GLM-OCR's documented raw-transcription
            # interface. One image is deliberately sent per generation.
            prompt = apply_chat_template(
                processor,
                config,
                "Text Recognition:",
                num_images=1,
            )
            result = generate(
                model,
                processor,
                prompt,
                image=[image_path],
                max_tokens=request["max_tokens"],
                temperature=0.0,
                verbose=False,
            )
            text = getattr(result, "text", result)
            if isinstance(text, str) and text.strip():
                pages.append({"page": page_number, "text": text.strip()})
        _emit(pages)
        return 0
    except Exception as exc:
        # Do not include the exception message: it can contain a document path
        # or model context.  The caller only needs a safe operational category.
        _emit([], type(exc).__name__)
        return 1
    finally:
        # Process exit is the primary release mechanism. These hints help the
        # interpreter drop Python references before teardown as well.
        del model, processor
        gc.collect()


if __name__ == "__main__":
    raise SystemExit(main())

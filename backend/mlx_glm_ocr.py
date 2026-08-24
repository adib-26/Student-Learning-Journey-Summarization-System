"""Local GLM-OCR ingestion for PDFs and images, designed for an 8 GB M1.

The recommended path uses one local MLX-VLM HTTP service for both GLM-OCR and
Qwen.  The service unloads the prior text-generation model before it loads the
next one; this module also shares the application's inference gate so OCR and
Qwen cannot race a model swap.  A short-lived subprocess worker remains an
explicit compatibility fallback and never imports MLX into Streamlit itself.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import re
import subprocess
import tempfile
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlsplit

import requests
from PIL import Image, ImageOps, UnidentifiedImageError

try:
    import pypdf
except ImportError:  # pragma: no cover - native fallback is optional at import time
    pypdf = None

try:
    import pypdfium2 as pdfium
except ImportError:  # pragma: no cover - visual PDF OCR is optional at import time
    pdfium = None


logger = logging.getLogger(__name__)
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
_SUPPORTED_SUFFIXES = (".pdf", *_IMAGE_SUFFIXES)
_RESULT_MARKER = "__GLM_OCR_RESULT__="
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


@dataclass(frozen=True)
class GlmOcrResult:
    """OCR text with safe provenance; no source bytes or server output are kept."""

    text: str = ""
    engine: str = "none"
    pages_attempted: int = 0
    pages_completed: int = 0
    native_pages: int = 0
    skipped_pages: int = 0
    truncated: bool = False
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _read_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def _read_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


def _read_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


def get_glm_ocr_settings() -> dict:
    """Read bounded M1/8 GB-safe settings without importing any MLX package."""
    # HTTP is deliberately the safe default: it uses the same MLX-VLM process
    # as Qwen rather than starting a second resident model in another process.
    mode = os.getenv("GLM_OCR_MODE", "http").strip().casefold()
    # Early builds used ``auto``/``force`` for a separate worker. Migrate
    # those values to the safe shared-server equivalents rather than silently
    # loading a second model process on an 8 GB machine.
    mode = {"auto": "http", "force": "http-force"}.get(mode, mode)
    enabled = _read_bool("GLM_OCR_ENABLED", True) and mode not in {"off", "disabled", "0"}
    default_python = Path.home() / ".venv-vllm-metal" / "bin" / "python"
    return {
        "mode": mode,
        "enabled": enabled,
        "transport": "http" if mode in {"http", "http-force"} else "subprocess",
        "force": mode in {"force", "http-force"} or _read_bool("GLM_OCR_FORCE", False),
        # The existing local cache contains this BF16 model.  We deliberately
        # run offline by default so an upload can never trigger a large model
        # download on the user's nearly-full disk.
        "model": os.getenv("GLM_OCR_MODEL", "mlx-community/GLM-OCR-bf16"),
        "python": os.getenv("GLM_OCR_PYTHON", str(default_python)),
        "base_url": os.getenv(
            "GLM_OCR_BASE_URL",
            os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        ).rstrip("/"),
        # Used only for an explicitly enabled remote HTTP endpoint. It is never
        # included in cache keys, logs, or worker request files.
        "api_key": os.getenv("GLM_OCR_API_KEY", ""),
        "timeout": _read_float("GLM_OCR_TIMEOUT_SECONDS", 180.0, 15.0, 600.0),
        "health_timeout": _read_float("GLM_OCR_HEALTH_TIMEOUT_SECONDS", 1.5, 0.2, 10.0),
        # These worker defaults are deliberately conservative.  Managed HTTP
        # model switching can increase them after it has freed Qwen's model.
        "max_pages": _read_int("GLM_OCR_MAX_PAGES", 1, 1, 20),
        # Native text extraction is inexpensive, but parsing arbitrarily many
        # PDF pages is not. Certificates are normally short; this separate cap
        # prevents a huge transcript from monopolising the desktop process.
        "max_total_pages": _read_int("GLM_OCR_MAX_TOTAL_PAGES", 20, 1, 100),
        "max_tokens": _read_int("GLM_OCR_MAX_TOKENS", 1024, 128, 4096),
        "render_dpi": _read_int("GLM_OCR_RENDER_DPI", 144, 72, 240),
        "max_dimension": _read_int("GLM_OCR_MAX_IMAGE_DIMENSION", 1280, 768, 4096),
        "max_pixels": _read_int("GLM_OCR_MAX_SOURCE_PIXELS", 40_000_000, 1_000_000, 100_000_000),
        "max_document_bytes": _read_int(
            "GLM_OCR_MAX_DOCUMENT_BYTES", 32 * 1024 * 1024, 1 * 1024 * 1024, 100 * 1024 * 1024
        ),
        "native_min_chars": _read_int("GLM_OCR_NATIVE_MIN_CHARS", 40, 1, 2000),
        "queue_timeout": _read_float("LLM_QUEUE_TIMEOUT_SECONDS", 5.0, 0.0, 30.0),
        "allow_model_download": _read_bool("GLM_OCR_ALLOW_MODEL_DOWNLOAD", False),
        "allow_remote": _read_bool("GLM_OCR_ALLOW_REMOTE", False),
        "allow_concurrent_qwen": _read_bool("GLM_OCR_ALLOW_CONCURRENT_QWEN", False),
    }


def get_glm_ocr_cache_identity() -> dict:
    """Configuration that materially changes text and invalidates session cache."""
    settings = get_glm_ocr_settings()
    identity = {
        key: settings[key]
        for key in (
            "mode", "enabled", "transport", "force", "model", "max_pages",
            "max_total_pages", "max_tokens", "render_dpi", "max_dimension",
            "max_pixels", "max_document_bytes", "native_min_chars",
            "allow_concurrent_qwen",
        )
    }
    # A URL can encode deployment topology; use a digest rather than retaining
    # its raw value in a Streamlit cache key or telemetry.
    identity["endpoint_digest"] = hashlib.sha256(
        settings["base_url"].encode("utf-8")
    ).hexdigest()
    return identity


def _clean_native_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = value.replace("\x00", "")
    # Preserve the project's useful cleanup for certificate PDFs even when
    # native and visual extraction are mixed in the same document.
    text = re.sub(r"([a-z.])(\s*Certificate of Completion)", r"\1\n\2", text, flags=re.IGNORECASE)
    text = re.sub(r"([a-z])\s+([A-Z]{3,}\s+[A-Z]{3,}\s+[A-Z]{3,})", r"\1\n\2", text)
    text = re.sub(r"([a-z.])\s*(THIS CERTIFICATE IS PROUDLY PRESENTED)", r"\1\n\2", text, flags=re.IGNORECASE)
    text = re.sub(r"([a-z])([A-Z][a-z]+\s+(?:of|is|to|in)\s+)", r"\1\n\2", text)
    text = re.sub(r"([a-z])([A-Z][a-z])", r"\1 \2", text)
    text = re.sub(r"([A-Z][a-z]+)\s*([A-Z][a-z]+)(Certificate)", r"\1 \2\n\3", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _native_pdf_pages(file_bytes: bytes, maximum_pages: int) -> tuple[int, dict[int, str], str]:
    """Extract only the bounded prefix of native PDF text.

    ``PdfReader.pages`` still needs to discover the document's page count, but
    limiting ``extract_text`` here prevents an oversized born-digital PDF from
    bypassing the M1 profile's total-page cap.
    """
    if pypdf is None:
        return 0, {}, "native PDF extraction unavailable"
    try:
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        page_count = len(reader.pages)
        return (
            page_count,
            {
                index + 1: _clean_native_text(reader.pages[index].extract_text() or "")
                for index in range(min(page_count, maximum_pages))
            },
            "",
        )
    except Exception:
        return 0, {}, "native PDF extraction failed"


def _pdf_page_count(file_bytes: bytes) -> int:
    if pdfium is None:
        return 0
    document = None
    try:
        document = pdfium.PdfDocument(file_bytes)
        return len(document)
    except Exception:
        return 0
    finally:
        close = getattr(document, "close", None)
        if callable(close):
            close()


def _has_native_text(text: str, minimum: int) -> bool:
    return len(text) >= minimum and sum(character.isalnum() for character in text) >= max(1, minimum // 2)


def _prepare_image(image: Image.Image, settings: dict) -> bytes:
    width, height = image.size
    if width * height > settings["max_pixels"]:
        raise ValueError("image exceeds OCR safety limit")
    prepared = ImageOps.exif_transpose(image)
    if prepared.mode != "RGB":
        prepared = prepared.convert("RGB")
    if max(prepared.size) > settings["max_dimension"]:
        scale = settings["max_dimension"] / max(prepared.size)
        prepared = prepared.resize(
            (max(1, round(prepared.width * scale)), max(1, round(prepared.height * scale))),
            Image.Resampling.LANCZOS,
        )
    output = io.BytesIO()
    prepared.save(output, format="PNG", optimize=True)
    return output.getvalue()


def _safe_pdf_render_scale(page, settings: dict) -> float:
    """Bound the bitmap before PDFium allocates it whenever page size is known."""
    requested_scale = settings["render_dpi"] / 72.0
    try:
        width, height = page.get_size()
        width, height = float(width), float(height)
        if width <= 0 or height <= 0:
            return requested_scale
        dimension_scale = settings["max_dimension"] / max(width, height)
        pixel_scale = (settings["max_pixels"] / (width * height)) ** 0.5
        return max(0.05, min(requested_scale, dimension_scale, pixel_scale))
    except Exception:
        # `_prepare_image` remains the second safety bound for older PDFium
        # builds that do not expose page dimensions.
        return requested_scale


def _render_pdf_pages(file_bytes: bytes, page_numbers: Iterable[int], settings: dict) -> dict[int, bytes]:
    if pdfium is None:
        raise ValueError("PDF rendering is unavailable")
    document = None
    images: dict[int, bytes] = {}
    try:
        document = pdfium.PdfDocument(file_bytes)
        for page_number in page_numbers:
            if not 1 <= page_number <= len(document):
                continue
            page = document[page_number - 1]
            bitmap = None
            try:
                bitmap = page.render(scale=_safe_pdf_render_scale(page, settings))
                images[page_number] = _prepare_image(bitmap.to_pil(), settings)
            finally:
                close = getattr(bitmap, "close", None)
                if callable(close):
                    close()
                close = getattr(page, "close", None)
                if callable(close):
                    close()
        return images
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("PDF rendering failed") from exc
    finally:
        close = getattr(document, "close", None)
        if callable(close):
            close()


def _render_image(file_bytes: bytes, settings: dict) -> dict[int, bytes]:
    try:
        # Convert PIL's decompression-bomb warning into a deterministic error
        # before a hostile or accidental image consumes unified memory.
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(file_bytes)) as image:
                return {1: _prepare_image(image, settings)}
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("image exceeds OCR safety limit") from exc
    except ValueError as exc:
        if str(exc) == "image exceeds OCR safety limit":
            raise
        raise ValueError("image preparation failed") from exc
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("image preparation failed") from exc


def _worker_path() -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / "glm_ocr_worker.py"


def _parse_worker_stdout(stdout: str) -> tuple[dict[int, str], str]:
    # MLX may log informational lines; only consume our final marker line.
    for line in reversed(stdout.splitlines()):
        if not line.startswith(_RESULT_MARKER):
            continue
        try:
            payload = json.loads(line[len(_RESULT_MARKER):])
        except json.JSONDecodeError:
            return {}, "worker returned invalid output"
        pages: dict[int, str] = {}
        for item in payload.get("pages", []) if isinstance(payload, dict) else []:
            if not isinstance(item, dict):
                continue
            page_number, text = item.get("page"), item.get("text")
            if isinstance(page_number, int) and isinstance(text, str) and text.strip():
                pages[page_number] = text.replace("\x00", "").strip()
        error = payload.get("error", "") if isinstance(payload, dict) else ""
        return pages, str(error)[:120]
    return {}, "worker returned no result"


_WORKER_ENVIRONMENT_KEYS = {
    "HOME",
    "PATH",
    "TMPDIR",
    "TMP",
    "TEMP",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "XDG_CACHE_HOME",
    "HF_HOME",
    "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "TOKENIZERS_PARALLELISM",
    "DYLD_LIBRARY_PATH",
    "DYLD_FALLBACK_LIBRARY_PATH",
}


def _worker_environment(settings: dict) -> dict[str, str]:
    """Pass only runtime settings needed by an isolated MLX worker.

    The parent Streamlit process can hold API keys and other user settings.
    The OCR worker needs neither, so do not inherit them into its environment.
    """
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in _WORKER_ENVIRONMENT_KEYS or key.startswith(("MLX_", "PYTORCH_MPS_"))
    }
    if not settings["allow_model_download"]:
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
    return environment


def _run_worker(page_images: dict[int, bytes], settings: dict) -> tuple[dict[int, str], str]:
    """Create private temp image files, run one worker, and remove everything."""
    executable = Path(settings["python"])
    worker = _worker_path()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        return {}, "MLX worker Python is unavailable"
    if not worker.is_file():
        return {}, "GLM-OCR worker script is unavailable"
    try:
        with tempfile.TemporaryDirectory(prefix="glm-ocr-") as directory:
            # The directory holds unredacted page images; enforce private mode.
            os.chmod(directory, 0o700)
            root = Path(directory)
            pages = []
            for page_number, image_bytes in page_images.items():
                image_path = root / f"page-{page_number}.png"
                image_path.write_bytes(image_bytes)
                os.chmod(image_path, 0o600)
                pages.append({"page": page_number, "path": str(image_path)})
            request_path = root / "request.json"
            request_path.write_text(
                json.dumps({"model": settings["model"], "max_tokens": settings["max_tokens"], "pages": pages}),
                encoding="utf-8",
            )
            os.chmod(request_path, 0o600)
            completed = subprocess.run(
                [str(executable), str(worker), "--request-json", str(request_path)],
                cwd=str(worker.parent.parent),
                env=_worker_environment(settings),
                stdout=subprocess.PIPE,
                # Model-library diagnostics occasionally include raw paths or
                # request fragments. They are not useful to the application.
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=settings["timeout"],
                check=False,
            )
            texts, worker_error = _parse_worker_stdout(completed.stdout or "")
            texts = {page: text for page, text in texts.items() if page in page_images}
            if completed.returncode != 0:
                # Do not propagate stderr: model libraries can echo OCR input.
                return texts, worker_error or "MLX worker failed"
            return texts, worker_error or ("" if texts else "no text returned")
    except subprocess.TimeoutExpired:
        return {}, "MLX worker timed out"
    except OSError:
        return {}, "MLX worker could not start"


def _http_urls(base_url: str) -> tuple[str, str, str]:
    """Accept either an MLX-VLM root URL or its conventional ``/v1`` URL."""
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1"):
        root = base_url[:-3]
        return f"{base_url}/chat/completions", f"{root}/health", f"{root}/unload"
    return f"{base_url}/chat/completions", f"{base_url}/health", f"{base_url}/unload"


def _allowed_http_endpoint(base_url: str, allow_remote: bool) -> bool:
    parsed = urlsplit(base_url)
    return (
        parsed.scheme in {"http", "https"}
        and bool(parsed.hostname)
        and (allow_remote or parsed.hostname.casefold() in _LOCAL_HOSTS)
    )


def _local_llm_service_is_live() -> bool:
    """Whether another local inference service could retain Qwen in memory.

    A subprocess GLM worker is a second process, so it is unsafe beside either
    classic vLLM *or* an MLX-VLM server that currently has (or can receive)
    Qwen work. A remote LLM endpoint does not consume this Mac's unified
    memory and therefore does not trigger the guard.
    """
    base_url = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
    parsed = urlsplit(base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.hostname.casefold() not in _LOCAL_HOSTS
    ):
        return False
    root_url = base_url[:-3] if base_url.endswith("/v1") else base_url
    try:
        response = requests.get(
            f"{root_url}/health",
            timeout=(0.5, 0.5),
            allow_redirects=False,
        )
        return 200 <= response.status_code < 300
    except requests.RequestException:
        return False


def _is_mlx_vlm_health_response(response: object) -> bool:
    """Identify the companion MLX-VLM health schema, not merely any 200."""
    try:
        payload = response.json()
    except (AttributeError, ValueError):
        return False
    return isinstance(payload, dict) and (
        "loaded_model" in payload
        or "loaded_models" in payload
        or "apc_enabled" in payload
    )


def _apply_qwen_memory_guard(settings: dict) -> dict:
    """Skip any second-process OCR worker while local Qwen could be resident."""
    if (
        not settings["enabled"]
        or settings["transport"] != "subprocess"
        or settings["allow_concurrent_qwen"]
        or not _local_llm_service_is_live()
    ):
        return settings
    guarded = dict(settings)
    guarded["enabled"] = False
    guarded["memory_guard_error"] = "GLM worker skipped while another local Qwen/MLX service is active"
    return guarded


def _response_text(payload: object) -> str:
    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return ""
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content", "")
    return content.replace("\x00", "").strip() if isinstance(content, str) else ""


def _http_headers(settings: dict) -> dict[str, str]:
    """Attach optional remote-service auth without retaining or logging it."""
    api_key = str(settings.get("api_key", "")).strip()
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _run_http_ocr(page_images: dict[int, bytes], settings: dict) -> tuple[dict[int, str], str]:
    """Use a model-switching MLX-VLM service, exactly one page per request.

    The endpoint is intended for the companion local service that keeps one
    model cache.  It loads GLM for this short batch, then ``/unload`` releases
    it so the next Qwen request can load without competing for unified memory.
    """
    if not _allowed_http_endpoint(settings["base_url"], settings["allow_remote"]):
        logger.warning("Refusing non-loopback GLM-OCR HTTP endpoint without opt-in")
        return {}, "endpoint is not permitted"
    chat_url, health_url, unload_url = _http_urls(settings["base_url"])
    headers = _http_headers(settings)
    try:
        ready = requests.get(
            health_url,
            headers=headers,
            timeout=(settings["health_timeout"], settings["health_timeout"]),
            allow_redirects=False,
        )
        if not 200 <= ready.status_code < 300:
            return {}, "service unavailable"
        if not _is_mlx_vlm_health_response(ready):
            return {}, "configured OCR service is not MLX-VLM"
    except requests.RequestException:
        return {}, "service unavailable"

    texts: dict[int, str] = {}
    try:
        for page_number, image_bytes in page_images.items():
            data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
            # Keep one image URL in the message. Some MLX-VLM versions silently
            # retained only the final image in a multi-image chat request.
            body = {
                "model": settings["model"],
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "Text Recognition:"},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]}],
                "temperature": 0.0,
                "max_tokens": settings["max_tokens"],
                "stream": False,
            }
            response = requests.post(
                chat_url,
                json=body,
                headers=headers,
                timeout=(settings["health_timeout"], settings["timeout"]),
                allow_redirects=False,
            )
            if not 200 <= response.status_code < 300:
                return texts, "request failed" if not texts else "partial extraction"
            text = _response_text(response.json())
            if text:
                texts[page_number] = text
    except (requests.RequestException, ValueError):
        return texts, "request failed" if not texts else "partial extraction"
    finally:
        # It is a best-effort cleanup. OCR output is already safely in memory,
        # and a failed unload must not erase it or block native fallback.
        try:
            requests.post(
                unload_url,
                headers=headers,
                timeout=(settings["health_timeout"], settings["health_timeout"]),
                allow_redirects=False,
            )
        except requests.RequestException:
            logger.info("GLM-OCR HTTP unload did not complete")
    return texts, "" if texts else "no text returned"


def _run_ocr(page_images: dict[int, bytes], settings: dict) -> tuple[dict[int, str], str]:
    """Run exactly one local inference task at a time across OCR and Qwen."""
    if not page_images:
        return {}, "no visual pages selected"
    try:
        from backend.llm_client import get_local_inference_lock

        inference_lock = get_local_inference_lock()
        acquired = inference_lock.acquire(timeout=settings.get("queue_timeout", 5.0))
    except Exception:
        # The isolated adapter remains usable by command-line callers that do
        # not have the application client installed.
        acquired = True
        inference_lock = None
    if not acquired:
        return {}, "local AI is busy; visual OCR was skipped"
    try:
        if settings.get("transport", "subprocess") == "http":
            return _run_http_ocr(page_images, settings)
        return _run_worker(page_images, settings)
    finally:
        if inference_lock is not None:
            inference_lock.release()


def _page_block(page_number: int, text: str) -> str:
    return f"===== Page {page_number} =====\n{text.strip()}"


def _page_limit_marker(processed: int, omitted: int) -> str:
    return f"[OCR PAGE LIMIT REACHED: processed {processed} visual page(s); omitted {omitted} later page(s).]"


def _document_page_limit_marker(processed: int, total: int) -> str:
    return f"[DOCUMENT PAGE LIMIT REACHED: used the first {processed} of {total} page(s).]"


def _incomplete_visual_marker(missing: int) -> str:
    return f"[OCR EXTRACTION INCOMPLETE: text was unavailable for {missing} requested visual page(s).]"


def _extract_pdf(file_bytes: bytes, settings: dict) -> GlmOcrResult:
    maximum_pages = settings.get("max_total_pages", 20)
    page_count, native, native_error = _native_pdf_pages(file_bytes, maximum_pages)
    page_count = page_count or _pdf_page_count(file_bytes)
    if not page_count:
        return GlmOcrResult(error=native_error or "PDF could not be read")

    considered_pages = min(page_count, settings.get("max_total_pages", page_count))
    document_omitted = page_count - considered_pages

    good_native_pages = {
        page_number for page_number, text in native.items() if _has_native_text(text, settings["native_min_chars"])
    }
    candidates = list(range(1, considered_pages + 1)) if settings["force"] else [
        page_number for page_number in range(1, considered_pages + 1) if page_number not in good_native_pages
    ]
    selected = candidates[: settings["max_pages"]] if settings["enabled"] else []
    skipped = candidates[len(selected):]
    visual_images: dict[int, bytes] = {}
    render_error = ""
    if selected:
        try:
            visual_images = _render_pdf_pages(file_bytes, selected, settings)
        except ValueError:
            render_error = "PDF rendering failed"
    ocr, ocr_error = _run_ocr(visual_images, settings) if visual_images else ({}, "")

    output, native_count = [], 0
    for page_number in range(1, considered_pages + 1):
        if ocr.get(page_number):
            output.append(_page_block(page_number, ocr[page_number]))
        elif native.get(page_number):
            output.append(_page_block(page_number, native[page_number]))
            native_count += 1
    # A marker is evidence context for Qwen only when at least one page yielded
    # readable text. It must never masquerade as a successful extraction.
    # Do not call a page "omitted" if its usable native text was retained.
    page_limit_omitted = [page for page in skipped if not native.get(page)]
    requested_but_missing = [
        page for page in selected if not ocr.get(page) and not native.get(page)
    ]
    if output:
        if document_omitted:
            output.append(_document_page_limit_marker(considered_pages, page_count))
        if settings["enabled"] and page_limit_omitted:
            output.append(_page_limit_marker(len(selected), len(page_limit_omitted)))
        if requested_but_missing:
            output.append(_incomplete_visual_marker(len(requested_but_missing)))
    text = "\n\n".join(output).strip()
    engine = "pypdf" if not ocr else f"glm-ocr-{settings.get('transport', 'subprocess')}"
    if ocr and native_count:
        engine += "+native-pdf"
    errors = [
        value for value in (
            settings.get("memory_guard_error", "") if candidates else "",
            render_error,
            ocr_error,
            native_error if not text else "",
        ) if value
    ]
    return GlmOcrResult(
        text=text,
        engine=engine if text else "none",
        pages_attempted=len(visual_images),
        pages_completed=len(ocr),
        native_pages=native_count,
        skipped_pages=document_omitted + len(page_limit_omitted),
        truncated=bool(document_omitted or page_limit_omitted or requested_but_missing),
        error="; ".join(errors)[:180],
    )


def _extract_image(file_bytes: bytes, settings: dict) -> GlmOcrResult:
    try:
        images = _render_image(file_bytes, settings)
    except ValueError as exc:
        return GlmOcrResult(error=str(exc))
    # Even when GLM is intentionally disabled, the safe preparation above
    # ensures the legacy Tesseract fallback cannot reopen a decompression bomb.
    if not settings["enabled"]:
        return GlmOcrResult(error=settings.get("memory_guard_error", "disabled"))
    ocr, error = _run_ocr(images, settings)
    text = ocr.get(1, "")
    return GlmOcrResult(
        text=_page_block(1, text) if text else "",
        engine=f"glm-ocr-{settings.get('transport', 'subprocess')}" if text else "none",
        pages_attempted=1,
        pages_completed=1 if text else 0,
        error=error,
    )


def extract_text_with_glm_ocr(file_name: str, file_bytes: bytes) -> GlmOcrResult:
    """Return native PDF text plus GLM text for scanned pages, or image OCR text.

    In default ``http`` mode a born-digital PDF page stays native: it is more
    faithful and costs no MLX memory. Images and scanned/no-text PDF pages go
    to the shared GLM model, then its client unloads it before Qwen receives
    evidence. Set ``GLM_OCR_MODE=http-force`` to visually OCR every retained
    PDF page; native text remains a resilience fallback if a forced page fails.
    """
    lower_name = (file_name or "").casefold()
    if not isinstance(file_bytes, (bytes, bytearray)) or not file_bytes:
        return GlmOcrResult(error="empty document")
    if not lower_name.endswith(_SUPPORTED_SUFFIXES):
        return GlmOcrResult(error="unsupported document type")
    settings = _apply_qwen_memory_guard(get_glm_ocr_settings())
    if len(file_bytes) > settings["max_document_bytes"]:
        return GlmOcrResult(error="document exceeds the configured OCR safety limit")
    if lower_name.endswith(".pdf"):
        return _extract_pdf(bytes(file_bytes), settings)
    return _extract_image(bytes(file_bytes), settings)


__all__ = ["GlmOcrResult", "extract_text_with_glm_ocr", "get_glm_ocr_cache_identity", "get_glm_ocr_settings"]

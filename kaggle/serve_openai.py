"""OpenAI-compatible Qwen2.5-VL server for a Kaggle GPU kernel.

This file is the fallback when Modal cannot deploy. ``scripts/deploy_inference.sh``
copies it into a private kernel and injects credentials there. Do not put a
Kaggle token or LLM API key in this file.

Do not add ``from __future__ import annotations``: FastAPI must resolve the
request model defined inside ``build_app`` at runtime. With string annotations
it treats the body as a missing query parameter and every chat call fails 422.
"""

import base64
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import secrets as secrets_mod

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
PORT = 8000
DATASET_SLUG = "student-learning-llm-url"
URL_PATTERN = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
SETUP_TIMEOUT_SECONDS = 25 * 60

_model = None
_processor = None
_generate_lock = threading.Lock()


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def install_dependencies() -> None:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "transformers>=4.49",
            "accelerate",
            "qwen-vl-utils",
            "fastapi",
            "uvicorn",
            "pillow",
        ]
    )
    kaggle_package = "kaggle" if _env("KAGGLE_API_TOKEN") else "kaggle==1.6.17"
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", kaggle_package])


def load_model() -> None:
    global _model, _processor
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto",
    )
    _processor = AutoProcessor.from_pretrained(MODEL_NAME)
    _model.eval()


def _decode_image(url: str):
    from PIL import Image

    if url.startswith("data:"):
        payload = url.split(",", 1)[1]
        raw = base64.b64decode(payload)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    with urllib.request.urlopen(url, timeout=30) as response:
        return Image.open(io.BytesIO(response.read())).convert("RGB")


def _normalize(messages: list) -> list:
    converted = []
    image_dir = Path(tempfile.mkdtemp(prefix="kaggle-vlm-"))
    image_index = 0
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        parts = []
        if isinstance(content, str):
            parts.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                kind = item.get("type")
                if kind == "text":
                    parts.append({"type": "text", "text": item.get("text", "")})
                elif kind == "image_url":
                    image_url = item.get("image_url") or {}
                    url = image_url.get("url") if isinstance(image_url, dict) else ""
                    if url:
                        image_index += 1
                        image_path = image_dir / f"{image_index}.png"
                        _decode_image(url).save(image_path)
                        parts.append({"type": "image", "image": str(image_path)})
        if parts:
            converted.append({"role": role, "content": parts})
    return converted


def generate_reply(messages: list, max_tokens: int, temperature: float) -> str:
    from qwen_vl_utils import process_vision_info

    converted = _normalize(messages)
    text = _processor.apply_chat_template(
        converted,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(converted)
    kwargs = {"text": [text], "padding": True, "return_tensors": "pt"}
    if image_inputs:
        kwargs["images"] = image_inputs
    if video_inputs:
        kwargs["videos"] = video_inputs
    inputs = _processor(**kwargs).to(_model.device)
    limit = max(1, min(int(max_tokens or 512), 1024))
    generate_kwargs = {"max_new_tokens": limit}
    if temperature and temperature > 0:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = float(temperature)
    else:
        generate_kwargs["do_sample"] = False
    with _generate_lock:
        generated = _model.generate(**inputs, **generate_kwargs)
    trimmed = generated[:, inputs["input_ids"].shape[-1] :]
    return _processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def build_app():
    from fastapi import FastAPI, Header, HTTPException
    from pydantic import BaseModel

    app = FastAPI()
    expected = _env("LLM_API_TOKEN")

    class ChatRequest(BaseModel):
        model: str | None = None
        messages: list
        max_tokens: int | None = 512
        temperature: float | None = 0.0

    def authorize(authorization: str | None) -> None:
        if not expected:
            return
        token = ""
        if authorization and authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
        if not secrets_mod.compare_digest(token, expected):
            raise HTTPException(status_code=401, detail="invalid API key")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    def models(authorization: str | None = Header(default=None)):
        authorize(authorization)
        return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]}

    @app.post("/v1/chat/completions")
    def chat(body: ChatRequest, authorization: str | None = Header(default=None)):
        authorize(authorization)
        reply = generate_reply(
            body.messages,
            body.max_tokens or 512,
            body.temperature or 0.0,
        )
        return {
            "id": "chatcmpl-kaggle",
            "object": "chat.completion",
            "model": MODEL_NAME,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": "stop",
                }
            ],
        }

    return app


def wait_until_ready() -> None:
    deadline = time.time() + 180
    url = f"http://127.0.0.1:{PORT}/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(1)
    raise RuntimeError("local inference server did not become ready")


def start_tunnel() -> str:
    binary = Path("/tmp/cloudflared")
    if not binary.exists():
        urllib.request.urlretrieve(
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
            binary,
        )
        binary.chmod(0o755)
    process = subprocess.Popen(
        [str(binary), "tunnel", "--url", f"http://127.0.0.1:{PORT}", "--no-autoupdate"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    found: dict[str, str] = {}

    def _read_tunnel_log() -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            print(line, end="", flush=True)
            match = URL_PATTERN.search(line)
            if match and "url" not in found:
                found["url"] = match.group(0)

    threading.Thread(target=_read_tunnel_log, daemon=True).start()
    deadline = time.time() + 90
    while time.time() < deadline:
        if "url" in found:
            return found["url"]
        if process.poll() is not None:
            break
        time.sleep(0.5)
    raise RuntimeError("cloudflared did not print a public URL")


def _redact(text: str) -> str:
    """Remove credentials before command output reaches the kernel log."""
    for name in ("KAGGLE_API_TOKEN", "KAGGLE_KEY", "LLM_API_TOKEN"):
        value = _env(name)
        if value:
            text = text.replace(value, "<redacted>")
    return text


def publish_url(base_url: str) -> None:
    """Publish the tunnel URL to a private dataset, or raise.

    A silent failure here leaves the kernel idling with no way for the local
    deploy script to discover the URL, and Kaggle hides logs until a run ends.
    """
    username = _env("KAGGLE_USERNAME")
    key = _env("KAGGLE_KEY")
    token = _env("KAGGLE_API_TOKEN")
    dataset_id = _env("DATASET_ID") or (
        f"{username}/{DATASET_SLUG}" if username else ""
    )
    if not username or not dataset_id or not (key or token):
        raise RuntimeError("Kaggle credentials were not injected; the URL cannot be published")
    os.environ["KAGGLE_USERNAME"] = username
    if token:
        os.environ["KAGGLE_API_TOKEN"] = token
    if key:
        os.environ["KAGGLE_KEY"] = key
    folder = Path("/tmp/llm-url-dataset")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "llm_url.txt").write_text(base_url + "\n", encoding="utf-8")
    # The dataset outlives a session. The deploy ID lets the local poller
    # ignore a stale URL left behind by an earlier, now-dead session.
    (folder / "deploy_id.txt").write_text(_env("DEPLOY_ID") + "\n", encoding="utf-8")
    metadata = {
        "title": "Student Learning LLM URL",
        "id": dataset_id,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (folder / "dataset-metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    kaggle_cmd = [sys.executable, "-m", "kaggle"]
    create = subprocess.run(
        [*kaggle_cmd, "datasets", "create", "-p", str(folder)],
        capture_output=True,
        text=True,
    )
    print(_redact(create.stdout + create.stderr), flush=True)
    if create.returncode == 0 and "error" not in (create.stdout + create.stderr).casefold():
        return
    # The dataset already exists from an earlier session: add a new version.
    version = subprocess.run(
        [*kaggle_cmd, "datasets", "version", "-p", str(folder), "-m", "llm url", "--dir-mode", "zip"],
        capture_output=True,
        text=True,
    )
    print(_redact(version.stdout + version.stderr), flush=True)
    if version.returncode != 0:
        raise RuntimeError("Kaggle rejected both dataset create and dataset version")


def main() -> None:
    # A hang (stalled download, stuck GPU) would otherwise idle silently for
    # hours with no visible log. Abort the run if setup does not finish.
    published = threading.Event()

    def _watchdog() -> None:
        if not published.wait(SETUP_TIMEOUT_SECONDS):
            print(f"FATAL: setup did not finish within {SETUP_TIMEOUT_SECONDS}s", flush=True)
            os._exit(1)

    threading.Thread(target=_watchdog, daemon=True).start()
    try:
        print("Installing Kaggle inference dependencies...", flush=True)
        install_dependencies()
        print(f"Loading {MODEL_NAME}...", flush=True)
        load_model()
        import uvicorn

        server = threading.Thread(
            target=lambda: uvicorn.run(build_app(), host="0.0.0.0", port=PORT, log_level="info"),
            daemon=True,
        )
        server.start()
        wait_until_ready()
        print("Local server is ready. Opening tunnel...", flush=True)
        public = start_tunnel()
        base_url = public.rstrip("/") + "/v1"
        print(f"KAGGLE_LLM_BASE_URL={base_url}", flush=True)
        publish_url(base_url)
        published.set()
        print("URL published. Serving until the session ends.", flush=True)
    except BaseException as exc:
        # End the run instead of idling: Kaggle only exposes the log of a
        # finished run, so a swallowed error would be impossible to diagnose.
        print(f"FATAL: {type(exc).__name__}: {_redact(str(exc))}", flush=True)
        raise SystemExit(1) from None
    threading.Event().wait()


if __name__ == "__main__":
    main()

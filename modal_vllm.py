"""Serve Qwen3-VL on Modal instead of a local NVIDIA GPU.

The Streamlit app already speaks the OpenAI-compatible API. This file is the
GPU side of that API: one vLLM process, the same model name the app expects.

One-time setup, from the project root:

    python3.12 -m venv .venv-modal
    .venv-modal/bin/pip install modal
    .venv-modal/bin/python -m modal setup
    .venv-modal/bin/python -m modal deploy modal_vllm.py

Deploy prints an HTTPS URL. Put that URL plus ``/v1`` in ``.env`` as
``LLM_BASE_URL``, set ``VLM_OCR_ALLOW_REMOTE=1``, and restart Streamlit.
"""
import json

import modal

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
VLLM_PORT = 8000

vllm_image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.11.1")
    .entrypoint([])
)

hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("student-learning-vllm")


@app.function(
    image=vllm_image,
    gpu="T4",
    timeout=10 * 60,
    scaledown_window=5 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
)
@modal.concurrent(max_inputs=8)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * 60)
def serve():
    import subprocess

    command = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--served-model-name",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--max-model-len",
        "4096",
        "--gpu-memory-utilization",
        "0.90",
        "--max-num-seqs",
        "2",
        "--limit-mm-per-prompt",
        json.dumps({"image": 1, "video": 0}),
        "--mm-processor-cache-gb",
        "0",
        "--enforce-eager",
        "--disable-log-requests",
    ]
    subprocess.Popen(command)

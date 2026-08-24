![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![vLLM](https://img.shields.io/badge/vLLM-Local_Inference-30a3dc.svg)
![DeepL](https://img.shields.io/badge/DeepL-API-blue.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)

# Student Learning Journey Summarization System

**AI-Powered Educational Data Analytics Platform**

**Streamlit • Local vLLM Inference (Apple Silicon Metal) • DeepL Translation • OCR • Docker**

**Demo Video:** https://youtu.be/pLLHletb9pE

**Sample Dataset:** https://github.com/user-attachments/files/29584548/Sample.Data.zip

**Sample Report:** https://drive.google.com/file/d/1g4QveYbhyFTo_bqDF5yw21BqoJC2DnnO/view?usp=sharing

---

## Overview

The **Student Learning Journey Summarization System** is an AI-powered educational analytics platform that transforms raw academic records into meaningful insights, multilingual reports, and professional AI-generated summaries.

Built with **Streamlit**, powered by a **self-hosted vLLM inference server** ([vllm-metal](https://github.com/vllm-project/vllm-metal) with Metal GPU acceleration on Apple Silicon), and integrated with the **DeepL Translation API**, the application processes both structured datasets and unstructured academic documents to generate comprehensive student learning reports.

All AI inference runs **locally on your machine** — student data is never sent to a cloud LLM provider, and no LLM API key is required. If the local AI server is down, the app degrades gracefully to built-in rule-based summaries.

Translation is a separate, optional external service: selecting a non-English
report language with `DEEPL_API_KEY` configured sends the selected report text
to DeepL. Use it only when your institution permits that processing. Translation
of PII-bearing report text is not cached. Before a request, the app replaces
the resolved learner name plus common email, phone, and student-ID patterns
with temporary placeholders and restores them locally afterward. This reduces
exposure; it is not a substitute for approval to use an external processor.

The Apple-Silicon setup is a **single-user desktop inference profile**, not a high-concurrency production server. The app deliberately serialises local LLM work, caches a completed result for each uploaded document, validates structured model output, and provides a deterministic rule-based fallback. For multi-user deployment, point `LLM_BASE_URL` at an approved Linux/NVIDIA vLLM service.

The platform automates data cleaning, OCR processing, analytics, behavioral analysis, interactive visualization, AI-powered summarization, and multilingual translation, enabling educators and institutions to make informed, data-driven decisions.

Supported report languages include:

- English
- Bahasa Indonesia
- Chinese

---

## Features

| Feature | Description |
|---|---|
| Structured Data Processing | Cleans and processes CSV/Excel academic datasets |
| Unstructured Document Processing | Extracts data from scanned documents using OCR |
| AI Executive Summary | Generates professional summaries using a local LLM served by vLLM |
| DeepL Translation | Translates reports into multiple languages using DeepL API |
| Interactive Analytics | Visualizes student performance through dynamic charts and dashboards |
| Behavioral Analysis | Identifies learning patterns and behavioral trends |
| Student Information Extraction | Automatically extracts student profiles from raw documents |
| Secure LLM Client | Timeouts, retries, circuit breaker, and one-at-a-time admission control for the local inference server |
| LLM Result Cache | Prevents a Streamlit rerun, language change, or download click from regenerating the same summary |
| Structured Output Guard | Validates and normalises local-model JSON before certificate data reaches the UI |
| PII Protection | Protects sensitive student information before AI processing |
| Audit Logging | Records system activities for traceability |
| Docker Support | Fully containerized for consistent, reproducible deployment |

---

## Technology Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | Python |
| AI Inference | vLLM (OpenAI-compatible API) — vllm-metal on Apple Silicon |
| AI Model | Qwen2.5-3B-Instruct 4-bit (MLX), configurable via `LLM_MODEL` |
| Translation | DeepL API |
| Data Processing | Pandas |
| OCR | GLM-OCR through a local MLX-VLM endpoint, with native PDF / Tesseract fallback |
| Visualization | Plotly / Matplotlib |
| Containerization | Docker & Docker Compose |

---

## Getting Started

### Prerequisites

- CPython 3.9–3.13 (Python 3.14 is not yet supported by the pinned
  scientific packages in `requirements.txt`)
- An Apple Silicon Mac on macOS 15+ for local GPU-accelerated inference
  (the vLLM installer bundles its own Python 3.12 environment), or any
  OpenAI-compatible vLLM server reachable over HTTP
- Docker and Docker Compose (for containerized deployment)
- DeepL API key (only for translations — no LLM API key is needed)

### Installation

**1. Clone the repository**

```bash
git clone <your-repository-url>
cd <repository-directory>
```

**2. Create a virtual environment**

```bash
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Use a Python 3.12 installation for a new environment; do not use Python 3.14
with this requirements file. If your terminal does not have `python3.12`,
install a CPython 3.12 release first. On this checkout, an already-provisioned
`.venv` can also be activated directly with `source .venv/bin/activate`.

If `python` or `pip` says “command not found,” do not repeat those bare
commands. macOS commonly exposes versioned commands such as `python3.12`; once
this checkout has `.venv`, the simplest route is `bash scripts/run_app.sh`,
which finds the project environment without relying on `python` being on PATH.

**3. Install dependencies**

```bash
python -m pip install -r requirements.txt
```

**4. Configure environment variables**

```bash
cp .env.example .env
```

Then edit `.env` with your DeepL API key (see [Environment Configuration](#environment-configuration)).

**5. Start the shared local AI inference server (recommended for PDFs/images on M1 / 8 GB)**

```bash
bash scripts/start_mlx_vlm.sh
```

The launcher uses the existing `~/.venv-vllm-metal` Python but starts
[MLX-VLM](https://github.com/Blaizzy/mlx-vlm) with **no model preloaded**. It
uses one local cache and swaps Qwen and GLM-OCR on demand, rather than keeping
both models resident. It is offline-only: both cached models must already be
present; this command never downloads anything. Keep this terminal open and do
not run `scripts/start_vllm.sh` at the same time (both use port 8000).

Verify the cached GLM-OCR runtime without loading a model, then make one small
Qwen endpoint check from a second terminal:

```bash
bash scripts/doctor_glm_ocr.sh
bash scripts/doctor_vllm.sh
```

The GLM doctor only checks the local runtime, cache, and shared endpoint. The
Qwen doctor verifies `/models`, performs one tiny deterministic completion, and
reports latency. Neither script installs or reconfigures packages.

Optional: run a full synthetic GLM-OCR → Qwen check. It creates no files and
never uses an uploaded document; it unloads the shared model afterward.

```bash
./.venv/bin/python scripts/smoke_mlx_document_pipeline.py
```

**6. Run the application (in a second terminal)**

```bash
bash scripts/run_app.sh
```

The application will be available at `http://localhost:8501`.

> If the local inference server is not running, the app still works — executive summaries fall back to the built-in rule-based narrative and a warning banner is shown.

### Verify the installation

Install the testing tools, then run the project’s quality gate:

```bash
python -m pip install -r backend/requirements-test.txt
python backend/run_tests.py
```

The test runner executes the full backend suite, reports coverage, and fails
when coverage falls below the current 45% floor.

---

## Environment Configuration

Create a `.env` file in the project root.

```env
# DeepL Translation API (only external API used)
DEEPL_API_KEY=your_deepl_api_key
```

### Local LLM settings (all optional, sensible defaults)

| Variable | Default | Purpose |
|---|---|---|
| `LLM_BASE_URL` | `http://127.0.0.1:8000/v1` | Shared local MLX-VLM endpoint (or another OpenAI-compatible server) |
| `LLM_MODEL` | `mlx-community/Qwen2.5-3B-Instruct-4bit` | Model to request (must match the served model) |
| `LLM_API_KEY` | *(none)* | Only if the server was started with `--api-key` |
| `LLM_TIMEOUT_SECONDS` | `120` | Per-request timeout |
| `LLM_MAX_RETRIES` | `2` | Retries for transient failures |
| `LLM_BREAKER_COOLDOWN_SECONDS` | `60` | Fast-fail window after the server is unreachable |
| `LLM_MAX_TOKENS` | `320` | Default completion token cap; task-specific calls use tighter limits |
| `LLM_TEMPERATURE` | `0.3` | Sampling temperature |
| `LLM_QUEUE_TIMEOUT_SECONDS` | `5` | Maximum wait for the single local inference slot |
| `LLM_VALIDATE_NAMES` | `0` | Enables an extra local name check only for legacy/direct callers; unstructured uploads already ask Qwen for the recipient in their single extraction request |
| `LLM_EXTRACTION_MAX_DOCUMENT_CHARS` | `6500` | Bounded evidence size for certificate extraction |

For the recommended shared launcher, configure `MLX_VLM_HOST`,
`MLX_VLM_PORT`, `MLX_VLM_VISION_CACHE_SIZE`, `MLX_VLM_PREFILL_STEP_SIZE`,
`MLX_VLM_MAX_TOKENS`, and `MLX_VLM_MAX_KV_SIZE`. The legacy
`scripts/start_vllm.sh` path instead uses `VLLM_PORT`, `VLLM_HOST`,
`VLLM_MAX_MODEL_LEN`, `VLLM_GPU_MEMORY_UTILIZATION`, and `VLLM_MAX_NUM_SEQS`.

The app, server launcher, and health check all read the project `.env`. Set
`LLM_MODEL` there when you want both terminals to use the same Qwen model.
For unstructured documents, Qwen extracts the evidence-backed recipient,
certificates, and skills once; the application then renders deterministic
Plotly charts from those facts rather than asking the model to invent a chart.

### GLM-OCR settings (PDFs and images)

For images and scanned PDF pages, the app first runs GLM-OCR locally through
MLX-VLM. Its clean, page-labelled text is then passed to Qwen for evidence-based
recipient, certificate, and skill extraction. Python and Plotly render the
charts from those validated facts, so a model never fabricates chart values.

For the recommended M1 / 8 GB profile, `scripts/start_mlx_vlm.sh` runs a single
local MLX-VLM process on `127.0.0.1:8000` with no model preloaded.
`GLM_OCR_MODE=http` sends a page to that endpoint using
`mlx-community/GLM-OCR-bf16`; MLX-VLM releases the current text model before it
loads the requested one, and the OCR client unloads GLM when extraction ends.
The next Qwen request reloads Qwen. This trades a little model-switch latency
for a much safer unified-memory peak than keeping both models resident.

Born-digital PDF pages with usable embedded text stay native in HTTP mode,
which is faster and preserves the original text. Images and scanned pages use
GLM-OCR; failures fall back safely to native PDF extraction or Tesseract. The
loopback-only endpoint and offline launcher mean an upload cannot send student
data off-device or silently download a model. Verify the runtime and endpoint
without loading a model:

```bash
bash scripts/doctor_glm_ocr.sh
```

On an M1 Pro with 8 GB unified memory, run `start_mlx_vlm.sh` instead of
`start_vllm.sh` for PDFs and images, close memory-heavy apps, and retain this
conservative OCR profile for large documents:

```env
GLM_OCR_MAX_PAGES=1
GLM_OCR_MAX_TOTAL_PAGES=20
GLM_OCR_MAX_TOKENS=1024
GLM_OCR_MAX_DOCUMENT_BYTES=33554432
```

Use `GLM_OCR_MODE=http-force` only when you specifically need visual OCR on
every PDF page. Set `GLM_OCR_MODE=off` to temporarily use the existing
native/Tesseract fallback. `GLM_OCR_MODE=subprocess` is an explicit
short-lived-worker fallback for diagnosing the shared server; it is not the
recommended M1 / 8 GB profile and is blocked while another local Qwen/MLX
service is live. See [`.env.example`](.env.example) for all bounded OCR
controls.

### Recommended M1 / 8 GB profile

The default `mlx-community/Qwen2.5-3B-Instruct-4bit` is the quality-focused
local choice for this project. For PDFs and images, use the shared MLX-VLM
server so GLM-OCR and Qwen are never kept as two simultaneously active
text-generation models:

```bash
bash scripts/start_mlx_vlm.sh
```

Do not set a 7B+ model as the default on an 8 GB Mac. If the 3B model cannot
stay responsive, use the smaller Qwen fallback in both the server configuration
and the app configuration:

Terminal 1:

```bash
LLM_MODEL=mlx-community/Qwen2.5-1.5B-Instruct-4bit bash scripts/start_mlx_vlm.sh
```

Terminal 2:

```bash
LLM_MODEL=mlx-community/Qwen2.5-1.5B-Instruct-4bit bash scripts/run_app.sh
```

For 16 GB+ machines, test a larger model with `scripts/doctor_vllm.sh` before changing the application default. Larger models are not a substitute for output validation or prompt contracts. `scripts/start_vllm.sh` remains available for a legacy text-only vLLM-Metal setup, but do not run it alongside the shared MLX-VLM server.

### Deployment boundary

Both `scripts/start_mlx_vlm.sh` and the legacy `scripts/start_vllm.sh` bind to
`127.0.0.1` by default, so the local endpoint is not exposed on the network.
Do not expose a desktop MLX/VLLM endpoint directly to other users. For a shared
or internet-facing deployment, run an approved Linux/NVIDIA vLLM service
behind your organisation's authentication and network controls, then configure
this app with its `LLM_BASE_URL`.

---

## Docker Deployment

### Using Docker Compose (Recommended)

```bash
docker compose up --build
```

The default Compose file is a local, immutable-image profile: it exposes the
UI only on `127.0.0.1`, runs as a non-root user, and keeps runtime logs/data in
named volumes. It does not copy `.env`, Streamlit secrets, or local student
data into the image.

For live source edits during local development, use the explicit development
override (on macOS/Linux):

```bash
LOCAL_UID=$(id -u) LOCAL_GID=$(id -g) \
  docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

### Using Docker directly

```bash
docker build -t student-learning-journey .

docker run -p 8501:8501 \
  -e DEEPL_API_KEY=your_deepl_api_key \
  -e LLM_BASE_URL=http://host.docker.internal:8000/v1 \
  student-learning-journey
```

The application will be available at `http://localhost:8501`.

> Note: the vLLM server runs natively on the host (Metal GPU access is not available inside Linux containers on macOS), so the container reaches it via `host.docker.internal`. On a Linux host with an NVIDIA GPU, run standard `vllm serve` and point `LLM_BASE_URL` at it.

> Security note: this Streamlit UI does not provide user authentication. Keep
> it on a trusted local network unless an approved SSO/authenticated reverse
> proxy is placed in front of it.

---

## Data Processing Pipeline

### Structured Data (CSV / Excel)

```
1.  Upload structured dataset
2.  Validate and inspect data
3.  Clean and preprocess records
4.  Detect and handle missing values
5.  Normalize and encode features
6.  Generate statistical summaries
7.  Perform behavioral analysis
8.  Render interactive visualizations
9.  Extract student profile information
10. Generate AI Executive Summary using the local LLM (vLLM)
11. Translate the report using DeepL
12. Export the final report
```

Where a dataset supplies both `Score` and `Maximum`, the application compares
assessments as percentages. Original values are retained in the processed data
for traceability when conversion is needed (for example, `18/20` becomes 90%).

### Unstructured Documents (Scanned / OCR)

```
1. Upload scanned academic document
2. Preprocess image for OCR
3. Extract raw text via OCR
4. Parse and structure extracted data
5. Perform behavioral and performance analysis
6. Generate AI Executive Summary
7. Translate into the selected language
8. Generate the final report
```

---

## Operational Features

- Fully local AI inference via vLLM (OpenAI-compatible API, Metal GPU acceleration)
- Student data never leaves the machine for LLM processing
- Circuit breaker, retries, and health checks around the inference server
- DeepL Translation API integration
- Secure configuration via environment variables
- Docker and Docker Compose deployment
- OCR document processing
- AI-powered executive summaries
- Student information extraction
- Interactive analytics dashboards
- PII protection before AI processing
- Audit logging for system traceability
- Modular backend architecture
- Scalable production-ready design

---

## Future Improvements

- Cloud-native deployment (AWS, Azure, GCP)
- Historical student performance tracking
- Database support (PostgreSQL / MySQL)
- Role-based access control
  
## REST API

The project includes a versioned Flask API alongside the Streamlit interface. Run it with:

```bash
export SECRET_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export JWT_SECRET_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
FLASK_APP=backend.app flask run --port 5000
```

Set distinct, high-entropy `SECRET_KEY` and `JWT_SECRET_KEY` values before
starting the API; it now fails fast without them. Configure explicit
`CORS_ORIGINS` for any browser client, and set `API_DATABASE_PATH` to durable
private storage in production. The API stores users and analysis results in
SQLite by default and scopes analysis resources to the authenticated user. Its
WAL/busy-timeout settings are for a single host with brief overlaps only; use a
managed database, server-side role provisioning, and rate limiting before a
multi-user or internet-facing deployment. The API verifies owner-only storage
permissions on macOS/Linux; Windows deployments must enforce equivalent NTFS
ACLs outside the application.

Available endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/v1/health` | Service health check |
| POST | `/api/v1/analyses` | Create analytics from JSON records or a CSV/XLSX/Image/PDF upload |
| GET | `/api/v1/analyses` | List the authenticated user's analyses with pagination |
| GET | `/api/v1/analyses/<id>` | Get one analysis |
| DELETE | `/api/v1/analyses/<id>` | Delete one analysis |

JSON analysis example:

```json
{
  "student": {"name": "Aisha"},
  "records": [
    {"subject": "Mathematics", "score": 90, "maximum": 100},
    {"subject": "Science", "score": 75, "maximum": 100}
  ]
}
```

All API errors use a predictable shape: `{"error": {"code": "...", "message": "..."}, "request_id": "..."}`.

---

## License

No license has been selected for this repository. The project owner must add an
approved `LICENSE` file before redistribution.

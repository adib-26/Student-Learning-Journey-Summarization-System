![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![Vertex AI](https://img.shields.io/badge/Vertex_AI-Model_Garden-4285F4.svg)
![DeepL](https://img.shields.io/badge/DeepL-API-blue.svg)

# Student Learning Journey Summarization System

**AI-Powered Educational Data Analytics Platform**

**Streamlit • Vertex AI vLLM • Qwen3.6 • DeepL Translation • OCR**

**Demo Video:** https://youtu.be/pLLHletb9pE

**Sample Dataset:** https://github.com/user-attachments/files/29584548/Sample.Data.zip

**Sample Report:** https://drive.google.com/file/d/1g4QveYbhyFTo_bqDF5yw21BqoJC2DnnO/view?usp=sharing

---

## Overview

The **Student Learning Journey Summarization System** is an AI-powered educational analytics platform that transforms raw academic records into meaningful insights, multilingual reports, and professional AI-generated summaries.

Built with **Streamlit** and **Vertex AI Model Garden**, the application uses one `Qwen3.6-35B-A3B` deployment for scanned-document transcription, evidence extraction, and report summarization. Vertex runs the model through an OpenAI-compatible vLLM serving container on a dedicated cloud GPU endpoint, so this Mac does not need an NVIDIA card.

Uploaded evidence is sent to the configured Vertex AI project for inference. If the endpoint is unavailable, the app degrades to native PDF extraction, Tesseract, and rule-based summaries.

Translation is a separate, optional external service: selecting a non-English
report language with `DEEPL_API_KEY` configured sends the selected report text
to DeepL. Use it only when your institution permits that processing. Translation
of PII-bearing report text is not cached. Before a request, the app replaces
the resolved learner name plus common email, phone, and student-ID patterns
with temporary placeholders and restores them locally afterward. This reduces
exposure; it is not a substitute for approval to use an external processor.

The GPU runs in the configured Google Cloud project. The application caches completed document results and validates every structured model response. Access uses short-lived Google Application Default Credential tokens; do not commit endpoint credentials or student data.

The platform automates data cleaning, OCR processing, analytics, behavioral analysis, interactive visualization, AI-powered summarization, and multilingual translation, enabling educators and institutions to make informed, data-driven decisions.

The language menu offers:

- English
- Bahasa Melayu
- Chinese

Bahasa Melayu is the Malay option. DeepL receives it as `MS`, not Indonesian. The built-in upload and service-warning text for that option is Malay. Other labels are translated from English when a DeepL key is configured.

---

## Features

| Feature | Description |
|---|---|
| Structured Data Processing | Cleans and processes CSV/Excel academic datasets |
| Unstructured Document Processing | Extracts data from scanned documents using OCR |
| AI Executive Summary | Generates professional summaries using Qwen3.6 on Vertex AI |
| DeepL Translation | Translates reports into multiple languages using DeepL API |
| Interactive Analytics | Visualizes student performance through dynamic charts and dashboards |
| Behavioral Analysis | Identifies learning patterns and behavioral trends |
| Student Information Extraction | Automatically extracts student profiles from raw documents |
| Secure LLM Client | Timeouts, retries, circuit breaker, and one-at-a-time admission control for the Vertex endpoint |
| LLM Result Cache | Prevents a Streamlit rerun, language change, or download click from regenerating the same summary |
| Structured Output Guard | Validates and normalises model JSON before certificate data reaches the UI |
| PII Protection | Protects sensitive student information before AI processing |
| Audit Logging | Records system activities for traceability |

---

## Technology Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | Python |
| AI Inference | Vertex AI Model Garden vLLM-compatible dedicated endpoint |
| AI Model | Qwen3.6-35B-A3B |
| Translation | DeepL API |
| Data Processing | Pandas |
| OCR | Qwen3.6 through Vertex AI, with native PDF / Tesseract fallback |
| Visualization | Plotly / Matplotlib |

---

## Getting Started

### Prerequisites

- CPython 3.9–3.13 (Python 3.14 is not yet supported by the pinned
  scientific packages in `requirements.txt`)
- A Google Cloud project with Vertex AI enabled and sufficient GPU quota
- Google Cloud CLI with Application Default Credentials
- DeepL API key (only for optional translations)

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

**5. Deploy Qwen3.6-35B-A3B on Vertex AI**

```bash
python3.12 -m venv .venv-vertex
.venv-vertex/bin/pip install 'google-cloud-aiplatform>=1.120' google-auth
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
gcloud config set compute/region YOUR_REGION
```

Put the same project and region in `.env`:

```env
VERTEX_PROJECT=YOUR_PROJECT_ID
VERTEX_REGION=YOUR_REGION
```

Then connect the app to the endpoint:

```bash
bash scripts/deploy_inference.sh
```

The script deploys `publishers/qwen/models/qwen3-6@qwen3.6-35b-a3b` in
`YOUR_REGION` and writes the endpoint URL and model settings into `.env`.
Keep `.env` out of Git. A later run
reuses that endpoint. Set `VERTEX_FORCE_DEPLOY=1` only when you intend to
create another GPU deployment. The endpoint keeps billing until you delete it
in the Vertex AI console. Google credentials stay in the gcloud login and are
not committed.

**6. Open the application**

```bash
bash scripts/run_app.sh
```

The application is available at `http://localhost:8501`.

> If the Vertex endpoint is unavailable, the app still works: executive summaries fall back to the built-in rule-based narrative and a warning is shown.

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
# DeepL Translation API (optional)
DEEPL_API_KEY=your_deepl_api_key
```

### Vertex AI

| Variable | Purpose |
|---|---|
| `VERTEX_PROJECT` | Google Cloud project that owns the endpoint. Put this only in `.env`. |
| `VERTEX_REGION` | Vertex region, such as `us-central1`. Put this only in `.env`. |
| `VERTEX_ENDPOINT_ID` | Existing endpoint to reuse. The deploy script writes it to `.env`. |
| `VERTEX_ENDPOINT_DNS` | Dedicated endpoint DNS, when Vertex assigns one. Keep it in `.env`. |
| `VERTEX_FORCE_DEPLOY` | Set to `1` to create another billed GPU endpoint instead of reusing `VERTEX_ENDPOINT_ID`. |

`bash scripts/deploy_inference.sh` accepts the model EULA and deploys one
replica on a dedicated endpoint. Requests use a short-lived Application
Default Credential token. Do not put that token in `.env`.

### Application model client

| Variable | Default | Purpose |
|---|---|---|
| `LLM_BASE_URL` | Vertex endpoint URL written by the deploy script | OpenAI-compatible base URL, without a trailing `/chat/completions` |
| `LLM_MODEL` | `qwen3.6-35b-a3b` | Must match the model served by the endpoint |
| `LLM_TIMEOUT_SECONDS` | `180` | Per-request timeout |
| `LLM_MAX_RETRIES` | `2` | SDK retries for transient failures |
| `LLM_BREAKER_COOLDOWN_SECONDS` | `60` | Fast-fail window after server failure |
| `LLM_MAX_TOKENS` | `512` | Default completion cap; tasks may override it |
| `LLM_TEMPERATURE` | `0.2` | Default sampling temperature |
| `LLM_ENABLE_THINKING` | `0` | Keep Qwen3.6 in concise instruct mode |
| `LLM_QUEUE_TIMEOUT_SECONDS` | `30` | Wait limit for duplicate work from one app process |
| `LLM_EXTRACTION_MAX_DOCUMENT_CHARS` | `12000` | Evidence budget for certificate extraction |

### Visual document settings

Born-digital PDF pages retain embedded text because it is faster and more
faithful. Images and scanned/no-text PDF pages are rendered within bounded
limits and sent to the same Qwen3.6 endpoint. The visual prompt asks for
literal transcription only; certificate extraction and summarization remain
separate validated stages.

| Variable | Default | Purpose |
|---|---|---|
| `VLM_OCR_MODE` | `http` | Use vision only for pages without usable native text |
| `VLM_OCR_MAX_PAGES` | `4` | Maximum visual pages per document |
| `VLM_OCR_MAX_TOTAL_PAGES` | `30` | Maximum PDF pages considered |
| `VLM_OCR_MAX_TOKENS` | `2048` | Transcription cap per rendered page |
| `VLM_OCR_RENDER_DPI` | `180` | PDF rendering resolution before bounding |
| `VLM_OCR_MAX_IMAGE_DIMENSION` | `2048` | Longest image edge sent to the Vertex model |
| `VLM_OCR_MAX_DOCUMENT_BYTES` | `67108864` | Hard source-document limit |
| `VLM_OCR_ALLOW_REMOTE` | `0` | Must be `1` before scanned pages may be sent to Vertex. The deploy script sets this. |

Use `VLM_OCR_MODE=http-force` only when every retained PDF page needs visual
transcription. Use `VLM_OCR_MODE=off` to test native PDF and Tesseract fallbacks.

### Deployment boundary

Streamlit binds to the local machine. For multi-user or internet-facing
deployment, add institutional SSO, TLS, request limits, and network policy in
front of the application. Page pixels reach Vertex AI before text can be
redacted, so that project must be approved for student-document processing.

---

## Running the application

```bash
bash scripts/run_app.sh
```

Open `http://localhost:8501`. Restart the app after `deploy_inference.sh`
changes `.env`, then refresh the browser.

`docker-compose.yml` remains an optional NVIDIA-host path for
`Qwen/Qwen3-VL-8B-Instruct`. It is not the Qwen3.6 Vertex deployment. The
Streamlit UI has no built-in user authentication. Keep it on a trusted network
or place approved SSO and TLS termination in front of it.

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
10. Generate AI Executive Summary using the configured Qwen endpoint
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

- Qwen3.6-35B-A3B on one dedicated Vertex AI endpoint in the configured region
- Student evidence is sent only to the configured Vertex AI project for model inference
- Circuit breaker, retries, and health checks around the inference server
- DeepL Translation API integration
- Secure configuration via environment variables
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

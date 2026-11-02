#!/usr/bin/env python3
"""Configure or deploy Qwen3.6-35B-A3B on Vertex AI Model Garden.

Project and region come from the environment or the gcloud configuration.
An existing ``VERTEX_ENDPOINT_ID`` is reused by default so rerunning this
script cannot accidentally create another continuously billed GPU replica.
Google credentials stay in Application Default Credentials, not in git.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL_NAME = "Qwen3.6-35B-A3B"
PUBLISHER_MODEL_NAME = f"publishers/qwen/models/qwen3-6@{BASE_MODEL_NAME.lower()}"
SERVED_MODEL_NAME = BASE_MODEL_NAME.lower()


def _gcloud_value(*args: str) -> str:
    try:
        result = subprocess.run(
            ["gcloud", "config", "get-value", *args],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if result.returncode != 0:
        return ""
    value = result.stdout.strip()
    if not value or value == "(unset)":
        return ""
    return value


def _dotenv_value(name: str) -> str:
    path = ROOT / ".env"
    if not path.is_file():
        return ""
    prefix = f"{name}="
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped[len(prefix):].strip().strip('"').strip("'")
    return ""


def resolve_project() -> str:
    return (
        os.environ.get("VERTEX_PROJECT", "").strip()
        or os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
        or _dotenv_value("VERTEX_PROJECT")
        or _dotenv_value("GOOGLE_CLOUD_PROJECT")
        or _gcloud_value("project")
    )


def resolve_region() -> str:
    return (
        os.environ.get("VERTEX_REGION", "").strip()
        or os.environ.get("GOOGLE_CLOUD_REGION", "").strip()
        or _dotenv_value("VERTEX_REGION")
        or _dotenv_value("GOOGLE_CLOUD_REGION")
        or _gcloud_value("compute/region")
        or _gcloud_value("ai/region")
        or "us-central1"
    )


def upsert_env(
    base_url: str,
    endpoint_id: str = "",
    endpoint_dns: str = "",
) -> None:
    path = ROOT / ".env"
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    values = {
        "LLM_BASE_URL": base_url,
        "LLM_MODEL": SERVED_MODEL_NAME,
        "VLM_OCR_ALLOW_REMOTE": "1",
        "VERTEX_PROJECT": resolve_project(),
        "VERTEX_REGION": resolve_region(),
    }
    if endpoint_id:
        values["VERTEX_ENDPOINT_ID"] = endpoint_id
    if endpoint_dns:
        values["VERTEX_ENDPOINT_DNS"] = endpoint_dns
    seen: set[str] = set()
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        matched = False
        for key, value in values.items():
            if stripped.startswith(f"{key}=") or stripped.startswith(f"# {key}="):
                if key not in seen:
                    kept.append(f"{key}={value}")
                    seen.add(key)
                matched = True
                break
        if not matched:
            kept.append(line)
    for key, value in values.items():
        if key not in seen:
            kept.append(f"{key}={value}")
    path.write_text("\n".join(kept).rstrip() + "\n", encoding="utf-8")


def _endpoint_identity(endpoint: object) -> tuple[str, str]:
    resource = str(getattr(endpoint, "resource_name", "") or getattr(endpoint, "name", ""))
    endpoint_id = resource.rstrip("/").split("/")[-1]
    endpoint_dns = str(getattr(endpoint, "dedicated_endpoint_dns", "") or "")
    if not endpoint_dns:
        resource_message = getattr(endpoint, "gca_resource", None)
        endpoint_dns = str(getattr(resource_message, "dedicated_endpoint_dns", "") or "")
    return endpoint_id, endpoint_dns


def _base_url_from_parts(
    endpoint_id: str,
    project: str,
    region: str,
    endpoint_dns: str = "",
) -> str:
    host = endpoint_dns or f"{region}-aiplatform.googleapis.com"
    return (
        f"https://{host}/v1beta1/projects/{project}/locations/{region}/"
        f"endpoints/{endpoint_id}"
    )


def endpoint_base_url(endpoint: object, project: str, region: str) -> str:
    endpoint_id, endpoint_dns = _endpoint_identity(endpoint)
    if not endpoint_id:
        raise RuntimeError("Vertex deployment returned no endpoint ID")
    return _base_url_from_parts(endpoint_id, project, region, endpoint_dns)


def main() -> None:
    project = resolve_project()
    region = resolve_region()
    if not project:
        raise SystemExit(
            "Vertex AI needs a Google Cloud project. Install the gcloud CLI, then run:\n"
            "  gcloud auth application-default login\n"
            "  gcloud config set project YOUR_PROJECT_ID\n"
            "  export GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID\n"
            "Then run bash scripts/deploy_inference.sh again.\n"
            "The login stays in your Google Cloud SDK config and is not committed."
        )

    existing_endpoint_id = (
        os.environ.get("VERTEX_ENDPOINT_ID", "").strip()
        or _dotenv_value("VERTEX_ENDPOINT_ID")
    )
    existing_endpoint_dns = (
        os.environ.get("VERTEX_ENDPOINT_DNS", "").strip()
        or _dotenv_value("VERTEX_ENDPOINT_DNS")
    )
    force_deploy = os.environ.get("VERTEX_FORCE_DEPLOY", "").strip().casefold() in {
        "1", "true", "yes", "on"
    }
    if existing_endpoint_id and not force_deploy:
        base_url = _base_url_from_parts(
            existing_endpoint_id,
            project,
            region,
            existing_endpoint_dns,
        )
        upsert_env(base_url, existing_endpoint_id, existing_endpoint_dns)
        print(f"Using existing Vertex endpoint {existing_endpoint_id}.")
        print("Updated the Vertex model settings in .env without deploying another GPU.")
        return

    try:
        import vertexai
        from vertexai import model_garden
    except ImportError as exc:
        raise SystemExit(
            "The Vertex AI SDK is not installed for this Python. Run:\n"
            "  python3.12 -m venv .venv-vertex\n"
            "  .venv-vertex/bin/pip install 'google-cloud-aiplatform>=1.120'\n"
            "Then run bash scripts/deploy_inference.sh again."
        ) from exc

    print(f"Deploying {PUBLISHER_MODEL_NAME} to {project} ({region})...")
    vertexai.init(project=project, location=region)
    model = model_garden.OpenModel(PUBLISHER_MODEL_NAME)
    endpoint = model.deploy(
        accept_eula=True,
        min_replica_count=1,
        max_replica_count=1,
        use_dedicated_endpoint=True,
        endpoint_display_name="student-learning-qwen3-6-35b-a3b",
        model_display_name="student-learning-qwen3-6-35b-a3b",
    )
    endpoint_id, endpoint_dns = _endpoint_identity(endpoint)
    base_url = endpoint_base_url(endpoint, project, region)
    upsert_env(base_url, endpoint_id, endpoint_dns)
    print(f"Vertex is ready. Wrote LLM_BASE_URL={base_url} to .env")
    print(f"LLM_MODEL={SERVED_MODEL_NAME}")
    print("Restart the Streamlit app, then refresh the browser.")
    print(
        "This endpoint keeps a GPU running until you delete it in the "
        "Vertex AI console."
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Deploy the OpenAI-compatible model to a private Kaggle GPU kernel.

Prefers a Kaggle access token in ~/.kaggle/access_token or KAGGLE_API_TOKEN.
The token is injected into a temporary private kernel and is never written
into the git repository.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL_SLUG = "student-learning-vllm"
DATASET_SLUG = "student-learning-llm-url"
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
POLL_SECONDS = 30
POLL_ATTEMPTS = 60


def read_access_token() -> str:
    """Return the KGAT access token from the environment or ~/.kaggle/access_token."""
    from_env = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if from_env:
        return from_env
    path = Path.home() / ".kaggle" / "access_token"
    if not path.is_file():
        return ""
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path.read_text(encoding="utf-8").strip()


def username_from_access_token(token: str, config_dir: Path) -> str:
    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = token
    env["KAGGLE_CONFIG_DIR"] = str(config_dir)
    env.pop("KAGGLE_USERNAME", None)
    env.pop("KAGGLE_KEY", None)
    script = (
        "from kaggle.api.kaggle_api_extended import KaggleApi\n"
        "api = KaggleApi()\n"
        "api.authenticate()\n"
        "print(api.config_values.get('username', ''))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
    )
    username = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    if result.returncode != 0 or not username:
        raise SystemExit(
            "Kaggle did not accept the access token in ~/.kaggle/access_token. "
            "Generate a new one at https://www.kaggle.com/settings/api."
        )
    return username


def legacy_credentials() -> tuple[str, str]:
    """Read ~/.kaggle/kaggle.json. The key stays in the private kernel only."""
    path = Path.home() / ".kaggle" / "kaggle.json"
    if not path.is_file():
        raise SystemExit(
            "Kaggle credentials were not found at ~/.kaggle/kaggle.json.\n"
            "Open https://www.kaggle.com/settings/api, generate a token, and "
            "save it there with mode 600. Do not commit that file.\n"
            "If the token is rejected, run `kaggle auth login` and try again."
        )
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    data = json.loads(path.read_text(encoding="utf-8"))
    username = str(data.get("username", "")).strip()
    key = str(data.get("key", "")).strip()
    if not username or not key:
        raise SystemExit(f"{path} must contain username and key.")
    return username, key


def kaggle_env(username: str, key: str) -> dict[str, str]:
    env = os.environ.copy()
    env["KAGGLE_USERNAME"] = username
    env["KAGGLE_KEY"] = key
    return env


def kaggle_executable() -> list[str]:
    for candidate in (
        ROOT / ".venv-kaggle" / "bin" / "kaggle",
        ROOT / ".venv" / "bin" / "kaggle",
        ROOT / ".venv-modal" / "bin" / "kaggle",
        Path(shutil.which("kaggle") or ""),
    ):
        if candidate and candidate.is_file():
            return [str(candidate)]
    raise SystemExit(
        "The Kaggle CLI is not installed. Run `pip install kaggle` and try again."
    )


def upsert_env(base_url: str, api_key: str) -> None:
    path = ROOT / ".env"
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    values = {
        "LLM_BASE_URL": base_url,
        "LLM_MODEL": MODEL_NAME,
        "LLM_API_KEY": api_key,
        "VLM_OCR_ALLOW_REMOTE": "1",
    }
    seen: set[str] = set()
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        matched = False
        for key in values:
            if stripped.startswith(f"{key}=") or stripped.startswith(f"# {key}="):
                if key not in seen:
                    kept.append(f"{key}={values[key]}")
                    seen.add(key)
                matched = True
                break
        if not matched:
            kept.append(line)
    for key, value in values.items():
        if key not in seen:
            kept.append(f"{key}={value}")
    path.write_text("\n".join(kept).rstrip() + "\n", encoding="utf-8")


def inject_header(source: str, header: str) -> str:
    """Insert runtime settings after any ``from __future__`` imports.

    Python requires future imports to precede every other statement, so
    prepending the header to the whole file raises a SyntaxError on Kaggle.
    """
    lines = source.splitlines(keepends=True)
    insert_at = 0
    for index, line in enumerate(lines):
        if line.startswith("from __future__ import"):
            insert_at = index + 1
    return "".join(lines[:insert_at]) + header + "".join(lines[insert_at:])


def build_kernel(
    directory: Path,
    username: str,
    api_token: str,
    deploy_id: str,
    kaggle_key: str = "",
    kaggle_token: str = "",
) -> None:
    dataset_id = f"{username}/{DATASET_SLUG}"
    header_lines = [
        "import os",
        f"os.environ['KAGGLE_USERNAME'] = {username!r}",
        f"os.environ['LLM_API_TOKEN'] = {api_token!r}",
        f"os.environ['DATASET_ID'] = {dataset_id!r}",
        f"os.environ['DEPLOY_ID'] = {deploy_id!r}",
    ]
    if kaggle_token:
        header_lines.append(f"os.environ['KAGGLE_API_TOKEN'] = {kaggle_token!r}")
    elif kaggle_key:
        header_lines.append(f"os.environ['KAGGLE_KEY'] = {kaggle_key!r}")
    header = "\n".join(header_lines) + "\n"
    source = (ROOT / "kaggle" / "serve_openai.py").read_text(encoding="utf-8")
    (directory / "serve_openai.py").write_text(inject_header(source, header), encoding="utf-8")
    metadata = {
        "id": f"{username}/{KERNEL_SLUG}",
        "title": "Student Learning vLLM",
        "code_file": "serve_openai.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        # Pin a T4: the default accelerator can be a P100, which current
        # PyTorch CUDA builds on Kaggle no longer support.
        "machine_shape": "NvidiaTeslaT4",
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": [],
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }
    (directory / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


class KaggleUnauthorized(RuntimeError):
    pass


def push_kernel(kaggle: list[str], directory: Path, env: dict[str, str]) -> None:
    result = subprocess.run(
        [*kaggle, "kernels", "push", "-p", str(directory)],
        text=True,
        capture_output=True,
        env=env,
    )
    output = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    if output:
        print(output)
    if result.returncode == 0:
        return
    if "401" in output or "Unauthorized" in output:
        raise KaggleUnauthorized(output)
    raise SystemExit(
        "Kaggle rejected the kernel. Confirm that GPU and internet are enabled "
        "on your account, then run bash scripts/deploy_inference.sh again."
    )


def read_published_url(
    kaggle: list[str], dataset_id: str, env: dict[str, str], deploy_id: str
) -> str | None:
    """Return this deployment's URL, ignoring one left by an earlier session."""
    with tempfile.TemporaryDirectory() as download_dir:
        result = subprocess.run(
            [*kaggle, "datasets", "download", "-d", dataset_id, "-p", download_dir, "--unzip", "--force"],
            capture_output=True,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            return None
        root = Path(download_dir)
        ids = list(root.rglob("deploy_id.txt"))
        if not ids or ids[0].read_text(encoding="utf-8").strip() != deploy_id:
            return None
        matches = list(root.rglob("llm_url.txt"))
        if not matches:
            return None
        url = matches[0].read_text(encoding="utf-8").strip()
        if url.startswith("https://") and url.endswith("/v1"):
            return url
    return None


def kernel_failed(kaggle: list[str], kernel_id: str, env: dict[str, str]) -> bool:
    result = subprocess.run(
        [*kaggle, "kernels", "status", kernel_id],
        capture_output=True,
        text=True,
        env=env,
    )
    status = result.stdout.casefold()
    return "error" in status or "cancel" in status or "complete" in status


def wait_for_url(
    kaggle: list[str],
    username: str,
    env: dict[str, str],
    api_token: str,
    deploy_id: str,
) -> None:
    kernel_id = f"{username}/{KERNEL_SLUG}"
    dataset_id = f"{username}/{DATASET_SLUG}"
    print(
        "Waiting for the Kaggle GPU session to load the model and open a tunnel. "
        "This often takes several minutes."
    )
    print(f"Kernel page: https://www.kaggle.com/code/{kernel_id}")
    for attempt in range(1, POLL_ATTEMPTS + 1):
        url = read_published_url(kaggle, dataset_id, env, deploy_id)
        if url:
            upsert_env(url, api_token)
            print(f"Kaggle is ready. Wrote LLM_BASE_URL={url} to .env")
            print("Restart the Streamlit app, then refresh the browser.")
            print(
                "A Kaggle GPU session stops after several hours. "
                "Run bash scripts/deploy_inference.sh again when the URL stops answering."
            )
            return
        # A failed kernel never publishes; stop now instead of waiting it out.
        if attempt > 2 and kernel_failed(kaggle, kernel_id, env):
            raise SystemExit(
                "The Kaggle kernel stopped before publishing a URL. Read its log with:\n"
                f"  kaggle kernels logs {kernel_id}"
            )
        print(f"URL not published yet ({attempt}/{POLL_ATTEMPTS}).", flush=True)
        time.sleep(POLL_SECONDS)
    raise SystemExit(
        "Kaggle did not publish a URL in time. Open the kernel page above and "
        "check whether the GPU quota or the model download failed."
    )


def modern_kaggle_executable() -> list[str]:
    candidate = Path(shutil.which("kaggle") or "")
    if candidate.is_file() and ".venv-kaggle" not in str(candidate):
        return [str(candidate)]
    raise SystemExit(
        "The legacy Kaggle API key was rejected, and the current Kaggle CLI "
        "was not found. Run `kaggle auth login`, then "
        "`bash scripts/deploy_inference.sh` again."
    )


def oauth_env(config_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = str(config_dir)
    for name in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
        env.pop(name, None)
    return env


def snapshot_oauth_credentials() -> Path | None:
    source = Path.home() / ".kaggle" / "credentials.json"
    if not source.is_file():
        return None
    handle, name = tempfile.mkstemp(prefix="kaggle-creds-")
    os.close(handle)
    destination = Path(name)
    shutil.copy2(source, destination)
    os.chmod(destination, 0o600)
    return destination


def restore_oauth_credentials(snapshot: Path | None) -> None:
    source = Path.home() / ".kaggle" / "credentials.json"
    if snapshot and snapshot.is_file() and not source.is_file():
        source.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(snapshot, source)
        os.chmod(source, 0o600)


def oauth_access_token(kaggle: list[str], env: dict[str, str]) -> str:
    snapshot = snapshot_oauth_credentials()
    try:
        result = subprocess.run(
            [*kaggle, "auth", "print-access-token"],
            capture_output=True,
            text=True,
            env=env,
        )
    finally:
        restore_oauth_credentials(snapshot)
        if snapshot:
            snapshot.unlink(missing_ok=True)
    token = result.stdout.strip()
    if result.returncode != 0 or not token or " " in token:
        raise SystemExit(
            "Kaggle rejected the API key in ~/.kaggle/kaggle.json, and the saved "
            "login could not be refreshed. Run `kaggle auth login`, then "
            "`bash scripts/deploy_inference.sh` again."
        )
    return token


def deploy_with_access_token(access_token: str) -> None:
    kaggle = modern_kaggle_executable()
    api_token = secrets.token_urlsafe(24)
    deploy_id = secrets.token_hex(8)
    with tempfile.TemporaryDirectory(prefix="kaggle-kernel-") as temp:
        directory = Path(temp)
        config_dir = directory / "kaggle-config"
        config_dir.mkdir()
        username = username_from_access_token(access_token, config_dir)
        env = os.environ.copy()
        env["KAGGLE_API_TOKEN"] = access_token
        env["KAGGLE_CONFIG_DIR"] = str(config_dir)
        env.pop("KAGGLE_USERNAME", None)
        env.pop("KAGGLE_KEY", None)
        build_kernel(directory, username, api_token, deploy_id, kaggle_token=access_token)
        print(f"Pushing private kernel {username}/{KERNEL_SLUG}...")
        push_kernel(kaggle, directory, env)
        wait_for_url(kaggle, username, env, api_token, deploy_id)


def main() -> None:
    access_token = read_access_token()
    if access_token:
        deploy_with_access_token(access_token)
        return
    username, key = legacy_credentials()
    kaggle = kaggle_executable()
    api_token = secrets.token_urlsafe(24)
    deploy_id = secrets.token_hex(8)
    env = kaggle_env(username, key)
    with tempfile.TemporaryDirectory(prefix="kaggle-kernel-") as temp:
        directory = Path(temp)
        build_kernel(directory, username, api_token, deploy_id, kaggle_key=key)
        print(f"Pushing private kernel {username}/{KERNEL_SLUG}...")
        try:
            push_kernel(kaggle, directory, env)
        except KaggleUnauthorized:
            print("The legacy API key was rejected. Trying the saved Kaggle login...")
            kaggle = modern_kaggle_executable()
            config_dir = Path(temp) / "kaggle-config"
            config_dir.mkdir()
            env = oauth_env(config_dir)
            kaggle_token = oauth_access_token(kaggle, env)
            build_kernel(directory, username, api_token, deploy_id, kaggle_token=kaggle_token)
            push_kernel(kaggle, directory, env)
        wait_for_url(kaggle, username, env, api_token, deploy_id)


if __name__ == "__main__":
    main()

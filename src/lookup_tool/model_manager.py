from __future__ import annotations

from pathlib import Path
import json
import shutil
import subprocess
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


DEFAULT_MODEL_URL = (
    "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/"
    "Qwen3-8B-Q4_K_M.gguf?download=true"
)


def model_status(settings: dict[str, Any]) -> dict[str, Any]:
    local_path = Path(str(settings.get("local_model_path") or ""))
    base_url = str(settings.get("base_url") or "").rstrip("/")
    return {
        "settings": settings,
        "local_model_exists": local_path.exists() if str(local_path) else False,
        "local_model_path": str(local_path) if str(local_path) else None,
        "local_model_size": local_path.stat().st_size if local_path.exists() else 0,
        "ollama_available": shutil.which("ollama") is not None,
        "llama_available": shutil.which("llama") is not None or shutil.which("llama-server") is not None,
        "endpoint_ok": endpoint_ok(base_url),
        "available_ollama_models": list_ollama_models() if shutil.which("ollama") else [],
    }


def endpoint_ok(base_url: str) -> bool:
    if not base_url:
        return False
    url = base_url.rstrip("/") + "/models"
    request = Request(url, method="GET")
    try:
        with urlopen(request, timeout=2) as response:
            return 200 <= response.status < 300
    except (OSError, URLError, TimeoutError):
        return False


def list_ollama_models() -> list[str]:
    try:
        result = subprocess.run(
            ["ollama", "list"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    models: list[str] = []
    for line in result.stdout.splitlines()[1:]:
        parts = line.split()
        if parts:
            models.append(parts[0])
    return models


def download_recommended_model(settings: dict[str, Any], url: str = DEFAULT_MODEL_URL) -> dict[str, Any]:
    target = Path(str(settings.get("local_model_path") or "data/models/Qwen3-8B-Q4_K_M.gguf"))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return {
            "status": "already_exists",
            "path": str(target),
            "size": target.stat().st_size,
            "expected_size": target.stat().st_size,
        }
    partial = target.with_suffix(target.suffix + ".partial")
    request = Request(url, headers={"User-Agent": "localMathRag/0.1"})
    with urlopen(request, timeout=60) as response:
        total = int(response.headers.get("Content-Length", "0") or "0")
        with partial.open("wb") as fh:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
    partial.replace(target)
    return {
        "status": "ok",
        "path": str(target),
        "size": target.stat().st_size,
        "expected_size": total,
    }


def settings_from_request(payload: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "enabled",
        "provider",
        "base_url",
        "model",
        "temperature",
        "timeout_seconds",
        "local_models_dir",
        "local_model_path",
        "recommended_repo",
        "recommended_file",
    }
    cleaned = {key: payload[key] for key in allowed if key in payload}
    if "enabled" in cleaned:
        cleaned["enabled"] = bool(cleaned["enabled"])
    if "temperature" in cleaned:
        cleaned["temperature"] = float(cleaned["temperature"])
    if "timeout_seconds" in cleaned:
        cleaned["timeout_seconds"] = int(cleaned["timeout_seconds"])
    return cleaned


def status_json(settings: dict[str, Any]) -> str:
    return json.dumps(model_status(settings), ensure_ascii=False, indent=2)

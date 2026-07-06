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
    local_path = find_installed_model(settings)
    base_url = str(settings.get("base_url") or "").rstrip("/")
    installed_llama = find_installed_llama_server(settings)
    return {
        "settings": settings,
        "local_model_exists": local_path.exists() if local_path else False,
        "local_model_path": str(local_path) if local_path else None,
        "local_model_size": local_path.stat().st_size if local_path and local_path.exists() else 0,
        "ollama_available": shutil.which("ollama") is not None,
        "llama_available": shutil.which("llama") is not None or shutil.which("llama-server") is not None or installed_llama is not None,
        "llama_server_path": str(installed_llama) if installed_llama else None,
        "endpoint_ok": endpoint_ok(base_url),
        "available_ollama_models": list_ollama_models() if shutil.which("ollama") else [],
    }


def find_installed_model(settings: dict[str, Any]) -> Path | None:
    configured_raw = str(settings.get("local_model_path") or "").strip()
    if configured_raw:
        configured = Path(configured_raw)
        if configured.exists():
            return configured
    model_name = str(settings.get("recommended_file") or "Qwen3-8B-Q4_K_M.gguf")
    search_roots: list[Path] = []
    local_models_dir_raw = str(settings.get("local_models_dir") or "").strip()
    if local_models_dir_raw:
        search_roots.append(Path(local_models_dir_raw))
    search_roots.extend(data_root / "models" for data_root in candidate_data_roots())
    for root in search_roots:
        candidate = root / model_name
        if candidate.exists():
            return candidate
    return None


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


def find_installed_llama_server(settings: dict[str, Any]) -> Path | None:
    configured_raw = str(settings.get("llama_server_path") or "").strip()
    if configured_raw:
        configured = Path(configured_raw)
        if configured.exists():
            return configured
    search_roots: list[Path] = []
    local_models_dir_raw = str(settings.get("local_models_dir") or "").strip()
    local_model_path_raw = str(settings.get("local_model_path") or "").strip()
    if local_models_dir_raw:
        search_roots.append(Path(local_models_dir_raw).parent / "runtime" / "llama.cpp")
    if local_model_path_raw:
        search_roots.append(Path(local_model_path_raw).parent.parent / "runtime" / "llama.cpp")
    search_roots.extend(data_root / "runtime" / "llama.cpp" for data_root in candidate_data_roots())
    for root in search_roots:
        if not root.exists():
            continue
        found = sorted(root.glob("**/llama-server.exe"), key=lambda item: item.stat().st_mtime, reverse=True)
        if found:
            return found[0]
    return None


def candidate_data_roots() -> list[Path]:
    roots: list[Path] = []
    current = Path.cwd().resolve()
    for item in [current, *current.parents]:
        roots.append(item / "data")
    seen: set[str] = set()
    unique: list[Path] = []
    for root in roots:
        key = str(root).lower()
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


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
        "llama_server_path",
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

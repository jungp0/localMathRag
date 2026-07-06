from __future__ import annotations

import hashlib
import fnmatch
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


SCHEMA_DIR = Path(os.environ.get("LOCALMATHRAG_SCHEMA_DIR", "/schemas"))
DATA_DIR = Path(os.environ.get("LOCALMATHRAG_DATA_DIR", "/data"))
MODEL_DIR = Path(os.environ.get("LOCALMATHRAG_MODEL_DIR", DATA_DIR / "models"))
MODEL_EXTENSIONS = {".gguf", ".safetensors", ".bin"}
MODEL_METADATA_NAME = ".localmathrag-model.json"
SNAPSHOT_FILE_EXTENSIONS = {
    ".bin",
    ".jinja",
    ".json",
    ".model",
    ".py",
    ".safetensors",
    ".txt",
    ".yaml",
    ".yml",
}
LLAMA_BASE_URL = os.environ.get("LOCALMATHRAG_LLAMA_BASE_URL", "http://host.docker.internal:8080/v1").rstrip("/")
EMBEDDING_BASE_URL = os.environ.get("LOCALMATHRAG_EMBEDDING_BASE_URL", "http://host.docker.internal:8081/v1").rstrip("/")
RERANK_BASE_URL = os.environ.get("LOCALMATHRAG_RERANK_BASE_URL", "http://host.docker.internal:8082/v1").rstrip("/")
VISION_BASE_URL = os.environ.get("LOCALMATHRAG_VISION_BASE_URL", "http://host.docker.internal:8083/v1").rstrip("/")

RECOMMENDED_MODELS = [
    {
        "id": "qwen3-8b-q4-k-m",
        "name": "Qwen3-8B Q4_K_M",
        "file_name": "Qwen3-8B-Q4_K_M.gguf",
        "runtime_model_name": "/models/Qwen3-8B-Q4_K_M.gguf",
        "repo": "Qwen/Qwen3-8B-GGUF",
        "url": "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf",
        "model_type": ["chat"],
        "max_tokens": 8192,
        "recommended_for": "Intel Ultra 9 + RTX 4060 + 32GB RAM",
        "provider": "OpenAI-API-Compatible",
        "base_url": LLAMA_BASE_URL,
        "downloadable": True,
        "group": "chat",
    },
    {
        "id": "qwen3-4b-q4-k-m",
        "name": "Qwen3-4B Q4_K_M",
        "file_name": "Qwen3-4B-Q4_K_M.gguf",
        "runtime_model_name": "/models/Qwen3-4B-Q4_K_M.gguf",
        "repo": "Qwen/Qwen3-4B-GGUF",
        "url": "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf",
        "model_type": ["chat"],
        "max_tokens": 8192,
        "recommended_for": "Lower memory fallback",
        "provider": "OpenAI-API-Compatible",
        "base_url": LLAMA_BASE_URL,
        "downloadable": True,
        "group": "chat",
    },
    {
        "id": "qwen3-embedding-06b",
        "name": "Qwen3-Embedding-0.6B",
        "file_name": "Qwen3-Embedding-0.6B",
        "runtime_model_name": "Qwen/Qwen3-Embedding-0.6B",
        "repo": "Qwen/Qwen3-Embedding-0.6B",
        "url": "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B",
        "download_kind": "snapshot",
        "model_type": ["embedding"],
        "max_tokens": 32768,
        "recommended_for": "Default local embedding model for multilingual technical retrieval",
        "provider": "OpenAI-API-Compatible",
        "base_url": EMBEDDING_BASE_URL,
        "downloadable": True,
        "group": "embedding",
    },
    {
        "id": "bge-m3",
        "name": "BAAI/bge-m3",
        "file_name": "bge-m3",
        "runtime_model_name": "BAAI/bge-m3",
        "repo": "BAAI/bge-m3",
        "url": "https://huggingface.co/BAAI/bge-m3",
        "download_kind": "snapshot",
        "model_type": ["embedding"],
        "max_tokens": 8192,
        "recommended_for": "Strong compact embedding fallback with broad RAG support",
        "provider": "OpenAI-API-Compatible",
        "base_url": EMBEDDING_BASE_URL,
        "downloadable": True,
        "group": "embedding",
    },
    {
        "id": "qwen3-reranker-06b",
        "name": "Qwen3-Reranker-0.6B",
        "file_name": "Qwen3-Reranker-0.6B",
        "runtime_model_name": "Qwen/Qwen3-Reranker-0.6B",
        "repo": "Qwen/Qwen3-Reranker-0.6B",
        "url": "https://huggingface.co/Qwen/Qwen3-Reranker-0.6B",
        "download_kind": "snapshot",
        "model_type": ["rerank"],
        "max_tokens": 32768,
        "recommended_for": "Default reranker for evidence ordering in technical documents",
        "provider": "OpenAI-API-Compatible",
        "base_url": RERANK_BASE_URL,
        "downloadable": True,
        "group": "rerank",
    },
    {
        "id": "bge-reranker-v2-m3",
        "name": "BAAI/bge-reranker-v2-m3",
        "file_name": "bge-reranker-v2-m3",
        "runtime_model_name": "BAAI/bge-reranker-v2-m3",
        "repo": "BAAI/bge-reranker-v2-m3",
        "url": "https://huggingface.co/BAAI/bge-reranker-v2-m3",
        "download_kind": "snapshot",
        "model_type": ["rerank"],
        "max_tokens": 8192,
        "recommended_for": "Compact rerank fallback",
        "provider": "OpenAI-API-Compatible",
        "base_url": RERANK_BASE_URL,
        "downloadable": True,
        "group": "rerank",
    },
    {
        "id": "qwen3-vl-8b-instruct",
        "name": "Qwen3-VL-8B-Instruct",
        "file_name": "Qwen3-VL-8B-Instruct",
        "runtime_model_name": "Qwen/Qwen3-VL-8B-Instruct",
        "repo": "Qwen/Qwen3-VL-8B-Instruct",
        "url": "https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct",
        "download_kind": "snapshot",
        "model_type": ["vision"],
        "max_tokens": 8192,
        "recommended_for": "Vision model for diagrams, figures, screenshots, and OCR-heavy pages",
        "provider": "OpenAI-API-Compatible",
        "base_url": VISION_BASE_URL,
        "downloadable": True,
        "group": "vision",
    },
    {
        "id": "qwen3-vl-4b-instruct",
        "name": "Qwen3-VL-4B-Instruct",
        "file_name": "Qwen3-VL-4B-Instruct",
        "runtime_model_name": "Qwen/Qwen3-VL-4B-Instruct",
        "repo": "Qwen/Qwen3-VL-4B-Instruct",
        "url": "https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct",
        "download_kind": "snapshot",
        "model_type": ["vision"],
        "max_tokens": 8192,
        "recommended_for": "Smaller vision fallback for 8 GB class GPUs",
        "provider": "OpenAI-API-Compatible",
        "base_url": VISION_BASE_URL,
        "downloadable": True,
        "group": "vision",
    },
]

app = FastAPI(
    title="LocalMathRAGFlow Object Service",
    version="0.1.0",
    description="Auxiliary structured evidence API for RAGFlow secondary development.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1",
        "http://127.0.0.1:80",
        "http://127.0.0.1:8765",
        "http://localhost",
        "http://localhost:80",
        "http://localhost:8765",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Source(BaseModel):
    source_file: str | None = None
    document_id: str | None = None
    section: str | None = None
    page: int | None = None
    sheet: str | None = None
    cell_range: str | None = None
    bbox: list[float] | None = None
    chunk_id: str | None = None
    citation_id: str | None = None


class NormalizeRequest(BaseModel):
    type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    source: Source = Field(default_factory=Source)


class DownloadModelRequest(BaseModel):
    id: str | None = None
    url: str | None = None
    file_name: str | None = None


def _schema_files() -> list[Path]:
    if not SCHEMA_DIR.exists():
        return []
    return sorted(SCHEMA_DIR.glob("*.schema.json"))


def _load_schema(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Invalid schema {path.name}: {exc}") from exc


def _stable_id(object_type: str, payload: dict[str, Any], source: Source) -> str:
    seed = {
        "type": object_type,
        "payload": payload,
        "source": source.model_dump(exclude_none=True),
    }
    digest = hashlib.sha1(json.dumps(seed, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return f"{object_type}_{digest[:12]}"


def _snapshot_model_dirs() -> list[Path]:
    if not MODEL_DIR.exists():
        return []
    return sorted(
        path
        for path in MODEL_DIR.iterdir()
        if path.is_dir() and (path / MODEL_METADATA_NAME).exists()
    )


def _is_inside_snapshot(path: Path, snapshot_dirs: list[Path]) -> bool:
    return any(snapshot_dir in path.parents for snapshot_dir in snapshot_dirs)


def _model_files(snapshot_dirs: list[Path] | None = None) -> list[Path]:
    if not MODEL_DIR.exists():
        return []
    snapshot_dirs = snapshot_dirs or _snapshot_model_dirs()
    return sorted(
        path
        for path in MODEL_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in MODEL_EXTENSIONS
        and not _is_inside_snapshot(path, snapshot_dirs)
    )


def _model_entries() -> list[Path]:
    if not MODEL_DIR.exists():
        return []
    snapshot_dirs = _snapshot_model_dirs()
    return sorted([*_model_files(snapshot_dirs), *snapshot_dirs])


def _load_model_metadata(path: Path) -> dict[str, Any]:
    metadata_path = path / MODEL_METADATA_NAME if path.is_dir() else None
    if not metadata_path or not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _infer_model_type(path: Path) -> list[str]:
    name = path.name.lower()
    if "embedding" in name or "embed" in name or "bge-m3" in name:
        return ["embedding"]
    if "rerank" in name or "reranker" in name:
        return ["rerank"]
    if "vl" in name or "vision" in name or "omni" in name:
        return ["vision"]
    if "ocr" in name:
        return ["ocr"]
    return ["chat"]


def _base_url_for_model_type(model_type: list[str]) -> str:
    first_type = model_type[0] if model_type else "chat"
    if first_type == "embedding":
        return EMBEDDING_BASE_URL
    if first_type == "rerank":
        return RERANK_BASE_URL
    if first_type in {"vision", "ocr"}:
        return VISION_BASE_URL
    return LLAMA_BASE_URL


def _llama_endpoint_status(timeout: float = 1.5) -> dict[str, Any]:
    url = f"{LLAMA_BASE_URL}/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            ok = 200 <= response.status < 300
            return {
                "base_url": LLAMA_BASE_URL,
                "models_url": url,
                "endpoint_ok": ok,
                "status_code": response.status,
            }
    except Exception as exc:
        return {
            "base_url": LLAMA_BASE_URL,
            "models_url": url,
            "endpoint_ok": False,
            "error": str(exc),
        }


def _model_payload(path: Path) -> dict[str, Any]:
    metadata = _load_model_metadata(path)
    model_type = metadata.get("model_type") or _infer_model_type(path)
    runtime_model_name = metadata.get("runtime_model_name") or path.stem
    if path.is_file() and path.suffix.lower() == ".gguf" and model_type[0] == "chat":
        runtime_model_name = f"/models/{path.name}"
    size_bytes = _directory_size(path) if path.is_dir() else path.stat().st_size
    return {
        "name": metadata.get("name") or path.stem,
        "runtime_model_name": runtime_model_name,
        "file_name": metadata.get("file_name") or path.name,
        "path": str(path),
        "relative_path": str(path.relative_to(MODEL_DIR)),
        "size_bytes": size_bytes,
        "size_gb": round(size_bytes / 1024 / 1024 / 1024, 2),
        "extension": "snapshot" if path.is_dir() else path.suffix.lower(),
        "provider": metadata.get("provider") or "OpenAI-API-Compatible",
        "base_url": metadata.get("base_url") or _base_url_for_model_type(model_type),
        "model_type": model_type,
        "max_tokens": metadata.get("max_tokens") or 8192,
    }


def _snapshot_metadata(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": model.get("id"),
        "name": model.get("name"),
        "file_name": model.get("file_name"),
        "runtime_model_name": model.get("runtime_model_name") or model.get("repo") or model.get("name"),
        "repo": model.get("repo"),
        "provider": model.get("provider") or "OpenAI-API-Compatible",
        "base_url": model.get("base_url"),
        "model_type": model.get("model_type") or ["chat"],
        "max_tokens": model.get("max_tokens") or 8192,
        "download_kind": "snapshot",
    }


def _hf_repo_tree(repo: str, revision: str = "main") -> list[dict[str, Any]]:
    repo_path = urllib.parse.quote(repo, safe="/")
    url = f"https://huggingface.co/api/models/{repo_path}/tree/{revision}?recursive=1"
    with urllib.request.urlopen(url, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))
    if not isinstance(data, list):
        raise HTTPException(status_code=502, detail="Hugging Face repository tree response is invalid")
    return [item for item in data if isinstance(item, dict)]


def _allowed_snapshot_file(path: str, allow_patterns: list[str] | None = None) -> bool:
    normalized = path.replace("\\", "/")
    if allow_patterns:
        return any(fnmatch.fnmatch(normalized, pattern) for pattern in allow_patterns)
    suffix = Path(normalized).suffix.lower()
    if suffix not in SNAPSHOT_FILE_EXTENSIONS:
        return False
    lower = normalized.lower()
    if lower.startswith((".git", "onnx/", "openvino/", "examples/")):
        return False
    if lower.endswith((".md", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".pdf")):
        return False
    return True


def _download_hf_file(repo: str, path: str, target: Path, revision: str = "main") -> None:
    repo_path = urllib.parse.quote(repo, safe="/")
    file_path = urllib.parse.quote(path, safe="/")
    url = f"https://huggingface.co/{repo_path}/resolve/{revision}/{file_path}"
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    urllib.request.urlretrieve(url, partial)
    partial.replace(target)


def _download_snapshot(model: dict[str, Any]) -> dict[str, Any]:
    repo = model.get("repo")
    file_name = model.get("file_name")
    if not repo or not file_name:
        raise HTTPException(status_code=400, detail="Snapshot model requires repo and file_name")
    target_dir = MODEL_DIR / Path(file_name).name
    metadata_path = target_dir / MODEL_METADATA_NAME
    if metadata_path.exists():
        return {"status": "exists", "model": _model_payload(target_dir)}

    target_dir.mkdir(parents=True, exist_ok=True)
    allow_patterns = model.get("allow_patterns")
    tree = _hf_repo_tree(repo)
    files = [
        item["path"]
        for item in tree
        if item.get("type") == "file"
        and isinstance(item.get("path"), str)
        and _allowed_snapshot_file(item["path"], allow_patterns)
    ]
    if not files:
        raise HTTPException(status_code=502, detail="No downloadable model files were found in the repository")

    try:
        for file_path in files:
            _download_hf_file(repo, file_path, target_dir / file_path)
        metadata_path.write_text(
            json.dumps(_snapshot_metadata(model), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        for partial in target_dir.rglob("*.partial"):
            partial.unlink(missing_ok=True)
        raise HTTPException(status_code=502, detail=f"Snapshot download failed: {exc}") from exc

    return {"status": "downloaded", "model": _model_payload(target_dir)}


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "schema_dir": str(SCHEMA_DIR),
        "schema_count": len(_schema_files()),
        "model_dir": str(MODEL_DIR),
        "model_count": len(_model_entries()),
    }


@app.get("/v1/models/local")
def list_local_models() -> dict[str, Any]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    endpoint_status = _llama_endpoint_status()
    return {
        "model_dir": str(MODEL_DIR),
        "endpoint_status": endpoint_status,
        "models": [_model_payload(path) for path in _model_entries()],
    }


@app.get("/v1/models/recommended")
def list_recommended_models() -> dict[str, Any]:
    local_names = {path.name for path in _model_entries()}
    endpoint_status = _llama_endpoint_status()
    models = []
    for model in RECOMMENDED_MODELS:
        item = dict(model)
        item["downloaded"] = item["file_name"] in local_names
        item["target_path"] = str(MODEL_DIR / item["file_name"])
        item["endpoint_status"] = endpoint_status
        item.setdefault("downloadable", False)
        models.append(item)
    return {"models": models}


@app.get("/v1/models/status")
def model_status() -> dict[str, Any]:
    return {
        "model_dir": str(MODEL_DIR),
        "model_count": len(_model_entries()),
        "endpoint_status": _llama_endpoint_status(timeout=3),
    }


@app.post("/v1/models/download")
def download_model(request: DownloadModelRequest) -> dict[str, Any]:
    selected = next(
        (model for model in RECOMMENDED_MODELS if model["id"] == request.id),
        None,
    )
    if selected and selected.get("download_kind") == "snapshot":
        if not selected.get("downloadable", False):
            raise HTTPException(status_code=400, detail="This model cannot be downloaded automatically")
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        return _download_snapshot(selected)

    url = request.url or (selected or {}).get("url")
    file_name = request.file_name or (selected or {}).get("file_name")
    if not url or not file_name:
        raise HTTPException(status_code=400, detail="Model url and file_name are required")
    if selected and not selected.get("downloadable", False):
        raise HTTPException(status_code=400, detail="This model is a repository recommendation. Open the model link and install it with the matching local runtime.")
    if not url.startswith("https://huggingface.co/"):
        raise HTTPException(status_code=400, detail="Only Hugging Face model downloads are supported")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    target = MODEL_DIR / Path(file_name).name
    if target.exists():
        return {"status": "exists", "model": _model_payload(target)}

    partial = target.with_suffix(target.suffix + ".partial")
    try:
        urllib.request.urlretrieve(url, partial)
        partial.replace(target)
    except Exception as exc:
        if partial.exists():
            partial.unlink()
        raise HTTPException(status_code=502, detail=f"Download failed: {exc}") from exc
    return {"status": "downloaded", "model": _model_payload(target)}


@app.get("/v1/schemas")
def list_schemas() -> dict[str, Any]:
    schemas = []
    for path in _schema_files():
        data = _load_schema(path)
        schemas.append(
            {
                "name": path.name,
                "id": data.get("$id"),
                "title": data.get("title"),
            }
        )
    return {"schemas": schemas}


@app.get("/v1/schemas/{name}")
def get_schema(name: str) -> dict[str, Any]:
    safe_name = Path(name).name
    path = SCHEMA_DIR / safe_name
    if not path.exists() and not safe_name.endswith(".schema.json"):
        path = SCHEMA_DIR / f"{safe_name}.schema.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Schema not found")
    return _load_schema(path)


@app.post("/v1/objects/normalize")
def normalize_object(request: NormalizeRequest) -> dict[str, Any]:
    payload = dict(request.payload)
    payload.setdefault("type", request.type)
    payload.setdefault("id", _stable_id(request.type, request.payload, request.source))
    if "source" not in payload:
        payload["source"] = request.source.model_dump(exclude_none=True)
    return {
        "object": payload,
        "display": {
            "default_state": "collapsed",
            "primary_surface": "citation_panel",
        },
    }


@app.post("/v1/search/objects")
def search_objects() -> dict[str, Any]:
    raise HTTPException(status_code=501, detail="Object search is reserved for phase 2.")


@app.post("/v1/export/objects")
def export_objects() -> dict[str, Any]:
    raise HTTPException(status_code=501, detail="Object export is reserved for phase 2.")

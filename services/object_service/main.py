from __future__ import annotations

import hashlib
import json
import os
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
LLAMA_BASE_URL = os.environ.get("LOCALMATHRAG_LLAMA_BASE_URL", "http://host.docker.internal:8080/v1").rstrip("/")
EMBEDDING_BASE_URL = os.environ.get("LOCALMATHRAG_EMBEDDING_BASE_URL", "http://host.docker.internal:8081/v1").rstrip("/")
RERANK_BASE_URL = os.environ.get("LOCALMATHRAG_RERANK_BASE_URL", "http://host.docker.internal:8082/v1").rstrip("/")
VISION_BASE_URL = os.environ.get("LOCALMATHRAG_VISION_BASE_URL", "http://host.docker.internal:8083/v1").rstrip("/")

RECOMMENDED_MODELS = [
    {
        "id": "qwen3-8b-q4-k-m",
        "name": "Qwen3-8B Q4_K_M",
        "file_name": "Qwen3-8B-Q4_K_M.gguf",
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
        "repo": "Qwen/Qwen3-Embedding-0.6B",
        "url": "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B",
        "model_type": ["embedding"],
        "max_tokens": 32768,
        "recommended_for": "Default local embedding model for multilingual technical retrieval",
        "provider": "OpenAI-API-Compatible",
        "base_url": EMBEDDING_BASE_URL,
        "downloadable": False,
        "group": "embedding",
    },
    {
        "id": "bge-m3",
        "name": "BAAI/bge-m3",
        "file_name": "bge-m3",
        "repo": "BAAI/bge-m3",
        "url": "https://huggingface.co/BAAI/bge-m3",
        "model_type": ["embedding"],
        "max_tokens": 8192,
        "recommended_for": "Strong compact embedding fallback with broad RAG support",
        "provider": "OpenAI-API-Compatible",
        "base_url": EMBEDDING_BASE_URL,
        "downloadable": False,
        "group": "embedding",
    },
    {
        "id": "qwen3-reranker-06b",
        "name": "Qwen3-Reranker-0.6B",
        "file_name": "Qwen3-Reranker-0.6B",
        "repo": "Qwen/Qwen3-Reranker-0.6B",
        "url": "https://huggingface.co/Qwen/Qwen3-Reranker-0.6B",
        "model_type": ["rerank"],
        "max_tokens": 32768,
        "recommended_for": "Default reranker for evidence ordering in technical documents",
        "provider": "OpenAI-API-Compatible",
        "base_url": RERANK_BASE_URL,
        "downloadable": False,
        "group": "rerank",
    },
    {
        "id": "bge-reranker-v2-m3",
        "name": "BAAI/bge-reranker-v2-m3",
        "file_name": "bge-reranker-v2-m3",
        "repo": "BAAI/bge-reranker-v2-m3",
        "url": "https://huggingface.co/BAAI/bge-reranker-v2-m3",
        "model_type": ["rerank"],
        "max_tokens": 8192,
        "recommended_for": "Compact rerank fallback",
        "provider": "OpenAI-API-Compatible",
        "base_url": RERANK_BASE_URL,
        "downloadable": False,
        "group": "rerank",
    },
    {
        "id": "qwen3-vl-8b-instruct",
        "name": "Qwen3-VL-8B-Instruct",
        "file_name": "Qwen3-VL-8B-Instruct",
        "repo": "Qwen/Qwen3-VL-8B-Instruct",
        "url": "https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct",
        "model_type": ["vision"],
        "max_tokens": 8192,
        "recommended_for": "Vision model for diagrams, figures, screenshots, and OCR-heavy pages",
        "provider": "OpenAI-API-Compatible",
        "base_url": VISION_BASE_URL,
        "downloadable": False,
        "group": "vision",
    },
    {
        "id": "qwen3-vl-4b-instruct",
        "name": "Qwen3-VL-4B-Instruct",
        "file_name": "Qwen3-VL-4B-Instruct",
        "repo": "Qwen/Qwen3-VL-4B-Instruct",
        "url": "https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct",
        "model_type": ["vision"],
        "max_tokens": 8192,
        "recommended_for": "Smaller vision fallback for 8 GB class GPUs",
        "provider": "OpenAI-API-Compatible",
        "base_url": VISION_BASE_URL,
        "downloadable": False,
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


def _model_files() -> list[Path]:
    if not MODEL_DIR.exists():
        return []
    return sorted(
        path
        for path in MODEL_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in MODEL_EXTENSIONS
    )


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
    stat = path.stat()
    model_type = _infer_model_type(path)
    return {
        "name": path.stem,
        "file_name": path.name,
        "path": str(path),
        "relative_path": str(path.relative_to(MODEL_DIR)),
        "size_bytes": stat.st_size,
        "size_gb": round(stat.st_size / 1024 / 1024 / 1024, 2),
        "extension": path.suffix.lower(),
        "provider": "OpenAI-API-Compatible",
        "base_url": _base_url_for_model_type(model_type),
        "model_type": model_type,
        "max_tokens": 8192,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "schema_dir": str(SCHEMA_DIR),
        "schema_count": len(_schema_files()),
        "model_dir": str(MODEL_DIR),
        "model_count": len(_model_files()),
    }


@app.get("/v1/models/local")
def list_local_models() -> dict[str, Any]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    endpoint_status = _llama_endpoint_status()
    return {
        "model_dir": str(MODEL_DIR),
        "endpoint_status": endpoint_status,
        "models": [_model_payload(path) for path in _model_files()],
    }


@app.get("/v1/models/recommended")
def list_recommended_models() -> dict[str, Any]:
    local_names = {path.name for path in _model_files()}
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
        "model_count": len(_model_files()),
        "endpoint_status": _llama_endpoint_status(timeout=3),
    }


@app.post("/v1/models/download")
def download_model(request: DownloadModelRequest) -> dict[str, Any]:
    selected = next(
        (model for model in RECOMMENDED_MODELS if model["id"] == request.id),
        None,
    )
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

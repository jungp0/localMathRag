from __future__ import annotations

import hashlib
import fnmatch
import http.client
import json
import math
import os
import re
import shutil
import socket
import threading
import time
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
DATASET_DIR = Path(os.environ.get("LOCALMATHRAG_DATASET_DIR", DATA_DIR / "dataset"))
DATASET_STATE_FILE = Path(os.environ.get("LOCALMATHRAG_DATASET_STATE_FILE", DATA_DIR / "cache" / "dataset-state.json"))
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
HF_ENDPOINT = os.environ.get("LOCALMATHRAG_HF_ENDPOINT", "https://huggingface.co").rstrip("/")
LLAMA_BASE_URL = os.environ.get("LOCALMATHRAG_LLAMA_BASE_URL", "http://host.docker.internal:8080/v1").rstrip("/")
EMBEDDING_BASE_URL = os.environ.get("LOCALMATHRAG_EMBEDDING_BASE_URL", "http://host.docker.internal:8081/v1").rstrip("/")
RERANK_BASE_URL = os.environ.get("LOCALMATHRAG_RERANK_BASE_URL", "http://localmathrag-object-service:8088/v1").rstrip("/")
EMBEDDING_RUNTIME_BASE_URL = os.environ.get("LOCALMATHRAG_EMBEDDING_RUNTIME_BASE_URL", "http://host.docker.internal:8081/v1").rstrip("/")
RERANK_RUNTIME_BASE_URL = os.environ.get("LOCALMATHRAG_RERANK_RUNTIME_BASE_URL", "http://host.docker.internal:8082/v1").rstrip("/")
VISION_BASE_URL = os.environ.get("LOCALMATHRAG_VISION_BASE_URL", "http://host.docker.internal:8083/v1").rstrip("/")
ASR_BASE_URL = os.environ.get("LOCALMATHRAG_ASR_BASE_URL", "http://host.docker.internal:8084/v1").rstrip("/")
TTS_BASE_URL = os.environ.get("LOCALMATHRAG_TTS_BASE_URL", "http://host.docker.internal:8085/v1").rstrip("/")
LOCAL_EMBEDDING_MODEL = os.environ.get("LOCALMATHRAG_FALLBACK_EMBEDDING_MODEL", "localmathrag-lexical-embedding")
LOCAL_EMBEDDING_DIM = int(os.environ.get("LOCALMATHRAG_FALLBACK_EMBEDDING_DIM", "1024"))
LOCAL_RERANK_MODEL = os.environ.get("LOCALMATHRAG_FALLBACK_RERANK_MODEL", "localmathrag-lexical-rerank")
DOCKER_SOCKET = os.environ.get("LOCALMATHRAG_DOCKER_SOCKET", "/var/run/docker.sock")
EMBEDDING_CONTAINER_NAME = os.environ.get("LOCALMATHRAG_EMBEDDING_CONTAINER", "docker-localmathrag-embedding-1")
RERANK_CONTAINER_NAME = os.environ.get("LOCALMATHRAG_RERANK_CONTAINER", "docker-localmathrag-rerank-1")
VISION_CONTAINER_NAME = os.environ.get("LOCALMATHRAG_VISION_CONTAINER", "docker-localmathrag-vlm-1")
ASR_CONTAINER_NAME = os.environ.get("LOCALMATHRAG_ASR_CONTAINER", "docker-localmathrag-asr-1")
TTS_CONTAINER_NAME = os.environ.get("LOCALMATHRAG_TTS_CONTAINER", "docker-localmathrag-tts-1")
RUNTIME_LAZY_ENABLED = os.environ.get("LOCALMATHRAG_RUNTIME_LAZY", "1").lower() not in {"0", "false", "no", "off"}
OPTIONAL_RUNTIME_MAX_ACTIVE = max(1, int(os.environ.get("LOCALMATHRAG_OPTIONAL_RUNTIME_MAX_ACTIVE", "1")))
RUNTIME_START_LOCK = threading.Lock()
DATASET_SCAN_LOCK = threading.Lock()

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
    {
        "id": "whisper-large-v3-turbo",
        "name": "Whisper Large v3 Turbo",
        "file_name": "whisper-large-v3-turbo",
        "runtime_model_name": "openai/whisper-large-v3-turbo",
        "repo": "openai/whisper-large-v3-turbo",
        "url": "https://huggingface.co/openai/whisper-large-v3-turbo",
        "download_kind": "snapshot",
        "model_type": ["asr"],
        "max_tokens": 0,
        "recommended_for": "Future local ASR endpoint for meeting audio and spoken engineering notes",
        "provider": "OpenAI-API-Compatible",
        "base_url": ASR_BASE_URL,
        "downloadable": True,
        "group": "asr",
    },
    {
        "id": "cosyvoice2-05b",
        "name": "CosyVoice2-0.5B",
        "file_name": "CosyVoice2-0.5B",
        "runtime_model_name": "FunAudioLLM/CosyVoice2-0.5B",
        "repo": "FunAudioLLM/CosyVoice2-0.5B",
        "url": "https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B",
        "download_kind": "snapshot",
        "model_type": ["tts"],
        "max_tokens": 0,
        "recommended_for": "Future local TTS endpoint for spoken answers and agent narration",
        "provider": "OpenAI-API-Compatible",
        "base_url": TTS_BASE_URL,
        "downloadable": True,
        "group": "tts",
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


class EmbeddingRequest(BaseModel):
    input: str | list[Any]
    model: str | None = None


class RerankRequest(BaseModel):
    query: str
    documents: list[Any] = Field(default_factory=list)
    model: str | None = None
    top_n: int | None = None
    return_documents: bool | str | None = None


class RuntimeEnsureRequest(BaseModel):
    kind: str = "vision"
    timeout_seconds: float | None = None
    start: bool = True


class RuntimeStopRequest(BaseModel):
    kind: str


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


def _quick_model_entry_count() -> int:
    if not MODEL_DIR.exists():
        return 0
    count = 0
    for path in MODEL_DIR.iterdir():
        if path.is_dir():
            count += 1
        elif path.suffix.lower() in MODEL_EXTENSIONS:
            count += 1
    return count


def _load_dataset_state() -> dict[str, Any]:
    if not DATASET_STATE_FILE.exists():
        return {}
    try:
        data = json.loads(DATASET_STATE_FILE.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_dataset_state(entries: list[dict[str, Any]]) -> None:
    DATASET_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    stable_entries = []
    for entry in entries:
        if entry.get("status") == "deleted":
            continue
        stable = {
            key: entry[key]
            for key in ("path", "name", "kind", "signature", "size", "modified_at")
            if key in entry
        }
        stable_entries.append(stable)
    DATASET_STATE_FILE.write_text(
        json.dumps(
            {
                "version": 1,
                "dataset_dir": str(DATASET_DIR),
                "scanned_at": time.time(),
                "entries": stable_entries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _entry_parent(path: str) -> str:
    if not path or "/" not in path:
        return ""
    return path.rsplit("/", 1)[0]


def _dataset_relative_path(path: Path) -> str:
    rel = path.relative_to(DATASET_DIR).as_posix()
    return "" if rel == "." else rel


def _path_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    signature = {
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
    }
    if path.is_file():
        signature["size"] = int(stat.st_size)
    return signature


def _dataset_entry(path: Path, kind: str, status: str, signature: dict[str, int], skipped: bool = False) -> dict[str, Any]:
    rel = _dataset_relative_path(path)
    entry: dict[str, Any] = {
        "path": rel,
        "name": path.name if rel else DATASET_DIR.name,
        "kind": kind,
        "status": status,
        "signature": signature,
        "modified_at": signature["mtime_ns"] / 1_000_000_000,
    }
    if kind == "file":
        entry["size"] = signature.get("size", 0)
    if skipped:
        entry["skipped"] = True
    return entry


def _cached_dataset_subtree(previous_entries: list[dict[str, Any]], rel: str, status: str) -> list[dict[str, Any]]:
    prefix = f"{rel}/" if rel else ""
    cached: list[dict[str, Any]] = []
    for entry in previous_entries:
        entry_path = str(entry.get("path", ""))
        if entry_path == rel or not rel or entry_path.startswith(prefix):
            copied = dict(entry)
            copied["status"] = status
            if copied.get("kind") == "directory":
                copied["skipped"] = status == "unchanged"
            cached.append(copied)
    return cached


def _deleted_dataset_subtree(previous_entries: list[dict[str, Any]], rel: str) -> list[dict[str, Any]]:
    deleted = _cached_dataset_subtree(previous_entries, rel, "deleted")
    for entry in deleted:
        entry.pop("skipped", None)
    return deleted


def _scan_dataset_directory(
    path: Path,
    previous_entries: list[dict[str, Any]],
    previous_by_path: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rel = _dataset_relative_path(path)
    signature = _path_signature(path)
    previous = previous_by_path.get(rel)
    if previous and previous.get("kind") == "directory" and previous.get("signature") == signature:
        cached = _cached_dataset_subtree(previous_entries, rel, "unchanged")
        if cached:
            return cached

    status = "new" if previous is None else "changed"
    entries = [_dataset_entry(path, "directory", status, signature)]
    current_children: set[str] = set()
    try:
        children = sorted(path.iterdir(), key=lambda item: item.name.lower())
    except OSError as exc:
        entries[0]["error"] = str(exc)
        return entries

    for child in children:
        try:
            child_rel = _dataset_relative_path(child)
            current_children.add(child_rel)
            if child.is_dir():
                entries.extend(_scan_dataset_directory(child, previous_entries, previous_by_path))
            elif child.is_file():
                file_signature = _path_signature(child)
                previous_file = previous_by_path.get(child_rel)
                file_status = "new"
                if previous_file and previous_file.get("kind") == "file":
                    file_status = "unchanged" if previous_file.get("signature") == file_signature else "changed"
                entries.append(_dataset_entry(child, "file", file_status, file_signature))
        except OSError as exc:
            entries.append(
                {
                    "path": _dataset_relative_path(child),
                    "name": child.name,
                    "kind": "unknown",
                    "status": "error",
                    "error": str(exc),
                }
            )

    for previous_entry in previous_entries:
        previous_path = str(previous_entry.get("path", ""))
        if previous_path and _entry_parent(previous_path) == rel and previous_path not in current_children:
            if previous_entry.get("kind") == "directory":
                entries.extend(_deleted_dataset_subtree(previous_entries, previous_path))
            else:
                deleted = dict(previous_entry)
                deleted["status"] = "deleted"
                entries.append(deleted)

    return entries


def _dataset_summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    active = [entry for entry in entries if entry.get("status") != "deleted"]
    changed_entries = [entry for entry in entries if entry.get("status") in {"new", "changed", "deleted"}]
    return {
        "file_count": sum(1 for entry in active if entry.get("kind") == "file"),
        "directory_count": sum(1 for entry in active if entry.get("kind") == "directory"),
        "new_count": sum(1 for entry in entries if entry.get("status") == "new"),
        "changed_count": sum(1 for entry in entries if entry.get("status") == "changed"),
        "deleted_count": sum(1 for entry in entries if entry.get("status") == "deleted"),
        "skipped_directory_count": sum(1 for entry in entries if entry.get("kind") == "directory" and entry.get("skipped")),
        "has_changes": bool(changed_entries),
    }


def _scan_dataset(include_entries: bool = False) -> dict[str, Any]:
    with DATASET_SCAN_LOCK:
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        previous_state = _load_dataset_state()
        previous_entries = [
            entry
            for entry in previous_state.get("entries", [])
            if isinstance(entry, dict) and isinstance(entry.get("path", ""), str)
        ]
        previous_by_path = {str(entry.get("path", "")): entry for entry in previous_entries}
        entries = _scan_dataset_directory(DATASET_DIR, previous_entries, previous_by_path)
        _write_dataset_state(entries)
        result: dict[str, Any] = {
            "dataset_dir": str(DATASET_DIR),
            "state_file": str(DATASET_STATE_FILE),
            "summary": _dataset_summary(entries),
        }
        if include_entries:
            result["entries"] = entries
        return result


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
    if "whisper" in name or "asr" in name or "speech-to-text" in name:
        return ["asr"]
    if "tts" in name or "cosyvoice" in name or "text-to-speech" in name:
        return ["tts"]
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
    if first_type == "asr":
        return ASR_BASE_URL
    if first_type == "tts":
        return TTS_BASE_URL
    return LLAMA_BASE_URL


def _base_url_for_payload(metadata: dict[str, Any], model_type: list[str]) -> str:
    first_type = model_type[0] if model_type else "chat"
    provider = metadata.get("provider") or "OpenAI-API-Compatible"
    if provider == "OpenAI-API-Compatible" and first_type in {
        "chat",
        "embedding",
        "rerank",
        "vision",
        "ocr",
        "asr",
        "tts",
    }:
        return _base_url_for_model_type(model_type)
    return metadata.get("base_url") or _base_url_for_model_type(model_type)


def _endpoint_key_for_model_type(model_type: str) -> str:
    if model_type == "embedding":
        return "embedding"
    if model_type == "rerank":
        return "rerank"
    if model_type in {"vision", "ocr"}:
        return "vision"
    if model_type == "asr":
        return "asr"
    if model_type == "tts":
        return "tts"
    return "chat"


def _base_url_for_endpoint_key(endpoint_key: str) -> str:
    if endpoint_key == "embedding":
        return EMBEDDING_BASE_URL
    if endpoint_key == "rerank":
        return RERANK_BASE_URL
    if endpoint_key == "vision":
        return VISION_BASE_URL
    if endpoint_key == "asr":
        return ASR_BASE_URL
    if endpoint_key == "tts":
        return TTS_BASE_URL
    return LLAMA_BASE_URL


def _runtime_base_url_for_endpoint_key(endpoint_key: str) -> str:
    if endpoint_key == "embedding":
        return EMBEDDING_RUNTIME_BASE_URL
    if endpoint_key == "rerank":
        return RERANK_RUNTIME_BASE_URL
    return _base_url_for_endpoint_key(endpoint_key)


def _endpoint_status(base_url: str, timeout: float = 1.5) -> dict[str, Any]:
    url = f"{base_url}/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            ok = 200 <= response.status < 300
            return {
                "base_url": base_url,
                "models_url": url,
                "endpoint_ok": ok,
                "status_code": response.status,
            }
    except Exception as exc:
        models_error = str(exc)
    health_url = f"{base_url.removesuffix('/v1')}/health"
    try:
        with urllib.request.urlopen(health_url, timeout=timeout) as response:
            ok = 200 <= response.status < 300
            return {
                "base_url": base_url,
                "models_url": url,
                "health_url": health_url,
                "endpoint_ok": ok,
                "status_code": response.status,
            }
    except Exception as exc:
        return {
            "base_url": base_url,
            "models_url": url,
            "health_url": health_url,
            "endpoint_ok": False,
            "error": str(exc),
            "models_error": models_error,
        }


def _llama_endpoint_status(timeout: float = 1.5) -> dict[str, Any]:
    return _endpoint_status(LLAMA_BASE_URL, timeout)


def _model_endpoint_statuses(timeout: float = 1.5) -> dict[str, Any]:
    return {
        "chat": _endpoint_status(LLAMA_BASE_URL, timeout),
        "embedding": _optional_runtime_endpoint_status("embedding", timeout),
        "rerank": _optional_runtime_endpoint_status("rerank", timeout),
        "vision": _optional_runtime_endpoint_status("vision", timeout),
        "asr": _optional_runtime_endpoint_status("asr", timeout),
        "tts": _optional_runtime_endpoint_status("tts", timeout),
    }


class _DockerSocketHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: str):
        super().__init__("localhost")
        self.socket_path = socket_path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self.socket_path)


def _docker_api(method: str, path: str, body: bytes | None = None) -> tuple[int, Any]:
    if not Path(DOCKER_SOCKET).exists():
        raise RuntimeError(f"Docker socket is not mounted: {DOCKER_SOCKET}")
    conn = _DockerSocketHTTPConnection(DOCKER_SOCKET)
    headers = {"Content-Type": "application/json"} if body is not None else {}
    try:
        conn.request(method, path, body=body, headers=headers)
        response = conn.getresponse()
        payload = response.read()
        if payload:
            decoded = payload.decode("utf-8", errors="replace")
            try:
                data = json.loads(decoded)
            except json.JSONDecodeError:
                data = decoded
        else:
            data = None
        return response.status, data
    finally:
        conn.close()


def _docker_find_container(name: str, service_name: str) -> str | None:
    status, data = _docker_api("GET", f"/containers/{urllib.parse.quote(name, safe='')}/json")
    if status == 200 and isinstance(data, dict):
        return data.get("Id")
    filters = urllib.parse.quote(json.dumps({"label": [f"com.docker.compose.service={service_name}"]}))
    status, data = _docker_api("GET", f"/containers/json?all=1&filters={filters}")
    if status != 200 or not isinstance(data, list) or not data:
        return None
    container = data[0]
    return container.get("Id") if isinstance(container, dict) else None


def _docker_container_state(name: str, service_name: str) -> dict[str, Any]:
    container_id = _docker_find_container(name, service_name)
    if not container_id:
        return {
            "container_found": False,
            "container_running": False,
            "container_name": name,
            "compose_service": service_name,
        }
    status, data = _docker_api("GET", f"/containers/{urllib.parse.quote(container_id, safe='')}/json")
    if status != 200 or not isinstance(data, dict):
        return {
            "container_found": True,
            "container_running": False,
            "container_id": container_id[:12],
            "container_name": name,
            "compose_service": service_name,
            "container_error": f"Docker inspect returned status {status}",
        }
    state = data.get("State") if isinstance(data.get("State"), dict) else {}
    return {
        "container_found": True,
        "container_running": bool(state.get("Running")),
        "container_status": state.get("Status"),
        "container_id": container_id[:12],
        "container_name": name,
        "compose_service": service_name,
    }


def _docker_start_container(container_id: str) -> dict[str, Any]:
    status, data = _docker_api("POST", f"/containers/{container_id}/start")
    if status in {204, 304}:
        return {"started": status == 204, "status_code": status}
    raise RuntimeError(f"Docker start failed with status {status}: {data}")


def _docker_stop_container(container_id: str, timeout_seconds: int = 10) -> dict[str, Any]:
    status, data = _docker_api("POST", f"/containers/{container_id}/stop?t={timeout_seconds}")
    if status in {204, 304}:
        return {"stopped": status == 204, "status_code": status}
    raise RuntimeError(f"Docker stop failed with status {status}: {data}")


def _docker_container_logs(container_id: str, tail: int = 80) -> str:
    status, data = _docker_api("GET", f"/containers/{urllib.parse.quote(container_id, safe='')}/logs?stdout=1&stderr=1&tail={tail}&timestamps=1")
    if status != 200:
        return ""
    return data if isinstance(data, str) else ""


def _has_local_model_type(model_type: str) -> bool:
    return any((_model_payload(path).get("model_type") or ["chat"])[0] == model_type for path in _model_entries())


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _available_memory_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except Exception:
        return None
    return None


def _runtime_resource_status(kind: str) -> dict[str, Any]:
    normalized = kind.upper()
    default_memory_gb = 8.0 if kind in {"vision", "vlm"} else 4.0
    min_memory_gb = _env_float(f"LOCALMATHRAG_{normalized}_MIN_AVAILABLE_MEMORY_GB", _env_float("LOCALMATHRAG_RUNTIME_MIN_AVAILABLE_MEMORY_GB", default_memory_gb))
    min_disk_gb = _env_float(f"LOCALMATHRAG_{normalized}_MIN_FREE_DISK_GB", _env_float("LOCALMATHRAG_RUNTIME_MIN_FREE_DISK_GB", 20.0))
    max_load_per_cpu = _env_float("LOCALMATHRAG_RUNTIME_MAX_LOAD_PER_CPU", 3.0)

    available_memory = _available_memory_bytes()
    disk = shutil.disk_usage(DATA_DIR)
    cpu_count = os.cpu_count() or 1
    try:
        load_1m = os.getloadavg()[0]
    except OSError:
        load_1m = 0.0

    min_memory_bytes = int(min_memory_gb * 1024 * 1024 * 1024)
    min_disk_bytes = int(min_disk_gb * 1024 * 1024 * 1024)
    load_limit = max_load_per_cpu * cpu_count
    reasons: list[str] = []
    if available_memory is not None and available_memory < min_memory_bytes:
        reasons.append(f"available memory {available_memory / 1024 / 1024 / 1024:.1f}GB is below {min_memory_gb:.1f}GB")
    if disk.free < min_disk_bytes:
        reasons.append(f"free disk {disk.free / 1024 / 1024 / 1024:.1f}GB is below {min_disk_gb:.1f}GB")
    if load_1m > load_limit:
        reasons.append(f"system load {load_1m:.1f} exceeds limit {load_limit:.1f}")
    return {
        "ok": not reasons,
        "reasons": reasons,
        "available_memory_gb": None if available_memory is None else round(available_memory / 1024 / 1024 / 1024, 2),
        "min_available_memory_gb": min_memory_gb,
        "free_disk_gb": round(disk.free / 1024 / 1024 / 1024, 2),
        "min_free_disk_gb": min_disk_gb,
        "load_1m": round(load_1m, 2),
        "load_limit": round(load_limit, 2),
    }


def _runtime_ready_timeout_seconds(kind: str) -> float:
    normalized = kind.upper()
    return _env_float(f"LOCALMATHRAG_{normalized}_READY_TIMEOUT_SECONDS", _env_float("LOCALMATHRAG_RUNTIME_READY_TIMEOUT_SECONDS", 240.0))


def _runtime_startup_stall_timeout_seconds(kind: str) -> float:
    normalized = kind.upper()
    return _env_float(f"LOCALMATHRAG_{normalized}_STARTUP_STALL_TIMEOUT_SECONDS", _env_float("LOCALMATHRAG_RUNTIME_STARTUP_STALL_TIMEOUT_SECONDS", 120.0))


def _runtime_target(kind: str) -> dict[str, Any]:
    normalized = kind.lower()
    if normalized in {"embedding", "embeddings", "embed"}:
        return {
            "kind": "embedding",
            "model_type": "embedding",
            "base_url": EMBEDDING_RUNTIME_BASE_URL,
            "container": EMBEDDING_CONTAINER_NAME,
            "compose_service": "localmathrag-embedding",
        }
    if normalized in {"rerank", "reranker"}:
        return {
            "kind": "rerank",
            "model_type": "rerank",
            "base_url": RERANK_RUNTIME_BASE_URL,
            "container": RERANK_CONTAINER_NAME,
            "compose_service": "localmathrag-rerank",
        }
    if normalized in {"vision", "vlm", "image2text"}:
        return {
            "kind": "vision",
            "model_type": "vision",
            "base_url": VISION_BASE_URL,
            "container": VISION_CONTAINER_NAME,
            "compose_service": "localmathrag-vlm",
        }
    if normalized in {"asr", "speech-to-text"}:
        return {
            "kind": "asr",
            "model_type": "asr",
            "base_url": ASR_BASE_URL,
            "container": ASR_CONTAINER_NAME,
            "compose_service": "localmathrag-asr",
        }
    if normalized in {"tts", "text-to-speech"}:
        return {
            "kind": "tts",
            "model_type": "tts",
            "base_url": TTS_BASE_URL,
            "container": TTS_CONTAINER_NAME,
            "compose_service": "localmathrag-tts",
        }
    raise HTTPException(status_code=400, detail=f"Unsupported runtime kind: {kind}")


def _optional_runtime_targets() -> list[dict[str, Any]]:
    return [_runtime_target(kind) for kind in ("embedding", "rerank", "vision", "asr", "tts")]


def _balance_optional_runtimes(target: dict[str, Any]) -> list[dict[str, Any]]:
    if OPTIONAL_RUNTIME_MAX_ACTIVE != 1:
        return []
    actions = []
    for other in _optional_runtime_targets():
        if other["kind"] == target["kind"]:
            continue
        state = _docker_container_state(other["container"], other["compose_service"])
        if not state.get("container_running"):
            continue
        container_id = state.get("container_id")
        if not container_id:
            continue
        stop_result = _docker_stop_container(container_id)
        actions.append(
            {
                "kind": other["kind"],
                "action": "stopped",
                "container_id": container_id,
                "reason": f"balancer reserved optional runtime slot for {target['kind']}",
                "result": stop_result,
            }
        )
    return actions


def _stop_optional_runtime_for_degrade(target: dict[str, Any], reason: str) -> dict[str, Any] | None:
    try:
        state = _docker_container_state(target["container"], target["compose_service"])
        if not state.get("container_running"):
            return None
        container_id = state.get("container_id")
        if not container_id:
            return None
        return {
            "kind": target["kind"],
            "action": "stopped",
            "container_id": container_id,
            "reason": reason,
            "result": _docker_stop_container(container_id),
        }
    except Exception as exc:
        return {
            "kind": target["kind"],
            "action": "stop_failed",
            "reason": reason,
            "error": str(exc),
        }


def _optional_runtime_endpoint_status(kind: str, timeout: float = 1.5) -> dict[str, Any]:
    target = _runtime_target(kind)
    state: dict[str, Any]
    try:
        state = _docker_container_state(target["container"], target["compose_service"])
    except Exception as exc:
        state = {
            "container_found": False,
            "container_running": False,
            "container_name": target["container"],
            "compose_service": target["compose_service"],
            "container_error": str(exc),
        }
    if RUNTIME_LAZY_ENABLED and not state.get("container_running"):
        return {
            "base_url": target["base_url"],
            "models_url": f"{target['base_url']}/models",
            "endpoint_ok": False,
            "runtime_kind": target["kind"],
            "reason": "runtime container is not running",
            **state,
        }
    status = _endpoint_status(target["base_url"], timeout)
    status["runtime_kind"] = target["kind"]
    status.update(state)
    if state.get("container_running") and state.get("container_id"):
        status["startup_progress"] = _runtime_startup_progress(str(state["container_id"]))
    return status


def _runtime_startup_progress(container_id: str | None) -> dict[str, Any]:
    if not container_id:
        return {"phase": "unknown", "progress": 0.0, "message": "container is not available"}
    logs = _docker_container_logs(container_id)
    clean_logs = re.sub(r"\x1b\[[0-9;]*m", "", logs)
    last_lines = [line for line in clean_logs.splitlines() if line.strip()]
    last_line = last_lines[-1] if last_lines else ""
    phase = "starting"
    progress = 0.1
    message = "Runtime container is starting."
    if "Starting model backend" in clean_logs:
        phase, progress, message = "backend-starting", 0.35, "Runtime backend is starting."
    if "Starting Flash" in clean_logs or "Starting Bert" in clean_logs or "Starting Qwen" in clean_logs:
        phase, progress, message = "cuda-model-loading", 0.6, "CUDA model weights are loading."
    if "Warming up model" in clean_logs:
        phase, progress, message = "warming-up", 0.85, "Runtime is warming up the model."
    if "Starting HTTP server" in clean_logs or " Ready" in clean_logs or "\nReady" in clean_logs:
        phase, progress, message = "ready", 1.0, "Runtime HTTP server is ready."
    if "ERROR" in clean_logs or "Backend error" in clean_logs or "error:" in clean_logs.lower():
        phase, progress, message = "error", progress, "Runtime logs contain an error."
    return {
        "phase": phase,
        "progress": progress,
        "message": message,
        "last_log_line": last_line[-500:],
    }


def _wait_for_endpoint(base_url: str, timeout_seconds: float, interval_seconds: float = 2.0, container_id: str | None = None, kind: str = "runtime") -> dict[str, Any]:
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    stall_timeout = _runtime_startup_stall_timeout_seconds(kind)
    stall_deadline = time.monotonic() + max(1.0, stall_timeout)
    last_progress_signature = ""
    last_progress: dict[str, Any] = _runtime_startup_progress(container_id) if container_id else {"phase": "unknown", "progress": 0.0}
    last_status = _endpoint_status(base_url, timeout=1.5)
    while time.monotonic() < deadline and time.monotonic() < stall_deadline:
        if last_status.get("endpoint_ok"):
            last_status["startup_progress"] = {"phase": "ready", "progress": 1.0, "message": "Runtime endpoint is ready."}
            return last_status
        if container_id:
            last_progress = _runtime_startup_progress(container_id)
            signature = f"{last_progress.get('phase')}|{last_progress.get('last_log_line')}"
            if signature and signature != last_progress_signature:
                last_progress_signature = signature
                stall_deadline = time.monotonic() + max(1.0, stall_timeout)
        time.sleep(interval_seconds)
        last_status = _endpoint_status(base_url, timeout=1.5)
    last_status["startup_progress"] = last_progress
    if time.monotonic() >= stall_deadline:
        last_status["startup_stalled"] = True
        last_status["startup_stall_timeout_seconds"] = stall_timeout
    return last_status


def _ensure_runtime_ready(kind: str, timeout_seconds: float = 90.0) -> dict[str, Any]:
    target = _runtime_target(kind)
    status = _optional_runtime_endpoint_status(target["kind"], timeout=1.5)
    if status.get("endpoint_ok"):
        return {"ready": True, "started": False, "target": target, "endpoint_status": status}
    if not RUNTIME_LAZY_ENABLED:
        return {"ready": False, "started": False, "target": target, "endpoint_status": status, "reason": "lazy runtime startup is disabled"}
    resource_status = _runtime_resource_status(target["kind"])
    if not resource_status["ok"]:
        return {
            "ready": False,
            "started": False,
            "target": target,
            "endpoint_status": status,
            "resource_status": resource_status,
            "reason": "; ".join(resource_status["reasons"]),
        }
    container_id = _docker_find_container(target["container"], target["compose_service"])
    if not container_id:
        return {"ready": False, "started": False, "target": target, "endpoint_status": status, "reason": f"container for {target['compose_service']} was not prepared"}
    with RUNTIME_START_LOCK:
        status = _endpoint_status(target["base_url"], timeout=1.5)
        if status.get("endpoint_ok"):
            return {"ready": True, "started": False, "target": target, "endpoint_status": status, "resource_status": resource_status}
        resource_status = _runtime_resource_status(target["kind"])
        if not resource_status["ok"]:
            return {
                "ready": False,
                "started": False,
                "target": target,
                "endpoint_status": status,
                "resource_status": resource_status,
                "reason": "; ".join(resource_status["reasons"]),
            }
        balance_actions = _balance_optional_runtimes(target)
        start_result = _docker_start_container(container_id)
        status = _wait_for_endpoint(target["base_url"], timeout_seconds, container_id=container_id, kind=target["kind"])
        if start_result.get("started") and not status.get("endpoint_ok"):
            degraded_stop = _stop_optional_runtime_for_degrade(target, f"runtime did not become ready within {timeout_seconds:.0f}s")
            if degraded_stop:
                balance_actions.append(degraded_stop)
    return {
        "ready": bool(status.get("endpoint_ok")),
        "started": bool(start_result.get("started")),
        "target": target,
        "endpoint_status": status,
        "resource_status": resource_status,
        "reason": None if status.get("endpoint_ok") else f"runtime did not become ready within {timeout_seconds:.0f}s",
        "container_id": container_id[:12],
        "balancer": {
            "max_active_optional": OPTIONAL_RUNTIME_MAX_ACTIVE,
            "actions": balance_actions,
        },
    }


def _post_runtime_json(base_url: str, route: str, payload: dict[str, Any], timeout: float = 120.0) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}{route}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read()
        return json.loads(data.decode("utf-8")) if data else {}


def _runtime_route_base_url(base_url: str, route: str) -> str:
    if route == "/rerank":
        return base_url.removesuffix("/v1")
    return base_url


def _embedding_inputs(value: str | list[Any]) -> list[str]:
    if isinstance(value, str):
        return [value]
    return ["" if item is None else str(item) for item in value]


def _embedding_tokens(text: str) -> list[str]:
    lowered = text.lower()
    tokens = re.findall(r"[a-z0-9_./:+-]+|[\u4e00-\u9fff]", lowered)
    if len(tokens) < 2:
        return tokens
    bigrams = [f"{tokens[index]}{tokens[index + 1]}" for index in range(len(tokens) - 1)]
    return [*tokens, *bigrams]


def _hash_to_index(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % LOCAL_EMBEDDING_DIM


def _lexical_embedding(text: str) -> list[float]:
    values = [0.0] * LOCAL_EMBEDDING_DIM
    for token in _embedding_tokens(text):
        values[_hash_to_index(token)] += 1.0
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        return values
    return [value / norm for value in values]


def _token_count_map(text: str) -> dict[str, float]:
    counts: dict[str, float] = {}
    for token in _embedding_tokens(text):
        counts[token] = counts.get(token, 0.0) + 1.0
    return counts


def _cosine_from_counts(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(token, 0.0) for token, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _document_text(document: Any) -> str:
    if isinstance(document, str):
        return document
    if isinstance(document, dict):
        for key in ("text", "content", "document", "page_content"):
            value = document.get(key)
            if value is not None:
                return str(value)
    return "" if document is None else str(document)


def _truthy(value: bool | str | None) -> bool:
    if isinstance(value, bool):
        return value
    return isinstance(value, str) and value.lower() in {"1", "true", "yes"}


def _normalize_runtime_rerank_response(response: Any, request: RerankRequest, model: str) -> dict[str, Any]:
    if isinstance(response, dict) and isinstance(response.get("results"), list):
        results = response["results"]
    elif isinstance(response, list):
        results = [
            {
                "index": int(item.get("index", index)),
                "relevance_score": float(item.get("score", item.get("relevance_score", 0.0))),
            }
            for index, item in enumerate(response)
            if isinstance(item, dict)
        ]
    else:
        results = []
    results.sort(key=lambda item: item.get("relevance_score", 0.0), reverse=True)
    top_n = request.top_n if request.top_n is not None and request.top_n > 0 else len(results)
    include_documents = _truthy(request.return_documents)
    normalized = []
    for item in results[:top_n]:
        index = int(item.get("index", 0))
        result = {
            "index": index,
            "relevance_score": float(item.get("relevance_score", item.get("score", 0.0))),
        }
        if include_documents and 0 <= index < len(request.documents):
            result["document"] = request.documents[index]
        normalized.append(result)
    return {
        "model": model,
        "results": normalized,
        "usage": {
            "total_tokens": len(_embedding_tokens(request.query)) + sum(len(_embedding_tokens(_document_text(document))) for document in request.documents),
        },
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
        "base_url": _base_url_for_payload(metadata, model_type),
        "model_type": model_type,
        "max_tokens": metadata.get("max_tokens") or 8192,
    }


def _model_identity_payload(path: Path) -> dict[str, Any]:
    metadata = _load_model_metadata(path)
    model_type = metadata.get("model_type") or _infer_model_type(path)
    runtime_model_name = metadata.get("runtime_model_name") or path.stem
    if path.is_file() and path.suffix.lower() == ".gguf" and model_type[0] == "chat":
        runtime_model_name = f"/models/{path.name}"
    return {
        "name": metadata.get("name") or path.stem,
        "runtime_model_name": runtime_model_name,
        "model_type": model_type,
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
    url = f"{HF_ENDPOINT}/api/models/{repo_path}/tree/{revision}?recursive=1"
    with urllib.request.urlopen(url, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))
    if not isinstance(data, list):
        raise RuntimeError("Hugging Face repository tree response is invalid")
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


DOWNLOAD_CHUNK_SIZE = 1024 * 1024
DOWNLOAD_RETRIES = 3
DOWNLOAD_JOBS: dict[str, dict[str, Any]] = {}
DOWNLOAD_JOBS_LOCK = threading.Lock()


def _job_snapshot(job_id: str) -> dict[str, Any] | None:
    with DOWNLOAD_JOBS_LOCK:
        job = DOWNLOAD_JOBS.get(job_id)
        return dict(job) if job else None


def _update_job(job_id: str, **fields: Any) -> None:
    with DOWNLOAD_JOBS_LOCK:
        job = DOWNLOAD_JOBS.get(job_id)
        if job is None:
            return
        job.update(fields)
        total = job.get("total_bytes") or 0
        done = job.get("downloaded_bytes") or 0
        if job.get("status") in {"completed", "exists"}:
            job["progress"] = 100
        elif total > 0:
            job["progress"] = min(int(done * 100 / total), 99)
        job["updated_at"] = time.time()


def _job_add_bytes(job_id: str, count: int) -> None:
    with DOWNLOAD_JOBS_LOCK:
        job = DOWNLOAD_JOBS.get(job_id)
        if job is None:
            return
        job["downloaded_bytes"] = max(job.get("downloaded_bytes", 0) + count, 0)
        total = job.get("total_bytes") or 0
        if total > 0:
            job["progress"] = min(int(job["downloaded_bytes"] * 100 / total), 99)
        job["updated_at"] = time.time()


def _download_url_to_file(url: str, target: Path, job_id: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    # Bytes left over from a previous run count towards the progress right
    # away, so resumed downloads do not appear to start from zero.
    if partial.exists():
        _job_add_bytes(job_id, partial.stat().st_size)
    last_error: Exception | None = None
    for _ in range(DOWNLOAD_RETRIES):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "LocalMathRAGFlow/0.1"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if offset and getattr(response, "status", 200) != 206:
                    # Server ignored the range request; restart from scratch.
                    _job_add_bytes(job_id, -offset)
                    offset = 0
                with open(partial, "ab" if offset else "wb") as out:
                    while True:
                        chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                        if not chunk:
                            break
                        out.write(chunk)
                        _job_add_bytes(job_id, len(chunk))
            partial.replace(target)
            return
        except Exception as exc:
            last_error = exc
            time.sleep(2)
    # Keep the .partial file so the next attempt resumes from this offset.
    raise RuntimeError(f"download failed for {url}: {last_error}")


def _hf_file_url(repo: str, path: str, revision: str = "main") -> str:
    repo_path = urllib.parse.quote(repo, safe="/")
    file_path = urllib.parse.quote(path, safe="/")
    return f"{HF_ENDPOINT}/{repo_path}/resolve/{revision}/{file_path}"


def _download_snapshot(model: dict[str, Any], job_id: str) -> dict[str, Any]:
    repo = model.get("repo")
    file_name = model.get("file_name")
    if not repo or not file_name:
        raise RuntimeError("Snapshot model requires repo and file_name")
    target_dir = MODEL_DIR / Path(file_name).name
    metadata_path = target_dir / MODEL_METADATA_NAME
    if metadata_path.exists():
        return {"status": "exists", "model": _model_payload(target_dir)}

    target_dir.mkdir(parents=True, exist_ok=True)
    allow_patterns = model.get("allow_patterns")
    tree = _hf_repo_tree(repo)
    files = [
        item
        for item in tree
        if item.get("type") == "file"
        and isinstance(item.get("path"), str)
        and _allowed_snapshot_file(item["path"], allow_patterns)
    ]
    if not files:
        raise RuntimeError("No downloadable model files were found in the repository")

    total_bytes = sum(int(item.get("size") or 0) for item in files)
    _update_job(job_id, total_bytes=total_bytes)

    # Resume support: files finished by a previous attempt are kept on disk
    # and only count towards the progress; .partial files are resumed inside
    # _download_url_to_file via HTTP range requests.
    pending: list[dict[str, Any]] = []
    for item in files:
        target = target_dir / item["path"]
        if target.exists():
            _job_add_bytes(job_id, target.stat().st_size)
        else:
            pending.append(item)

    try:
        for item in pending:
            file_path = item["path"]
            _update_job(job_id, current_file=file_path)
            _download_url_to_file(_hf_file_url(repo, file_path), target_dir / file_path, job_id)
        metadata_path.write_text(
            json.dumps(_snapshot_metadata(model), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        # Keep finished files and .partial files so the next attempt resumes.
        raise RuntimeError(f"Snapshot download failed: {exc}") from exc

    return {"status": "downloaded", "model": _model_payload(target_dir)}


def _download_single_file(url: str, file_name: str, job_id: str) -> dict[str, Any]:
    target = MODEL_DIR / Path(file_name).name
    if target.exists():
        return {"status": "exists", "model": _model_payload(target)}
    try:
        head = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "LocalMathRAGFlow/0.1"})
        with urllib.request.urlopen(head, timeout=30) as response:
            length = int(response.headers.get("Content-Length") or 0)
        if length:
            _update_job(job_id, total_bytes=length)
    except Exception:
        pass
    _update_job(job_id, current_file=Path(file_name).name)
    _download_url_to_file(url, target, job_id)
    return {"status": "downloaded", "model": _model_payload(target)}


def _run_download_job(job_id: str, selected: dict[str, Any] | None, url: str | None, file_name: str | None) -> None:
    _update_job(job_id, status="downloading")
    try:
        if selected and selected.get("download_kind") == "snapshot":
            result = _download_snapshot(selected, job_id)
        else:
            result = _download_single_file(url or "", file_name or "", job_id)
        _update_job(
            job_id,
            status="completed" if result["status"] == "downloaded" else result["status"],
            model=result["model"],
        )
    except Exception as exc:
        _update_job(job_id, status="failed", error=str(exc))


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "schema_dir": str(SCHEMA_DIR),
        "schema_count": len(_schema_files()),
        "model_dir": str(MODEL_DIR),
        "model_count": len(_model_entries()),
        "dataset_dir": str(DATASET_DIR),
    }


@app.get("/v1/dataset/status")
def dataset_status() -> dict[str, Any]:
    return _scan_dataset(include_entries=False)


@app.get("/v1/dataset/files")
def dataset_files() -> dict[str, Any]:
    return _scan_dataset(include_entries=True)


@app.get("/v1/models")
def openai_models() -> dict[str, Any]:
    data: list[dict[str, Any]] = []
    seen: set[str] = set()
    endpoint_statuses: dict[str, dict[str, Any]] = {}

    def endpoint_ok(endpoint_key: str) -> bool:
        if endpoint_key not in endpoint_statuses:
            endpoint_statuses[endpoint_key] = _endpoint_status(_runtime_base_url_for_endpoint_key(endpoint_key), timeout=0.8)
        return bool(endpoint_statuses[endpoint_key].get("endpoint_ok", False))

    def add_model(model_id: str | None, model_type: str, fallback: bool = False) -> None:
        if not model_id or model_id in seen:
            return
        seen.add(model_id)
        data.append(
            {
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "localmathrag",
                "model_type": model_type,
                "fallback": fallback,
            }
        )

    add_model(LOCAL_EMBEDDING_MODEL, "embedding", True)
    add_model(LOCAL_RERANK_MODEL, "rerank", True)
    for path in _model_entries():
        payload = _model_identity_payload(path)
        model_type = (payload.get("model_type") or ["chat"])[0]
        endpoint_key = _endpoint_key_for_model_type(model_type)
        if not endpoint_ok(endpoint_key):
            continue
        add_model(payload.get("runtime_model_name"), model_type)
        add_model(payload.get("name"), model_type)

    return {"object": "list", "data": data}


@app.post("/v1/embeddings")
def openai_embeddings(request: EmbeddingRequest) -> dict[str, Any]:
    try:
        runtime = _ensure_runtime_ready("embedding", _runtime_ready_timeout_seconds("embedding"))
        if runtime.get("ready"):
            resource_status = _runtime_resource_status("embedding")
            if not resource_status["ok"]:
                runtime["degraded_stop"] = _stop_optional_runtime_for_degrade(runtime["target"], "; ".join(resource_status["reasons"]))
                raise RuntimeError("; ".join(resource_status["reasons"]))
            payload = request.model_dump(exclude_none=True)
            payload.setdefault("model", request.model or "bge-m3")
            response = _post_runtime_json(runtime["target"]["base_url"], "/embeddings", payload)
            response.setdefault("model", payload["model"])
            response["runtime"] = {
                "kind": "embedding",
                "fallback": False,
                "started": runtime.get("started", False),
                "balancer": runtime.get("balancer"),
            }
            return response
    except Exception as exc:
        runtime = {"reason": str(exc)}

    inputs = _embedding_inputs(request.input)
    data = [
        {
            "object": "embedding",
            "index": index,
            "embedding": _lexical_embedding(text),
        }
        for index, text in enumerate(inputs)
    ]
    token_count = sum(len(_embedding_tokens(text)) for text in inputs)
    return {
        "object": "list",
        "model": request.model or LOCAL_EMBEDDING_MODEL,
        "data": data,
        "usage": {
            "prompt_tokens": token_count,
            "total_tokens": token_count,
        },
        "runtime": {
            "kind": "embedding",
            "fallback": True,
            "reason": runtime.get("reason", "runtime endpoint is not ready"),
        },
    }


@app.post("/v1/rerank")
def openai_rerank(request: RerankRequest) -> dict[str, Any]:
    try:
        runtime = _ensure_runtime_ready("rerank", _runtime_ready_timeout_seconds("rerank"))
        if runtime.get("ready"):
            resource_status = _runtime_resource_status("rerank")
            if not resource_status["ok"]:
                runtime["degraded_stop"] = _stop_optional_runtime_for_degrade(runtime["target"], "; ".join(resource_status["reasons"]))
                raise RuntimeError("; ".join(resource_status["reasons"]))
            model = request.model or "bge-reranker-v2-m3"
            payload = {
                "query": request.query,
                "texts": [_document_text(document) for document in request.documents],
            }
            response = _post_runtime_json(_runtime_route_base_url(runtime["target"]["base_url"], "/rerank"), "/rerank", payload)
            response = _normalize_runtime_rerank_response(response, request, model)
            response["runtime"] = {
                "kind": "rerank",
                "fallback": False,
                "started": runtime.get("started", False),
                "balancer": runtime.get("balancer"),
            }
            return response
    except Exception as exc:
        runtime = {"reason": str(exc)}

    query_counts = _token_count_map(request.query)
    scored = []
    for index, document in enumerate(request.documents):
        text = _document_text(document)
        scored.append(
            {
                "index": index,
                "relevance_score": _cosine_from_counts(query_counts, _token_count_map(text)),
                "document": document,
            }
        )
    scored.sort(key=lambda item: item["relevance_score"], reverse=True)
    top_n = request.top_n if request.top_n is not None and request.top_n > 0 else len(scored)
    include_documents = _truthy(request.return_documents)
    results = []
    for item in scored[:top_n]:
        result = {
            "index": item["index"],
            "relevance_score": item["relevance_score"],
        }
        if include_documents:
            result["document"] = item["document"]
        results.append(result)
    return {
        "model": request.model or LOCAL_RERANK_MODEL,
        "results": results,
        "usage": {
            "total_tokens": len(_embedding_tokens(request.query)) + sum(len(_embedding_tokens(_document_text(document))) for document in request.documents),
        },
        "runtime": {
            "kind": "rerank",
            "fallback": True,
            "reason": runtime.get("reason", "runtime endpoint is not ready"),
        },
    }


@app.get("/v1/models/local")
def list_local_models() -> dict[str, Any]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    endpoint_statuses = _model_endpoint_statuses()
    return {
        "model_dir": str(MODEL_DIR),
        "endpoint_status": endpoint_statuses["chat"],
        "endpoint_statuses": endpoint_statuses,
        "models": [_model_payload(path) for path in _model_entries()],
    }


@app.get("/v1/models/recommended")
def list_recommended_models() -> dict[str, Any]:
    local_names = {path.name for path in _model_entries()}
    endpoint_statuses = _model_endpoint_statuses()
    models = []
    for model in RECOMMENDED_MODELS:
        item = dict(model)
        item["downloaded"] = item["file_name"] in local_names
        item["target_path"] = str(MODEL_DIR / item["file_name"])
        item["endpoint_status"] = endpoint_statuses.get((item.get("model_type") or ["chat"])[0], endpoint_statuses["chat"])
        item.setdefault("downloadable", False)
        job = _job_snapshot(item["id"])
        if job and job.get("status") in {"queued", "downloading"}:
            item["active_job"] = job
        models.append(item)
    return {"models": models, "endpoint_statuses": endpoint_statuses}


@app.get("/v1/models/status")
def model_status() -> dict[str, Any]:
    endpoint_statuses = _model_endpoint_statuses(timeout=0.25)
    return {
        "model_dir": str(MODEL_DIR),
        "model_count": _quick_model_entry_count(),
        "endpoint_status": endpoint_statuses["chat"],
        "endpoint_statuses": endpoint_statuses,
    }


@app.post("/v1/runtime/ensure")
def ensure_runtime(request: RuntimeEnsureRequest) -> dict[str, Any]:
    target = _runtime_target(request.kind)
    timeout_seconds = request.timeout_seconds if request.timeout_seconds is not None else _runtime_ready_timeout_seconds(target["kind"])
    status = _optional_runtime_endpoint_status(target["kind"], timeout=1.5)
    if status.get("endpoint_ok"):
        resource_status = _runtime_resource_status(target["kind"])
        if not resource_status["ok"]:
            degraded_stop = _stop_optional_runtime_for_degrade(target, "; ".join(resource_status["reasons"]))
            return {
                "kind": target["kind"],
                "ready": False,
                "started": False,
                "degraded": True,
                "reason": "; ".join(resource_status["reasons"]),
                "endpoint_status": status,
                "resource_status": resource_status,
                "balancer": {"max_active_optional": OPTIONAL_RUNTIME_MAX_ACTIVE, "actions": [degraded_stop] if degraded_stop else []},
            }
        return {
            "kind": target["kind"],
            "ready": True,
            "started": False,
            "degraded": False,
            "endpoint_status": status,
            "resource_status": resource_status,
            "balancer": {"max_active_optional": OPTIONAL_RUNTIME_MAX_ACTIVE, "actions": []},
        }
    if not request.start:
        return {
            "kind": target["kind"],
            "ready": False,
            "started": False,
            "degraded": True,
            "reason": "runtime endpoint is not ready and start=false",
            "endpoint_status": status,
            "balancer": {"max_active_optional": OPTIONAL_RUNTIME_MAX_ACTIVE, "actions": []},
        }
    if not RUNTIME_LAZY_ENABLED:
        return {
            "kind": target["kind"],
            "ready": False,
            "started": False,
            "degraded": True,
            "reason": "lazy runtime startup is disabled",
            "endpoint_status": status,
            "balancer": {"max_active_optional": OPTIONAL_RUNTIME_MAX_ACTIVE, "actions": []},
        }
    resource_status = _runtime_resource_status(target["kind"])
    if not resource_status["ok"]:
        return {
            "kind": target["kind"],
            "ready": False,
            "started": False,
            "degraded": True,
            "reason": "; ".join(resource_status["reasons"]),
            "endpoint_status": status,
            "resource_status": resource_status,
            "balancer": {"max_active_optional": OPTIONAL_RUNTIME_MAX_ACTIVE, "actions": []},
        }
    if not _has_local_model_type(target["model_type"]):
        return {
            "kind": target["kind"],
            "ready": False,
            "started": False,
            "degraded": True,
            "reason": f"no local {target['model_type']} model found",
            "endpoint_status": status,
            "balancer": {"max_active_optional": OPTIONAL_RUNTIME_MAX_ACTIVE, "actions": []},
        }

    try:
        runtime = _ensure_runtime_ready(target["kind"], timeout_seconds)
        return {
            "kind": target["kind"],
            "ready": bool(runtime.get("ready")),
            "started": bool(runtime.get("started")),
            "degraded": not bool(runtime.get("ready")),
            "reason": runtime.get("reason"),
            "endpoint_status": runtime.get("endpoint_status"),
            "resource_status": runtime.get("resource_status"),
            "container_id": runtime.get("container_id"),
            "balancer": runtime.get("balancer"),
        }
    except Exception as exc:
        return {
            "kind": target["kind"],
            "ready": False,
            "started": False,
            "degraded": True,
            "reason": str(exc),
            "endpoint_status": status,
            "balancer": {"max_active_optional": OPTIONAL_RUNTIME_MAX_ACTIVE, "actions": []},
        }


@app.get("/v1/runtime/status")
def runtime_status(kind: str = "embedding") -> dict[str, Any]:
    target = _runtime_target(kind)
    status = _optional_runtime_endpoint_status(target["kind"], timeout=0.25)
    resource_status = _runtime_resource_status(target["kind"])
    return {
        "kind": target["kind"],
        "ready": bool(status.get("endpoint_ok")),
        "endpoint_status": status,
        "resource_status": resource_status,
        "balancer": {"max_active_optional": OPTIONAL_RUNTIME_MAX_ACTIVE},
    }


@app.post("/v1/runtime/stop")
def stop_runtime(request: RuntimeStopRequest) -> dict[str, Any]:
    target = _runtime_target(request.kind)
    action = _stop_optional_runtime_for_degrade(target, "manual optional runtime cleanup")
    status = _optional_runtime_endpoint_status(target["kind"], timeout=0.25)
    return {
        "kind": target["kind"],
        "stopped": bool(action and action.get("action") == "stopped"),
        "action": action,
        "endpoint_status": status,
    }


@app.post("/v1/models/download")
def download_model(request: DownloadModelRequest) -> dict[str, Any]:
    selected = next(
        (model for model in RECOMMENDED_MODELS if model["id"] == request.id),
        None,
    )
    is_snapshot = bool(selected and selected.get("download_kind") == "snapshot")
    url = request.url or (selected or {}).get("url")
    file_name = request.file_name or (selected or {}).get("file_name")

    if selected and not selected.get("downloadable", False):
        raise HTTPException(status_code=400, detail="This model is a repository recommendation. Open the model link and install it with the matching local runtime.")
    if not is_snapshot:
        if not url or not file_name:
            raise HTTPException(status_code=400, detail="Model url and file_name are required")
        if not url.startswith(("https://huggingface.co/", f"{HF_ENDPOINT}/")):
            raise HTTPException(status_code=400, detail="Only Hugging Face model downloads are supported")
        if HF_ENDPOINT != "https://huggingface.co" and url.startswith("https://huggingface.co/"):
            url = HF_ENDPOINT + url[len("https://huggingface.co"):]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Fast path: already downloaded, no job required.
    if is_snapshot:
        target_dir = MODEL_DIR / Path(selected["file_name"]).name
        if (target_dir / MODEL_METADATA_NAME).exists():
            return {"status": "exists", "model": _model_payload(target_dir)}
    else:
        target = MODEL_DIR / Path(file_name).name
        if target.exists():
            return {"status": "exists", "model": _model_payload(target)}

    job_id = request.id or Path(file_name or "").name
    if not job_id:
        raise HTTPException(status_code=400, detail="Model id or file_name is required")

    with DOWNLOAD_JOBS_LOCK:
        existing = DOWNLOAD_JOBS.get(job_id)
        if existing and existing.get("status") in {"queued", "downloading"}:
            return {"status": "downloading", "job": dict(existing)}
        DOWNLOAD_JOBS[job_id] = {
            "id": job_id,
            "status": "queued",
            "progress": 0,
            "downloaded_bytes": 0,
            "total_bytes": 0,
            "current_file": None,
            "error": None,
            "model": None,
            "updated_at": time.time(),
        }

    worker = threading.Thread(
        target=_run_download_job,
        args=(job_id, selected, url, file_name),
        name=f"model-download-{job_id}",
        daemon=True,
    )
    worker.start()
    return {"status": "started", "job": _job_snapshot(job_id)}


@app.get("/v1/models/download/{job_id}/status")
def download_status(job_id: str) -> dict[str, Any]:
    job = _job_snapshot(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No download job found for this model")
    return {"job": job}


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

from __future__ import annotations

import hashlib
import fnmatch
import http.client
import json
import logging
import math
import os
import re
import shutil
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


SCHEMA_DIR = Path(os.environ.get("LOCALMATHRAG_SCHEMA_DIR", "/schemas"))
DATA_DIR = Path(os.environ.get("LOCALMATHRAG_DATA_DIR", "/data"))
MODEL_DIR = Path(os.environ.get("LOCALMATHRAG_MODEL_DIR", DATA_DIR / "models"))
DATASET_DIR = Path(os.environ.get("LOCALMATHRAG_DATASET_DIR", DATA_DIR / "dataset"))
DATASET_STATE_FILE = Path(os.environ.get("LOCALMATHRAG_DATASET_STATE_FILE", DATA_DIR / "cache" / "dataset-state.json"))
RUNTIME_CONFIG_FILE = Path(os.environ.get("LOCALMATHRAG_RUNTIME_CONFIG_FILE", DATA_DIR / "cache" / "runtime-config.json"))
MODEL_EXTENSIONS = {".gguf", ".safetensors", ".bin"}
MODEL_METADATA_NAME = ".localmathrag-model.json"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_is_set(name: str) -> bool:
    return name in os.environ and str(os.environ.get(name) or "").strip() != ""


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def _coerce_int(value: Any, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(value))
    except (TypeError, ValueError):
        return max(minimum, default)


def _read_runtime_config_unlocked() -> dict[str, Any]:
    if not RUNTIME_CONFIG_FILE.exists():
        return {}
    try:
        data = json.loads(RUNTIME_CONFIG_FILE.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _runtime_host_fingerprint() -> str:
    explicit = os.environ.get("LOCALMATHRAG_HOST_FINGERPRINT")
    if explicit and explicit.strip():
        return explicit.strip()
    mem_total_kb = "unknown"
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                mem_total_kb = str(_coerce_int(line.split()[1], 0, 0))
                break
    except OSError:
        pass
    return f"cpu={os.cpu_count() or 'unknown'};mem_kb={mem_total_kb}"


def _scheduler_host_matches(scheduler: dict[str, Any]) -> bool:
    saved = str(scheduler.get("host_fingerprint") or "").strip()
    return not saved or saved == _runtime_host_fingerprint()


def _normalize_scheduler_policy(
    scheduler: dict[str, Any],
    *,
    mode: str = "auto",
    source: str = "runtime-config",
) -> dict[str, Any]:
    max_active = _coerce_int(scheduler.get("max_active_optional"), 1, 1)
    allow_chat_embedding = _coerce_bool(scheduler.get("allow_chat_embedding_concurrency"), max_active > 1)
    degrade_small_embedding = _coerce_bool(
        scheduler.get("degrade_small_embedding_when_chat_ready"),
        not allow_chat_embedding,
    )
    return {
        "mode": mode,
        "source": str(scheduler.get("source") or source),
        "max_active_optional": max_active,
        "allow_chat_embedding_concurrency": allow_chat_embedding,
        "degrade_small_embedding_when_chat_ready": degrade_small_embedding,
        "small_embedding_max_inputs": _coerce_int(scheduler.get("small_embedding_max_inputs"), 2, 1),
        "small_embedding_max_tokens": _coerce_int(scheduler.get("small_embedding_max_tokens"), 512, 1),
        "reason": str(scheduler.get("reason") or ""),
        "updated_at": scheduler.get("updated_at"),
        "host_fingerprint": scheduler.get("host_fingerprint") or _runtime_host_fingerprint(),
    }


def _resolve_optional_runtime_policy() -> dict[str, Any]:
    raw = str(os.environ.get("LOCALMATHRAG_OPTIONAL_RUNTIME_MAX_ACTIVE", "auto")).strip()
    normalized = raw.lower()
    env_source = os.environ.get("LOCALMATHRAG_OPTIONAL_RUNTIME_MAX_ACTIVE_SOURCE") or "env"
    if normalized not in {"", "auto", "adaptive"}:
        max_active = _coerce_int(raw, 1, 1)
        return _normalize_scheduler_policy(
            {
                "max_active_optional": max_active,
                "allow_chat_embedding_concurrency": max_active > 1,
                "degrade_small_embedding_when_chat_ready": max_active <= 1,
                "reason": "explicit LOCALMATHRAG_OPTIONAL_RUNTIME_MAX_ACTIVE override",
                "source": env_source,
            },
            mode="manual",
            source=env_source,
        )

    config = _read_runtime_config_unlocked()
    scheduler = config.get("scheduler") if isinstance(config.get("scheduler"), dict) else None
    if scheduler and _scheduler_host_matches(scheduler):
        return _normalize_scheduler_policy(scheduler, mode="auto", source="runtime-config")

    return _normalize_scheduler_policy(
        {
            "max_active_optional": 1,
            "allow_chat_embedding_concurrency": False,
            "degrade_small_embedding_when_chat_ready": True,
            "reason": "no persisted scheduler probe result for this host",
            "source": "adaptive-unprobed",
        },
        mode="auto",
        source="adaptive-unprobed",
    )


def _scheduler_policy_bool(policy: dict[str, Any], env_name: str, policy_key: str, default: bool) -> bool:
    if _env_is_set(env_name):
        return _env_bool(env_name, default)
    return _coerce_bool(policy.get(policy_key), default)


def _scheduler_policy_int(policy: dict[str, Any], env_name: str, policy_key: str, default: int, minimum: int = 1) -> int:
    if _env_is_set(env_name):
        return _env_int(env_name, default)
    return _coerce_int(policy.get(policy_key), default, minimum)


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
LLAMA_BASE_URL = os.environ.get("LOCALMATHRAG_LLAMA_BASE_URL", "http://localmathrag-object-service:8088/v1").rstrip("/")
LLAMA_RUNTIME_BASE_URL = os.environ.get("LOCALMATHRAG_LLAMA_RUNTIME_BASE_URL", "http://host.docker.internal:8080/v1").rstrip("/")
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
LLAMA_CONTAINER_NAME = os.environ.get("LOCALMATHRAG_LLAMA_CONTAINER", "docker-localmathrag-llama-cpp-cuda-1")
LLAMA_COMPOSE_SERVICE = os.environ.get("LOCALMATHRAG_LLAMA_COMPOSE_SERVICE", "localmathrag-llama-cpp-cuda")
EMBEDDING_CONTAINER_NAME = os.environ.get("LOCALMATHRAG_EMBEDDING_CONTAINER", "docker-localmathrag-embedding-1")
RERANK_CONTAINER_NAME = os.environ.get("LOCALMATHRAG_RERANK_CONTAINER", "docker-localmathrag-rerank-1")
RERANK_MODEL_NAME = os.environ.get("LOCALMATHRAG_RERANK_MODEL", "bge-reranker-v2-m3")
RERANK_PROFILE = os.environ.get("LOCALMATHRAG_RERANK_PROFILE", "cuda")
VISION_CONTAINER_NAME = os.environ.get("LOCALMATHRAG_VISION_CONTAINER", "docker-localmathrag-vlm-1")
ASR_CONTAINER_NAME = os.environ.get("LOCALMATHRAG_ASR_CONTAINER", "docker-localmathrag-asr-1")
TTS_CONTAINER_NAME = os.environ.get("LOCALMATHRAG_TTS_CONTAINER", "docker-localmathrag-tts-1")
RUNTIME_LAZY_ENABLED = os.environ.get("LOCALMATHRAG_RUNTIME_LAZY", "1").lower() not in {"0", "false", "no", "off"}
SCHEDULER_POLICY = _resolve_optional_runtime_policy()
OPTIONAL_RUNTIME_MAX_ACTIVE = int(SCHEDULER_POLICY["max_active_optional"])
OPTIONAL_RUNTIME_MAX_ACTIVE_SOURCE = str(SCHEDULER_POLICY.get("source") or "adaptive")
CHAT_BACKGROUND_START = os.environ.get("LOCALMATHRAG_CHAT_BACKGROUND_START", "1").lower() not in {"0", "false", "no", "off"}
CHAT_BACKGROUND_START_DELAY_SECONDS = max(0.0, float(os.environ.get("LOCALMATHRAG_CHAT_BACKGROUND_START_DELAY_SECONDS", "20")))
CHAT_RESTORE_AFTER_EMBEDDING = _env_bool("LOCALMATHRAG_CHAT_RESTORE_AFTER_EMBEDDING", True)
CHAT_RESTORE_AFTER_EMBEDDING_DELAY_SECONDS = max(0.0, _env_float("LOCALMATHRAG_CHAT_RESTORE_AFTER_EMBEDDING_DELAY_SECONDS", 30.0))
EMBEDDING_DEGRADE_SMALL_REQUESTS_WHEN_CHAT_READY = _scheduler_policy_bool(
    SCHEDULER_POLICY,
    "LOCALMATHRAG_EMBEDDING_DEGRADE_SMALL_REQUESTS_WHEN_CHAT_READY",
    "degrade_small_embedding_when_chat_ready",
    OPTIONAL_RUNTIME_MAX_ACTIVE <= 1,
)
EMBEDDING_SMALL_REQUEST_MAX_INPUTS = max(
    1,
    _scheduler_policy_int(
        SCHEDULER_POLICY,
        "LOCALMATHRAG_EMBEDDING_SMALL_REQUEST_MAX_INPUTS",
        "small_embedding_max_inputs",
        2,
    ),
)
EMBEDDING_SMALL_REQUEST_MAX_TOKENS = max(
    1,
    _scheduler_policy_int(
        SCHEDULER_POLICY,
        "LOCALMATHRAG_EMBEDDING_SMALL_REQUEST_MAX_TOKENS",
        "small_embedding_max_tokens",
        512,
    ),
)
EMBEDDING_AUTOSTART_COOLDOWN_SECONDS = max(0.0, _env_float("LOCALMATHRAG_EMBEDDING_AUTOSTART_COOLDOWN_SECONDS", 86400.0))
EMBEDDING_DOCUMENT_REQUEST_MIN_INPUTS = max(1, _env_int("LOCALMATHRAG_EMBEDDING_DOCUMENT_REQUEST_MIN_INPUTS", 8))
EMBEDDING_DOCUMENT_REQUEST_MIN_TOKENS = max(1, _env_int("LOCALMATHRAG_EMBEDDING_DOCUMENT_REQUEST_MIN_TOKENS", 2048))
EMBEDDING_DOCUMENT_READY_TIMEOUT_SECONDS = max(1.0, _env_float("LOCALMATHRAG_EMBEDDING_DOCUMENT_READY_TIMEOUT_SECONDS", 480.0))
EMBEDDING_CITATION_READY_TIMEOUT_SECONDS = max(1.0, _env_float("LOCALMATHRAG_EMBEDDING_CITATION_READY_TIMEOUT_SECONDS", 18.0))
EMBEDDING_DOCUMENT_PREEMPT_CHAT = _env_bool("LOCALMATHRAG_EMBEDDING_DOCUMENT_PREEMPT_CHAT", True)
RERANK_BACKGROUND_PREWARM = os.environ.get("LOCALMATHRAG_RERANK_BACKGROUND_PREWARM", "1").lower() not in {"0", "false", "no", "off"}
EMBEDDING_BACKGROUND_PREWARM = _env_bool("LOCALMATHRAG_EMBEDDING_BACKGROUND_PREWARM", True)
NO_COLD_START_IN_REQUEST = _env_bool("LOCALMATHRAG_RUNTIME_NO_COLD_START_IN_REQUEST", True)
AUXILIARY_PREWARM_ENABLED = _env_bool("LOCALMATHRAG_AUXILIARY_PREWARM_ENABLED", True)
AUXILIARY_PREWARM_IDLE_SECONDS = max(0.0, _env_float("LOCALMATHRAG_AUXILIARY_PREWARM_IDLE_SECONDS", 8.0))
AUXILIARY_PREWARM_MAX_WAIT_SECONDS = max(1.0, _env_float("LOCALMATHRAG_AUXILIARY_PREWARM_MAX_WAIT_SECONDS", 300.0))
RECENT_TASK_WINDOW_SECONDS = max(30.0, _env_float("LOCALMATHRAG_RECENT_TASK_WINDOW_SECONDS", 900.0))
RERANK_PREWARM_RECENT_THRESHOLD = max(1, _env_int("LOCALMATHRAG_RERANK_PREWARM_RECENT_THRESHOLD", 1))
EMBEDDING_PREWARM_RECENT_THRESHOLD = max(1, _env_int("LOCALMATHRAG_EMBEDDING_PREWARM_RECENT_THRESHOLD", 2))
SCHEDULER_AUTO_PROBE = _env_bool("LOCALMATHRAG_SCHEDULER_AUTO_PROBE", True)
SCHEDULER_AUTO_PROBE_TIMEOUT_SECONDS = max(1.0, _env_float("LOCALMATHRAG_SCHEDULER_AUTO_PROBE_TIMEOUT_SECONDS", 240.0))
CHAT_REQUEST_TIMEOUT_SECONDS = float(os.environ.get("LOCALMATHRAG_CHAT_REQUEST_TIMEOUT_SECONDS", "600"))
CHAT_CONTEXT_SIZE = max(2048, _env_int("LOCALMATHRAG_CTX_SIZE", 8192))
CHAT_CONTEXT_MIN_PROMPT_TOKENS = max(1024, _env_int("LOCALMATHRAG_CHAT_CONTEXT_MIN_PROMPT_TOKENS", 6144))
CHAT_CONTEXT_RESPONSE_RESERVE_TOKENS = max(128, _env_int("LOCALMATHRAG_CHAT_CONTEXT_RESPONSE_RESERVE_TOKENS", 1024))
CHAT_CONTEXT_SAFETY_MARGIN_TOKENS = max(128, _env_int("LOCALMATHRAG_CHAT_CONTEXT_SAFETY_MARGIN_TOKENS", 512))
CHAT_CONTEXT_PROMPT_BUDGET_RATIO = max(0.25, min(0.9, _env_float("LOCALMATHRAG_CHAT_CONTEXT_PROMPT_BUDGET_RATIO", 0.82)))
CHAT_CONTEXT_CLAMP_ENABLED = _env_bool("LOCALMATHRAG_CHAT_CONTEXT_CLAMP_ENABLED", True)
CHAT_RUNTIME_REQUEST_RETRIES = max(0, _env_int("LOCALMATHRAG_CHAT_RUNTIME_REQUEST_RETRIES", 1))
CHAT_RUNTIME_RETRY_PROMPT_BUDGET_RATIO = max(0.25, min(0.8, _env_float("LOCALMATHRAG_CHAT_RUNTIME_RETRY_PROMPT_BUDGET_RATIO", 0.65)))
RERANK_REQUEST_TIMEOUT_SECONDS = float(os.environ.get("LOCALMATHRAG_RERANK_REQUEST_TIMEOUT_SECONDS", "8"))
RERANK_RUNTIME_BATCH_SIZE = max(1, int(os.environ.get("LOCALMATHRAG_RERANK_RUNTIME_BATCH_SIZE", "32")))
RERANK_START_MAX_FAILURES = max(1, _env_int("LOCALMATHRAG_RERANK_START_MAX_FAILURES", 2))
RERANK_DISABLE_AFTER_FAILURES = _env_bool("LOCALMATHRAG_RERANK_DISABLE_AFTER_FAILURES", True)
RERANK_CONTEXT_MIN_TOKENS = max(1024, _env_int("LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS_MIN", 8192))
RERANK_CONTEXT_STEP_TOKENS = max(256, _env_int("LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS_STEP", 4096))
RERANK_CONTEXT_RECREATE_RETRIES = max(0, _env_int("LOCALMATHRAG_RERANK_CONTEXT_RECREATE_RETRIES", 2))
OPTIONAL_RUNTIME_DISABLE_AFTER_FAILURES = _env_bool("LOCALMATHRAG_OPTIONAL_RUNTIME_DISABLE_AFTER_FAILURES", True)
OPTIONAL_RUNTIME_START_MAX_FAILURES = max(1, _env_int("LOCALMATHRAG_OPTIONAL_RUNTIME_START_MAX_FAILURES", 2))
OPTIONAL_RUNTIME_DEGRADE_STOP_TIMEOUT_SECONDS = max(0, int(os.environ.get("LOCALMATHRAG_OPTIONAL_RUNTIME_DEGRADE_STOP_TIMEOUT_SECONDS", "1")))
OPTIONAL_RUNTIME_STOP_ON_READY_TIMEOUT = os.environ.get("LOCALMATHRAG_OPTIONAL_RUNTIME_STOP_ON_READY_TIMEOUT", "0").lower() in {"1", "true", "yes", "on"}
RUNTIME_START_FAILURE_COOLDOWN_SECONDS = max(0.0, float(os.environ.get("LOCALMATHRAG_RUNTIME_START_FAILURE_COOLDOWN_SECONDS", "60")))
RUNTIME_START_FAILURE_PROBE_TIMEOUT_SECONDS = max(0.05, float(os.environ.get("LOCALMATHRAG_RUNTIME_START_FAILURE_PROBE_TIMEOUT_SECONDS", "0.2")))
RUNTIME_READY_PROBE_TIMEOUT_SECONDS = max(0.05, float(os.environ.get("LOCALMATHRAG_RUNTIME_READY_PROBE_TIMEOUT_SECONDS", "0.5")))
RERANK_BACKGROUND_START = os.environ.get("LOCALMATHRAG_RERANK_BACKGROUND_START", "1").lower() not in {"0", "false", "no", "off"}
RUNTIME_START_LOCK = threading.Lock()
RUNTIME_START_FAILURE_LOCK = threading.Lock()
RUNTIME_START_FAILURES: dict[str, dict[str, Any]] = {}
RUNTIME_START_FAILURE_COUNTS: dict[str, int] = {}
RUNTIME_CONFIG_LOCK = threading.Lock()
RUNTIME_DISABLED_LOCK = threading.Lock()
RUNTIME_DISABLED: dict[str, dict[str, Any]] = {}
RUNTIME_BACKGROUND_START_LOCK = threading.Lock()
RUNTIME_BACKGROUND_STARTS: set[str] = set()
RUNTIME_CHAT_RESTORE_LOCK = threading.Lock()
RUNTIME_CHAT_RESTORE_SEQUENCE = 0
RUNTIME_CHAT_RESTORE_THREAD_RUNNING = False
PERSISTENT_OPTIONAL_RUNTIME_KINDS = {"embedding", "vision", "asr", "tts"}
RUNTIME_DEGRADATION_LOCK = threading.Lock()
RUNTIME_DEGRADATIONS: dict[str, dict[str, Any]] = {}
RUNTIME_REQUEST_ACTIVITY_LOCK = threading.Lock()
RUNTIME_ACTIVE_REQUESTS: dict[str, int] = {}
RUNTIME_RECENT_TASKS: list[dict[str, Any]] = []
RUNTIME_AUX_PREWARM_LOCK = threading.Lock()
RUNTIME_AUX_PREWARM_THREAD_RUNNING = False
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
        "max_tokens": 16384,
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
        "max_tokens": 16384,
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
    localmathrag_embedding_purpose: str | None = None
    localmathrag_strong_embedding: bool | str | None = None


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


class RuntimeProbeRequest(BaseModel):
    pair: str = "chat_embedding"
    timeout_seconds: float | None = None
    persist: bool = True


class RuntimeStopRequest(BaseModel):
    kind: str


class RuntimeSwitchModelRequest(BaseModel):
    kind: str = "chat"
    model: str
    timeout_seconds: float | None = None
    start: bool = True
    force: bool = False


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
    if endpoint_key == "chat":
        return LLAMA_RUNTIME_BASE_URL
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
        "chat": _optional_runtime_endpoint_status("chat", timeout),
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


def _docker_inspect_container(container_id: str) -> dict[str, Any]:
    status, data = _docker_api("GET", f"/containers/{urllib.parse.quote(container_id, safe='')}/json")
    if status != 200 or not isinstance(data, dict):
        raise RuntimeError(f"Docker inspect failed with status {status}: {data}")
    return data


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


def _docker_remove_container(container_id: str, force: bool = True) -> dict[str, Any]:
    force_flag = "1" if force else "0"
    status, data = _docker_api("DELETE", f"/containers/{urllib.parse.quote(container_id, safe='')}?force={force_flag}&v=1")
    if status in {204, 404}:
        return {"removed": status == 204, "status_code": status}
    raise RuntimeError(f"Docker remove failed with status {status}: {data}")


def _docker_container_logs(container_id: str, tail: int = 80) -> str:
    status, data = _docker_api("GET", f"/containers/{urllib.parse.quote(container_id, safe='')}/logs?stdout=1&stderr=1&tail={tail}&timestamps=1")
    if status != 200:
        return ""
    return data if isinstance(data, str) else ""


def _container_name_from_inspect(inspected: dict[str, Any], fallback: str) -> str:
    name = inspected.get("Name")
    if isinstance(name, str) and name.strip("/"):
        return name.strip("/")
    return fallback


def _copy_dict_keys(source: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    copied = {}
    for key in keys:
        value = source.get(key)
        if value is not None:
            copied[key] = value
    return copied


def _docker_create_container_from_inspect(inspected: dict[str, Any], name: str, cmd: list[str], env: list[str]) -> str:
    config = inspected.get("Config") if isinstance(inspected.get("Config"), dict) else {}
    host_config = inspected.get("HostConfig") if isinstance(inspected.get("HostConfig"), dict) else {}
    network_settings = inspected.get("NetworkSettings") if isinstance(inspected.get("NetworkSettings"), dict) else {}
    networks = network_settings.get("Networks") if isinstance(network_settings.get("Networks"), dict) else {}

    create_payload: dict[str, Any] = {
        "Image": config.get("Image"),
        "Cmd": cmd,
        "Env": env,
        "Entrypoint": config.get("Entrypoint"),
        "Labels": config.get("Labels") or {},
        "ExposedPorts": config.get("ExposedPorts") or {},
        "Volumes": config.get("Volumes") or {},
        "Healthcheck": config.get("Healthcheck"),
        "Tty": bool(config.get("Tty")),
        "OpenStdin": bool(config.get("OpenStdin")),
        "StdinOnce": bool(config.get("StdinOnce")),
        "AttachStdin": bool(config.get("AttachStdin")),
        "AttachStdout": bool(config.get("AttachStdout")),
        "AttachStderr": bool(config.get("AttachStderr")),
        "WorkingDir": config.get("WorkingDir") or "",
        "User": config.get("User") or "",
        "StopSignal": config.get("StopSignal"),
        "StopTimeout": config.get("StopTimeout"),
    }
    create_payload = {key: value for key, value in create_payload.items() if value not in (None, {}, [])}

    host_keys = [
        "AutoRemove",
        "Binds",
        "CapAdd",
        "CapDrop",
        "CgroupnsMode",
        "CpuShares",
        "CpusetCpus",
        "DeviceRequests",
        "Devices",
        "Dns",
        "DnsOptions",
        "DnsSearch",
        "ExtraHosts",
        "GroupAdd",
        "IpcMode",
        "LogConfig",
        "Memory",
        "MemoryReservation",
        "NanoCpus",
        "NetworkMode",
        "OomKillDisable",
        "PortBindings",
        "Privileged",
        "PublishAllPorts",
        "ReadonlyRootfs",
        "RestartPolicy",
        "Runtime",
        "SecurityOpt",
        "ShmSize",
        "Ulimits",
        "UsernsMode",
        "VolumeDriver",
        "VolumesFrom",
    ]
    copied_host_config = _copy_dict_keys(host_config, host_keys)
    if copied_host_config:
        create_payload["HostConfig"] = copied_host_config

    endpoints: dict[str, Any] = {}
    for network_name, network in networks.items():
        if not isinstance(network, dict):
            continue
        endpoint = _copy_dict_keys(network, ["Aliases", "DriverOpts", "IPAMConfig", "Links", "MacAddress"])
        endpoints[network_name] = endpoint
    if endpoints:
        create_payload["NetworkingConfig"] = {"EndpointsConfig": endpoints}

    body = json.dumps(create_payload).encode("utf-8")
    status, data = _docker_api("POST", f"/containers/create?name={urllib.parse.quote(name, safe='')}", body)
    if status != 201 or not isinstance(data, dict) or not data.get("Id"):
        raise RuntimeError(f"Docker create failed with status {status}: {data}")
    return str(data["Id"])


def _has_local_model_type(model_type: str) -> bool:
    return any((_model_payload(path).get("model_type") or ["chat"])[0] == model_type for path in _model_entries())


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


def _rerank_model_name() -> str:
    return os.environ.get("LOCALMATHRAG_RERANK_MODEL") or RERANK_MODEL_NAME


def _runtime_startup_stall_timeout_seconds(kind: str) -> float:
    normalized = kind.upper()
    return _env_float(f"LOCALMATHRAG_{normalized}_STARTUP_STALL_TIMEOUT_SECONDS", _env_float("LOCALMATHRAG_RUNTIME_STARTUP_STALL_TIMEOUT_SECONDS", 120.0))


def _load_runtime_config() -> dict[str, Any]:
    return _read_runtime_config_unlocked()


def _write_runtime_config(config: dict[str, Any]) -> None:
    RUNTIME_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp_path = RUNTIME_CONFIG_FILE.with_suffix(RUNTIME_CONFIG_FILE.suffix + ".tmp")
    temp_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(RUNTIME_CONFIG_FILE)


def _rerank_runtime_config() -> dict[str, Any] | None:
    config = _load_runtime_config()
    rerank = config.get("rerank")
    if not isinstance(rerank, dict):
        return None
    model = str(rerank.get("model") or "")
    if model and model != _rerank_model_name():
        return None
    profile = str(rerank.get("profile") or "")
    if profile and RERANK_PROFILE and profile != RERANK_PROFILE and not (rerank.get("disabled") and RERANK_PROFILE in {"none", "off", "disabled"}):
        return None
    return dict(rerank)


def _persist_rerank_runtime_config(
    *,
    max_batch_tokens: int | None = None,
    disabled: bool | None = None,
    reason: str = "",
    source: str = "",
    failure_count: int | None = None,
) -> dict[str, Any]:
    with RUNTIME_CONFIG_LOCK:
        config = _load_runtime_config()
        existing = config.get("rerank") if isinstance(config.get("rerank"), dict) else {}
        existing_tokens = existing.get("max_batch_tokens") if isinstance(existing, dict) else None
        tokens = max_batch_tokens if max_batch_tokens is not None else existing_tokens
        if tokens is None:
            tokens = _env_int("LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS", 16384)
        payload: dict[str, Any] = {
            "model": _rerank_model_name(),
            "profile": RERANK_PROFILE,
            "max_batch_tokens": int(tokens),
            "disabled": bool(disabled) if disabled is not None else bool(existing.get("disabled", False)),
            "reason": reason or str(existing.get("reason") or ""),
            "source": source or str(existing.get("source") or "runtime"),
            "updated_at": time.time(),
        }
        if failure_count is not None:
            payload["failure_count"] = failure_count
        config["version"] = 1
        config["updated_at"] = time.time()
        config["rerank"] = payload
        _write_runtime_config(config)
        return payload


def _persist_optional_runtime_disabled_config(
    kind: str,
    *,
    disabled: bool,
    reason: str = "",
    source: str = "",
    failure_count: int | None = None,
) -> dict[str, Any] | None:
    if kind not in PERSISTENT_OPTIONAL_RUNTIME_KINDS:
        return None
    with RUNTIME_CONFIG_LOCK:
        config = _load_runtime_config()
        disabled_runtimes = config.get("disabled_runtimes")
        if not isinstance(disabled_runtimes, dict):
            disabled_runtimes = {}
        if not disabled:
            disabled_runtimes.pop(kind, None)
            config["disabled_runtimes"] = disabled_runtimes
            config["version"] = 1
            config["updated_at"] = time.time()
            _write_runtime_config(config)
            return None
        payload: dict[str, Any] = {
            "kind": kind,
            "disabled": True,
            "reason": reason,
            "source": source or "runtime-disabled",
            "updated_at": time.time(),
        }
        if failure_count is not None:
            payload["failure_count"] = failure_count
        disabled_runtimes[kind] = payload
        config["disabled_runtimes"] = disabled_runtimes
        config["version"] = 1
        config["updated_at"] = time.time()
        _write_runtime_config(config)
        return payload


def _optional_runtime_disabled_config(kind: str) -> dict[str, Any] | None:
    if kind not in PERSISTENT_OPTIONAL_RUNTIME_KINDS:
        return None
    config = _load_runtime_config()
    disabled_runtimes = config.get("disabled_runtimes")
    if not isinstance(disabled_runtimes, dict):
        return None
    payload = disabled_runtimes.get(kind)
    if not isinstance(payload, dict) or not payload.get("disabled"):
        return None
    return dict(payload)


def _runtime_scheduler_policy_snapshot() -> dict[str, Any]:
    policy = dict(SCHEDULER_POLICY)
    policy["effective_max_active_optional"] = OPTIONAL_RUNTIME_MAX_ACTIVE
    policy["effective_source"] = OPTIONAL_RUNTIME_MAX_ACTIVE_SOURCE
    policy["host_fingerprint"] = _runtime_host_fingerprint()
    policy["manual_override"] = policy.get("mode") == "manual"
    policy["embedding_degrade_small_requests_when_chat_ready"] = EMBEDDING_DEGRADE_SMALL_REQUESTS_WHEN_CHAT_READY
    policy["embedding_small_request_max_inputs"] = EMBEDDING_SMALL_REQUEST_MAX_INPUTS
    policy["embedding_small_request_max_tokens"] = EMBEDDING_SMALL_REQUEST_MAX_TOKENS
    policy["no_cold_start_in_request"] = NO_COLD_START_IN_REQUEST
    policy["auxiliary_prewarm_enabled"] = AUXILIARY_PREWARM_ENABLED
    policy["auxiliary_prewarm_idle_seconds"] = AUXILIARY_PREWARM_IDLE_SECONDS
    policy["recent_task_window_seconds"] = RECENT_TASK_WINDOW_SECONDS
    policy["recent_task_counts"] = _recent_runtime_task_counts()
    with RUNTIME_REQUEST_ACTIVITY_LOCK:
        policy["active_requests"] = dict(RUNTIME_ACTIVE_REQUESTS)
    return policy


def _apply_scheduler_runtime_policy(policy: dict[str, Any]) -> None:
    global SCHEDULER_POLICY
    global OPTIONAL_RUNTIME_MAX_ACTIVE
    global OPTIONAL_RUNTIME_MAX_ACTIVE_SOURCE
    global EMBEDDING_DEGRADE_SMALL_REQUESTS_WHEN_CHAT_READY
    global EMBEDDING_SMALL_REQUEST_MAX_INPUTS
    global EMBEDDING_SMALL_REQUEST_MAX_TOKENS
    if SCHEDULER_POLICY.get("mode") == "manual":
        return
    normalized = _normalize_scheduler_policy(policy, mode="auto", source=str(policy.get("source") or "runtime-config"))
    SCHEDULER_POLICY = normalized
    OPTIONAL_RUNTIME_MAX_ACTIVE = int(normalized["max_active_optional"])
    OPTIONAL_RUNTIME_MAX_ACTIVE_SOURCE = str(normalized.get("source") or "runtime-config")
    EMBEDDING_DEGRADE_SMALL_REQUESTS_WHEN_CHAT_READY = _scheduler_policy_bool(
        normalized,
        "LOCALMATHRAG_EMBEDDING_DEGRADE_SMALL_REQUESTS_WHEN_CHAT_READY",
        "degrade_small_embedding_when_chat_ready",
        OPTIONAL_RUNTIME_MAX_ACTIVE <= 1,
    )
    EMBEDDING_SMALL_REQUEST_MAX_INPUTS = max(
        1,
        _scheduler_policy_int(
            normalized,
            "LOCALMATHRAG_EMBEDDING_SMALL_REQUEST_MAX_INPUTS",
            "small_embedding_max_inputs",
            2,
        ),
    )
    EMBEDDING_SMALL_REQUEST_MAX_TOKENS = max(
        1,
        _scheduler_policy_int(
            normalized,
            "LOCALMATHRAG_EMBEDDING_SMALL_REQUEST_MAX_TOKENS",
            "small_embedding_max_tokens",
            512,
        ),
    )


def _persist_scheduler_runtime_config(
    *,
    max_active_optional: int,
    allow_chat_embedding_concurrency: bool,
    reason: str,
    source: str,
    probe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    with RUNTIME_CONFIG_LOCK:
        config = _load_runtime_config()
        existing = config.get("scheduler") if isinstance(config.get("scheduler"), dict) else {}
        payload: dict[str, Any] = {
            "version": 1,
            "mode": "auto",
            "source": source,
            "max_active_optional": max(1, int(max_active_optional)),
            "allow_chat_embedding_concurrency": bool(allow_chat_embedding_concurrency),
            "degrade_small_embedding_when_chat_ready": not bool(allow_chat_embedding_concurrency),
            "small_embedding_max_inputs": _coerce_int(existing.get("small_embedding_max_inputs"), EMBEDDING_SMALL_REQUEST_MAX_INPUTS, 1),
            "small_embedding_max_tokens": _coerce_int(existing.get("small_embedding_max_tokens"), EMBEDDING_SMALL_REQUEST_MAX_TOKENS, 1),
            "reason": reason,
            "host_fingerprint": _runtime_host_fingerprint(),
            "updated_at": time.time(),
        }
        if probe:
            payload["probe"] = probe
        config["version"] = 1
        config["updated_at"] = time.time()
        config["scheduler"] = payload
        _write_runtime_config(config)
    _apply_scheduler_runtime_policy(payload)
    return payload


def _runtime_config_disabled_payload(kind: str) -> dict[str, Any] | None:
    optional_config = _optional_runtime_disabled_config(kind)
    if optional_config:
        reason = str(optional_config.get("reason") or f"{kind} runtime is disabled by persisted runtime config")
        payload = {
            "kind": kind,
            "disabled": True,
            "reason": reason,
            "message": _runtime_degraded_message(kind, reason),
            "recorded_at": optional_config.get("updated_at", time.time()),
            "source": "runtime-config",
            "runtime_config": optional_config,
        }
        if optional_config.get("failure_count") is not None:
            payload["failure_count"] = optional_config.get("failure_count")
        return payload
    if kind != "rerank":
        return None
    config = _rerank_runtime_config()
    if not config or not config.get("disabled"):
        return None
    reason = str(config.get("reason") or "rerank runtime is disabled by persisted runtime config")
    payload = {
        "kind": kind,
        "disabled": True,
        "reason": reason,
        "message": _runtime_degraded_message(kind, reason),
        "recorded_at": config.get("updated_at", time.time()),
        "source": "runtime-config",
        "runtime_config": config,
    }
    if config.get("failure_count") is not None:
        payload["failure_count"] = config.get("failure_count")
    if config.get("max_batch_tokens") is not None:
        payload["max_batch_tokens"] = config.get("max_batch_tokens")
    return payload


def _cached_runtime_start_failure(kind: str) -> dict[str, Any] | None:
    if _runtime_start_failure_cooldown_seconds(kind) <= 0:
        return None
    now = time.monotonic()
    with RUNTIME_START_FAILURE_LOCK:
        failure = RUNTIME_START_FAILURES.get(kind)
        if not failure:
            return None
        if failure["expires_at"] <= now:
            RUNTIME_START_FAILURES.pop(kind, None)
            return None
        cached = dict(failure)
    cached["cooldown_seconds_remaining"] = round(max(0.0, cached["expires_at"] - now), 1)
    cached.pop("expires_at", None)
    return cached


def _disable_runtime(kind: str, reason: str, failure_count: int | None = None) -> dict[str, Any]:
    runtime_config = None
    if kind in PERSISTENT_OPTIONAL_RUNTIME_KINDS:
        runtime_config = _persist_optional_runtime_disabled_config(
            kind,
            disabled=True,
            reason=reason,
            source="runtime-disabled",
            failure_count=failure_count,
        )
    if kind == "rerank":
        rerank_config = _persist_rerank_runtime_config(
            disabled=True,
            reason=reason,
            source="runtime-disabled",
            failure_count=failure_count,
        )
        runtime_config = runtime_config or rerank_config
    payload = {
        "kind": kind,
        "disabled": True,
        "reason": reason,
        "message": _runtime_degraded_message(kind, reason),
        "recorded_at": time.time(),
    }
    if failure_count is not None:
        payload["failure_count"] = failure_count
    if runtime_config:
        payload["runtime_config"] = runtime_config
    with RUNTIME_DISABLED_LOCK:
        RUNTIME_DISABLED[kind] = payload
    return dict(payload)


def _runtime_disabled(kind: str) -> dict[str, Any] | None:
    with RUNTIME_DISABLED_LOCK:
        payload = RUNTIME_DISABLED.get(kind)
        if payload:
            return dict(payload)
    payload = _runtime_config_disabled_payload(kind)
    if payload:
        with RUNTIME_DISABLED_LOCK:
            RUNTIME_DISABLED[kind] = payload
        return dict(payload)
    return None


def _runtime_disabled_snapshot() -> dict[str, dict[str, Any]]:
    for kind in PERSISTENT_OPTIONAL_RUNTIME_KINDS:
        _runtime_disabled(kind)
    _runtime_disabled("rerank")
    with RUNTIME_DISABLED_LOCK:
        return {kind: dict(payload) for kind, payload in RUNTIME_DISABLED.items()}


def _clear_runtime_disabled(kind: str) -> None:
    with RUNTIME_DISABLED_LOCK:
        RUNTIME_DISABLED.pop(kind, None)
    if kind in PERSISTENT_OPTIONAL_RUNTIME_KINDS:
        _persist_optional_runtime_disabled_config(kind, disabled=False)
    if kind == "rerank":
        _persist_rerank_runtime_config(
            disabled=False,
            reason="runtime endpoint is ready",
            source="runtime-ready",
        )


def _runtime_start_failure_cooldown_seconds(kind: str) -> float:
    if kind == "embedding":
        return EMBEDDING_AUTOSTART_COOLDOWN_SECONDS
    return RUNTIME_START_FAILURE_COOLDOWN_SECONDS


def _runtime_disable_after_failures_enabled(kind: str) -> bool:
    if kind == "rerank":
        return RERANK_DISABLE_AFTER_FAILURES
    if kind in PERSISTENT_OPTIONAL_RUNTIME_KINDS:
        return OPTIONAL_RUNTIME_DISABLE_AFTER_FAILURES
    return False


def _runtime_start_max_failures(kind: str) -> int:
    if kind == "rerank":
        return RERANK_START_MAX_FAILURES
    if kind in PERSISTENT_OPTIONAL_RUNTIME_KINDS:
        return OPTIONAL_RUNTIME_START_MAX_FAILURES
    return 0


def _should_persistently_disable_runtime_after_failures(kind: str, failure_count: int) -> bool:
    max_failures = _runtime_start_max_failures(kind)
    return (
        _runtime_disable_after_failures_enabled(kind)
        and max_failures > 0
        and failure_count >= max_failures
    )


def _record_runtime_start_failure(kind: str, reason: str) -> dict[str, Any]:
    cooldown_seconds = _runtime_start_failure_cooldown_seconds(kind)
    failure = {
        "kind": kind,
        "reason": reason,
        "recorded_at": time.time(),
        "cooldown_seconds": cooldown_seconds,
        "expires_at": time.monotonic() + cooldown_seconds,
    }
    with RUNTIME_START_FAILURE_LOCK:
        failure_count = RUNTIME_START_FAILURE_COUNTS.get(kind, 0) + 1
        RUNTIME_START_FAILURE_COUNTS[kind] = failure_count
        failure["failure_count"] = failure_count
        if cooldown_seconds > 0:
            RUNTIME_START_FAILURES[kind] = failure
    if _should_persistently_disable_runtime_after_failures(kind, failure_count):
        failure["runtime_disabled"] = _disable_runtime(
            kind,
            f"{kind} runtime disabled after {failure_count} failed starts: {reason}",
            failure_count,
        )
    cached = dict(failure)
    cached.pop("expires_at", None)
    return cached


def _clear_runtime_start_failure(kind: str) -> None:
    with RUNTIME_START_FAILURE_LOCK:
        RUNTIME_START_FAILURES.pop(kind, None)
        RUNTIME_START_FAILURE_COUNTS.pop(kind, None)
    _clear_runtime_disabled(kind)


def _runtime_degraded_message(kind: str, reason: str) -> str:
    if kind == "rerank":
        return f"Rerank model is configured but the local rerank runtime was degraded; using ES/KNN scoring instead. {reason}"
    if kind == "embedding":
        return f"Embedding runtime was degraded; using lexical fallback embeddings. {reason}"
    return f"Local {kind} runtime was degraded. {reason}"


def _record_runtime_degradation(
    kind: str,
    reason: str,
    actions: list[dict[str, Any]] | None = None,
    resource_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "kind": kind,
        "degraded": True,
        "reason": reason,
        "message": _runtime_degraded_message(kind, reason),
        "recorded_at": time.time(),
    }
    if actions:
        payload["actions"] = actions
    if resource_status:
        payload["resource_status"] = resource_status
    with RUNTIME_DEGRADATION_LOCK:
        RUNTIME_DEGRADATIONS[kind] = payload
    return payload


def _last_runtime_degradation(kind: str) -> dict[str, Any] | None:
    with RUNTIME_DEGRADATION_LOCK:
        degradation = RUNTIME_DEGRADATIONS.get(kind)
        return dict(degradation) if degradation else None


def _clear_runtime_degradation(kind: str) -> None:
    with RUNTIME_DEGRADATION_LOCK:
        RUNTIME_DEGRADATIONS.pop(kind, None)


def _runtime_degradation_snapshot() -> dict[str, dict[str, Any]]:
    with RUNTIME_DEGRADATION_LOCK:
        return {kind: dict(payload) for kind, payload in RUNTIME_DEGRADATIONS.items()}


def _runtime_background_start_enabled(kind: str) -> bool:
    return (kind == "rerank" and RERANK_BACKGROUND_START) or (kind == "embedding" and EMBEDDING_BACKGROUND_PREWARM)


def _active_runtime_request_count() -> int:
    with RUNTIME_REQUEST_ACTIVITY_LOCK:
        return sum(RUNTIME_ACTIVE_REQUESTS.values())


def _persist_recent_runtime_task(kind: str, request_kind: str, quality: bool) -> None:
    try:
        with RUNTIME_CONFIG_LOCK:
            config = _load_runtime_config()
            recent = config.get("recent_tasks")
            if not isinstance(recent, dict):
                recent = {}
            payload = recent.get(kind)
            if not isinstance(payload, dict):
                payload = {"kind": kind, "observed_count": 0, "quality_count": 0}
            payload["kind"] = kind
            payload["last_request_kind"] = request_kind
            payload["last_seen"] = time.time()
            payload["observed_count"] = int(payload.get("observed_count") or 0) + 1
            if quality:
                payload["quality_count"] = int(payload.get("quality_count") or 0) + 1
            recent[kind] = payload
            config["recent_tasks"] = recent
            config["version"] = 1
            config["updated_at"] = time.time()
            _write_runtime_config(config)
    except Exception:
        logging.debug("Failed to persist recent runtime task observation", exc_info=True)


def _persist_runtime_prewarm_result(kind: str, result: dict[str, Any], reason: str) -> None:
    try:
        with RUNTIME_CONFIG_LOCK:
            config = _load_runtime_config()
            prewarm = config.get("prewarm")
            if not isinstance(prewarm, dict):
                prewarm = {}
            prewarm[kind] = {
                "kind": kind,
                "ready": bool(result.get("ready")),
                "skipped": bool(result.get("skipped")),
                "reason": result.get("reason"),
                "trigger": reason,
                "updated_at": time.time(),
            }
            config["prewarm"] = prewarm
            config["version"] = 1
            config["updated_at"] = time.time()
            _write_runtime_config(config)
    except Exception:
        logging.debug("Failed to persist runtime prewarm result", exc_info=True)


def _begin_runtime_request(kind: str, request_kind: str, quality: bool = False) -> dict[str, Any]:
    normalized = _runtime_target(kind)["kind"]
    token = {
        "kind": normalized,
        "request_kind": request_kind,
        "quality": bool(quality),
        "started_at": time.time(),
    }
    now = time.time()
    with RUNTIME_REQUEST_ACTIVITY_LOCK:
        RUNTIME_ACTIVE_REQUESTS[normalized] = RUNTIME_ACTIVE_REQUESTS.get(normalized, 0) + 1
        RUNTIME_RECENT_TASKS.append({**token, "recorded_at": now})
        cutoff = now - RECENT_TASK_WINDOW_SECONDS
        del RUNTIME_RECENT_TASKS[:-200]
        RUNTIME_RECENT_TASKS[:] = [item for item in RUNTIME_RECENT_TASKS if item.get("recorded_at", 0) >= cutoff]
    _persist_recent_runtime_task(normalized, request_kind, bool(quality))
    return token


def _end_runtime_request(token: dict[str, Any]) -> None:
    kind = str(token.get("kind") or "")
    with RUNTIME_REQUEST_ACTIVITY_LOCK:
        current = RUNTIME_ACTIVE_REQUESTS.get(kind, 0)
        if current <= 1:
            RUNTIME_ACTIVE_REQUESTS.pop(kind, None)
        else:
            RUNTIME_ACTIVE_REQUESTS[kind] = current - 1
    _schedule_recent_task_prewarm(f"recent {token.get('request_kind') or kind} request completed")


@contextmanager
def _runtime_request_context(kind: str, request_kind: str, quality: bool = False):
    token = _begin_runtime_request(kind, request_kind, quality)
    try:
        yield token
    finally:
        _end_runtime_request(token)


def _stream_with_runtime_activity(token: dict[str, Any], iterator):
    try:
        for chunk in iterator:
            yield chunk
    finally:
        _end_runtime_request(token)


def _finish_runtime_request(token: dict[str, Any] | None) -> None:
    if token is not None:
        _end_runtime_request(token)


def _recent_runtime_task_counts() -> dict[str, dict[str, Any]]:
    now = time.time()
    cutoff = now - RECENT_TASK_WINDOW_SECONDS
    counts: dict[str, dict[str, Any]] = {}
    with RUNTIME_REQUEST_ACTIVITY_LOCK:
        recent = [item for item in RUNTIME_RECENT_TASKS if item.get("recorded_at", 0) >= cutoff]
    for item in recent:
        kind = str(item.get("kind") or "")
        if not kind:
            continue
        payload = counts.setdefault(kind, {"count": 0, "quality_count": 0, "last_seen": 0.0})
        payload["count"] += 1
        if item.get("quality"):
            payload["quality_count"] += 1
        payload["last_seen"] = max(float(payload["last_seen"]), float(item.get("recorded_at") or 0.0))
    persisted = _load_runtime_config().get("recent_tasks")
    if isinstance(persisted, dict):
        for kind, item in persisted.items():
            if not isinstance(item, dict):
                continue
            last_seen = float(item.get("last_seen") or 0.0)
            if last_seen < cutoff:
                continue
            payload = counts.setdefault(str(kind), {"count": 0, "quality_count": 0, "last_seen": 0.0})
            payload["count"] = max(int(payload["count"]), int(item.get("observed_count") or 0))
            payload["quality_count"] = max(int(payload["quality_count"]), int(item.get("quality_count") or 0))
            payload["last_seen"] = max(float(payload["last_seen"]), last_seen)
    return counts


def _recent_task_prewarm_candidates() -> list[str]:
    counts = _recent_runtime_task_counts()
    candidates: list[str] = []
    rerank_count = int((counts.get("rerank") or {}).get("count") or 0)
    embedding_count = int((counts.get("embedding") or {}).get("count") or 0)
    embedding_quality_count = int((counts.get("embedding") or {}).get("quality_count") or 0)

    if RERANK_BACKGROUND_PREWARM and rerank_count >= RERANK_PREWARM_RECENT_THRESHOLD:
        candidates.append("rerank")
    if EMBEDDING_BACKGROUND_PREWARM and (embedding_quality_count > 0 or embedding_count >= EMBEDDING_PREWARM_RECENT_THRESHOLD):
        candidates.append("embedding")
    return candidates


def _wait_for_runtime_request_idle() -> bool:
    if AUXILIARY_PREWARM_IDLE_SECONDS > 0:
        time.sleep(AUXILIARY_PREWARM_IDLE_SECONDS)
    deadline = time.monotonic() + AUXILIARY_PREWARM_MAX_WAIT_SECONDS
    while time.monotonic() <= deadline:
        if _active_runtime_request_count() == 0:
            return True
        time.sleep(1.0)
    return False


def _schedule_recent_task_prewarm(reason: str) -> None:
    global RUNTIME_AUX_PREWARM_THREAD_RUNNING
    if not (RUNTIME_LAZY_ENABLED and AUXILIARY_PREWARM_ENABLED):
        return
    if _active_runtime_request_count() > 0:
        return
    with RUNTIME_AUX_PREWARM_LOCK:
        if RUNTIME_AUX_PREWARM_THREAD_RUNNING:
            return
        RUNTIME_AUX_PREWARM_THREAD_RUNNING = True

    def run() -> None:
        global RUNTIME_AUX_PREWARM_THREAD_RUNNING
        try:
            if not _wait_for_runtime_request_idle():
                return
            for kind in _recent_task_prewarm_candidates():
                if _active_runtime_request_count() > 0:
                    return
                result = _prewarm_runtime(kind)
                _persist_runtime_prewarm_result(kind, result, reason)
                if result.get("ready"):
                    logging.info("LocalMathRAG prewarmed %s after recent tasks: %s", kind, reason)
        except Exception as exc:
            _record_runtime_degradation("scheduler", f"recent-task prewarm failed: {exc}")
        finally:
            with RUNTIME_AUX_PREWARM_LOCK:
                RUNTIME_AUX_PREWARM_THREAD_RUNNING = False

    thread = threading.Thread(target=run, name="localmathrag-recent-task-prewarm", daemon=True)
    thread.start()


def _runtime_background_ready_timeout_seconds(kind: str) -> float:
    normalized = kind.upper()
    return _env_float(
        f"LOCALMATHRAG_{normalized}_BACKGROUND_READY_TIMEOUT_SECONDS",
        _env_float("LOCALMATHRAG_RUNTIME_BACKGROUND_READY_TIMEOUT_SECONDS", 240.0),
    )


def _schedule_runtime_background_start(target: dict[str, Any], container_id: str) -> dict[str, Any]:
    kind = target["kind"]
    with RUNTIME_BACKGROUND_START_LOCK:
        if kind in RUNTIME_BACKGROUND_STARTS:
            return {"scheduled": False, "in_progress": True}
        RUNTIME_BACKGROUND_STARTS.add(kind)

    def run() -> None:
        try:
            if not _wait_for_runtime_request_idle():
                _record_runtime_degradation(kind, "background runtime start deferred because user tasks stayed active")
                return
            with RUNTIME_START_LOCK:
                if _active_runtime_request_count() > 0:
                    return
                prepared = _prepare_runtime_start(target)
                if not prepared["ok"]:
                    _record_runtime_degradation(
                        kind,
                        prepared["reason"],
                        prepared.get("actions"),
                        prepared.get("resource_status"),
                    )
                    return
                _docker_start_container(container_id)
            timeout_seconds = _runtime_background_ready_timeout_seconds(kind)
            status = _wait_for_endpoint(target["base_url"], timeout_seconds, container_id=container_id, kind=kind)
            if status.get("endpoint_ok"):
                _clear_runtime_start_failure(kind)
            else:
                reason = f"runtime did not become ready within {timeout_seconds:.0f}s"
                retry_result = _retry_runtime_with_lower_config(
                    target,
                    container_id,
                    reason,
                    timeout_seconds,
                    status,
                    prepared.get("resource_status", {}),
                    prepared.get("actions", []),
                )
                if retry_result is None:
                    _record_runtime_start_failure(kind, reason)
        except Exception as exc:
            _record_runtime_start_failure(kind, f"runtime background start failed: {exc}")
        finally:
            with RUNTIME_BACKGROUND_START_LOCK:
                RUNTIME_BACKGROUND_STARTS.discard(kind)

    thread = threading.Thread(target=run, name=f"localmathrag-{kind}-background-start", daemon=True)
    thread.start()
    return {"scheduled": True, "in_progress": True}


def _runtime_target(kind: str) -> dict[str, Any]:
    normalized = kind.lower()
    if normalized in {"chat", "llm", "llama", "llama-cpp"}:
        return {
            "kind": "chat",
            "model_type": "chat",
            "base_url": LLAMA_RUNTIME_BASE_URL,
            "container": LLAMA_CONTAINER_NAME,
            "compose_service": LLAMA_COMPOSE_SERVICE,
        }
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
    return [_runtime_target(kind) for kind in ("chat", "embedding", "rerank", "vision", "asr", "tts")]


def _runtime_preemption_order(kind: str) -> list[str]:
    if kind == "embedding":
        return ["rerank", "chat", "vision", "asr", "tts"]
    if kind == "chat":
        return ["embedding", "rerank", "vision", "asr", "tts"]
    if kind == "rerank":
        return ["embedding", "vision", "asr", "tts"]
    return ["embedding", "rerank", "chat", "vision", "asr", "tts"]


def _running_optional_runtime_states(exclude_kind: str | None = None) -> list[dict[str, Any]]:
    running = []
    for runtime in _optional_runtime_targets():
        if runtime["kind"] == exclude_kind:
            continue
        state = _docker_container_state(runtime["container"], runtime["compose_service"])
        if state.get("container_running"):
            running.append({"target": runtime, "state": state})
    return running


def _stop_runtime_for_scheduler(kind: str, reason: str) -> dict[str, Any] | None:
    target = _runtime_target(kind)
    return _stop_optional_runtime_for_degrade(target, reason)


def _prepare_quality_embedding_priority(target: dict[str, Any]) -> list[dict[str, Any]]:
    if target["kind"] != "embedding":
        return []

    actions: list[dict[str, Any]] = []
    cached_failure = _cached_runtime_start_failure("embedding")
    disabled = _runtime_disabled("embedding")
    degradation = _last_runtime_degradation("embedding")
    if cached_failure or disabled or degradation:
        _clear_runtime_start_failure("embedding")
        _clear_runtime_degradation("embedding")
        actions.append(
            {
                "kind": "embedding",
                "action": "quality_embedding_priority_reset",
                "reason": "high priority embedding requires the real embedding runtime; cleared previous embedding fallback state",
                "had_cached_failure": bool(cached_failure),
                "had_runtime_disabled": bool(disabled),
                "had_runtime_degradation": bool(degradation),
            }
        )

    if EMBEDDING_DOCUMENT_PREEMPT_CHAT and _has_local_model_type("chat"):
        action = _stop_runtime_for_scheduler(
            "chat",
            "high priority embedding request reserved resources",
        )
        if action:
            actions.append(action)
    return actions


def _balance_optional_runtimes(target: dict[str, Any]) -> list[dict[str, Any]]:
    actions = []
    stopped: set[str] = set()
    running = _running_optional_runtime_states(exclude_kind=target["kind"])
    running_kinds = {item["target"]["kind"] for item in running}
    running_count = len(running)

    for victim_kind in _runtime_preemption_order(target["kind"]):
        if running_count < OPTIONAL_RUNTIME_MAX_ACTIVE:
            break
        if victim_kind not in running_kinds:
            continue
        action = _stop_runtime_for_scheduler(
            victim_kind,
            f"scheduler reserved runtime slot for {target['kind']}",
        )
        if action:
            actions.append(action)
            stopped.add(victim_kind)
            if action.get("action") == "stopped":
                running_count -= 1

    resource_status = _runtime_resource_status(target["kind"])
    if resource_status["ok"]:
        return actions

    for victim_kind in _runtime_preemption_order(target["kind"]):
        if victim_kind in stopped or victim_kind not in running_kinds:
            continue
        action = _stop_runtime_for_scheduler(
            victim_kind,
            f"scheduler freed resources for {target['kind']}: {'; '.join(resource_status['reasons'])}",
        )
        if action:
            actions.append(action)
            stopped.add(victim_kind)
        resource_status = _runtime_resource_status(target["kind"])
        if resource_status["ok"]:
            break
    return actions


def _prepare_runtime_start(target: dict[str, Any]) -> dict[str, Any]:
    actions = _balance_optional_runtimes(target)
    running_count = len(_running_optional_runtime_states(exclude_kind=target["kind"]))
    if running_count >= OPTIONAL_RUNTIME_MAX_ACTIVE:
        return {
            "ok": False,
            "actions": actions,
            "resource_status": _runtime_resource_status(target["kind"]),
            "reason": f"no runtime slot available for {target['kind']} without stopping protected runtimes",
        }
    resource_status = _runtime_resource_status(target["kind"])
    if resource_status["ok"]:
        return {
            "ok": True,
            "actions": actions,
            "resource_status": resource_status,
        }
    return {
        "ok": False,
        "actions": actions,
        "resource_status": resource_status,
        "reason": "; ".join(resource_status["reasons"]),
    }


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
            "result": _docker_stop_container(container_id, OPTIONAL_RUNTIME_DEGRADE_STOP_TIMEOUT_SECONDS),
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
    last_status = _endpoint_status(base_url, timeout=RUNTIME_READY_PROBE_TIMEOUT_SECONDS)
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
        last_status = _endpoint_status(base_url, timeout=RUNTIME_READY_PROBE_TIMEOUT_SECONDS)
    last_status["startup_progress"] = last_progress
    if time.monotonic() >= stall_deadline:
        last_status["startup_stalled"] = True
        last_status["startup_stall_timeout_seconds"] = stall_timeout
    return last_status


def _replace_cmd_value(cmd: list[str], flag: str, value: Any) -> list[str]:
    updated = list(cmd)
    for index, token in enumerate(updated[:-1]):
        if token == flag:
            updated[index + 1] = str(value)
            return updated
    updated.extend([flag, str(value)])
    return updated


def _cmd_int_value(cmd: list[str], flag: str, default: int) -> int:
    for index, token in enumerate(cmd[:-1]):
        if token == flag:
            try:
                return int(cmd[index + 1])
            except (TypeError, ValueError):
                return default
    return default


def _replace_env_value(env: list[str], name: str, value: Any) -> list[str]:
    prefix = f"{name}="
    updated = [item for item in env if not item.startswith(prefix)]
    updated.append(f"{name}={value}")
    return updated


def _next_lower_rerank_context(current_tokens: int) -> int | None:
    if current_tokens <= RERANK_CONTEXT_MIN_TOKENS:
        return None
    lowered = max(RERANK_CONTEXT_MIN_TOKENS, current_tokens - RERANK_CONTEXT_STEP_TOKENS)
    if lowered >= current_tokens:
        return None
    return lowered


def _recreate_rerank_container_with_lower_context(target: dict[str, Any], container_id: str, reason: str) -> dict[str, Any] | None:
    inspected = _docker_inspect_container(container_id)
    config = inspected.get("Config") if isinstance(inspected.get("Config"), dict) else {}
    cmd = config.get("Cmd") if isinstance(config.get("Cmd"), list) else []
    env = config.get("Env") if isinstance(config.get("Env"), list) else []
    current_tokens = _cmd_int_value(cmd, "--max-batch-tokens", _env_int("LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS", 16384))
    lowered_tokens = _next_lower_rerank_context(current_tokens)
    if lowered_tokens is None:
        failure_count = RUNTIME_START_FAILURE_COUNTS.get(target["kind"], 0)
        disabled = _disable_runtime(
            target["kind"],
            f"rerank context is at minimum {RERANK_CONTEXT_MIN_TOKENS} and runtime still failed: {reason}",
            failure_count,
        )
        return {
            "kind": target["kind"],
            "action": "disabled",
            "reason": disabled["reason"],
            "runtime_disabled": disabled,
            "from_max_batch_tokens": current_tokens,
            "min_max_batch_tokens": RERANK_CONTEXT_MIN_TOKENS,
        }

    name = _container_name_from_inspect(inspected, target["container"])
    updated_cmd = _replace_cmd_value(cmd, "--max-batch-tokens", lowered_tokens)
    updated_env = _replace_env_value(env, "LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS", lowered_tokens)
    _docker_remove_container(container_id, force=True)
    new_container_id = _docker_create_container_from_inspect(inspected, name, updated_cmd, updated_env)
    runtime_config = _persist_rerank_runtime_config(
        max_batch_tokens=lowered_tokens,
        disabled=False,
        reason=reason,
        source="runtime-context-retry",
    )
    return {
        "kind": target["kind"],
        "action": "rerank_context_recreated",
        "reason": reason,
        "container_id": new_container_id[:12],
        "container_id_full": new_container_id,
        "from_max_batch_tokens": current_tokens,
        "to_max_batch_tokens": lowered_tokens,
        "min_max_batch_tokens": RERANK_CONTEXT_MIN_TOKENS,
        "runtime_config": runtime_config,
    }


def _retry_rerank_with_lower_context(
    target: dict[str, Any],
    container_id: str,
    reason: str,
    timeout_seconds: float,
    endpoint_status: dict[str, Any],
    resource_status: dict[str, Any],
    balance_actions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if target["kind"] != "rerank" or RERANK_CONTEXT_RECREATE_RETRIES <= 0:
        return None

    context_actions: list[dict[str, Any]] = []
    current_container_id = container_id
    last_reason = reason
    last_status = endpoint_status
    start_result: dict[str, Any] = {"started": False}

    for _ in range(RERANK_CONTEXT_RECREATE_RETRIES):
        failure = _record_runtime_start_failure(target["kind"], last_reason)
        disabled = failure.get("runtime_disabled") or _runtime_disabled(target["kind"])
        if disabled:
            return {
                "ready": False,
                "started": False,
                "target": target,
                "endpoint_status": last_status,
                "resource_status": resource_status,
                "reason": disabled["reason"],
                "container_id": current_container_id[:12],
                "runtime_start_failure": failure,
                "runtime_disabled": disabled,
                "balancer": {
                    "max_active_optional": OPTIONAL_RUNTIME_MAX_ACTIVE,
                    "actions": [*balance_actions, *context_actions],
                },
            }

        try:
            action = _recreate_rerank_container_with_lower_context(target, current_container_id, last_reason)
        except Exception as exc:
            failure = _record_runtime_start_failure(target["kind"], f"rerank context downgrade failed: {exc}")
            return {
                "ready": False,
                "started": False,
                "target": target,
                "endpoint_status": last_status,
                "resource_status": resource_status,
                "reason": failure["reason"],
                "container_id": current_container_id[:12],
                "runtime_start_failure": failure,
                "runtime_disabled": failure.get("runtime_disabled") or _runtime_disabled(target["kind"]),
                "balancer": {
                    "max_active_optional": OPTIONAL_RUNTIME_MAX_ACTIVE,
                    "actions": [*balance_actions, *context_actions],
                },
            }

        if not action:
            return None
        context_actions.append(action)
        if action.get("runtime_disabled"):
            disabled = action["runtime_disabled"]
            return {
                "ready": False,
                "started": False,
                "target": target,
                "endpoint_status": last_status,
                "resource_status": resource_status,
                "reason": disabled["reason"],
                "container_id": current_container_id[:12],
                "runtime_disabled": disabled,
                "balancer": {
                    "max_active_optional": OPTIONAL_RUNTIME_MAX_ACTIVE,
                    "actions": [*balance_actions, *context_actions],
                },
            }

        current_container_id = str(action["container_id_full"])
        try:
            start_result = _docker_start_container(current_container_id)
        except Exception as exc:
            last_reason = f"runtime start failed after lowering rerank context to {action['to_max_batch_tokens']}: {exc}"
            continue

        last_status = _wait_for_endpoint(target["base_url"], timeout_seconds, container_id=current_container_id, kind=target["kind"])
        if last_status.get("endpoint_ok"):
            _clear_runtime_start_failure(target["kind"])
            return {
                "ready": True,
                "started": bool(start_result.get("started")),
                "target": target,
                "endpoint_status": last_status,
                "resource_status": resource_status,
                "reason": None,
                "container_id": current_container_id[:12],
                "balancer": {
                    "max_active_optional": OPTIONAL_RUNTIME_MAX_ACTIVE,
                    "actions": [*balance_actions, *context_actions],
                },
            }
        last_reason = f"runtime did not become ready after lowering rerank context to {action['to_max_batch_tokens']} within {timeout_seconds:.0f}s"

    failure = _record_runtime_start_failure(
        target["kind"],
        f"{last_reason}; rerank context retry limit {RERANK_CONTEXT_RECREATE_RETRIES} reached",
    )
    return {
        "ready": False,
        "started": bool(start_result.get("started")),
        "target": target,
        "endpoint_status": last_status,
        "resource_status": resource_status,
        "reason": failure["reason"],
        "container_id": current_container_id[:12],
        "runtime_start_failure": failure,
        "runtime_disabled": failure.get("runtime_disabled") or _runtime_disabled(target["kind"]),
        "balancer": {
            "max_active_optional": OPTIONAL_RUNTIME_MAX_ACTIVE,
            "actions": [*balance_actions, *context_actions],
        },
    }


def _retry_runtime_with_lower_config(
    target: dict[str, Any],
    container_id: str,
    reason: str,
    timeout_seconds: float,
    endpoint_status: dict[str, Any],
    resource_status: dict[str, Any],
    balance_actions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if target["kind"] == "rerank":
        return _retry_rerank_with_lower_context(
            target,
            container_id,
            reason,
            timeout_seconds,
            endpoint_status,
            resource_status,
            balance_actions,
        )
    return None


def _switch_runtime_model(request: RuntimeSwitchModelRequest) -> dict[str, Any]:
    target = _runtime_target(request.kind)
    kind = target["kind"]
    payload = _find_local_model_for_runtime(kind, request.model)
    timeout_seconds = request.timeout_seconds if request.timeout_seconds is not None else _runtime_ready_timeout_seconds(kind)
    warnings: list[str] = []
    if kind == "embedding":
        warnings.append("Embedding model switched. Existing knowledge-base vectors should be reparsed or rebuilt before comparing semantic scores.")

    with RUNTIME_START_LOCK:
        container_id = _docker_find_container(target["container"], target["compose_service"])
        if not container_id:
            raise HTTPException(status_code=409, detail=f"container for {target['compose_service']} was not prepared")

        inspected = _docker_inspect_container(container_id)
        config = inspected.get("Config") if isinstance(inspected.get("Config"), dict) else {}
        cmd = config.get("Cmd") if isinstance(config.get("Cmd"), list) else []
        env = config.get("Env") if isinstance(config.get("Env"), list) else []
        name = _container_name_from_inspect(inspected, target["container"])
        updated_cmd = _runtime_command_for_model(kind, payload, cmd)
        updated_env = _runtime_env_for_model(kind, payload, env)

        actions: list[dict[str, Any]] = []
        _docker_remove_container(container_id, force=True)
        actions.append({"kind": kind, "action": "removed_old_container", "container_id": container_id[:12]})
        new_container_id = _docker_create_container_from_inspect(inspected, name, updated_cmd, updated_env)
        actions.append({"kind": kind, "action": "created_container", "container_id": new_container_id[:12]})

        _apply_runtime_model_to_process_env(kind, payload)
        selection = _persist_runtime_model_selection(kind, payload)
        if kind == "rerank":
            _persist_rerank_runtime_config(disabled=False, reason="runtime model switched", source="runtime-switch")

        with RUNTIME_START_FAILURE_LOCK:
            RUNTIME_START_FAILURES.pop(kind, None)
            RUNTIME_START_FAILURE_COUNTS.pop(kind, None)
        with RUNTIME_DISABLED_LOCK:
            RUNTIME_DISABLED.pop(kind, None)
        _clear_runtime_degradation(kind)

        endpoint_status = _endpoint_status(target["base_url"], timeout=0.25)
        started = False
        ready = False
        reason: str | None = None
        balancer_actions: list[dict[str, Any]] = []
        resource_status = _runtime_resource_status(kind)

        if request.start:
            prepared = _prepare_runtime_start(target)
            resource_status = prepared["resource_status"]
            balancer_actions = prepared["actions"]
            actions.extend(balancer_actions)
            if prepared["ok"] or request.force:
                try:
                    start_result = _docker_start_container(new_container_id)
                    started = bool(start_result.get("started"))
                    actions.append({"kind": kind, "action": "started_container", "container_id": new_container_id[:12], "started": started})
                    endpoint_status = _wait_for_endpoint(target["base_url"], timeout_seconds, container_id=new_container_id, kind=kind)
                    ready = bool(endpoint_status.get("endpoint_ok"))
                    if ready:
                        _clear_runtime_start_failure(kind)
                    else:
                        reason = f"runtime did not become ready within {timeout_seconds:.0f}s"
                        retry_result = _retry_runtime_with_lower_config(
                            target,
                            new_container_id,
                            reason,
                            timeout_seconds,
                            endpoint_status,
                            resource_status,
                            balancer_actions,
                        )
                        if retry_result is not None:
                            endpoint_status = retry_result.get("endpoint_status", endpoint_status)
                            ready = bool(retry_result.get("ready"))
                            started = bool(retry_result.get("started", started))
                            reason = retry_result.get("reason")
                            actions.extend((retry_result.get("balancer") or {}).get("actions") or [])
                        if not ready and retry_result is None:
                            _record_runtime_start_failure(kind, reason)
                except Exception as exc:
                    reason = f"runtime start failed after model switch: {exc}"
                    _record_runtime_start_failure(kind, reason)
            else:
                reason = prepared["reason"]
                _record_runtime_degradation(kind, reason, balancer_actions, resource_status)
        else:
            reason = "runtime model switched; start=false"

    return {
        "ok": ready if request.start else True,
        "kind": kind,
        "ready": ready,
        "started": started,
        "reason": reason,
        "model": {
            "name": payload.get("name"),
            "runtime_model_name": payload.get("runtime_model_name"),
            "model_type": payload.get("model_type"),
            "relative_path": payload.get("relative_path"),
        },
        "selection": selection,
        "container_id": new_container_id[:12],
        "endpoint_status": endpoint_status,
        "resource_status": resource_status,
        "warnings": warnings,
        "actions": actions,
        "runtime_config": _load_runtime_config(),
    }


def _ensure_runtime_ready(
    kind: str,
    timeout_seconds: float = 90.0,
    allow_background: bool = False,
    bypass_cached_failure: bool = False,
    bypass_runtime_disabled: bool = False,
    prefer_quality: bool = False,
) -> dict[str, Any]:
    target = _runtime_target(kind)
    cached_failure = _cached_runtime_start_failure(target["kind"])
    status_timeout = RUNTIME_START_FAILURE_PROBE_TIMEOUT_SECONDS if cached_failure else 1.5
    status = _optional_runtime_endpoint_status(target["kind"], timeout=status_timeout)
    if status.get("endpoint_ok"):
        _clear_runtime_start_failure(target["kind"])
        if prefer_quality:
            _clear_runtime_degradation(target["kind"])
        balance_actions = _balance_optional_runtimes(target)
        return {
            "ready": True,
            "started": False,
            "target": target,
            "endpoint_status": status,
            "balancer": {
                "max_active_optional": OPTIONAL_RUNTIME_MAX_ACTIVE,
                "actions": balance_actions,
            },
        }
    disabled = None if bypass_runtime_disabled else _runtime_disabled(target["kind"])
    if disabled:
        return {
            "ready": False,
            "started": False,
            "target": target,
            "endpoint_status": status,
            "reason": disabled["reason"],
            "runtime_disabled": disabled,
        }
    if not RUNTIME_LAZY_ENABLED:
        return {"ready": False, "started": False, "target": target, "endpoint_status": status, "reason": "lazy runtime startup is disabled"}
    if cached_failure and target["kind"] != "rerank" and not bypass_cached_failure:
        return {
            "ready": False,
            "started": False,
            "target": target,
            "endpoint_status": status,
            "reason": cached_failure["reason"],
            "runtime_start_failure": cached_failure,
        }
    container_id = _docker_find_container(target["container"], target["compose_service"])
    if not container_id:
        return {"ready": False, "started": False, "target": target, "endpoint_status": status, "reason": f"container for {target['compose_service']} was not prepared"}
    if allow_background and _runtime_background_start_enabled(target["kind"]):
        background_start = _schedule_runtime_background_start(target, container_id)
        reason = "runtime is loading in background" if status.get("container_running") else "runtime is starting in background"
        return {
            "ready": False,
            "started": False,
            "target": target,
            "endpoint_status": status,
            "reason": reason,
            "container_id": container_id[:12],
            "background_start": background_start,
        }
    with RUNTIME_START_LOCK:
        priority_actions = _prepare_quality_embedding_priority(target) if prefer_quality else []
        cached_failure = _cached_runtime_start_failure(target["kind"])
        status_timeout = RUNTIME_START_FAILURE_PROBE_TIMEOUT_SECONDS if cached_failure else 1.5
        status = _endpoint_status(target["base_url"], timeout=status_timeout)
        if status.get("endpoint_ok"):
            _clear_runtime_start_failure(target["kind"])
            if prefer_quality:
                _clear_runtime_degradation(target["kind"])
            balance_actions = _balance_optional_runtimes(target)
            return {
                "ready": True,
                "started": False,
                "target": target,
                "endpoint_status": status,
                "balancer": {
                    "max_active_optional": OPTIONAL_RUNTIME_MAX_ACTIVE,
                    "actions": [*priority_actions, *balance_actions],
                },
            }
        disabled = None if bypass_runtime_disabled else _runtime_disabled(target["kind"])
        if disabled:
            return {
                "ready": False,
                "started": False,
                "target": target,
                "endpoint_status": status,
                "reason": disabled["reason"],
                "container_id": container_id[:12],
                "runtime_disabled": disabled,
            }
        if cached_failure and target["kind"] != "rerank" and not bypass_cached_failure:
            return {
                "ready": False,
                "started": False,
                "target": target,
                "endpoint_status": status,
                "reason": cached_failure["reason"],
                "container_id": container_id[:12],
                "runtime_start_failure": cached_failure,
            }
        prepared = _prepare_runtime_start(target)
        resource_status = prepared["resource_status"]
        balance_actions = [*priority_actions, *prepared["actions"]]
        if not prepared["ok"]:
            if prefer_quality and target["kind"] == "embedding":
                balance_actions.append(
                    {
                        "kind": target["kind"],
                        "action": "quality_embedding_resource_override",
                        "reason": prepared["reason"],
                    }
                )
            else:
                degradation = _record_runtime_degradation(
                    target["kind"],
                    prepared["reason"],
                    balance_actions,
                    resource_status,
                )
                return {
                    "ready": False,
                    "started": False,
                    "target": target,
                    "endpoint_status": status,
                    "resource_status": resource_status,
                    "reason": prepared["reason"],
                    "runtime_degradation": degradation,
                    "balancer": {
                        "max_active_optional": OPTIONAL_RUNTIME_MAX_ACTIVE,
                        "actions": balance_actions,
                    },
                }
        if not prepared["ok"]:
            resource_status = {
                **resource_status,
                "quality_override": True,
                "quality_override_reason": prepared["reason"],
            }
        try:
            start_result = _docker_start_container(container_id)
        except Exception as exc:
            reason = f"runtime start failed: {exc}"
            retry_result = _retry_runtime_with_lower_config(
                target,
                container_id,
                reason,
                timeout_seconds,
                status,
                resource_status,
                balance_actions,
            )
            if retry_result is not None:
                return retry_result
            failure = _record_runtime_start_failure(target["kind"], reason)
            return {
                "ready": False,
                "started": False,
                "target": target,
                "endpoint_status": status,
                "resource_status": resource_status,
                "reason": reason,
                "container_id": container_id[:12],
                "runtime_start_failure": failure,
                "balancer": {
                    "max_active_optional": OPTIONAL_RUNTIME_MAX_ACTIVE,
                    "actions": balance_actions,
                },
            }
        status = _wait_for_endpoint(target["base_url"], timeout_seconds, container_id=container_id, kind=target["kind"])
        if start_result.get("started") and not status.get("endpoint_ok") and OPTIONAL_RUNTIME_STOP_ON_READY_TIMEOUT:
            degraded_stop = _stop_optional_runtime_for_degrade(target, f"runtime did not become ready within {timeout_seconds:.0f}s")
            if degraded_stop:
                balance_actions.append(degraded_stop)
        if status.get("endpoint_ok"):
            _clear_runtime_start_failure(target["kind"])
        else:
            reason = f"runtime did not become ready within {timeout_seconds:.0f}s"
            retry_result = _retry_runtime_with_lower_config(
                target,
                container_id,
                reason,
                timeout_seconds,
                status,
                resource_status,
                balance_actions,
            )
            if retry_result is not None:
                return retry_result
            _record_runtime_start_failure(target["kind"], reason)
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


def _wait_for_prepared_runtime(kind: str, timeout_seconds: float = 300.0) -> dict[str, Any]:
    target = _runtime_target(kind)
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    last_status: dict[str, Any] = {"target": target, "container_found": False}
    while time.monotonic() <= deadline:
        try:
            container_id = _docker_find_container(target["container"], target["compose_service"])
        except Exception as exc:
            last_status = {"target": target, "container_found": False, "reason": str(exc)}
            time.sleep(2.0)
            continue
        if container_id:
            return {"target": target, "container_found": True, "container_id": container_id[:12]}
        last_status = {
            "target": target,
            "container_found": False,
            "reason": f"container for {target['compose_service']} was not prepared",
        }
        time.sleep(2.0)
    return last_status


def _prewarm_runtime(kind: str) -> dict[str, Any]:
    target = _runtime_target(kind)
    disabled = _runtime_disabled(target["kind"])
    if disabled:
        return {"kind": target["kind"], "skipped": True, "reason": disabled["reason"], "runtime_disabled": disabled}
    if not _has_local_model_type(target["model_type"]):
        return {"kind": target["kind"], "skipped": True, "reason": f"no local {target['model_type']} model found"}
    prepared = _wait_for_prepared_runtime(target["kind"])
    if not prepared.get("container_found"):
        return {"kind": target["kind"], "skipped": True, "reason": prepared.get("reason", "runtime container was not prepared")}
    timeout_seconds = _runtime_background_ready_timeout_seconds(target["kind"])
    return _ensure_runtime_ready(target["kind"], timeout_seconds, allow_background=False)


def _schedule_chat_restore_after_embedding() -> None:
    global RUNTIME_CHAT_RESTORE_SEQUENCE, RUNTIME_CHAT_RESTORE_THREAD_RUNNING
    if not (RUNTIME_LAZY_ENABLED and CHAT_BACKGROUND_START and CHAT_RESTORE_AFTER_EMBEDDING):
        return
    if not _has_local_model_type("chat"):
        return
    with RUNTIME_CHAT_RESTORE_LOCK:
        RUNTIME_CHAT_RESTORE_SEQUENCE += 1
        if RUNTIME_CHAT_RESTORE_THREAD_RUNNING:
            return
        RUNTIME_CHAT_RESTORE_THREAD_RUNNING = True

    def run() -> None:
        global RUNTIME_CHAT_RESTORE_THREAD_RUNNING
        while True:
            with RUNTIME_CHAT_RESTORE_LOCK:
                sequence = RUNTIME_CHAT_RESTORE_SEQUENCE
            time.sleep(CHAT_RESTORE_AFTER_EMBEDDING_DELAY_SECONDS)
            with RUNTIME_CHAT_RESTORE_LOCK:
                if sequence == RUNTIME_CHAT_RESTORE_SEQUENCE:
                    RUNTIME_CHAT_RESTORE_THREAD_RUNNING = False
                    break
        try:
            _prewarm_runtime("chat")
        except Exception as exc:
            _record_runtime_start_failure("chat", f"chat restore after embedding failed: {exc}")

    thread = threading.Thread(target=run, name="localmathrag-chat-restore-after-embedding", daemon=True)
    thread.start()


def _scheduler_auto_probe_needed() -> bool:
    if not (RUNTIME_LAZY_ENABLED and SCHEDULER_AUTO_PROBE):
        return False
    if SCHEDULER_POLICY.get("mode") == "manual":
        return False
    if SCHEDULER_POLICY.get("source") != "adaptive-unprobed":
        return False
    return _has_local_model_type("chat") and _has_local_model_type("embedding")


def _run_scheduler_auto_probe() -> None:
    try:
        _probe_chat_embedding_runtime(
            RuntimeProbeRequest(
                pair="chat_embedding",
                timeout_seconds=SCHEDULER_AUTO_PROBE_TIMEOUT_SECONDS,
                persist=True,
            )
        )
    except Exception as exc:
        _persist_scheduler_runtime_config(
            max_active_optional=1,
            allow_chat_embedding_concurrency=False,
            reason=f"scheduler auto probe failed: {exc}",
            source="runtime-auto-probe",
            probe={
                "pair": "chat_embedding",
                "ok": False,
                "error": str(exc),
                "recorded_at": time.time(),
            },
        )


def _runtime_startup_scheduler() -> None:
    if not RUNTIME_LAZY_ENABLED:
        return
    time.sleep(CHAT_BACKGROUND_START_DELAY_SECONDS)
    if CHAT_BACKGROUND_START:
        _prewarm_runtime("chat")
    if RERANK_BACKGROUND_PREWARM:
        _prewarm_runtime("rerank")
    if _scheduler_auto_probe_needed():
        _run_scheduler_auto_probe()


@app.on_event("startup")
def _schedule_runtime_startup() -> None:
    if not CHAT_BACKGROUND_START and not RERANK_BACKGROUND_PREWARM:
        return
    thread = threading.Thread(target=_runtime_startup_scheduler, name="localmathrag-runtime-startup-scheduler", daemon=True)
    thread.start()


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


def _runtime_model_name_for_type(model_type: str) -> str:
    for path in _model_entries():
        payload = _model_identity_payload(path)
        types = payload.get("model_type") or ["chat"]
        if model_type in types:
            name = payload.get("runtime_model_name") or payload.get("name")
            if name:
                return str(name)
    if model_type == "chat" and os.environ.get("LOCALMATHRAG_GGUF_MODEL"):
        return f"/models/{os.environ['LOCALMATHRAG_GGUF_MODEL']}"
    if model_type == "embedding" and os.environ.get("LOCALMATHRAG_EMBEDDING_MODEL"):
        return os.environ["LOCALMATHRAG_EMBEDDING_MODEL"]
    return "bge-m3" if model_type == "embedding" else model_type


def _runtime_chat_model_name() -> str:
    model = os.environ.get("LOCALMATHRAG_GGUF_MODEL", "").strip()
    if model:
        if model.startswith("/models/"):
            return model
        model_name = model.replace("\\", "/").rsplit("/", 1)[-1]
        return f"/models/{model_name}"
    return _runtime_model_name_for_type("chat")


def _rough_chat_token_count(text: str) -> int:
    ascii_chars = 0
    non_ascii_chars = 0
    for char in text:
        if ord(char) < 128:
            ascii_chars += 1
        else:
            non_ascii_chars += 1
    return max(1, int(math.ceil(non_ascii_chars + ascii_chars / 4.0)))


def _runtime_tokenize_count(text: str) -> int:
    if not text:
        return 0
    try:
        body = json.dumps({"content": text, "add_special": False}, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{LLAMA_RUNTIME_BASE_URL.removesuffix('/v1')}/tokenize",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10.0) as response:
            data = json.loads(response.read().decode("utf-8"))
        tokens = data.get("tokens") if isinstance(data, dict) else None
        if isinstance(tokens, list):
            return len(tokens)
    except Exception:
        pass
    return _rough_chat_token_count(text)


def _message_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return ""


def _set_message_text_content(message: dict[str, Any], text: str) -> None:
    content = message.get("content")
    if isinstance(content, str):
        message["content"] = text
        return
    if isinstance(content, list):
        updated = []
        replaced = False
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str) and not replaced:
                copied = dict(item)
                copied["text"] = text
                updated.append(copied)
                replaced = True
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                copied = dict(item)
                copied["text"] = ""
                updated.append(copied)
            else:
                updated.append(item)
        message["content"] = updated


def _truncate_text_to_token_budget(text: str, max_tokens: int) -> str:
    if not text or max_tokens <= 0:
        return ""
    current_tokens = _runtime_tokenize_count(text)
    if current_tokens <= max_tokens:
        return text

    marker = "\n\n[LocalMathRAGFlow truncated retrieved context to fit the local model context window.]\n\n"
    minimum_chars = min(len(text), 256)
    target_chars = max(minimum_chars, int(len(text) * (max_tokens / max(1, current_tokens)) * 0.9))
    for _ in range(4):
        available = max(0, target_chars - len(marker))
        if available <= 0:
            candidate = text[:target_chars]
        else:
            head_chars = int(available * 0.72)
            tail_chars = available - head_chars
            candidate = f"{text[:head_chars]}{marker}{text[-tail_chars:] if tail_chars else ''}"
        if _runtime_tokenize_count(candidate) <= max_tokens:
            return candidate
        target_chars = max(minimum_chars, int(target_chars * 0.82))
    return candidate


def _chat_response_token_reserve(payload: dict[str, Any]) -> int:
    for key in ("max_tokens", "max_completion_tokens"):
        value = payload.get(key)
        if isinstance(value, int):
            return max(128, value)
        if isinstance(value, str) and value.isdigit():
            return max(128, int(value))
    return CHAT_CONTEXT_RESPONSE_RESERVE_TOKENS


def _chat_prompt_token_budget(payload: dict[str, Any], prompt_budget_tokens: int | None = None) -> int:
    if prompt_budget_tokens is not None:
        return max(1024, min(CHAT_CONTEXT_SIZE - 256, int(prompt_budget_tokens)))
    response_reserve = _chat_response_token_reserve(payload)
    hard_budget = CHAT_CONTEXT_SIZE - response_reserve - CHAT_CONTEXT_SAFETY_MARGIN_TOKENS
    ratio_budget = int(CHAT_CONTEXT_SIZE * CHAT_CONTEXT_PROMPT_BUDGET_RATIO)
    budget = min(hard_budget, ratio_budget)
    floor = min(CHAT_CONTEXT_MIN_PROMPT_TOKENS, max(1024, hard_budget))
    return max(1024, min(CHAT_CONTEXT_SIZE - 256, max(floor, budget)))


def _message_token_count(message: dict[str, Any]) -> int:
    return _runtime_tokenize_count(_message_text_content(message.get("content"))) + 8


def _fit_generation_payload_to_chat_context(payload: dict[str, Any], prompt_budget_tokens: int | None = None) -> dict[str, Any]:
    if not CHAT_CONTEXT_CLAMP_ENABLED:
        return payload
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return payload

    normalized_messages = [dict(message) if isinstance(message, dict) else message for message in messages]
    normalized = dict(payload)
    normalized["messages"] = normalized_messages
    budget = _chat_prompt_token_budget(normalized, prompt_budget_tokens)
    overhead = 64
    token_counts = [
        _message_token_count(message) if isinstance(message, dict) else 0
        for message in normalized_messages
    ]
    total_tokens = sum(token_counts) + overhead
    if total_tokens <= budget:
        return normalized
    original_total_tokens = total_tokens

    for _ in range(6):
        candidates = []
        for index, message in enumerate(normalized_messages):
            if not isinstance(message, dict):
                continue
            text = _message_text_content(message.get("content"))
            if not text:
                continue
            is_last_user = index == len(normalized_messages) - 1 and message.get("role") == "user"
            if is_last_user and len(normalized_messages) > 1:
                continue
            candidates.append((token_counts[index], index))
        if not candidates:
            break

        current_tokens, index = max(candidates)
        if current_tokens <= 256:
            break
        excess = max(1, total_tokens - budget)
        target_tokens = max(256, current_tokens - excess - 128)
        message = normalized_messages[index]
        original_text = _message_text_content(message.get("content"))
        truncated_text = _truncate_text_to_token_budget(original_text, target_tokens)
        if truncated_text == original_text:
            break
        _set_message_text_content(message, truncated_text)
        token_counts[index] = _message_token_count(message)
        total_tokens = sum(token_counts) + overhead
        if total_tokens <= budget:
            logging.info(
                "Clamped local chat prompt from approximately %s to %s tokens for ctx=%s budget=%s",
                original_total_tokens,
                total_tokens,
                CHAT_CONTEXT_SIZE,
                budget,
            )
            return normalized

    logging.warning(
        "Local chat prompt remains near context limit after clamp: tokens=%s budget=%s ctx=%s",
        total_tokens,
        budget,
        CHAT_CONTEXT_SIZE,
    )
    return normalized


def _generation_body_for_chat_runtime(payload: Any, prompt_budget_tokens: int | None = None) -> bytes:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="OpenAI generation request body must be a JSON object")
    normalized = _fit_generation_payload_to_chat_context(dict(payload), prompt_budget_tokens)
    normalized["model"] = _runtime_chat_model_name()
    return json.dumps(normalized, ensure_ascii=False).encode("utf-8")


def _scheduler_probe_response(
    *,
    ok: bool,
    pair: str,
    reason: str,
    persist: bool,
    timings: dict[str, float] | None = None,
    actions: list[dict[str, Any]] | None = None,
    resource_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    probe = {
        "pair": pair,
        "ok": ok,
        "reason": reason,
        "timings": timings or {},
        "actions": actions or [],
        "resource_status": resource_status or {},
        "recorded_at": time.time(),
    }
    if persist:
        policy = _persist_scheduler_runtime_config(
            max_active_optional=2 if ok else 1,
            allow_chat_embedding_concurrency=ok,
            reason=reason,
            source="runtime-probe",
            probe=probe,
        )
    else:
        policy = _runtime_scheduler_policy_snapshot()
    return {
        "pair": pair,
        "ok": ok,
        "reason": reason,
        "timings": timings or {},
        "actions": actions or [],
        "resource_status": resource_status or {},
        "scheduler_policy": policy,
    }


def _maybe_record_chat_embedding_concurrency_observed(timings: dict[str, float] | None = None) -> None:
    if SCHEDULER_POLICY.get("mode") == "manual":
        return
    if SCHEDULER_POLICY.get("allow_chat_embedding_concurrency") and OPTIONAL_RUNTIME_MAX_ACTIVE >= 2:
        return
    try:
        chat_status = _optional_runtime_endpoint_status("chat", timeout=0.1)
    except Exception:
        return
    if not chat_status.get("endpoint_ok"):
        return
    _persist_scheduler_runtime_config(
        max_active_optional=2,
        allow_chat_embedding_concurrency=True,
        reason="chat and embedding runtimes were observed ready during a successful embedding request",
        source="runtime-observed",
        probe={
            "pair": "chat_embedding",
            "ok": True,
            "kind": "passive-observation",
            "timings": timings or {},
            "recorded_at": time.time(),
        },
    )


def _reset_scheduler_runtime_policy(reason: str = "runtime policy reset") -> dict[str, Any]:
    payload = {
        "version": 1,
        "mode": "auto",
        "source": "adaptive-unprobed",
        "max_active_optional": 1,
        "allow_chat_embedding_concurrency": False,
        "degrade_small_embedding_when_chat_ready": True,
        "small_embedding_max_inputs": 2,
        "small_embedding_max_tokens": 512,
        "reason": reason,
        "host_fingerprint": _runtime_host_fingerprint(),
        "updated_at": time.time(),
    }
    _apply_scheduler_runtime_policy(payload)
    return payload


def _probe_chat_embedding_runtime(request: RuntimeProbeRequest) -> dict[str, Any]:
    timeout_seconds = max(1.0, request.timeout_seconds or _env_float("LOCALMATHRAG_RUNTIME_PROBE_TIMEOUT_SECONDS", 240.0))
    actions: list[dict[str, Any]] = []
    timings: dict[str, float] = {}

    if not _has_local_model_type("chat"):
        return _scheduler_probe_response(
            ok=False,
            pair=request.pair,
            reason="no local chat model found",
            persist=False,
            actions=actions,
        )
    if not _has_local_model_type("embedding"):
        return _scheduler_probe_response(
            ok=False,
            pair=request.pair,
            reason="no local embedding model found",
            persist=False,
            actions=actions,
        )

    chat_start = time.monotonic()
    chat_runtime = _ensure_runtime_ready("chat", timeout_seconds)
    timings["chat_ready_seconds"] = round(time.monotonic() - chat_start, 3)
    if not chat_runtime.get("ready"):
        return _scheduler_probe_response(
            ok=False,
            pair=request.pair,
            reason=str(chat_runtime.get("reason") or "chat runtime did not become ready"),
            persist=False,
            timings=timings,
            actions=actions,
            resource_status={"chat": chat_runtime.get("resource_status")},
        )

    embedding_target = _runtime_target("embedding")
    embedding_container_id = _docker_find_container(embedding_target["container"], embedding_target["compose_service"])
    if not embedding_container_id:
        return _scheduler_probe_response(
            ok=False,
            pair=request.pair,
            reason=f"container for {embedding_target['compose_service']} was not prepared",
            persist=False,
            timings=timings,
            actions=actions,
        )

    try:
        with RUNTIME_START_LOCK:
            embedding_status = _optional_runtime_endpoint_status("embedding", timeout=0.5)
            if not embedding_status.get("endpoint_ok"):
                start_result = _docker_start_container(embedding_container_id)
                actions.append(
                    {
                        "kind": "embedding",
                        "action": "started_for_probe",
                        "started": bool(start_result.get("started")),
                        "container_id": embedding_container_id[:12],
                    }
                )
        embedding_ready_start = time.monotonic()
        embedding_status = _wait_for_endpoint(
            embedding_target["base_url"],
            timeout_seconds,
            container_id=embedding_container_id,
            kind="embedding",
        )
        timings["embedding_ready_seconds"] = round(time.monotonic() - embedding_ready_start, 3)
        if not embedding_status.get("endpoint_ok"):
            degraded_stop = _stop_optional_runtime_for_degrade(embedding_target, "scheduler probe failed; keeping chat available")
            if degraded_stop:
                actions.append(degraded_stop)
            reason = f"embedding runtime did not become ready within {timeout_seconds:.0f}s while chat was running"
            _record_runtime_start_failure("embedding", reason)
            return _scheduler_probe_response(
                ok=False,
                pair=request.pair,
                reason=reason,
                persist=request.persist,
                timings=timings,
                actions=actions,
                resource_status={
                    "chat": _runtime_resource_status("chat"),
                    "embedding": _runtime_resource_status("embedding"),
                    "embedding_endpoint": embedding_status,
                },
            )

        embed_start = time.monotonic()
        _post_runtime_json(
            embedding_target["base_url"],
            "/embeddings",
            {
                "model": _runtime_model_name_for_type("embedding"),
                "input": ["LocalMathRAGFlow scheduler probe."],
            },
            timeout=min(timeout_seconds, 120.0),
        )
        timings["embedding_request_seconds"] = round(time.monotonic() - embed_start, 3)

        chat_request_start = time.monotonic()
        _post_runtime_json(
            chat_runtime["target"]["base_url"],
            "/chat/completions",
            {
                "model": _runtime_model_name_for_type("chat"),
                "messages": [{"role": "user", "content": "Reply with OK."}],
                "max_tokens": 4,
                "stream": False,
            },
            timeout=min(max(timeout_seconds, 30.0), CHAT_REQUEST_TIMEOUT_SECONDS),
        )
        timings["chat_request_seconds"] = round(time.monotonic() - chat_request_start, 3)
    except Exception as exc:
        degraded_stop = _stop_optional_runtime_for_degrade(embedding_target, f"scheduler probe failed; keeping chat available: {exc}")
        if degraded_stop:
            actions.append(degraded_stop)
        _record_runtime_start_failure("embedding", f"scheduler probe failed while chat was running: {exc}")
        return _scheduler_probe_response(
            ok=False,
            pair=request.pair,
            reason=str(exc),
            persist=request.persist,
            timings=timings,
            actions=actions,
            resource_status={
                "chat": _runtime_resource_status("chat"),
                "embedding": _runtime_resource_status("embedding"),
            },
        )

    return _scheduler_probe_response(
        ok=True,
        pair=request.pair,
        reason="chat and embedding runtimes completed live requests concurrently",
        persist=request.persist,
        timings=timings,
        actions=actions,
        resource_status={
            "chat": _runtime_resource_status("chat"),
            "embedding": _runtime_resource_status("embedding"),
        },
    )


def _runtime_proxy_request(base_url: str, route: str, body: bytes, timeout: float) -> urllib.request.Request:
    return urllib.request.Request(
        f"{base_url}{route}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )


def _runtime_exception_message(exc: Exception) -> tuple[str, int | None]:
    if isinstance(exc, urllib.error.HTTPError):
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        message = f"HTTP {exc.code}: {body or exc.reason}"
        return message, int(exc.code)
    return str(exc), None


def _is_recoverable_chat_runtime_error(message: str, status_code: int | None = None) -> bool:
    lowered = message.lower()
    recoverable_patterns = (
        "exceeds the available context size",
        "out of memory",
        "cuda out of memory",
        "failed to allocate",
        "kv cache",
        "no slot available",
        "connection refused",
        "connection reset",
        "remote end closed",
        "timed out",
        "temporarily unavailable",
    )
    if any(pattern in lowered for pattern in recoverable_patterns):
        return True
    return status_code in {408, 409, 429, 500, 502, 503, 504}


def _runtime_retry_prompt_budget() -> int:
    return max(1024, min(CHAT_CONTEXT_SIZE - 256, int(CHAT_CONTEXT_SIZE * CHAT_RUNTIME_RETRY_PROMPT_BUDGET_RATIO)))


def _recover_chat_runtime_for_retry(runtime: dict[str, Any], reason: str) -> dict[str, Any]:
    target = runtime.get("target") if isinstance(runtime.get("target"), dict) else _runtime_target("chat")
    actions = _balance_optional_runtimes(target)
    resource_status = _runtime_resource_status("chat")
    degradation = _record_runtime_degradation("chat", reason, actions, resource_status)

    status = _optional_runtime_endpoint_status("chat", timeout=RUNTIME_READY_PROBE_TIMEOUT_SECONDS)
    if not status.get("endpoint_ok"):
        try:
            state = _docker_container_state(target["container"], target["compose_service"])
            container_id = state.get("container_id")
            if container_id and not state.get("container_running"):
                actions.append(
                    {
                        "kind": "chat",
                        "action": "started_for_retry",
                        "container_id": str(container_id)[:12],
                        "reason": reason,
                        "result": _docker_start_container(str(container_id)),
                    }
                )
            ready = _ensure_runtime_ready("chat", _runtime_ready_timeout_seconds("chat"))
        except Exception as exc:
            ready = {
                "ready": False,
                "target": target,
                "reason": f"chat runtime recovery failed: {exc}",
                "balancer": {"actions": actions},
            }
    else:
        ready = {
            "ready": True,
            "target": target,
            "endpoint_status": status,
            "balancer": {"actions": actions},
        }

    ready.setdefault("balancer", {})
    ready["balancer"]["actions"] = [*actions, *ready.get("balancer", {}).get("actions", [])]
    ready["runtime_degradation"] = degradation
    ready["request_retry"] = {
        "enabled": True,
        "reason": reason,
        "prompt_budget_tokens": _runtime_retry_prompt_budget(),
    }
    return ready


def _stream_runtime_proxy(base_url: str, route: str, body: bytes, timeout: float):
    try:
        request = _runtime_proxy_request(base_url, route, body, timeout)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                yield chunk
    except Exception as exc:
        payload = json.dumps({"error": {"message": str(exc), "type": "localmathrag_runtime_error"}}, ensure_ascii=False)
        yield f"data: {payload}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"


def _stream_chat_runtime_proxy(runtime: dict[str, Any], route: str, body: bytes, retry_body: bytes, timeout: float):
    try:
        request = _runtime_proxy_request(runtime["target"]["base_url"], route, body, timeout)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                yield chunk
            return
    except Exception as exc:
        message, status_code = _runtime_exception_message(exc)
        if CHAT_RUNTIME_REQUEST_RETRIES <= 0 or not _is_recoverable_chat_runtime_error(message, status_code):
            payload = json.dumps({"error": {"message": message, "type": "localmathrag_runtime_error"}}, ensure_ascii=False)
            yield f"data: {payload}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"
            return

    retry_runtime = _recover_chat_runtime_for_retry(runtime, message)
    if not retry_runtime.get("ready"):
        payload = json.dumps(
            {
                "error": {
                    "message": retry_runtime.get("reason", message),
                    "type": "localmathrag_runtime_recovery_failed",
                    "runtime": retry_runtime,
                }
            },
            ensure_ascii=False,
        )
        yield f"data: {payload}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"
        return
    try:
        request = _runtime_proxy_request(retry_runtime["target"]["base_url"], route, retry_body, timeout)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                yield chunk
    except Exception as retry_exc:
        retry_message, _ = _runtime_exception_message(retry_exc)
        payload = json.dumps(
            {
                "error": {
                    "message": retry_message,
                    "type": "localmathrag_runtime_retry_error",
                    "runtime": retry_runtime,
                }
            },
            ensure_ascii=False,
        )
        yield f"data: {payload}\n\n".encode("utf-8")
    yield b"data: [DONE]\n\n"


def _json_runtime_proxy(base_url: str, route: str, body: bytes, timeout: float) -> Response:
    request = _runtime_proxy_request(base_url, route, body, timeout)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read()
        content_type = response.headers.get("Content-Type") or "application/json"
        return Response(content=data, status_code=response.status, media_type=content_type.split(";", 1)[0])


def _json_chat_runtime_proxy(runtime: dict[str, Any], route: str, body: bytes, retry_body: bytes, timeout: float) -> Response:
    try:
        return _json_runtime_proxy(runtime["target"]["base_url"], route, body, timeout)
    except Exception as exc:
        message, status_code = _runtime_exception_message(exc)
        if CHAT_RUNTIME_REQUEST_RETRIES <= 0 or not _is_recoverable_chat_runtime_error(message, status_code):
            raise
        retry_runtime = _recover_chat_runtime_for_retry(runtime, message)
        if not retry_runtime.get("ready"):
            raise RuntimeError(retry_runtime.get("reason", message)) from exc
        response = _json_runtime_proxy(retry_runtime["target"]["base_url"], route, retry_body, timeout)
        response.headers["X-LocalMathRAG-Runtime-Retry"] = "1"
        return response


def _runtime_route_base_url(base_url: str, route: str) -> str:
    if route == "/rerank":
        return base_url.removesuffix("/v1")
    return base_url


def _fallback_rerank_response(request: RerankRequest, reason: str, degraded_stop: dict[str, Any] | None = None) -> dict[str, Any]:
    degradation = _record_runtime_degradation("rerank", reason, [degraded_stop] if degraded_stop else None)
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
    runtime = {
        "kind": "rerank",
        "fallback": True,
        "reason": reason,
        "degraded": True,
        "warning": degradation["message"],
        "degradation": degradation,
    }
    disabled = _runtime_disabled("rerank")
    if disabled:
        runtime["runtime_disabled"] = disabled
    if degraded_stop:
        runtime["degraded_stop"] = degraded_stop
    return {
        "model": request.model or LOCAL_RERANK_MODEL,
        "results": results,
        "usage": {
            "total_tokens": len(_embedding_tokens(request.query)) + sum(len(_embedding_tokens(_document_text(document))) for document in request.documents),
        },
        "runtime": runtime,
    }


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


def _should_degrade_small_embedding_for_chat(inputs: list[str], token_count: int) -> str | None:
    if not EMBEDDING_DEGRADE_SMALL_REQUESTS_WHEN_CHAT_READY:
        return None
    if OPTIONAL_RUNTIME_MAX_ACTIVE > 1:
        return None
    if len(inputs) > EMBEDDING_SMALL_REQUEST_MAX_INPUTS or token_count > EMBEDDING_SMALL_REQUEST_MAX_TOKENS:
        return None
    if not (CHAT_BACKGROUND_START and _has_local_model_type("chat")):
        return None
    return (
        "small embedding request used lexical fallback to keep the chat runtime available "
        f"on low VRAM (inputs={len(inputs)}, tokens={token_count})"
    )


def _is_document_embedding_request(inputs: list[str], token_count: int) -> bool:
    return (
        len(inputs) >= EMBEDDING_DOCUMENT_REQUEST_MIN_INPUTS
        or token_count >= EMBEDDING_DOCUMENT_REQUEST_MIN_TOKENS
    )


def _embedding_request_purpose(request: EmbeddingRequest) -> str:
    return str(request.localmathrag_embedding_purpose or "").strip().lower()


def _is_quality_embedding_request(request: EmbeddingRequest, document_request: bool) -> bool:
    purpose = _embedding_request_purpose(request)
    return (
        document_request
        or _coerce_bool(request.localmathrag_strong_embedding, False)
        or purpose in {"citation", "document", "parsing", "ingestion"}
    )


def _fallback_embedding_response(
    request: EmbeddingRequest,
    inputs: list[str],
    token_count: int,
    reason: str,
    degradation: dict[str, Any],
) -> dict[str, Any]:
    document_request = _is_document_embedding_request(inputs, token_count)
    purpose = _embedding_request_purpose(request)
    quality_request = _is_quality_embedding_request(request, document_request)
    return {
        "object": "list",
        "model": request.model or LOCAL_EMBEDDING_MODEL,
        "data": [
            {
                "object": "embedding",
                "index": index,
                "embedding": _lexical_embedding(text),
            }
            for index, text in enumerate(inputs)
        ],
        "usage": {
            "prompt_tokens": token_count,
            "total_tokens": token_count,
        },
        "runtime": {
            "kind": "embedding",
            "fallback": True,
            "reason": reason,
            "degraded": True,
            "warning": degradation["message"],
            "degradation": degradation,
            "document_request": document_request,
            "quality_embedding": quality_request,
            "embedding_purpose": purpose or None,
        },
    }


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


def _runtime_rerank_response(runtime: dict[str, Any], request: RerankRequest, model: str) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    route_base_url = _runtime_route_base_url(runtime["target"]["base_url"], "/rerank")
    for offset in range(0, len(request.documents), RERANK_RUNTIME_BATCH_SIZE):
        batch_documents = request.documents[offset:offset + RERANK_RUNTIME_BATCH_SIZE]
        payload = {
            "query": request.query,
            "texts": [_document_text(document) for document in batch_documents],
        }
        batch_response = _post_runtime_json(route_base_url, "/rerank", payload, timeout=RERANK_REQUEST_TIMEOUT_SECONDS)
        batch_request = RerankRequest(
            query=request.query,
            documents=batch_documents,
            model=request.model,
            top_n=None,
            return_documents=False,
        )
        normalized = _normalize_runtime_rerank_response(batch_response, batch_request, model)
        for item in normalized["results"]:
            shifted = dict(item)
            shifted["index"] = int(shifted.get("index", 0)) + offset
            results.append(shifted)
    results.sort(key=lambda item: item.get("relevance_score", 0.0), reverse=True)
    top_n = request.top_n if request.top_n is not None and request.top_n > 0 else len(results)
    include_documents = _truthy(request.return_documents)
    normalized_results = []
    for item in results[:top_n]:
        index = int(item.get("index", 0))
        result = {
            "index": index,
            "relevance_score": float(item.get("relevance_score", item.get("score", 0.0))),
        }
        if include_documents and 0 <= index < len(request.documents):
            result["document"] = request.documents[index]
        normalized_results.append(result)
    return {
        "model": model,
        "results": normalized_results,
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


def _normalize_model_switch_key(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    if not text:
        return ""
    if text.startswith("/models/"):
        text = text[len("/models/"):]
    text = text.strip("/")
    return text.lower()


def _model_switch_keys(payload: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for field in ("name", "runtime_model_name", "file_name", "relative_path", "path"):
        value = payload.get(field)
        if not value:
            continue
        normalized = _normalize_model_switch_key(value)
        if normalized:
            keys.add(normalized)
            keys.add(Path(normalized).name.lower())
            keys.add(Path(normalized).stem.lower())
    for recommended in RECOMMENDED_MODELS:
        if recommended.get("file_name") == payload.get("file_name"):
            for field in ("id", "name", "runtime_model_name", "file_name"):
                normalized = _normalize_model_switch_key(recommended.get(field))
                if normalized:
                    keys.add(normalized)
    return keys


def _find_local_model_for_runtime(kind: str, model: str) -> dict[str, Any]:
    target = _runtime_target(kind)
    requested = _normalize_model_switch_key(model)
    if not requested:
        raise HTTPException(status_code=400, detail="Runtime model switch requires a non-empty model")
    for path in _model_entries():
        payload = _model_payload(path)
        model_types = payload.get("model_type") or ["chat"]
        endpoint_key = _endpoint_key_for_model_type(str(model_types[0]))
        if endpoint_key != target["model_type"]:
            continue
        if requested in _model_switch_keys(payload):
            payload["switch_model_path"] = payload.get("relative_path") or payload.get("file_name") or path.name
            return payload
    raise HTTPException(
        status_code=404,
        detail=f"Local {target['kind']} model is not installed under {MODEL_DIR}: {model}",
    )


def _runtime_command_for_model(kind: str, payload: dict[str, Any], cmd: list[str]) -> list[str]:
    model_path = str(payload.get("switch_model_path") or payload.get("relative_path") or payload.get("file_name") or payload.get("name"))
    normalized_model_path = model_path.replace("\\", "/")
    container_model_path = f"/models/{normalized_model_path}"
    if kind == "chat":
        return _replace_cmd_value(cmd, "-m", container_model_path)
    if kind in {"embedding", "rerank"}:
        return _replace_cmd_value(cmd, "--model-id", container_model_path)
    return cmd


def _runtime_env_for_model(kind: str, payload: dict[str, Any], env: list[str]) -> list[str]:
    model_path = str(payload.get("switch_model_path") or payload.get("relative_path") or payload.get("file_name") or payload.get("name")).replace("\\", "/")
    updated = list(env)
    if kind == "chat":
        updated = _replace_env_value(updated, "LOCALMATHRAG_GGUF_MODEL", model_path)
    elif kind == "embedding":
        updated = _replace_env_value(updated, "LOCALMATHRAG_EMBEDDING_MODEL", model_path)
        lowered_name = model_path.lower()
        if "qwen3" in lowered_name:
            updated = _replace_env_value(updated, "LOCALMATHRAG_EMBEDDING_POOLING", "last-token")
        elif "bge" in lowered_name:
            updated = _replace_env_value(updated, "LOCALMATHRAG_EMBEDDING_POOLING", "cls")
    elif kind == "rerank":
        updated = _replace_env_value(updated, "LOCALMATHRAG_RERANK_MODEL", model_path)
    return updated


def _apply_runtime_model_to_process_env(kind: str, payload: dict[str, Any]) -> None:
    model_path = str(payload.get("switch_model_path") or payload.get("relative_path") or payload.get("file_name") or payload.get("name")).replace("\\", "/")
    if kind == "chat":
        os.environ["LOCALMATHRAG_GGUF_MODEL"] = model_path
    elif kind == "embedding":
        os.environ["LOCALMATHRAG_EMBEDDING_MODEL"] = model_path
    elif kind == "rerank":
        os.environ["LOCALMATHRAG_RERANK_MODEL"] = model_path


def _persist_runtime_model_selection(kind: str, payload: dict[str, Any], source: str = "runtime-switch") -> dict[str, Any]:
    model_path = str(payload.get("switch_model_path") or payload.get("relative_path") or payload.get("file_name") or payload.get("name")).replace("\\", "/")
    selection = {
        "model": model_path,
        "name": payload.get("name"),
        "runtime_model_name": payload.get("runtime_model_name"),
        "model_type": payload.get("model_type"),
        "source": source,
        "updated_at": time.time(),
    }
    with RUNTIME_CONFIG_LOCK:
        config = _load_runtime_config()
        models = config.get("models") if isinstance(config.get("models"), dict) else {}
        models[kind] = selection
        config["models"] = models
        config["version"] = 1
        config["updated_at"] = time.time()
        _write_runtime_config(config)
    return selection


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
            endpoint_statuses[endpoint_key] = _optional_runtime_endpoint_status(endpoint_key, timeout=0.25)
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
        if endpoint_key != "chat" and not endpoint_ok(endpoint_key):
            continue
        add_model(payload.get("runtime_model_name"), model_type)
        add_model(payload.get("name"), model_type)

    return {"object": "list", "data": data}


@app.post("/v1/embeddings")
def openai_embeddings(request: EmbeddingRequest) -> dict[str, Any]:
    inputs = _embedding_inputs(request.input)
    token_count = sum(len(_embedding_tokens(text)) for text in inputs)
    document_request = _is_document_embedding_request(inputs, token_count)
    purpose = _embedding_request_purpose(request)
    quality_request = _is_quality_embedding_request(request, document_request)
    request_kind = f"embedding:{purpose}" if purpose else "embedding"
    if document_request:
        request_kind = "embedding:document"
    with _runtime_request_context("embedding", request_kind, quality_request):
        return _openai_embeddings_inner(request, inputs, token_count, document_request, purpose, quality_request)


def _openai_embeddings_inner(
    request: EmbeddingRequest,
    inputs: list[str],
    token_count: int,
    document_request: bool,
    purpose: str | None,
    quality_request: bool,
) -> dict[str, Any]:
    small_request_reason = None if quality_request else _should_degrade_small_embedding_for_chat(inputs, token_count)
    if small_request_reason:
        degradation = _record_runtime_degradation("embedding", small_request_reason)
        _schedule_chat_restore_after_embedding()
        return _fallback_embedding_response(request, inputs, token_count, small_request_reason, degradation)

    try:
        ready_timeout_seconds = _runtime_ready_timeout_seconds("embedding")
        if document_request:
            ready_timeout_seconds = max(ready_timeout_seconds, EMBEDDING_DOCUMENT_READY_TIMEOUT_SECONDS)
        elif purpose == "citation":
            ready_timeout_seconds = min(ready_timeout_seconds, EMBEDDING_CITATION_READY_TIMEOUT_SECONDS)
        elif quality_request:
            ready_timeout_seconds = max(ready_timeout_seconds, EMBEDDING_CITATION_READY_TIMEOUT_SECONDS)
        runtime = _ensure_runtime_ready(
            "embedding",
            ready_timeout_seconds,
            allow_background=NO_COLD_START_IN_REQUEST and not quality_request,
            bypass_cached_failure=quality_request,
            bypass_runtime_disabled=quality_request,
            prefer_quality=quality_request,
        )
        if runtime.get("ready"):
            resource_status = _runtime_resource_status("embedding")
            if not resource_status["ok"]:
                if quality_request:
                    runtime["resource_warning"] = resource_status
                else:
                    runtime["degraded_stop"] = _stop_optional_runtime_for_degrade(runtime["target"], "; ".join(resource_status["reasons"]))
                    raise RuntimeError("; ".join(resource_status["reasons"]))
            payload = request.model_dump(
                exclude_none=True,
                exclude={"localmathrag_embedding_purpose", "localmathrag_strong_embedding"},
            )
            payload.setdefault("model", request.model or "bge-m3")
            request_start = time.monotonic()
            runtime_request_timeout = EMBEDDING_CITATION_READY_TIMEOUT_SECONDS if purpose == "citation" and not document_request else 120.0
            response = _post_runtime_json(runtime["target"]["base_url"], "/embeddings", payload, timeout=runtime_request_timeout)
            response.setdefault("model", payload["model"])
            embedding_seconds = round(time.monotonic() - request_start, 3)
            response["runtime"] = {
                "kind": "embedding",
                "fallback": False,
                "started": runtime.get("started", False),
                "document_request": document_request,
                "quality_embedding": quality_request,
                "embedding_purpose": purpose or None,
                "balancer": runtime.get("balancer"),
            }
            if runtime.get("resource_warning"):
                response["runtime"]["resource_warning"] = runtime["resource_warning"]
            if not quality_request:
                _maybe_record_chat_embedding_concurrency_observed({"embedding_request_seconds": embedding_seconds})
            _schedule_chat_restore_after_embedding()
            return response
    except Exception as exc:
        runtime = {"reason": str(exc)}

    degradation = _record_runtime_degradation("embedding", runtime.get("reason", "runtime endpoint is not ready"))
    _schedule_chat_restore_after_embedding()
    return _fallback_embedding_response(
        request,
        inputs,
        token_count,
        runtime.get("reason", "runtime endpoint is not ready"),
        degradation,
    )


@app.post("/v1/rerank")
def openai_rerank(request: RerankRequest) -> dict[str, Any]:
    with _runtime_request_context("rerank", "rerank", False):
        try:
            runtime = _ensure_runtime_ready(
                "rerank",
                _runtime_ready_timeout_seconds("rerank"),
                allow_background=NO_COLD_START_IN_REQUEST,
            )
            if runtime.get("ready"):
                resource_status = _runtime_resource_status("rerank")
                if not resource_status["ok"]:
                    runtime["degraded_stop"] = _stop_optional_runtime_for_degrade(runtime["target"], "; ".join(resource_status["reasons"]))
                    raise RuntimeError("; ".join(resource_status["reasons"]))
                model = request.model or _rerank_model_name()
                response = _runtime_rerank_response(runtime, request, model)
                response["runtime"] = {
                    "kind": "rerank",
                    "fallback": False,
                    "started": runtime.get("started", False),
                    "balancer": runtime.get("balancer"),
                }
                return response
        except Exception as exc:
            target = _runtime_target("rerank")
            degraded_stop = _stop_optional_runtime_for_degrade(target, f"rerank runtime failed; falling back: {exc}")
            return _fallback_rerank_response(request, str(exc), degraded_stop)

        return _fallback_rerank_response(request, runtime.get("reason", "runtime endpoint is not ready"))


async def _proxy_openai_generation(request: Request, route: str):
    body = await request.body()
    try:
        payload = json.loads(body.decode("utf-8")) if body else {}
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}") from exc
    token = _begin_runtime_request("chat", "chat_stream" if payload.get("stream") else "chat", False)
    stream_response = False
    try:
        runtime_body = _generation_body_for_chat_runtime(payload)
        retry_body = _generation_body_for_chat_runtime(payload, _runtime_retry_prompt_budget())

        runtime = _ensure_runtime_ready("chat", _runtime_ready_timeout_seconds("chat"))
        if not runtime.get("ready"):
            raise HTTPException(
                status_code=503,
                detail={
                    "message": runtime.get("reason", "chat runtime endpoint is not ready"),
                    "runtime": runtime,
                },
            )

        if payload.get("stream"):
            stream_response = True
            return StreamingResponse(
                _stream_with_runtime_activity(
                    token,
                    _stream_chat_runtime_proxy(runtime, route, runtime_body, retry_body, CHAT_REQUEST_TIMEOUT_SECONDS),
                ),
                media_type="text/event-stream",
            )
        try:
            return _json_chat_runtime_proxy(runtime, route, runtime_body, retry_body, CHAT_REQUEST_TIMEOUT_SECONDS)
        except Exception as exc:
            degraded_stop = _stop_optional_runtime_for_degrade(runtime["target"], f"chat runtime failed: {exc}")
            raise HTTPException(
                status_code=503,
                detail={
                    "message": str(exc),
                    "runtime": {
                        "kind": "chat",
                        "fallback": False,
                        "degraded_stop": degraded_stop,
                    },
                },
            ) from exc
    finally:
        if not stream_response:
            _finish_runtime_request(token)


@app.post("/v1/chat/completions", response_model=None)
async def openai_chat_completions(request: Request):
    return await _proxy_openai_generation(request, "/chat/completions")


@app.post("/v1/completions", response_model=None)
async def openai_completions(request: Request):
    return await _proxy_openai_generation(request, "/completions")


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
        "runtime_degradations": _runtime_degradation_snapshot(),
        "runtime_disabled": _runtime_disabled_snapshot(),
        "runtime_config": _load_runtime_config(),
        "scheduler_policy": _runtime_scheduler_policy_snapshot(),
    }


@app.get("/v1/runtime/policy")
def runtime_policy() -> dict[str, Any]:
    return {
        "scheduler_policy": _runtime_scheduler_policy_snapshot(),
        "runtime_config": _load_runtime_config(),
    }


@app.post("/v1/runtime/probe")
def probe_runtime(request: RuntimeProbeRequest) -> dict[str, Any]:
    pair = request.pair.strip().lower().replace("-", "_")
    if pair not in {"chat_embedding", "llm_embedding"}:
        raise HTTPException(status_code=400, detail=f"Unsupported runtime probe pair: {request.pair}")
    return _probe_chat_embedding_runtime(request)


@app.post("/v1/runtime/policy/reset")
def reset_runtime_policy() -> dict[str, Any]:
    stopped = []
    for kind in ("embedding", "rerank", "vision", "asr", "tts", "chat"):
        action = _stop_runtime_for_scheduler(kind, "runtime policy reset")
        if action:
            stopped.append(action)
    with RUNTIME_START_FAILURE_LOCK:
        RUNTIME_START_FAILURES.clear()
        RUNTIME_START_FAILURE_COUNTS.clear()
    with RUNTIME_DISABLED_LOCK:
        RUNTIME_DISABLED.clear()
    with RUNTIME_DEGRADATION_LOCK:
        RUNTIME_DEGRADATIONS.clear()
    with RUNTIME_CONFIG_LOCK:
        config = _load_runtime_config()
        removed = {
            "scheduler": config.pop("scheduler", None) is not None,
            "rerank": config.pop("rerank", None) is not None,
            "recent_tasks": config.pop("recent_tasks", None) is not None,
            "prewarm": config.pop("prewarm", None) is not None,
        }
        config["version"] = 1
        config["updated_at"] = time.time()
        _write_runtime_config(config)
    policy = _reset_scheduler_runtime_policy()
    return {
        "ok": True,
        "removed": removed,
        "stopped": stopped,
        "scheduler_policy": policy,
        "runtime_config": _load_runtime_config(),
    }


@app.post("/v1/runtime/ensure")
def ensure_runtime(request: RuntimeEnsureRequest) -> dict[str, Any]:
    target = _runtime_target(request.kind)
    timeout_seconds = request.timeout_seconds if request.timeout_seconds is not None else _runtime_ready_timeout_seconds(target["kind"])
    status = _optional_runtime_endpoint_status(target["kind"], timeout=1.5)
    if status.get("endpoint_ok"):
        _clear_runtime_start_failure(target["kind"])
        resource_status = _runtime_resource_status(target["kind"])
        if not resource_status["ok"]:
            degraded_stop = _stop_optional_runtime_for_degrade(target, "; ".join(resource_status["reasons"]))
            degradation = _record_runtime_degradation(
                target["kind"],
                "; ".join(resource_status["reasons"]),
                [degraded_stop] if degraded_stop else None,
                resource_status,
            )
            return {
                "kind": target["kind"],
                "ready": False,
                "started": False,
                "degraded": True,
                "reason": "; ".join(resource_status["reasons"]),
                "endpoint_status": status,
                "resource_status": resource_status,
                "runtime_degradation": degradation,
                "balancer": {"max_active_optional": OPTIONAL_RUNTIME_MAX_ACTIVE, "actions": [degraded_stop] if degraded_stop else []},
            }
        return {
            "kind": target["kind"],
            "ready": True,
            "started": False,
            "degraded": False,
            "endpoint_status": status,
            "resource_status": resource_status,
            "runtime_degradation": _last_runtime_degradation(target["kind"]),
            "runtime_disabled": _runtime_disabled(target["kind"]),
            "balancer": {"max_active_optional": OPTIONAL_RUNTIME_MAX_ACTIVE, "actions": []},
        }
    disabled = _runtime_disabled(target["kind"])
    if disabled:
        return {
            "kind": target["kind"],
            "ready": False,
            "started": False,
            "degraded": True,
            "reason": disabled["reason"],
            "endpoint_status": status,
            "runtime_degradation": _last_runtime_degradation(target["kind"]),
            "runtime_disabled": disabled,
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
            "runtime_degradation": _last_runtime_degradation(target["kind"]),
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
            "runtime_degradation": _last_runtime_degradation(target["kind"]),
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
            "runtime_degradation": _last_runtime_degradation(target["kind"]),
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
            "runtime_degradation": runtime.get("runtime_degradation") or _last_runtime_degradation(target["kind"]),
            "runtime_disabled": runtime.get("runtime_disabled") or _runtime_disabled(target["kind"]),
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
            "runtime_degradation": _last_runtime_degradation(target["kind"]),
            "balancer": {"max_active_optional": OPTIONAL_RUNTIME_MAX_ACTIVE, "actions": []},
        }


@app.post("/v1/runtime/switch-model")
def switch_runtime_model(request: RuntimeSwitchModelRequest) -> dict[str, Any]:
    return _switch_runtime_model(request)


@app.get("/v1/runtime/status")
def runtime_status(kind: str = "embedding") -> dict[str, Any]:
    target = _runtime_target(kind)
    status = _optional_runtime_endpoint_status(target["kind"], timeout=0.25)
    if status.get("endpoint_ok"):
        _clear_runtime_start_failure(target["kind"])
    resource_status = _runtime_resource_status(target["kind"])
    return {
        "kind": target["kind"],
        "ready": bool(status.get("endpoint_ok")),
        "endpoint_status": status,
        "resource_status": resource_status,
        "runtime_degradation": _last_runtime_degradation(target["kind"]),
        "runtime_disabled": _runtime_disabled(target["kind"]),
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

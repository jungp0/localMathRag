from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import ipaddress
import json
import mimetypes
import secrets
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .formula import stable_id


class RagFlowError(RuntimeError):
    pass


@dataclass(slots=True)
class RagFlowChunk:
    id: str
    text: str
    document_id: str | None = None
    document_name: str | None = None
    score: float | None = None
    page: int | None = None
    metadata: dict[str, Any] | None = None


class RagFlowClient:
    def __init__(self, settings: dict[str, Any]):
        self.settings = settings
        self.base_url = str(settings.get("base_url") or "").rstrip("/")
        self.timeout = int(settings.get("timeout_seconds") or 20)

    @property
    def enabled(self) -> bool:
        return bool(self.settings.get("enabled"))

    @property
    def dataset_id(self) -> str:
        return str(self.settings.get("dataset_id") or "").strip()

    def offline_allowed(self) -> bool:
        return is_offline_url_allowed(self.base_url)

    def status(self) -> dict[str, Any]:
        status = {
            "enabled": self.enabled,
            "mode": self.settings.get("mode", "local_only"),
            "base_url": self.base_url,
            "dataset_id": self.dataset_id,
            "offline_allowed": self.offline_allowed(),
            "endpoint_ok": False,
            "reachable": False,
            "http_status": None,
            "error": None,
        }
        if not self.enabled:
            return status
        if not status["offline_allowed"]:
            status["error"] = "RAGFlow base_url is outside localhost/private network and remote hosts are disabled."
            return status
        try:
            response_status, _ = self.request_json("GET", str(self.settings.get("status_path") or "/api/v1/datasets"))
            status["http_status"] = response_status
            status["reachable"] = True
            status["endpoint_ok"] = 200 <= response_status < 300
        except RagFlowError as exc:
            status["error"] = str(exc)
        return status

    def retrieve(self, query: str, top_k: int | None = None) -> tuple[list[RagFlowChunk], list[str]]:
        warnings: list[str] = []
        if not self.enabled:
            return [], warnings
        if not self.offline_allowed():
            return [], ["RAGFlow skipped: base_url is not localhost/private network."]
        if not self.dataset_id:
            return [], ["RAGFlow skipped: dataset_id is not configured."]
        payload = {
            "question": query,
            "query": query,
            "dataset_ids": [self.dataset_id],
            "top_k": top_k or int(self.settings.get("top_k") or 8),
        }
        try:
            _, data = self.request_json("POST", str(self.settings.get("retrieval_path") or "/api/v1/retrieval"), payload)
        except RagFlowError as exc:
            return [], [f"RAGFlow retrieval failed: {exc}"]
        return normalize_chunks(data), warnings

    def upload_documents(self, paths: list[str | Path]) -> dict[str, Any]:
        result = {"status": "skipped", "uploaded": [], "warnings": []}
        if not self.enabled:
            return result
        if not self.offline_allowed():
            result["warnings"].append("RAGFlow sync skipped: base_url is not localhost/private network.")
            return result
        if not self.dataset_id:
            result["warnings"].append("RAGFlow sync skipped: dataset_id is not configured.")
            return result
        upload_path = str(self.settings.get("upload_path_template") or "/api/v1/datasets/{dataset_id}/documents")
        upload_path = upload_path.replace("{dataset_id}", self.dataset_id)
        for item in paths:
            path = Path(item)
            if not path.exists() or not path.is_file():
                result["warnings"].append(f"RAGFlow sync skipped missing file: {path}")
                continue
            try:
                status_code, data = self.request_multipart(
                    upload_path,
                    field_name=str(self.settings.get("upload_field") or "file"),
                    file_path=path,
                )
                result["uploaded"].append(
                    {
                        "path": str(path),
                        "http_status": status_code,
                        "response": compact_response(data),
                    }
                )
            except RagFlowError as exc:
                result["warnings"].append(f"{path}: {exc}")
        result["status"] = "ok" if result["uploaded"] and not result["warnings"] else "partial" if result["uploaded"] else "skipped"
        return result

    def request_json(self, method: str, path: str, payload: dict[str, Any] | None = None) -> tuple[int, Any]:
        url = join_url(self.base_url, path)
        headers = self.headers()
        body = None
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(url, data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8-sig", errors="replace")
                return response.status, json.loads(raw) if raw.strip() else {}
        except HTTPError as exc:
            body_text = exc.read().decode("utf-8-sig", errors="replace")[:800]
            raise RagFlowError(f"HTTP {exc.code}: {body_text}") from exc
        except (OSError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RagFlowError(str(exc)) from exc

    def request_multipart(self, endpoint: str, *, field_name: str, file_path: Path) -> tuple[int, Any]:
        boundary = "----localMathRag" + secrets.token_hex(12)
        media_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        chunks = [
            f"--{boundary}\r\n".encode("utf-8"),
            (
                f'Content-Disposition: form-data; name="{field_name}"; filename="{file_path.name}"\r\n'
                f"Content-Type: {media_type}\r\n\r\n"
            ).encode("utf-8"),
            file_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
        body = b"".join(chunks)
        headers = self.headers()
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        request = Request(join_url(self.base_url, endpoint), data=body, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8-sig", errors="replace")
                return response.status, json.loads(raw) if raw.strip() else {}
        except HTTPError as exc:
            body_text = exc.read().decode("utf-8-sig", errors="replace")[:800]
            raise RagFlowError(f"HTTP {exc.code}: {body_text}") from exc
        except (OSError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RagFlowError(str(exc)) from exc

    def headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "User-Agent": "localMathRag-offline/0.1"}
        api_key = str(self.settings.get("api_key") or "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers


def ragflow_settings_from_request(payload: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "enabled",
        "mode",
        "base_url",
        "api_key",
        "dataset_id",
        "timeout_seconds",
        "top_k",
        "auto_sync_uploads",
        "status_path",
        "retrieval_path",
        "upload_path_template",
        "upload_field",
    }
    cleaned = {key: payload[key] for key in allowed if key in payload}
    for key in ["enabled", "auto_sync_uploads"]:
        if key in cleaned:
            cleaned[key] = bool(cleaned[key])
    for key in ["timeout_seconds", "top_k"]:
        if key in cleaned:
            cleaned[key] = int(cleaned[key])
    if "mode" in cleaned and cleaned["mode"] not in {"local_only", "ragflow_only", "hybrid"}:
        cleaned["mode"] = "local_only"
    return cleaned


def ragflow_status(settings: dict[str, Any]) -> dict[str, Any]:
    return RagFlowClient(settings).status()


def sync_paths_to_ragflow(settings: dict[str, Any], paths: list[str | Path]) -> dict[str, Any]:
    return RagFlowClient(settings).upload_documents(paths)


def apply_ragflow_retrieval(
    payload: dict[str, Any],
    *,
    settings: dict[str, Any],
    query: str,
    top_k: int | None,
) -> dict[str, Any]:
    mode = str(settings.get("mode") or "local_only")
    if not settings.get("enabled") or mode == "local_only":
        return payload
    client = RagFlowClient(settings)
    chunks, warnings = client.retrieve(query, top_k=top_k)
    payload.setdefault("warnings", []).extend(warnings)
    payload.setdefault("meta", {})["ragflow_mode"] = mode
    payload.setdefault("meta", {})["ragflow_enabled"] = True
    if not chunks:
        if mode == "ragflow_only" and not payload.get("items"):
            payload["status"] = "not_found"
        return payload
    if mode == "ragflow_only":
        payload["items"] = []
        payload["evidence"] = {}
    evidence = payload.setdefault("evidence", {})
    items = payload.setdefault("items", [])
    for ordinal, chunk in enumerate(chunks, start=1):
        ev_id = f"ev.ragflow.{chunk.id}"
        evidence[ev_id] = {
            "doc_id": chunk.document_id or "ragflow",
            "path": chunk.document_name or "ragflow",
            "block_id": f"ragflow.{chunk.id}",
            "page": chunk.page,
            "section": "ragflow_chunk",
            "text_preview": chunk.text[:500],
        }
        items.append(
            {
                "id": f"ragflow.chunk.{ordinal:03d}",
                "type": "ragflow_chunk",
                "text": chunk.text,
                "score": chunk.score,
                "source": chunk.document_name,
                "evidence": [ev_id],
            }
        )
    payload["status"] = "ok"
    payload.setdefault("meta", {})["ragflow_chunks"] = len(chunks)
    return payload


def normalize_chunks(response: Any) -> list[RagFlowChunk]:
    raw_chunks = find_chunk_list(response)
    chunks: list[RagFlowChunk] = []
    for index, item in enumerate(raw_chunks, start=1):
        if isinstance(item, str):
            text = item.strip()
            metadata: dict[str, Any] = {}
        elif isinstance(item, dict):
            text = str(
                item.get("content")
                or item.get("text")
                or item.get("chunk")
                or item.get("answer")
                or item.get("document")
                or ""
            ).strip()
            metadata = dict(item)
        else:
            continue
        if not text:
            continue
        document_id = pick_first(metadata, "document_id", "doc_id", "documentId")
        document_name = pick_first(metadata, "document_name", "doc_name", "filename", "name", "source")
        score_value = pick_first(metadata, "score", "similarity", "rank")
        page_value = pick_first(metadata, "page", "page_number", "page_no")
        chunk_id = str(pick_first(metadata, "id", "chunk_id", "chunkId") or stable_id("ragflow", text, str(index), length=12))
        chunks.append(
            RagFlowChunk(
                id=chunk_id,
                text=text,
                document_id=str(document_id) if document_id else None,
                document_name=str(document_name) if document_name else None,
                score=parse_float(score_value),
                page=parse_int(page_value),
                metadata=metadata,
            )
        )
    return chunks


def find_chunk_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not isinstance(value, dict):
        return []
    for key in ["chunks", "documents", "refs", "references", "items", "data"]:
        item = value.get(key)
        if isinstance(item, list):
            return item
        if isinstance(item, dict):
            found = find_chunk_list(item)
            if found:
                return found
    return []


def is_offline_url_allowed(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    host = (parsed.hostname or "").lower()
    if not host:
        return False
    if host in {"localhost", "localhost.localdomain"}:
        return True
    if "." not in host and ":" not in host:
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return host.endswith(".local")
    return bool(address.is_loopback or address.is_private or address.is_link_local)


def join_url(base_url: str, path: str) -> str:
    if path.startswith("http://") or path.startswith("https://"):
        return path
    return base_url.rstrip("/") + "/" + path.lstrip("/")


def pick_first(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data and data[key] not in {None, ""}:
            return data[key]
    return None


def parse_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def compact_response(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: compact_response(item) for key, item in list(value.items())[:12] if key not in {"content", "text"}}
    if isinstance(value, list):
        return [compact_response(item) for item in value[:4]]
    return value

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


SCHEMA_DIR = Path(os.environ.get("LOCALMATHRAG_SCHEMA_DIR", "/schemas"))

app = FastAPI(
    title="LocalMathRAGFlow Object Service",
    version="0.1.0",
    description="Auxiliary structured evidence API for RAGFlow secondary development.",
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


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "schema_dir": str(SCHEMA_DIR),
        "schema_count": len(_schema_files()),
    }


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

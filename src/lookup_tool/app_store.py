from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import json
import os
import shutil
import sqlite3
import time
import uuid
from typing import Any, Iterator


def now_ts() -> float:
    return time.time()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


class AppStore:
    def __init__(self, db_path: str | Path = "data/app.sqlite"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init(self) -> None:
        needs_default = False
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_bases (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    root_path TEXT NOT NULL,
                    db_path TEXT NOT NULL,
                    upload_dir TEXT NOT NULL,
                    artifact_dir TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    archived INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    kb_id TEXT NOT NULL,
                    parent_id TEXT,
                    name TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    archived INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY(kb_id) REFERENCES knowledge_bases(id)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_projects_kb ON projects(kb_id, archived)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS questions (
                    id TEXT PRIMARY KEY,
                    kb_id TEXT NOT NULL,
                    project_id TEXT,
                    title TEXT,
                    query TEXT NOT NULL,
                    task TEXT NOT NULL,
                    top_k INTEGER,
                    result_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    archived INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY(kb_id) REFERENCES knowledge_bases(id),
                    FOREIGN KEY(project_id) REFERENCES projects(id)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_questions_kb ON questions(kb_id, archived, created_at)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS app_settings (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            row = conn.execute("SELECT COUNT(*) AS count FROM knowledge_bases").fetchone()
            needs_default = int(row["count"] if row else 0) == 0
            model_row = conn.execute("SELECT 1 FROM app_settings WHERE key = 'model'").fetchone()
            needs_model_default = model_row is None
            ragflow_row = conn.execute("SELECT 1 FROM app_settings WHERE key = 'ragflow'").fetchone()
            needs_ragflow_default = ragflow_row is None
        if needs_default:
            self.create_kb("Default", self.db_path.parent / "kbs" / "default")
        if needs_model_default:
            self.update_model_settings(default_model_settings(self.db_path.parent))
        if needs_ragflow_default:
            self.update_ragflow_settings(default_ragflow_settings())

    def create_kb(self, name: str, root_path: str | Path) -> dict[str, Any]:
        kb_id = new_id("kb")
        root = Path(root_path)
        paths = kb_paths(root)
        ensure_kb_dirs(paths)
        timestamp = now_ts()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO knowledge_bases(
                    id, name, root_path, db_path, upload_dir, artifact_dir, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    kb_id,
                    name.strip() or "Knowledge Base",
                    str(root),
                    str(paths["db_path"]),
                    str(paths["upload_dir"]),
                    str(paths["artifact_dir"]),
                    timestamp,
                    timestamp,
                ),
            )
            conn.execute(
                """
                INSERT INTO projects(id, kb_id, parent_id, name, created_at, updated_at)
                VALUES (?, ?, NULL, ?, ?, ?)
                """,
                (new_id("project"), kb_id, "Inbox", timestamp, timestamp),
            )
        return self.get_kb(kb_id)

    def list_kbs(self, include_archived: bool = False) -> list[dict[str, Any]]:
        clause = "" if include_archived else "WHERE archived = 0"
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM knowledge_bases {clause} ORDER BY updated_at DESC, created_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_kb(self, kb_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM knowledge_bases WHERE id = ?", (kb_id,)).fetchone()
        if row is None:
            raise KeyError(f"Knowledge base not found: {kb_id}")
        return dict(row)

    def update_kb(self, kb_id: str, *, name: str | None = None) -> dict[str, Any]:
        if name is None:
            return self.get_kb(kb_id)
        with self.connect() as conn:
            conn.execute(
                "UPDATE knowledge_bases SET name = ?, updated_at = ? WHERE id = ?",
                (name.strip() or "Knowledge Base", now_ts(), kb_id),
            )
        return self.get_kb(kb_id)

    def archive_kb(self, kb_id: str) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE knowledge_bases SET archived = 1, updated_at = ? WHERE id = ?", (now_ts(), kb_id))

    def migrate_kb(self, kb_id: str, new_root_path: str | Path) -> dict[str, Any]:
        kb = self.get_kb(kb_id)
        old_root = Path(kb["root_path"])
        new_root = Path(new_root_path)
        old_paths = kb_paths(old_root)
        new_paths = kb_paths(new_root)
        ensure_kb_dirs(new_paths)

        copy_if_exists(old_paths["db_path"], new_paths["db_path"])
        copytree_contents(old_paths["upload_dir"], new_paths["upload_dir"])
        copytree_contents(old_paths["artifact_dir"], new_paths["artifact_dir"])

        with self.connect() as conn:
            conn.execute(
                """
                UPDATE knowledge_bases
                SET root_path = ?, db_path = ?, upload_dir = ?, artifact_dir = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    str(new_root),
                    str(new_paths["db_path"]),
                    str(new_paths["upload_dir"]),
                    str(new_paths["artifact_dir"]),
                    now_ts(),
                    kb_id,
                ),
            )
        return self.get_kb(kb_id)

    def create_project(self, kb_id: str, name: str, parent_id: str | None = None) -> dict[str, Any]:
        project_id = new_id("project")
        timestamp = now_ts()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO projects(id, kb_id, parent_id, name, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (project_id, kb_id, parent_id, name.strip() or "Project", timestamp, timestamp),
            )
        return self.get_project(project_id)

    def list_projects(self, kb_id: str, include_archived: bool = False) -> list[dict[str, Any]]:
        archived_clause = "" if include_archived else "AND archived = 0"
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM projects
                WHERE kb_id = ? {archived_clause}
                ORDER BY parent_id IS NOT NULL, name COLLATE NOCASE
                """,
                (kb_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_project(self, project_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        if row is None:
            raise KeyError(f"Project not found: {project_id}")
        return dict(row)

    def update_project(self, project_id: str, *, name: str | None = None, parent_id: str | None = None) -> dict[str, Any]:
        project = self.get_project(project_id)
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE projects SET name = ?, parent_id = ?, updated_at = ? WHERE id = ?
                """,
                (
                    name.strip() if name is not None and name.strip() else project["name"],
                    parent_id,
                    now_ts(),
                    project_id,
                ),
            )
        return self.get_project(project_id)

    def archive_project(self, project_id: str) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE projects SET archived = 1, updated_at = ? WHERE id = ?", (now_ts(), project_id))
            conn.execute(
                "UPDATE questions SET archived = 1, updated_at = ? WHERE project_id = ?",
                (now_ts(), project_id),
            )

    def create_question(
        self,
        *,
        kb_id: str,
        project_id: str | None,
        query: str,
        task: str,
        top_k: int | None,
        result: dict[str, Any],
        title: str | None = None,
    ) -> dict[str, Any]:
        question_id = new_id("question")
        timestamp = now_ts()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO questions(
                    id, kb_id, project_id, title, query, task, top_k,
                    result_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    question_id,
                    kb_id,
                    project_id,
                    title or query[:80],
                    query,
                    task,
                    top_k,
                    json.dumps(result, ensure_ascii=False),
                    timestamp,
                    timestamp,
                ),
            )
        return self.get_question(question_id)

    def list_questions(self, kb_id: str, project_id: str | None = None, include_archived: bool = False) -> list[dict[str, Any]]:
        clauses = ["kb_id = ?"]
        params: list[Any] = [kb_id]
        if project_id:
            clauses.append("project_id = ?")
            params.append(project_id)
        if not include_archived:
            clauses.append("archived = 0")
        where = " AND ".join(clauses)
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM questions WHERE {where} ORDER BY created_at DESC LIMIT 500",
                params,
            ).fetchall()
        return [question_row_to_dict(row) for row in rows]

    def get_question(self, question_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM questions WHERE id = ?", (question_id,)).fetchone()
        if row is None:
            raise KeyError(f"Question not found: {question_id}")
        return question_row_to_dict(row)

    def update_question(
        self,
        question_id: str,
        *,
        title: str | None = None,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        question = self.get_question(question_id)
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE questions SET title = ?, project_id = ?, updated_at = ? WHERE id = ?
                """,
                (
                    title.strip() if title is not None and title.strip() else question["title"],
                    project_id,
                    now_ts(),
                    question_id,
                ),
            )
        return self.get_question(question_id)

    def archive_question(self, question_id: str) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE questions SET archived = 1, updated_at = ? WHERE id = ?", (now_ts(), question_id))

    def get_model_settings(self) -> dict[str, Any]:
        return self.get_settings("model", default_model_settings(self.db_path.parent))

    def update_model_settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        return self.update_settings("model", settings, default_model_settings(self.db_path.parent))

    def get_ragflow_settings(self) -> dict[str, Any]:
        return self.get_settings("ragflow", default_ragflow_settings())

    def update_ragflow_settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        return self.update_settings("ragflow", settings, default_ragflow_settings())

    def get_settings(self, key: str, defaults: dict[str, Any]) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute("SELECT value_json FROM app_settings WHERE key = ?", (key,)).fetchone()
        if row is None:
            self.update_settings(key, defaults, defaults)
            return defaults
        try:
            stored = json.loads(row["value_json"])
        except json.JSONDecodeError:
            stored = {}
        return {**defaults, **stored}

    def update_settings(self, key: str, settings: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
        current = dict(defaults)
        with self.connect() as conn:
            row = conn.execute("SELECT value_json FROM app_settings WHERE key = ?", (key,)).fetchone()
        if row is not None:
            try:
                current.update(json.loads(row["value_json"]))
            except json.JSONDecodeError:
                pass
        current.update({key: value for key, value in settings.items() if value is not None})
        timestamp = now_ts()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO app_settings(key, value_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json, updated_at = excluded.updated_at
                """,
                (key, json.dumps(current, ensure_ascii=False), timestamp),
            )
        return current


def kb_paths(root: Path) -> dict[str, Path]:
    return {
        "root": root,
        "db_path": root / "index.sqlite",
        "upload_dir": root / "uploads",
        "artifact_dir": root / "artifacts",
    }


def ensure_kb_dirs(paths: dict[str, Path]) -> None:
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["upload_dir"].mkdir(parents=True, exist_ok=True)
    paths["artifact_dir"].mkdir(parents=True, exist_ok=True)


def copy_if_exists(source: Path, target: Path) -> None:
    if not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copytree_contents(source: Path, target: Path) -> None:
    if not source.exists():
        return
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        destination = target / item.name
        if item.is_dir():
            if destination.exists():
                copytree_contents(item, destination)
            else:
                shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)


def question_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    try:
        item["result"] = json.loads(item.pop("result_json"))
    except (json.JSONDecodeError, TypeError):
        item["result"] = {}
    return item


def default_model_settings(base_dir: Path) -> dict[str, Any]:
    models_dir = base_dir / "models"
    default_model = str(models_dir / "Qwen3-8B-Q4_K_M.gguf")
    return {
        "enabled": env_bool("LOOKUP_TOOL_MODEL_ENABLED", False),
        "provider": os.environ.get("LOOKUP_TOOL_MODEL_PROVIDER", "openai_compatible"),
        "base_url": os.environ.get("LOOKUP_TOOL_MODEL_BASE_URL", "http://127.0.0.1:8080/v1"),
        "model": os.environ.get("LOOKUP_TOOL_MODEL_ID", default_model),
        "temperature": float(os.environ.get("LOOKUP_TOOL_MODEL_TEMPERATURE", "0")),
        "timeout_seconds": int(os.environ.get("LOOKUP_TOOL_MODEL_TIMEOUT_SECONDS", "60")),
        "local_models_dir": os.environ.get("LOOKUP_TOOL_LOCAL_MODELS_DIR", str(models_dir)),
        "local_model_path": os.environ.get("LOOKUP_TOOL_LOCAL_MODEL_PATH", default_model),
        "llama_server_path": os.environ.get("LOOKUP_TOOL_LLAMA_SERVER_PATH", ""),
        "recommended_repo": "Qwen/Qwen3-8B-GGUF",
        "recommended_file": "Qwen3-8B-Q4_K_M.gguf",
    }


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def default_ragflow_settings() -> dict[str, Any]:
    return {
        "enabled": False,
        "mode": "local_only",
        "base_url": "http://127.0.0.1:9380",
        "api_key": "",
        "dataset_id": "",
        "timeout_seconds": 20,
        "top_k": 8,
        "auto_sync_uploads": False,
        "status_path": "/api/v1/datasets",
        "retrieval_path": "/api/v1/retrieval",
        "upload_path_template": "/api/v1/datasets/{dataset_id}/documents",
        "upload_field": "file",
    }

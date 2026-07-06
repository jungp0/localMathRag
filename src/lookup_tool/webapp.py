from __future__ import annotations

from dataclasses import replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from .app_store import AppStore
from .config import ParserConfig, RetrievalConfig
from .extractor import AgentExtractor
from .index import SQLiteIndex
from .llm import synthesize_answer
from .model_manager import download_recommended_model, model_status, settings_from_request
from .models import IngestReport
from .parsers import DocumentParser


STATIC_DIR = Path(__file__).parent / "static"


class WebApp:
    def __init__(
        self,
        *,
        app_store: AppStore | None = None,
        parser_config: ParserConfig | None = None,
        retrieval_config: RetrievalConfig | None = None,
    ):
        self.store = app_store or AppStore()
        self.parser_config = parser_config or ParserConfig()
        self.retrieval_config = retrieval_config or RetrievalConfig()

    def components(self, kb_id: str) -> tuple[DocumentParser, SQLiteIndex, AgentExtractor, dict[str, Any]]:
        kb = self.store.get_kb(kb_id)
        parser_config = replace(self.parser_config, artifact_dir=Path(kb["artifact_dir"]))
        parser = DocumentParser(parser_config)
        index = SQLiteIndex(kb["db_path"], self.retrieval_config)
        extractor = AgentExtractor(index)
        return parser, index, extractor, kb


def serve_webapp(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    app_store: AppStore | None = None,
    parser_config: ParserConfig | None = None,
    retrieval_config: RetrievalConfig | None = None,
) -> None:
    app = WebApp(app_store=app_store, parser_config=parser_config, retrieval_config=retrieval_config)
    handler = make_web_handler(app)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Lookup Tool WebApp listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping Lookup Tool WebApp")
    finally:
        server.server_close()


def make_web_handler(app: WebApp):
    class LookupWebHandler(BaseHTTPRequestHandler):
        server_version = "LookupToolWeb/0.1"

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/app"}:
                self.write_static(STATIC_DIR / "index.html", "text/html; charset=utf-8")
                return
            if parsed.path == "/health":
                self.write_json({"status": "ok"})
                return
            if parsed.path.startswith("/static/"):
                self.handle_static(parsed.path)
                return
            if parsed.path.startswith("/api/"):
                self.handle_api("GET", parsed)
                return
            self.write_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path.startswith("/api/"):
                self.handle_api("POST", parsed)
                return
            self.write_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

        def do_PATCH(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path.startswith("/api/"):
                self.handle_api("PATCH", parsed)
                return
            self.write_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

        def do_DELETE(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path.startswith("/api/"):
                self.handle_api("DELETE", parsed)
                return
            self.write_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

        def handle_static(self, request_path: str) -> None:
            relative = unquote(request_path.removeprefix("/static/")).replace("\\", "/")
            if ".." in relative.split("/"):
                self.write_json({"error": "Invalid static path"}, HTTPStatus.BAD_REQUEST)
                return
            target = STATIC_DIR / relative
            content_type = {
                ".css": "text/css; charset=utf-8",
                ".js": "application/javascript; charset=utf-8",
                ".html": "text/html; charset=utf-8",
                ".svg": "image/svg+xml",
            }.get(target.suffix.lower(), "application/octet-stream")
            self.write_static(target, content_type)

        def handle_api(self, method: str, parsed) -> None:
            try:
                response, status = self.route_api(method, parsed)
            except KeyError as exc:
                self.write_json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
                return
            except ValueError as exc:
                self.write_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            except Exception as exc:
                self.write_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self.write_json(response, status)

        def route_api(self, method: str, parsed) -> tuple[dict[str, Any], HTTPStatus]:
            parts = [part for part in parsed.path.split("/") if part]
            query = parse_qs(parsed.query)
            if parts == ["api", "health"] and method == "GET":
                return {"status": "ok"}, HTTPStatus.OK

            if parts == ["api", "kbs"] and method == "GET":
                return {"items": app.store.list_kbs()}, HTTPStatus.OK
            if parts == ["api", "kbs"] and method == "POST":
                body = self.read_json()
                kb = app.store.create_kb(str(body.get("name", "Knowledge Base")), str(body.get("root_path", "data/kbs/new")))
                return {"item": kb}, HTTPStatus.CREATED

            if parts == ["api", "model", "settings"] and method == "GET":
                return {"item": app.store.get_model_settings()}, HTTPStatus.OK
            if parts == ["api", "model", "settings"] and method == "PATCH":
                body = self.read_json()
                return {"item": app.store.update_model_settings(settings_from_request(body))}, HTTPStatus.OK
            if parts == ["api", "model", "status"] and method == "GET":
                return {"item": model_status(app.store.get_model_settings())}, HTTPStatus.OK
            if parts == ["api", "model", "download"] and method == "POST":
                settings = app.store.get_model_settings()
                return {"item": download_recommended_model(settings)}, HTTPStatus.OK

            if len(parts) >= 3 and parts[0:2] == ["api", "kbs"]:
                kb_id = parts[2]
                if len(parts) == 3 and method == "PATCH":
                    body = self.read_json()
                    return {"item": app.store.update_kb(kb_id, name=body.get("name"))}, HTTPStatus.OK
                if len(parts) == 3 and method == "DELETE":
                    app.store.archive_kb(kb_id)
                    return {"status": "ok"}, HTTPStatus.OK
                if len(parts) == 4 and parts[3] == "migrate" and method == "POST":
                    body = self.read_json()
                    new_root = body.get("root_path")
                    if not new_root:
                        raise ValueError("root_path is required")
                    return {"item": app.store.migrate_kb(kb_id, str(new_root))}, HTTPStatus.OK
                if len(parts) == 4 and parts[3] == "documents" and method == "GET":
                    _, index, _, _ = app.components(kb_id)
                    return {"items": index.list_documents()}, HTTPStatus.OK
                if len(parts) == 4 and parts[3] == "upload" and method == "POST":
                    return self.handle_upload(kb_id)
                if len(parts) == 4 and parts[3] == "ingest" and method == "POST":
                    return self.handle_ingest(kb_id)
                if len(parts) == 5 and parts[3] == "documents" and method == "DELETE":
                    _, index, _, _ = app.components(kb_id)
                    ok = index.delete_document(parts[4])
                    return {"status": "ok" if ok else "not_found"}, HTTPStatus.OK if ok else HTTPStatus.NOT_FOUND
                if len(parts) == 4 and parts[3] == "projects" and method == "GET":
                    return {"items": app.store.list_projects(kb_id)}, HTTPStatus.OK
                if len(parts) == 4 and parts[3] == "projects" and method == "POST":
                    body = self.read_json()
                    item = app.store.create_project(kb_id, str(body.get("name", "Project")), body.get("parent_id"))
                    return {"item": item}, HTTPStatus.CREATED
                if len(parts) == 5 and parts[3] == "projects" and method == "PATCH":
                    body = self.read_json()
                    item = app.store.update_project(parts[4], name=body.get("name"), parent_id=body.get("parent_id"))
                    return {"item": item}, HTTPStatus.OK
                if len(parts) == 5 and parts[3] == "projects" and method == "DELETE":
                    app.store.archive_project(parts[4])
                    return {"status": "ok"}, HTTPStatus.OK
                if len(parts) == 4 and parts[3] == "questions" and method == "GET":
                    project_id = query.get("project_id", [None])[0]
                    return {"items": app.store.list_questions(kb_id, project_id=project_id)}, HTTPStatus.OK
                if len(parts) == 4 and parts[3] == "ask" and method == "POST":
                    return self.handle_ask(kb_id)
                if len(parts) == 5 and parts[3] == "questions" and method == "GET":
                    return {"item": app.store.get_question(parts[4])}, HTTPStatus.OK
                if len(parts) == 5 and parts[3] == "questions" and method == "PATCH":
                    body = self.read_json()
                    item = app.store.update_question(parts[4], title=body.get("title"), project_id=body.get("project_id"))
                    return {"item": item}, HTTPStatus.OK
                if len(parts) == 5 and parts[3] == "questions" and method == "DELETE":
                    app.store.archive_question(parts[4])
                    return {"status": "ok"}, HTTPStatus.OK

            raise KeyError("Route not found")

        def handle_ingest(self, kb_id: str) -> tuple[dict[str, Any], HTTPStatus]:
            body = self.read_json()
            paths = body.get("paths")
            if isinstance(paths, str):
                paths = [paths]
            if not isinstance(paths, list) or not paths:
                raise ValueError("paths must be a non-empty string or list")
            parser, index, _, _ = app.components(kb_id)
            report = ingest_paths(parser, index, [str(path) for path in paths], recursive=bool(body.get("recursive", True)))
            return report.model_dump(mode="json", by_alias=True, exclude_none=True), HTTPStatus.OK

        def handle_upload(self, kb_id: str) -> tuple[dict[str, Any], HTTPStatus]:
            parser, index, _, kb = app.components(kb_id)
            files = self.read_multipart_files()
            if not files:
                raise ValueError("No files uploaded")
            saved_paths = []
            upload_dir = Path(kb["upload_dir"])
            upload_dir.mkdir(parents=True, exist_ok=True)
            for item in files:
                filename = safe_filename(item["filename"])
                target = unique_path(upload_dir / filename)
                target.write_bytes(item["data"])
                saved_paths.append(str(target))
            report = ingest_paths(parser, index, saved_paths, recursive=False)
            return report.model_dump(mode="json", by_alias=True, exclude_none=True), HTTPStatus.OK

        def handle_ask(self, kb_id: str) -> tuple[dict[str, Any], HTTPStatus]:
            body = self.read_json()
            query = str(body.get("query", "")).strip()
            if not query:
                raise ValueError("query is required")
            task = str(body.get("task", "answer"))
            top_k = body.get("top_k")
            project_id = body.get("project_id")
            _, _, extractor, _ = app.components(kb_id)
            result = extractor.extract(query, task=task, top_k=top_k)
            payload = result.model_dump(mode="json", by_alias=True, exclude_none=True)
            generated = synthesize_answer(app.store.get_model_settings(), query, payload)
            if generated:
                payload.setdefault("items", []).insert(0, generated)
                payload.setdefault("meta", {})["model_used"] = generated.get("model")
            elif app.store.get_model_settings().get("enabled"):
                payload.setdefault("warnings", []).append("Model is enabled but the configured endpoint did not return an answer.")
            record = app.store.create_question(
                kb_id=kb_id,
                project_id=project_id,
                query=query,
                task=task,
                top_k=top_k,
                result=payload,
                title=body.get("title"),
            )
            return {"record": record, "result": payload}, HTTPStatus.OK

        def read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length <= 0:
                return {}
            raw = self.rfile.read(length).decode("utf-8-sig")
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError("JSON body must be an object")
            return value

        def read_multipart_files(self) -> list[dict[str, Any]]:
            content_type = self.headers.get("Content-Type", "")
            match = re.search(r"boundary=(?P<boundary>[^;]+)", content_type)
            if not match:
                raise ValueError("multipart boundary is missing")
            boundary = match.group("boundary").strip('"').encode("utf-8")
            length = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(length)
            files: list[dict[str, Any]] = []
            for part in body.split(b"--" + boundary):
                part = part.strip()
                if not part or part == b"--":
                    continue
                if part.endswith(b"--"):
                    part = part[:-2].strip()
                if b"\r\n\r\n" not in part:
                    continue
                header_bytes, data = part.split(b"\r\n\r\n", 1)
                headers = header_bytes.decode("utf-8", errors="replace")
                disposition = next((line for line in headers.splitlines() if line.lower().startswith("content-disposition:")), "")
                filename_match = re.search(r'filename="(?P<filename>[^"]*)"', disposition)
                if not filename_match:
                    continue
                filename = filename_match.group("filename")
                if data.endswith(b"\r\n"):
                    data = data[:-2]
                files.append({"filename": filename, "data": data})
            return files

        def write_static(self, path: Path, content_type: str) -> None:
            if not path.exists() or not path.is_file():
                self.write_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
                return
            raw = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def write_json(self, payload: dict[str, Any], status: int | HTTPStatus = HTTPStatus.OK) -> None:
            raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"{self.address_string()} - {format % args}")

    return LookupWebHandler


def ingest_paths(parser: DocumentParser, index: SQLiteIndex, paths: list[str], recursive: bool = True) -> IngestReport:
    documents = []
    warnings = []
    for item in paths:
        try:
            parsed_docs = parser.parse_path(item, recursive=recursive)
            for document in parsed_docs:
                index.upsert_document(document)
                documents.append(
                    {
                        "doc_id": document.doc_id,
                        "path": document.path,
                        "blocks": len(document.blocks),
                        "sha256": document.sha256,
                    }
                )
        except Exception as exc:
            warnings.append(f"{item}: {exc}")
    return IngestReport(status="ok" if not warnings else "partial", documents=documents, warnings=warnings)


def safe_filename(filename: str) -> str:
    name = Path(filename).name
    name = re.sub(r"[^A-Za-z0-9_.()\-\u4e00-\u9fff ]+", "_", name).strip()
    return name or "upload.bin"


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(2, 10000):
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise ValueError(f"Cannot allocate upload path for {path}")


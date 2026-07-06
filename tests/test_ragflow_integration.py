from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.request import Request, urlopen

from lookup_tool.app_store import AppStore
from lookup_tool.ragflow import (
    RagFlowClient,
    apply_ragflow_retrieval,
    is_offline_url_allowed,
    ragflow_settings_from_request,
)
from lookup_tool.webapp import WebApp, make_web_handler


class FakeRagFlowHandler(BaseHTTPRequestHandler):
    uploads: list[bytes] = []

    def do_GET(self) -> None:
        if self.path == "/api/v1/datasets":
            self.write_json({"code": 0, "data": [{"id": "dataset-a", "name": "offline"}]})
            return
        self.send_error(404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length)
        if self.path == "/api/v1/retrieval":
            payload = json.loads(body.decode("utf-8"))
            self.write_json(
                {
                    "code": 0,
                    "data": {
                        "chunks": [
                            {
                                "id": "chunk-1",
                                "content": f"RAGFlow local evidence for {payload['question']}",
                                "document_name": "ragflow-spec.pdf",
                                "document_id": "doc-ragflow",
                                "score": 0.91,
                                "page": 3,
                            }
                        ]
                    },
                }
            )
            return
        if self.path == "/api/v1/datasets/dataset-a/documents":
            self.uploads.append(body)
            self.write_json({"code": 0, "data": {"document_id": "uploaded-doc"}})
            return
        self.send_error(404)

    def write_json(self, payload: dict) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, format: str, *args) -> None:
        return


class RagFlowIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeRagFlowHandler.uploads = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeRagFlowHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def settings(self) -> dict:
        return {
            "enabled": True,
            "mode": "hybrid",
            "base_url": self.base_url,
            "dataset_id": "dataset-a",
            "timeout_seconds": 5,
            "top_k": 3,
            "auto_sync_uploads": True,
            "status_path": "/api/v1/datasets",
            "retrieval_path": "/api/v1/retrieval",
            "upload_path_template": "/api/v1/datasets/{dataset_id}/documents",
            "upload_field": "file",
            "api_key": "local-token",
        }

    def test_offline_url_guard_blocks_public_hosts(self) -> None:
        self.assertTrue(is_offline_url_allowed("http://127.0.0.1:9380"))
        self.assertTrue(is_offline_url_allowed("http://192.168.1.8:9380"))
        self.assertFalse(is_offline_url_allowed("https://example.com"))

        client = RagFlowClient({**self.settings(), "base_url": "https://example.com"})
        chunks, warnings = client.retrieve("test", top_k=1)
        self.assertFalse(chunks)
        self.assertTrue(any("base_url" in warning for warning in warnings))

    def test_client_status_retrieve_and_upload_are_local(self) -> None:
        client = RagFlowClient(self.settings())
        self.assertTrue(client.status()["endpoint_ok"])

        chunks, warnings = client.retrieve("wheel diameter requirement", top_k=1)
        self.assertFalse(warnings)
        self.assertEqual(chunks[0].document_name, "ragflow-spec.pdf")
        self.assertIn("wheel diameter requirement", chunks[0].text)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.txt"
            path.write_text("offline sample", encoding="utf-8")
            result = client.upload_documents([path])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(FakeRagFlowHandler.uploads), 1)

    def test_apply_ragflow_retrieval_keeps_local_items_in_hybrid_mode(self) -> None:
        payload = {
            "schema": "lookup.result.v1",
            "task": "answer",
            "status": "ok",
            "items": [{"id": "local.1", "type": "supporting_block", "evidence": []}],
            "evidence": {},
            "warnings": [],
        }
        merged = apply_ragflow_retrieval(payload, settings=self.settings(), query="ATP requirement", top_k=2)
        self.assertEqual(merged["status"], "ok")
        self.assertTrue(any(item["type"] == "supporting_block" for item in merged["items"]))
        self.assertTrue(any(item["type"] == "ragflow_chunk" for item in merged["items"]))
        self.assertEqual(merged["meta"]["ragflow_chunks"], 1)

    def test_app_store_persists_ragflow_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AppStore(Path(tmp) / "app.sqlite")
            settings = store.update_ragflow_settings(
                ragflow_settings_from_request(
                    {
                        "enabled": True,
                        "mode": "hybrid",
                        "base_url": self.base_url,
                        "dataset_id": "dataset-a",
                    }
                )
            )
            self.assertTrue(settings["enabled"])
            self.assertEqual(store.get_ragflow_settings()["dataset_id"], "dataset-a")

    def test_webapp_exposes_ragflow_settings_and_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = WebApp(app_store=AppStore(Path(tmp) / "app.sqlite"))
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_web_handler(app))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            payload = json.dumps(
                {
                    "enabled": True,
                    "mode": "hybrid",
                    "base_url": self.base_url,
                    "dataset_id": "dataset-a",
                }
            ).encode("utf-8")
            request = Request(
                base + "/api/ragflow/settings",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="PATCH",
            )
            saved = json.loads(urlopen(request, timeout=5).read().decode("utf-8"))
            status = json.loads(urlopen(base + "/api/ragflow/status", timeout=5).read().decode("utf-8"))
            server.shutdown()
            server.server_close()
        self.assertEqual(saved["item"]["mode"], "hybrid")
        self.assertTrue(status["item"]["endpoint_ok"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from lookup_tool.llm import extract_choice_text
from lookup_tool.model_manager import find_installed_llama_server, find_installed_model, model_status


class LlmClientTest(unittest.TestCase):
    def test_extract_choice_text_reads_reasoning_fallback(self) -> None:
        raw = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "fallback answer",
                    }
                }
            ]
        }
        self.assertEqual(extract_choice_text(raw), "fallback answer")

    def test_extract_choice_text_reads_openai_content_parts(self) -> None:
        raw = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "part one"}, {"text": "part two"}],
                    }
                }
            ]
        }
        self.assertEqual(extract_choice_text(raw), "part one\npart two")

    def test_find_installed_llama_server_ignores_empty_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch("lookup_tool.model_manager.Path.cwd", return_value=Path(tmp)):
                self.assertIsNone(
                    find_installed_llama_server(
                        {
                            "llama_server_path": "",
                            "local_models_dir": "",
                            "local_model_path": "",
                        }
                    )
                )

    def test_model_status_ignores_empty_model_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch("lookup_tool.model_manager.Path.cwd", return_value=Path(tmp)):
                status = model_status({"local_model_path": "", "base_url": ""})
                self.assertFalse(status["local_model_exists"])
                self.assertIsNone(status["local_model_path"])

    def test_model_and_llama_search_walks_up_from_release_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            release_dir = root / "dist" / "LocalMathRAG-win-x64"
            model = root / "data" / "models" / "Qwen3-8B-Q4_K_M.gguf"
            server = root / "data" / "runtime" / "llama.cpp" / "b9878-cuda12" / "llama-server.exe"
            release_dir.mkdir(parents=True)
            model.parent.mkdir(parents=True)
            server.parent.mkdir(parents=True)
            model.write_bytes(b"model")
            server.write_bytes(b"server")
            with patch("lookup_tool.model_manager.Path.cwd", return_value=release_dir):
                settings = {
                    "local_model_path": "",
                    "local_models_dir": "",
                    "llama_server_path": "",
                    "recommended_file": "Qwen3-8B-Q4_K_M.gguf",
                }
                self.assertEqual(find_installed_model(settings), model)
                self.assertEqual(find_installed_llama_server(settings), server)


if __name__ == "__main__":
    unittest.main()

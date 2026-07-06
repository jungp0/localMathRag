from pathlib import Path
import tempfile
import unittest

from lookup_tool.config import ParserConfig, RetrievalConfig
from lookup_tool.app_store import AppStore
from lookup_tool.extractor import AgentExtractor
from lookup_tool.index import SQLiteIndex
from lookup_tool.parsers import DocumentParser


class FormulaPipelineTest(unittest.TestCase):
    def test_non_ascii_text_files_use_utf8_bom(self) -> None:
        checked_suffixes = {".py", ".md", ".csv", ".txt", ".html", ".css", ".js"}
        offenders: list[str] = []
        for path in Path(".").rglob("*"):
            if not path.is_file() or ".git" in path.parts or path.suffix.lower() not in checked_suffixes:
                continue
            data = path.read_bytes()
            try:
                text = data.decode("utf-8-sig")
            except UnicodeDecodeError:
                continue
            if any(ord(char) > 127 for char in text) and not data.startswith(b"\xef\xbb\xbf"):
                offenders.append(str(path))
        self.assertFalse(offenders, "Non-ASCII text files must use UTF-8 BOM: " + ", ".join(offenders))

    def test_app_store_manages_kbs_projects_and_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = AppStore(root / "app.sqlite")
            kb = store.create_kb("Rail Specs", root / "kb-a")
            self.assertTrue(Path(kb["db_path"]).parent.exists())

            project = store.create_project(kb["id"], "ATP")
            self.assertEqual(project["name"], "ATP")

            question = store.create_question(
                kb_id=kb["id"],
                project_id=project["id"],
                query="轮径校正条件",
                task="requirement_extract",
                top_k=6,
                result={"schema": "lookup.result.v1", "items": []},
            )
            records = store.list_questions(kb["id"], project_id=project["id"])
            self.assertEqual(records[0]["id"], question["id"])

            migrated = store.migrate_kb(kb["id"], root / "kb-b")
            self.assertIn("kb-b", migrated["root_path"])

    def test_kalman_formula_extract(self) -> None:
        sample = Path("examples/kalman_sample.md")
        parser = DocumentParser(ParserConfig(chunk_target_chars=800, chunk_overlap_chars=80))
        docs = parser.parse_file(sample)

        equation_blocks = [block for block in docs.blocks if block.kind == "equation"]
        self.assertTrue(equation_blocks)
        self.assertTrue(any("P_{k|k-1}" in (block.latex or "") for block in equation_blocks))

        with tempfile.TemporaryDirectory() as tmp:
            index = SQLiteIndex(Path(tmp) / "lookup.sqlite", RetrievalConfig(top_k=8))
            index.upsert_document(docs)
            extractor = AgentExtractor(index)

            result = extractor.extract("Kalman covariance prediction formula", task="formula_extract", top_k=6)
            self.assertEqual(result.status, "ok")
            self.assertTrue(result.items)
            self.assertTrue(any("covariance_prediction" in item.get("domain", []) for item in result.items))
            self.assertTrue(result.evidence)


    def test_parameter_extract(self) -> None:
        parser = DocumentParser(ParserConfig(chunk_target_chars=800, chunk_overlap_chars=80))
        doc = parser.parse_file(Path("examples/kalman_sample.md"))
        with tempfile.TemporaryDirectory() as tmp:
            index = SQLiteIndex(Path(tmp) / "lookup.sqlite")
            index.upsert_document(doc)
            extractor = AgentExtractor(index)

            result = extractor.extract("sampling period parameter", task="parameter_extract", top_k=5)
            self.assertEqual(result.status, "ok")
            self.assertTrue(any(item.get("name") == "sampling_period_ms" for item in result.items))

    def test_visual_extract_uses_caption_and_nearby_text(self) -> None:
        parser = DocumentParser(ParserConfig(chunk_target_chars=800, chunk_overlap_chars=80))
        doc = parser.parse_file(Path("examples/kalman_sample.md"))
        visual_blocks = [block for block in doc.blocks if block.kind == "visual_object"]
        self.assertTrue(visual_blocks)
        self.assertTrue(any(block.metadata.get("caption") for block in visual_blocks))
        self.assertTrue(any("covariance prediction" in (block.metadata.get("nearby_text") or "") for block in visual_blocks))

        with tempfile.TemporaryDirectory() as tmp:
            index = SQLiteIndex(Path(tmp) / "lookup.sqlite", RetrievalConfig(top_k=8))
            index.upsert_document(doc)
            extractor = AgentExtractor(index)

            result = extractor.extract("Kalman covariance propagation diagram", task="visual_extract", top_k=6)
            self.assertEqual(result.status, "ok")
            self.assertTrue(result.items)
            self.assertTrue(any(item.get("type") == "visual_object" for item in result.items))
            self.assertTrue(any(item.get("visual_type") == "diagram" for item in result.items))

    def test_table_internal_requirement_extract(self) -> None:
        parser = DocumentParser(ParserConfig(chunk_target_chars=800, chunk_overlap_chars=80))
        doc = parser.parse_file(Path("examples/requirement_table.csv"))
        table_rows = [block for block in doc.blocks if block.kind == "table_row"]
        table_cells = [block for block in doc.blocks if block.kind == "table_cell"]
        self.assertTrue(table_rows)
        self.assertTrue(table_cells)
        self.assertTrue(any(block.metadata.get("row_label") == "描述" for block in table_rows))

        with tempfile.TemporaryDirectory() as tmp:
            index = SQLiteIndex(Path(tmp) / "lookup.sqlite", RetrievalConfig(top_k=12))
            index.upsert_document(doc)
            extractor = AgentExtractor(index)

            result = extractor.extract("轮径校正 描述 条件检查", task="requirement_extract", top_k=10)
            self.assertEqual(result.status, "ok")
            self.assertTrue(any("条件检查" in item.get("predicate", "") for item in result.items))
            self.assertTrue(any(item.get("subject") == "描述" for item in result.items))


if __name__ == "__main__":
    unittest.main()

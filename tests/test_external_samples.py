from __future__ import annotations

from collections import Counter
import os
from pathlib import Path
import tempfile
import unittest

from lookup_tool.config import ParserConfig, RetrievalConfig
from lookup_tool.extractor import AgentExtractor
from lookup_tool.index import SQLiteIndex
from lookup_tool.parsers import DocumentParser


def external_samples_dir() -> Path | None:
    value = os.environ.get("LOOKUP_TOOL_EXTERNAL_SAMPLES")
    if not value:
        return None
    path = Path(value)
    return path if path.exists() else None


@unittest.skipUnless(external_samples_dir(), "Set LOOKUP_TOOL_EXTERNAL_SAMPLES to run public document smoke tests.")
class ExternalSamplesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sample_dir = external_samples_dir()
        assert sample_dir is not None
        cls.sample_dir = sample_dir
        cls.parser = DocumentParser(
            ParserConfig(
                artifact_dir=Path("data/external_artifacts"),
                chunk_target_chars=1000,
                chunk_overlap_chars=100,
            )
        )
        cls.documents = {
            "pdf": cls.parser.parse_file(sample_dir / "kalman_arxiv_1710.04055.pdf"),
            "docx": cls.parser.parse_file(sample_dir / "calibre_demo.docx"),
            "xlsx": cls.parser.parse_file(sample_dir / "financial_sample.xlsx"),
        }

    def test_real_documents_produce_expected_block_shapes(self) -> None:
        pdf_counts = Counter(block.kind for block in self.documents["pdf"].blocks)
        docx_counts = Counter(block.kind for block in self.documents["docx"].blocks)
        xlsx_counts = Counter(block.kind for block in self.documents["xlsx"].blocks)

        self.assertGreaterEqual(pdf_counts["equation"], 20)
        self.assertGreaterEqual(pdf_counts["visual_object"], 1)
        self.assertGreaterEqual(docx_counts["table_cell"], 20)
        self.assertGreaterEqual(docx_counts["visual_object"], 1)
        self.assertGreaterEqual(xlsx_counts["table_cell"], 1000)

    def test_real_documents_have_unique_block_ids(self) -> None:
        for name, document in self.documents.items():
            block_ids = [block.block_id for block in document.blocks]
            duplicates = [block_id for block_id, count in Counter(block_ids).items() if count > 1]
            self.assertFalse(duplicates, f"{name} has duplicate block ids: {duplicates[:5]}")

    def test_xlsx_wide_table_uses_column_names_for_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index = SQLiteIndex(Path(tmp) / "lookup.sqlite", RetrievalConfig(top_k=12))
            index.upsert_document(self.documents["xlsx"])
            result = AgentExtractor(index).extract("Gross Sales Profit parameter value", task="parameter_extract", top_k=10)

        self.assertEqual(result.status, "ok")
        self.assertTrue(any(item.get("name") == "Gross Sales" for item in result.items))
        self.assertFalse(any("2014-06-01 00" in str(item.get("name", "")) for item in result.items))

    def test_mixed_real_documents_can_be_indexed_and_extracted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index = SQLiteIndex(Path(tmp) / "lookup.sqlite", RetrievalConfig(top_k=12))
            for document in self.documents.values():
                index.upsert_document(document)
            extractor = AgentExtractor(index)
            formula_result = extractor.extract("Kalman gain covariance formula", task="formula_extract", top_k=10)
            visual_result = extractor.extract("figure plot Kalman", task="visual_extract", top_k=10)

        self.assertEqual(formula_result.status, "ok")
        self.assertTrue(any("kalman_filter" in item.get("domain", []) for item in formula_result.items))
        self.assertEqual(visual_result.status, "ok")
        self.assertTrue(any(item.get("type") == "visual_object" for item in visual_result.items))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
import json
import math
import re
import sqlite3
import time
from typing import Iterator

from .config import RetrievalConfig
from .formula import classify_domain, extract_symbols
from .models import EvidenceRef, ParsedBlock, ParsedDocument, SearchHit, SourceRef


TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[\u4e00-\u9fff]{2,}|\d+(?:\.\d+)?")
FORMULA_INTENT_PATTERN = re.compile(
    r"formula|equation|latex|variance|covariance|kalman|matrix|gain|"
    r"\u516c\u5f0f|\u65b9\u5dee|\u534f\u65b9\u5dee|\u77e9\u9635|"
    r"\u5361\u5c14\u66fc|\u589e\u76ca|\u63a8\u5bfc|\u72b6\u6001\u8f6c\u79fb",
    re.I,
)
VISUAL_INTENT_PATTERN = re.compile(
    r"figure|fig\.?|image|visual|picture|photo|chart|plot|graph|curve|diagram|"
    r"screenshot|flow|architecture|block diagram|"
    r"\u56fe|\u56fe\u7247|\u56fe\u8868|\u66f2\u7ebf|\u622a\u56fe|\u6d41\u7a0b|"
    r"\u67b6\u6784|\u6846\u56fe|\u793a\u610f",
    re.I,
)


class SQLiteIndex:
    def __init__(self, db_path: str | Path, retrieval: RetrievalConfig | None = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.retrieval = retrieval or RetrievalConfig()
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
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    parser_version TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    indexed_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS blocks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    block_id TEXT UNIQUE NOT NULL,
                    doc_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    text TEXT NOT NULL,
                    latex TEXT,
                    normalized_latex TEXT,
                    symbols_json TEXT NOT NULL,
                    operators_json TEXT NOT NULL,
                    source_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_blocks_doc_seq ON blocks(doc_id, seq)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_blocks_kind ON blocks(kind)")
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS blocks_fts USING fts5(
                    block_id UNINDEXED,
                    doc_id UNINDEXED,
                    kind UNINDEXED,
                    text,
                    latex,
                    symbols
                )
                """
            )

    def upsert_document(self, document: ParsedDocument) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (document.doc_id,))
            conn.execute("DELETE FROM blocks WHERE doc_id = ?", (document.doc_id,))
            conn.execute(
                """
                INSERT INTO documents(doc_id, path, sha256, parser_version, metadata_json, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    document.doc_id,
                    document.path,
                    document.sha256,
                    document.parser_version,
                    json.dumps(document.metadata, ensure_ascii=False),
                    time.time(),
                ),
            )
            for seq, block in enumerate(document.blocks):
                self._insert_block(conn, block, seq)
            self._rebuild_fts(conn)

    def upsert_documents(self, documents: list[ParsedDocument]) -> None:
        for document in documents:
            self.upsert_document(document)

    def list_documents(self) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    d.doc_id,
                    d.path,
                    d.sha256,
                    d.parser_version,
                    d.metadata_json,
                    d.indexed_at,
                    COUNT(b.block_id) AS block_count,
                    SUM(CASE WHEN b.kind = 'equation' THEN 1 ELSE 0 END) AS equation_count,
                    SUM(CASE WHEN b.kind IN ('table', 'table_row', 'table_cell') THEN 1 ELSE 0 END) AS table_block_count,
                    SUM(CASE WHEN b.kind = 'visual_object' THEN 1 ELSE 0 END) AS visual_count
                FROM documents d
                LEFT JOIN blocks b ON b.doc_id = d.doc_id
                GROUP BY d.doc_id
                ORDER BY d.indexed_at DESC
                """
            ).fetchall()
        documents = []
        for row in rows:
            item = dict(row)
            try:
                item["metadata"] = json.loads(item.pop("metadata_json"))
            except (json.JSONDecodeError, TypeError):
                item["metadata"] = {}
            documents.append(item)
        return documents

    def delete_document(self, doc_id: str) -> bool:
        with self.connect() as conn:
            exists = conn.execute("SELECT 1 FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
            if exists is None:
                return False
            conn.execute("DELETE FROM blocks WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            self._rebuild_fts(conn)
        return True

    def _insert_block(self, conn: sqlite3.Connection, block: ParsedBlock, seq: int) -> None:
        conn.execute(
            """
            INSERT INTO blocks(
                block_id, doc_id, seq, kind, text, latex, normalized_latex,
                symbols_json, operators_json, source_json, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                block.block_id,
                block.doc_id,
                seq,
                block.kind,
                block.text,
                block.latex,
                block.normalized_latex,
                json.dumps(block.symbols, ensure_ascii=False),
                json.dumps(block.operators, ensure_ascii=False),
                block.source.model_dump_json(by_alias=True),
                json.dumps(block.metadata, ensure_ascii=False),
            ),
        )

    def _rebuild_fts(self, conn: sqlite3.Connection) -> None:
        conn.execute("DELETE FROM blocks_fts")
        rows = conn.execute(
            "SELECT block_id, doc_id, kind, text, latex, symbols_json FROM blocks ORDER BY doc_id, seq"
        ).fetchall()
        for row in rows:
            symbols = " ".join(json.loads(row["symbols_json"]))
            conn.execute(
                """
                INSERT INTO blocks_fts(block_id, doc_id, kind, text, latex, symbols)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (row["block_id"], row["doc_id"], row["kind"], row["text"], row["latex"] or "", symbols),
            )

    def search(self, query: str, top_k: int | None = None, kinds: list[str] | None = None) -> list[SearchHit]:
        limit = top_k or self.retrieval.top_k
        candidates: dict[str, tuple[sqlite3.Row, float]] = {}
        with self.connect() as conn:
            for row, score in self._fts_candidates(conn, query, limit * 8, kinds):
                candidates[row["block_id"]] = (row, max(score, candidates.get(row["block_id"], (row, 0.0))[1]))

            if self._is_formula_intent(query):
                for row in self._equation_candidates(conn, limit * 20):
                    old = candidates.get(row["block_id"])
                    candidates[row["block_id"]] = (row, old[1] if old else 0.05)

            if self._is_visual_intent(query):
                for row in self._visual_candidates(conn, limit * 20):
                    old = candidates.get(row["block_id"])
                    candidates[row["block_id"]] = (row, old[1] if old else 0.05)

            if len(candidates) < limit:
                for row in self._scan_candidates(conn, kinds):
                    old = candidates.get(row["block_id"])
                    if old is None:
                        candidates[row["block_id"]] = (row, 0.01)

            ranked = []
            for row, base_score in candidates.values():
                ranked.append((row, self._rerank_score(query, row, base_score)))
            ranked.sort(key=lambda item: item[1], reverse=True)

            top_rows = ranked[:limit]
            if self.retrieval.context_window > 0:
                top_rows = self._expand_context(conn, top_rows, self.retrieval.context_window, limit)

        return [row_to_hit(row, score) for row, score in top_rows[:limit]]

    def _fts_candidates(
        self,
        conn: sqlite3.Connection,
        query: str,
        limit: int,
        kinds: list[str] | None,
    ) -> list[tuple[sqlite3.Row, float]]:
        fts_query = build_fts_query(query)
        if not fts_query:
            return []
        kind_filter = ""
        params: list[object] = [fts_query]
        if kinds:
            placeholders = ",".join("?" for _ in kinds)
            kind_filter = f"AND b.kind IN ({placeholders})"
            params.extend(kinds)
        params.append(limit)
        try:
            rows = conn.execute(
                f"""
                SELECT b.*, bm25(blocks_fts) AS rank
                FROM blocks_fts
                JOIN blocks b ON b.block_id = blocks_fts.block_id
                WHERE blocks_fts MATCH ? {kind_filter}
                ORDER BY rank
                LIMIT ?
                """,
                params,
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(row, 1.0 / (1.0 + abs(float(row["rank"] or 0.0)))) for row in rows]

    def _equation_candidates(self, conn: sqlite3.Connection, limit: int) -> list[sqlite3.Row]:
        return conn.execute(
            "SELECT * FROM blocks WHERE kind = 'equation' ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()

    def _visual_candidates(self, conn: sqlite3.Connection, limit: int) -> list[sqlite3.Row]:
        return conn.execute(
            "SELECT * FROM blocks WHERE kind = 'visual_object' ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()

    def _scan_candidates(self, conn: sqlite3.Connection, kinds: list[str] | None) -> list[sqlite3.Row]:
        if kinds:
            placeholders = ",".join("?" for _ in kinds)
            return conn.execute(f"SELECT * FROM blocks WHERE kind IN ({placeholders})", kinds).fetchall()
        return conn.execute("SELECT * FROM blocks").fetchall()

    def _rerank_score(self, query: str, row: sqlite3.Row, base_score: float) -> float:
        text = row["text"] or ""
        latex = row["latex"] or ""
        normalized = row["normalized_latex"] or latex
        symbols = json.loads(row["symbols_json"])
        query_tokens = set(tokenize(query))
        block_tokens = set(tokenize(f"{text} {latex}"))
        token_score = jaccard(query_tokens, block_tokens)

        query_symbols = set(extract_symbols(query))
        symbol_score = jaccard(query_symbols, set(symbols)) if query_symbols else 0.0
        query_domains = set(classify_domain(query, query))
        block_domains = set(classify_domain(latex or text, text))
        domain_score = jaccard(query_domains, block_domains) if query_domains else 0.0

        exact_formula_score = 0.0
        compact_query = re.sub(r"\s+", "", query)
        if compact_query and compact_query in re.sub(r"\s+", "", normalized):
            exact_formula_score = 1.0
        elif any(sym in normalized for sym in query_symbols):
            exact_formula_score = 0.35

        kind_boost = 0.0
        if row["kind"] == "equation" and self._is_formula_intent(query):
            kind_boost = self.retrieval.formula_boost
        elif row["kind"] in {"table", "table_row", "table_cell", "cell"} and looks_parameter_query(query):
            kind_boost = 0.65
        elif row["kind"] in {"table_row", "table_cell"} and looks_table_detail_query(query):
            kind_boost = 0.9
        elif row["kind"] == "visual_object" and self._is_visual_intent(query):
            kind_boost = 1.75 + visual_type_match_bonus(query, json.loads(row["metadata_json"]))

        return (
            base_score
            + 1.8 * token_score
            + 2.4 * symbol_score
            + 2.0 * domain_score
            + 1.4 * exact_formula_score
            + kind_boost
        )

    def _expand_context(
        self,
        conn: sqlite3.Connection,
        ranked: list[tuple[sqlite3.Row, float]],
        window: int,
        limit: int,
    ) -> list[tuple[sqlite3.Row, float]]:
        merged: dict[str, tuple[sqlite3.Row, float]] = {row["block_id"]: (row, score) for row, score in ranked}
        for row, score in ranked[: max(1, math.ceil(limit / 2))]:
            neighbors = conn.execute(
                """
                SELECT * FROM blocks
                WHERE doc_id = ? AND seq BETWEEN ? AND ?
                ORDER BY seq
                """,
                (row["doc_id"], int(row["seq"]) - window, int(row["seq"]) + window),
            ).fetchall()
            for neighbor in neighbors:
                if neighbor["block_id"] not in merged:
                    merged[neighbor["block_id"]] = (neighbor, score * 0.65)
        values = list(merged.values())
        values.sort(key=lambda item: item[1], reverse=True)
        return values

    def get_block(self, block_id: str) -> SearchHit | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM blocks WHERE block_id = ?", (block_id,)).fetchone()
        return row_to_hit(row, 1.0) if row else None

    def evidence_for_hits(self, hits: list[SearchHit], include_preview: bool = False) -> dict[str, EvidenceRef]:
        evidence: dict[str, EvidenceRef] = {}
        for hit in hits:
            evidence_id = f"ev.{hit.block_id}"
            src = hit.source
            metadata = hit.metadata
            evidence[evidence_id] = EvidenceRef(
                doc_id=src.doc_id,
                path=src.path,
                block_id=src.block_id,
                page=src.page,
                section=src.section,
                sheet=src.sheet,
                cell_range=src.cell_range,
                bbox=src.bbox,
                equation_no=src.equation_no,
                visual_no=src.visual_no,
                asset_path=src.asset_path or metadata.get("asset_path"),
                table_no=src.table_no,
                row_index=src.row_index or metadata.get("row_index"),
                col_index=src.col_index or metadata.get("col_index"),
                row_label=metadata.get("row_label"),
                column_name=metadata.get("column_name"),
                visual_type=metadata.get("visual_type"),
                caption=metadata.get("caption"),
                text_preview=hit.text[:500] if include_preview else None,
            )
        return evidence

    def _is_formula_intent(self, query: str) -> bool:
        return bool(FORMULA_INTENT_PATTERN.search(query)) or bool(extract_symbols(query))

    def _is_visual_intent(self, query: str) -> bool:
        return bool(VISUAL_INTENT_PATTERN.search(query))


def row_to_hit(row: sqlite3.Row, score: float) -> SearchHit:
    source = SourceRef.model_validate_json(row["source_json"])
    return SearchHit(
        block_id=row["block_id"],
        doc_id=row["doc_id"],
        kind=row["kind"],
        score=float(score),
        text=row["text"],
        source=source,
        latex=row["latex"],
        normalized_latex=row["normalized_latex"],
        symbols=json.loads(row["symbols_json"]),
        operators=json.loads(row["operators_json"]),
        metadata=json.loads(row["metadata_json"]),
    )


def build_fts_query(query: str) -> str | None:
    tokens = tokenize(query)
    if not tokens:
        return None
    return " OR ".join(f'"{token}"' for token in tokens[:24])


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def looks_parameter_query(query: str) -> bool:
    text = query.lower()
    return any(
        needle in text
        for needle in [
            "parameter",
            "param",
            "value",
            "threshold",
            "unit",
            "\u53c2\u6570",
            "\u9608\u503c",
            "\u5355\u4f4d",
        ]
    )


def looks_table_detail_query(query: str) -> bool:
    text = query.lower()
    return any(
        needle in text
        for needle in [
            "table",
            "row",
            "cell",
            "description",
            "input",
            "output",
            "exception",
            "handling",
            "requirement",
            "shall",
            "should",
            "\u8868\u683c",
            "\u63cf\u8ff0",
            "\u8f93\u5165",
            "\u8f93\u51fa",
            "\u5f02\u5e38",
            "\u5904\u7406",
            "\u8981\u6c42",
        ]
    )


def visual_type_match_bonus(query: str, metadata: dict) -> float:
    visual_type = str(metadata.get("visual_type") or "")
    text = query.lower()
    aliases = {
        "chart": ["chart", "plot", "graph", "curve", "\u56fe\u8868", "\u66f2\u7ebf"],
        "diagram": ["diagram", "flow", "architecture", "block", "\u6d41\u7a0b", "\u6846\u56fe", "\u67b6\u6784"],
        "screenshot": ["screenshot", "screen", "\u622a\u56fe", "\u754c\u9762"],
        "formula_image": ["formula", "equation", "\u516c\u5f0f"],
        "table_image": ["table", "\u8868\u683c"],
        "photo": ["photo", "picture", "\u7167\u7247", "\u56fe\u7247"],
    }
    if any(alias.lower() in text for alias in aliases.get(visual_type, [])):
        return 0.75
    return 0.0


def group_hits_by_doc(hits: list[SearchHit]) -> dict[str, list[SearchHit]]:
    grouped: dict[str, list[SearchHit]] = defaultdict(list)
    for hit in hits:
        grouped[hit.doc_id].append(hit)
    return dict(grouped)

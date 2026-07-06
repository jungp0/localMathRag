from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .config import load_config
from .extractor import AgentExtractor
from .index import SQLiteIndex
from .models import IngestReport
from .parsers import DocumentParser
from .webapp import serve_webapp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lookup-tool", description="Local formula-aware document lookup tool")
    parser.add_argument("--config", help="Path to TOML config file")
    parser.add_argument("--db", help="Override SQLite DB path")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Parse and index files/directories")
    ingest.add_argument("paths", nargs="+")
    ingest.add_argument("--no-recursive", action="store_true")

    search = sub.add_parser("search", help="Search indexed documents")
    search.add_argument("query")
    search.add_argument("--top-k", type=int)

    extract = sub.add_parser("extract", help="Extract agent-compact facts")
    extract.add_argument("query")
    extract.add_argument(
        "--task",
        default="formula_extract",
        choices=["formula_extract", "visual_extract", "parameter_extract", "requirement_extract", "answer"],
    )
    extract.add_argument("--top-k", type=int)

    serve = sub.add_parser("serve", help="Start local WebApp and HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)

    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.db:
        config.storage.db_path = Path(args.db)

    if args.command == "serve":
        serve_webapp(
            host=args.host,
            port=args.port,
            parser_config=config.parser,
            retrieval_config=config.retrieval,
        )
        return 0

    doc_parser = DocumentParser(config.parser)
    index = SQLiteIndex(config.storage.db_path, config.retrieval)
    extractor = AgentExtractor(index)

    if args.command == "ingest":
        report = run_ingest(doc_parser, index, args.paths, recursive=not args.no_recursive)
        print_json(report.model_dump(mode="json", by_alias=True, exclude_none=True))
        return 0 if report.status in {"ok", "partial"} else 1

    if args.command == "search":
        result = extractor.search(args.query, top_k=args.top_k)
        print_json(result.model_dump(mode="json", by_alias=True, exclude_none=True))
        return 0

    if args.command == "extract":
        result = extractor.extract(args.query, task=args.task, top_k=args.top_k)
        print_json(result.model_dump(mode="json", by_alias=True, exclude_none=True))
        return 0

    return 1


def run_ingest(doc_parser: DocumentParser, index: SQLiteIndex, paths: list[str], recursive: bool = True) -> IngestReport:
    documents = []
    warnings = []
    for item in paths:
        try:
            parsed_docs = doc_parser.parse_path(item, recursive=recursive)
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


def print_json(payload: dict) -> None:
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())

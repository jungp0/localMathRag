from __future__ import annotations

from collections import OrderedDict
import re
from typing import Any

from .formula import (
    classify_domain,
    extract_formula_candidates,
    infer_assumptions,
    infer_variable_roles,
    stable_id,
)
from .index import SQLiteIndex
from .models import EquationItem, LookupResult, ParameterItem, RequirementItem, SearchHit, TaskKind, VisualItem


PARAMETER_PATTERN = re.compile(
    r"(?P<name>[A-Za-z][A-Za-z0-9_. /()\-]{1,80}|[\u4e00-\u9fffA-Za-z0-9_ /()\-]{2,80})"
    r"\s*(?:=|:|\uff1a|\u4e3a|\u53d6\u503c\u4e3a)\s*"
    r"(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|true|false|TRUE|FALSE|[A-Za-z0-9_.+\-/]+)"
    r"\s*(?P<unit>[%A-Za-z/\u00b0\u03bc\u03a9ohmmskgNPaV]+)?"
)
REQUIREMENT_PATTERN = re.compile(
    r"(?P<sentence>[^\u3002\n.;\uff1b]*(?:shall|should|must|may|\u5fc5\u987b|"
    r"\u5e94\u5f53|\u5e94\u8be5|\u4e0d\u5f97|\u7981\u6b62|\u9700\u8981|"
    r"\u8981\u6c42|\u5e94|\u5141\u8bb8|\u65e0\u6548|\u6709\u6548|"
    r"\u6210\u529f|\u5931\u8d25)[^\u3002\n.;\uff1b]*)",
    re.I,
)
TABLE_REQUIREMENT_LABELS = {
    "description",
    "desc",
    "input",
    "output",
    "exception",
    "exception handling",
    "overview",
    "notes",
    "requirement",
    "\u63cf\u8ff0",
    "\u8f93\u5165",
    "\u8f93\u51fa",
    "\u5f02\u5e38\u5904\u7406",
    "\u6982\u8ff0",
    "\u8bf4\u660e",
    "\u9002\u7528\u6a21\u5f0f",
    "\u786e\u8ba4\u65b9\u5f0f",
}


class AgentExtractor:
    def __init__(self, index: SQLiteIndex):
        self.index = index

    def search(self, query: str, top_k: int | None = None) -> LookupResult:
        hits = self.index.search(query, top_k=top_k)
        evidence = self.index.evidence_for_hits(hits)
        items = []
        for hit in hits:
            ev_id = f"ev.{hit.block_id}"
            items.append(
                compact_dict(
                    {
                        "id": hit.block_id,
                        "type": hit.kind,
                        "score": round(hit.score, 6),
                        "latex": hit.latex,
                        "normalized_latex": hit.normalized_latex,
                        "symbols": hit.symbols,
                        "visual_type": hit.metadata.get("visual_type"),
                        "caption": hit.metadata.get("caption"),
                        "asset_path": hit.metadata.get("asset_path"),
                        "evidence": [ev_id],
                    }
                )
            )
        return LookupResult(task="search", status="ok" if items else "not_found", items=items, evidence=evidence)

    def extract(self, query: str, task: TaskKind = "formula_extract", top_k: int | None = None) -> LookupResult:
        if task == "search":
            return self.search(query, top_k)
        hits = self.index.search(query, top_k=top_k)
        if task == "formula_extract":
            return self.extract_formulas(query, hits)
        if task == "visual_extract":
            return self.extract_visuals(query, hits)
        if task == "parameter_extract":
            return self.extract_parameters(query, hits)
        if task == "requirement_extract":
            return self.extract_requirements(query, hits)
        if task == "answer":
            return self.answer_with_evidence(query, hits)
        return LookupResult(task=task, status="error", warnings=[f"Unsupported task: {task}"])

    def extract_formulas(self, query: str, hits: list[SearchHit]) -> LookupResult:
        evidence = self.index.evidence_for_hits(hits)
        items: list[dict[str, Any]] = []
        seen: set[str] = set()
        formula_hits = [hit for hit in hits if hit.kind == "equation"]

        for hit in formula_hits:
            item = equation_item_from_hit(hit, query, len(items) + 1)
            key = item.normalized_latex or item.latex
            if key in seen:
                continue
            seen.add(key)
            items.append(item.model_dump(exclude_none=True))

        if not items:
            for hit in hits:
                for candidate in extract_formula_candidates(hit.text):
                    pseudo_hit = hit.model_copy(
                        update={
                            "kind": "equation",
                            "text": candidate.latex,
                            "latex": candidate.latex,
                            "normalized_latex": candidate.normalized_latex,
                            "symbols": candidate.symbols,
                        }
                    )
                    item = equation_item_from_hit(pseudo_hit, query, len(items) + 1)
                    key = item.normalized_latex or item.latex
                    if key not in seen:
                        seen.add(key)
                        items.append(item.model_dump(exclude_none=True))

        return LookupResult(
            task="formula_extract",
            status="ok" if items else "not_found",
            items=items,
            evidence=evidence,
            warnings=[] if items else ["No formula-like evidence found."],
        )

    def extract_visuals(self, query: str, hits: list[SearchHit]) -> LookupResult:
        evidence = self.index.evidence_for_hits(hits, include_preview=True)
        items: list[dict[str, Any]] = []
        seen: set[str] = set()
        visual_hits = [hit for hit in hits if hit.kind == "visual_object"]
        for hit in visual_hits:
            item = visual_item_from_hit(hit, len(items) + 1)
            key = item.asset_path or item.caption or hit.block_id
            if key in seen:
                continue
            seen.add(key)
            items.append(item.model_dump(exclude_none=True))
        return LookupResult(
            task="visual_extract",
            status="ok" if items else "not_found",
            items=items,
            evidence=evidence,
            warnings=[] if items else ["No visual_object evidence found."],
        )

    def extract_parameters(self, query: str, hits: list[SearchHit]) -> LookupResult:
        evidence = self.index.evidence_for_hits(hits)
        items: list[dict[str, Any]] = []
        seen: set[str] = set()
        for hit in hits:
            if hit.kind in {"table", "table_row", "table_cell"}:
                table_param = parameter_from_table_hit(hit)
                if table_param:
                    key = f"{table_param.name.lower()}={str(table_param.value).lower()}"
                    if key not in seen:
                        seen.add(key)
                        items.append(table_param.model_dump(exclude_none=True))
                continue
            for line in hit.text.splitlines():
                line = cleanup(line)
                if not line or line.startswith("$$") or line.endswith("$$"):
                    continue
                if looks_like_formula_line(line):
                    continue
                for match in PARAMETER_PATTERN.finditer(line):
                    name = cleanup(match.group("name"))
                    value = cleanup(match.group("value"))
                    unit = cleanup(match.group("unit") or "") or None
                    if len(name) > 80 or not value:
                        continue
                    key = f"{name.lower()}={value.lower()}:{unit or ''}"
                    if key in seen:
                        continue
                    seen.add(key)
                    item = ParameterItem(
                        id=stable_id("param", hit.block_id, key),
                        name=name,
                        value=parse_scalar(value),
                        unit=unit,
                        condition=infer_condition(hit.text, hit.text.find(line)),
                        domain=classify_domain(hit.latex or hit.text, hit.text),
                        evidence=[f"ev.{hit.block_id}"],
                        confidence=min(0.95, 0.45 + hit.score / 5),
                    )
                    items.append(item.model_dump(exclude_none=True))
        return LookupResult(
            task="parameter_extract",
            status="ok" if items else "not_found",
            items=items,
            evidence=evidence,
            warnings=[] if items else ["No parameter-like evidence found."],
        )

    def extract_requirements(self, query: str, hits: list[SearchHit]) -> LookupResult:
        evidence = self.index.evidence_for_hits(hits)
        items: list[dict[str, Any]] = []
        seen: set[str] = set()
        for hit in hits:
            for sentence in requirement_fragments(hit):
                if len(sentence) < 8 or sentence.lower() in seen:
                    continue
                seen.add(sentence.lower())
                item = RequirementItem(
                    id=stable_id("req", hit.block_id, sentence),
                    modality=infer_modality(sentence),
                    subject=hit.metadata.get("row_label") or hit.metadata.get("column_name"),
                    predicate=sentence,
                    threshold=infer_threshold(sentence),
                    domain=classify_domain(hit.latex or hit.text, hit.text),
                    evidence=[f"ev.{hit.block_id}"],
                    confidence=min(0.92, 0.4 + hit.score / 6),
                )
                items.append(item.model_dump(exclude_none=True))
        return LookupResult(
            task="requirement_extract",
            status="ok" if items else "not_found",
            items=items,
            evidence=evidence,
            warnings=[] if items else ["No requirement-like evidence found."],
        )

    def answer_with_evidence(self, query: str, hits: list[SearchHit]) -> LookupResult:
        evidence = self.index.evidence_for_hits(hits, include_preview=True)
        items = [
            compact_dict(
                {
                    "id": stable_id("answer_support", hit.block_id, query),
                    "type": "supporting_block",
                    "block_id": hit.block_id,
                    "kind": hit.kind,
                    "score": round(hit.score, 6),
                    "evidence": [f"ev.{hit.block_id}"],
                }
            )
            for hit in hits
        ]
        return LookupResult(
            task="answer",
            status="ok" if items else "not_found",
            items=items,
            evidence=evidence,
            warnings=["Natural-language synthesis is intentionally omitted in agent_compact mode."],
        )


def equation_item_from_hit(hit: SearchHit, query: str, ordinal: int) -> EquationItem:
    context = hit.metadata.get("context", hit.text)
    domains = classify_domain(hit.latex or hit.text, f"{context} {query}")
    domain_slug = ".".join(domains[:2]) if domains else "math"
    item_id = f"eq.{domain_slug}.{ordinal:03d}"
    latex = hit.latex or hit.text
    return EquationItem(
        id=item_id,
        latex=latex,
        normalized_latex=hit.normalized_latex,
        vars=infer_variable_roles(latex, context),
        operators=hit.operators,
        domain=domains,
        assumptions=infer_assumptions(latex, context),
        evidence=[f"ev.{hit.block_id}"],
        confidence=min(0.97, 0.52 + hit.score / 5),
    )


def visual_item_from_hit(hit: SearchHit, ordinal: int) -> VisualItem:
    metadata = hit.metadata
    visual_type = metadata.get("visual_type") or "unknown"
    item_id = f"vis.{visual_type}.{ordinal:03d}"
    return VisualItem(
        id=item_id,
        visual_type=visual_type,
        caption=metadata.get("caption"),
        nearby_text=metadata.get("nearby_text"),
        asset_path=metadata.get("asset_path") or hit.source.asset_path,
        media_type=metadata.get("media_type"),
        width=metadata.get("width"),
        height=metadata.get("height"),
        domain=classify_domain(metadata.get("nearby_text") or hit.text, hit.text),
        linked_text_blocks=metadata.get("linked_text_blocks") or [],
        evidence=[f"ev.{hit.block_id}"],
        confidence=min(0.96, 0.5 + hit.score / 5),
    )


def parameter_from_table_hit(hit: SearchHit) -> ParameterItem | None:
    metadata = hit.metadata
    row_label = cleanup(str(metadata.get("row_label") or ""))
    column_name = cleanup(str(metadata.get("column_name") or ""))
    cell_text = cleanup(str(metadata.get("cell_text") or ""))
    if hit.kind == "table_cell" and cell_text:
        if metadata.get("row_index") in {1, "1"} and column_name and cell_text == column_name:
            return None
        name = column_name or row_label or f"col_{metadata.get('col_index') or 'unknown'}"
        if not name or name.lower() in {"table_cell", "row", "col"}:
            return None
        return ParameterItem(
            id=stable_id("param", hit.block_id, name, cell_text),
            name=name,
            value=parse_scalar(cell_text) if is_scalar_like(cell_text) else cell_text,
            condition=f"row_label: {row_label}" if row_label else None,
            domain=classify_domain(hit.latex or hit.text, hit.text),
            evidence=[f"ev.{hit.block_id}"],
            confidence=min(0.95, 0.6 + hit.score / 5),
        )

    if hit.kind != "table_row" or not row_label:
        return None
    values = metadata.get("values")
    value: str | float | int | bool | None = None
    if isinstance(values, dict):
        if row_label in values:
            value = values[row_label]
        elif len(values) == 1:
            value = next(iter(values.values()))
        elif len(values) >= 2:
            non_label_values = [str(item).strip() for key, item in values.items() if str(item).strip() and str(key).strip() != row_label]
            if non_label_values:
                value = "\n".join(non_label_values)
    elif metadata.get("cell_text") and metadata.get("col_index") not in {1, "1"}:
        value = metadata.get("cell_text")
    if value is None or str(value).strip() == "":
        return None
    return ParameterItem(
        id=stable_id("param", hit.block_id, row_label, str(value)),
        name=row_label,
        value=parse_scalar(str(value).strip()) if is_scalar_like(str(value).strip()) else str(value).strip(),
        domain=classify_domain(hit.latex or hit.text, hit.text),
        evidence=[f"ev.{hit.block_id}"],
        confidence=min(0.95, 0.55 + hit.score / 5),
    )


def is_scalar_like(value: str) -> bool:
    return bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|true|false|TRUE|FALSE", value.strip()))


def requirement_fragments(hit: SearchHit) -> list[str]:
    fragments: list[str] = []
    texts = table_semantic_texts(hit) if hit.kind in {"table_row", "table_cell"} else [hit.text]
    for text in texts:
        for match in REQUIREMENT_PATTERN.finditer(text):
            sentence = cleanup(match.group("sentence"))
            if sentence:
                fragments.append(sentence)
        if hit.kind in {"table_row", "table_cell"} and is_table_requirement_context(hit):
            for piece in split_requirement_text(text):
                if piece and piece not in fragments:
                    fragments.append(piece)
    return fragments


def table_semantic_texts(hit: SearchHit) -> list[str]:
    metadata = hit.metadata
    texts: list[str] = []
    cell_text = metadata.get("cell_text")
    if isinstance(cell_text, str) and cell_text.strip():
        texts.append(cell_text)
    values = metadata.get("values")
    if isinstance(values, dict):
        for value in values.values():
            if isinstance(value, str) and value.strip():
                texts.append(value)
    if not texts:
        texts.append(hit.text)
    return texts


def is_table_requirement_context(hit: SearchHit) -> bool:
    row_label = cleanup(str(hit.metadata.get("row_label") or "")).lower()
    column_name = cleanup(str(hit.metadata.get("column_name") or "")).lower()
    return row_label in TABLE_REQUIREMENT_LABELS or column_name in TABLE_REQUIREMENT_LABELS


def split_requirement_text(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[➢•●○]\s*", "\n", normalized)
    normalized = re.sub(r"(?m)^\s*(?:\d+[、.．)]|[-*])\s*", "", normalized)
    pieces = re.split(r"[\n\u3002\uff1b;]+", normalized)
    return [cleanup(piece) for piece in pieces if len(cleanup(piece)) >= 8]


def cleanup(text: str) -> str:
    return " ".join(text.strip().strip("-* \t").split())


def parse_scalar(value: str) -> str | float | int | bool:
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    try:
        if re.fullmatch(r"[-+]?\d+", value):
            return int(value)
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", value):
            return float(value)
    except ValueError:
        pass
    return value


def infer_condition(text: str, position: int) -> str | None:
    prefix = text[max(0, position - 160) : position]
    for marker in ["if", "when", "under", "若", "如果", "当", "在"]:
        idx = prefix.lower().rfind(marker)
        if idx >= 0:
            return cleanup(prefix[idx:])
    return None


def infer_modality(sentence: str) -> str:
    lower = sentence.lower()
    if "must not" in lower or "shall not" in lower or "不得" in sentence or "禁止" in sentence:
        return "must_not"
    if "shall" in lower or "应当" in sentence or "必须" in sentence:
        return "shall"
    if "must" in lower or "需要" in sentence:
        return "must"
    if "should" in lower or "应该" in sentence:
        return "should"
    if "may" in lower:
        return "may"
    return "unknown"


def infer_threshold(sentence: str) -> str | None:
    match = re.search(r"(?:<=|>=|<|>|=|不少于|不大于|至少|至多)\s*[-+]?\d+(?:\.\d+)?\s*[%A-Za-z/°μΩ]*", sentence)
    return cleanup(match.group(0)) if match else None


def looks_like_formula_line(line: str) -> bool:
    if "=" not in line:
        return False
    return any(marker in line for marker in ["_{", "^", "\\", "$", "|"]) or bool(re.search(r"\b[A-Za-z]_[A-Za-z0-9]", line))


def compact_dict(data: dict[str, Any]) -> dict[str, Any]:
    compact: OrderedDict[str, Any] = OrderedDict()
    for key, value in data.items():
        if value is None:
            continue
        if value == [] or value == {}:
            continue
        compact[key] = value
    return dict(compact)

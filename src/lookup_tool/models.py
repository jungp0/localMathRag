from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


BlockKind = Literal["text", "table", "table_row", "table_cell", "equation", "cell", "visual_object", "metadata"]
VisualObjectKind = Literal[
    "chart",
    "diagram",
    "plot",
    "screenshot",
    "photo",
    "formula_image",
    "table_image",
    "mixed",
    "unknown",
]
TaskKind = Literal[
    "search",
    "formula_extract",
    "visual_extract",
    "parameter_extract",
    "requirement_extract",
    "answer",
]


class SourceRef(BaseModel):
    doc_id: str
    path: str
    block_id: str | None = None
    page: int | None = None
    section: str | None = None
    sheet: str | None = None
    cell_range: str | None = None
    bbox: list[float] | None = None
    equation_no: str | None = None
    visual_no: str | None = None
    asset_path: str | None = None
    table_no: str | None = None
    row_index: int | None = None
    col_index: int | None = None


class ParsedBlock(BaseModel):
    block_id: str
    doc_id: str
    kind: BlockKind
    text: str
    source: SourceRef
    latex: str | None = None
    normalized_latex: str | None = None
    symbols: list[str] = Field(default_factory=list)
    operators: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    doc_id: str
    path: str
    sha256: str
    parser_version: str
    blocks: list[ParsedBlock]
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchHit(BaseModel):
    block_id: str
    doc_id: str
    kind: BlockKind
    score: float
    text: str
    source: SourceRef
    latex: str | None = None
    normalized_latex: str | None = None
    symbols: list[str] = Field(default_factory=list)
    operators: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceRef(BaseModel):
    doc_id: str
    path: str
    block_id: str | None = None
    page: int | None = None
    section: str | None = None
    sheet: str | None = None
    cell_range: str | None = None
    bbox: list[float] | None = None
    equation_no: str | None = None
    visual_no: str | None = None
    asset_path: str | None = None
    table_no: str | None = None
    row_index: int | None = None
    col_index: int | None = None
    row_label: str | None = None
    column_name: str | None = None
    visual_type: str | None = None
    caption: str | None = None
    text_preview: str | None = None


class EquationItem(BaseModel):
    id: str
    type: Literal["equation"] = "equation"
    domain: list[str] = Field(default_factory=list)
    latex: str
    normalized_latex: str | None = None
    sympy: str | None = None
    vars: dict[str, dict[str, Any]] = Field(default_factory=dict)
    operators: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class ParameterItem(BaseModel):
    id: str
    type: Literal["parameter"] = "parameter"
    name: str
    value: str | float | int | bool | None = None
    unit: str | None = None
    condition: str | None = None
    domain: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class RequirementItem(BaseModel):
    id: str
    type: Literal["requirement"] = "requirement"
    modality: Literal["shall", "should", "must", "may", "must_not", "unknown"] = "unknown"
    subject: str | None = None
    predicate: str
    condition: str | None = None
    threshold: str | None = None
    domain: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class VisualItem(BaseModel):
    id: str
    type: Literal["visual_object"] = "visual_object"
    visual_type: VisualObjectKind = "unknown"
    caption: str | None = None
    nearby_text: str | None = None
    asset_path: str | None = None
    media_type: str | None = None
    width: int | None = None
    height: int | None = None
    domain: list[str] = Field(default_factory=list)
    linked_text_blocks: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class LookupResult(BaseModel):
    schema_: str = Field(default="lookup.result.v1", serialization_alias="schema")
    task: TaskKind
    status: Literal["ok", "partial", "not_found", "error"] = "ok"
    items: list[dict[str, Any]] = Field(default_factory=list)
    evidence: dict[str, EvidenceRef] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


class IngestReport(BaseModel):
    schema_: str = Field(default="lookup.ingest.v1", serialization_alias="schema")
    status: Literal["ok", "partial", "error"] = "ok"
    documents: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

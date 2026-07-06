from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import tomllib


@dataclass(slots=True)
class StorageConfig:
    db_path: Path = Path("data/lookup.sqlite")
    artifact_dir: Path = Path("data/artifacts")


@dataclass(slots=True)
class ParserConfig:
    enable_formula_detection: bool = True
    enable_ocr: bool = False
    prefer_docling: bool = True
    artifact_dir: Path = Path("data/artifacts")
    chunk_target_chars: int = 1200
    chunk_overlap_chars: int = 180


@dataclass(slots=True)
class RetrievalConfig:
    top_k: int = 12
    formula_boost: float = 2.2
    context_window: int = 1


@dataclass(slots=True)
class GenerationConfig:
    enabled: bool = False
    provider: str = "openai_compatible"
    base_url: str = "http://127.0.0.1:8080/v1"
    model: str = "Qwen/Qwen3-8B-GGUF:Q4_K_M"
    temperature: float = 0.0
    timeout_seconds: int = 60


@dataclass(slots=True)
class OutputConfig:
    mode: str = "agent_compact"


@dataclass(slots=True)
class AppConfig:
    storage: StorageConfig = field(default_factory=StorageConfig)
    parser: ParserConfig = field(default_factory=ParserConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _section(data: dict, name: str) -> dict:
    value = data.get(name, {})
    return value if isinstance(value, dict) else {}


def load_config(path: str | Path | None = None) -> AppConfig:
    if path is None:
        return AppConfig()
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("rb") as fh:
        raw = tomllib.load(fh)

    storage = _section(raw, "storage")
    parser = _section(raw, "parser")
    retrieval = _section(raw, "retrieval")
    generation = _section(raw, "generation")
    output = _section(raw, "output")

    storage_config = StorageConfig(
        db_path=Path(storage.get("db_path", StorageConfig.db_path)),
        artifact_dir=Path(storage.get("artifact_dir", StorageConfig.artifact_dir)),
    )
    parser_defaults = {**asdict(ParserConfig()), "artifact_dir": storage_config.artifact_dir}
    if "artifact_dir" in parser:
        parser = {**parser, "artifact_dir": Path(parser["artifact_dir"])}

    return AppConfig(
        storage=storage_config,
        parser=ParserConfig(**{**parser_defaults, **parser}),
        retrieval=RetrievalConfig(**{**asdict(RetrievalConfig()), **retrieval}),
        generation=GenerationConfig(**{**asdict(GenerationConfig()), **generation}),
        output=OutputConfig(**{**asdict(OutputConfig()), **output}),
    )

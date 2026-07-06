from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def test_json_schemas_are_valid() -> None:
    schema_dir = ROOT / "extensions" / "local_math_rag" / "schemas"
    files = sorted(schema_dir.glob("*.json"))
    assert files, "No schema files found"
    for path in files:
        data = json.loads(read_text(path))
        assert "$schema" in data
        assert "title" in data


def test_openapi_and_pipeline_exist() -> None:
    assert (ROOT / "extensions" / "local_math_rag" / "api" / "openapi.yaml").exists()
    assert (ROOT / "extensions" / "local_math_rag" / "config" / "pipeline.yaml").exists()
    assert "chat_first: true" in read_text(ROOT / "extensions" / "local_math_rag" / "config" / "pipeline.yaml")


def test_chinese_text_files_use_utf8_bom() -> None:
    suffixes = {".md", ".txt", ".ps1", ".py", ".yaml", ".yml"}
    ignored = {"data", "dist", "third_party", ".git"}
    offenders: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if any(part in ignored for part in path.relative_to(ROOT).parts):
            continue
        raw = path.read_bytes()
        if any(byte >= 0x80 for byte in raw) and not raw.startswith(b"\xef\xbb\xbf"):
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, "Non-ASCII text files missing UTF-8 BOM: " + ", ".join(offenders)


def test_object_service_imports() -> None:
    service = ROOT / "services" / "object_service" / "main.py"
    text = read_text(service)
    assert "FastAPI" in text
    assert "/v1/objects/normalize" in text


def test_windows_launcher_exists() -> None:
    project = ROOT / "launcher" / "LocalMathRAGFlow" / "LocalMathRAGFlow.csproj"
    program = ROOT / "launcher" / "LocalMathRAGFlow" / "Program.cs"
    build_script = ROOT / "scripts" / "build-launcher.ps1"
    assert project.exists()
    assert program.exists()
    assert build_script.exists()
    assert "UseWindowsForms" in read_text(project)
    program_text = read_text(program)
    assert "StartDockerDesktop" in program_text
    assert "installedRoot" in program_text
    assert "third_party" in program_text
    assert "ragflow" in program_text


def main() -> None:
    test_json_schemas_are_valid()
    test_openapi_and_pipeline_exist()
    test_chinese_text_files_use_utf8_bom()
    test_object_service_imports()
    test_windows_launcher_exists()
    print("contract checks passed")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
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
    for current_root, dirs, files in os.walk(ROOT):
        dirs[:] = [name for name in dirs if name not in ignored]
        root_path = Path(current_root)
        for filename in files:
            path = root_path / filename
            if path.suffix.lower() not in suffixes:
                continue
            raw = path.read_bytes()
            if any(byte >= 0x80 for byte in raw) and not raw.startswith(b"\xef\xbb\xbf"):
                offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, "Non-ASCII text files missing UTF-8 BOM: " + ", ".join(offenders)


def test_no_unpruned_bom_scan() -> None:
    text = read_text(ROOT / "tests" / "test_contracts.py")
    assert "os.walk" in text
    assert "dirs[:]" in text


def test_object_service_imports() -> None:
    service = ROOT / "services" / "object_service" / "main.py"
    text = read_text(service)
    assert "FastAPI" in text
    assert "/v1/objects/normalize" in text
    assert "/v1/models/local" in text
    assert "/v1/models/recommended" in text
    assert "/v1/models/download" in text
    assert "/v1/models/status" in text
    assert "LLAMA_BASE_URL" in text
    assert "EMBEDDING_BASE_URL" in text
    assert "RERANK_BASE_URL" in text
    assert "VISION_BASE_URL" in text
    assert "qwen3-embedding-06b" in text
    assert "qwen3-reranker-06b" in text
    assert "qwen3-vl-8b-instruct" in text
    assert "runtime_model_name" in text
    assert '"/models/Qwen3-8B-Q4_K_M.gguf"' in text
    assert "download_kind" in text
    assert "_download_snapshot" in text
    assert "_snapshot_model_dirs" in text
    assert "_is_inside_snapshot" in text
    assert "Qwen/Qwen3-Embedding-0.6B" in text
    assert "Qwen/Qwen3-Reranker-0.6B" in text


def test_windows_launcher_exists() -> None:
    project = ROOT / "launcher" / "LocalMathRAGFlow" / "LocalMathRAGFlow.csproj"
    program = ROOT / "launcher" / "LocalMathRAGFlow" / "Program.cs"
    build_script = ROOT / "scripts" / "build-launcher.ps1"
    icon = ROOT / "launcher" / "LocalMathRAGFlow" / "Assets" / "ragflow.ico"
    assert project.exists()
    assert program.exists()
    assert build_script.exists()
    assert icon.exists()
    project_text = read_text(project)
    assert "UseWindowsForms" in project_text
    assert "ApplicationIcon" in project_text
    program_text = read_text(program)
    assert "StartDockerDesktop" in program_text
    assert "TryBuildAutoLoginUrlAsync" in program_text
    assert "EncryptRagflowPassword" in program_text
    assert "ModernMenuRenderer" in program_text
    assert "MenuGlyph" in program_text
    assert "NativeTrayIcon" in program_text
    assert "Shell_NotifyIcon" in program_text
    assert "RegisterWindowMessage(\"TaskbarCreated\")" in program_text
    assert "CallbackMessage" in program_text
    assert "menuHost" in program_text
    assert "OpenFromTrayAsync" in program_text
    assert "TRAY menu opening" in program_text
    assert "TRAY native right click" in program_text
    assert "TRAY native icon created" in program_text
    assert "ShowTrayMenu" in program_text
    assert "OpenOrFocusAppWindow" in program_text
    assert "SetForegroundWindow" in program_text
    assert "BuildComposeProfiles" in program_text
    assert "ConfirmLocalModelRuntimeAsync" in program_text
    assert "DockerImageExistsAsync" in program_text
    assert "FindDefaultGgufModelName" in program_text
    assert "LOCALMATHRAG_GGUF_MODEL" in program_text
    assert "llama-cpp-cpu" in program_text
    assert "up -d --build" in program_text
    assert "installedRoot" in program_text
    assert "HandleComposeOutput" in program_text
    assert "downloading Docker images" in program_text
    assert "WaitForHttpOkAsync" in program_text
    assert "waiting for RAGFlow web" in program_text
    assert "third_party" in program_text
    assert "ragflow" in program_text


def test_ragflow_patch_workflow_exists() -> None:
    apply_script = ROOT / "scripts" / "apply-ragflow-patches.ps1"
    build_script = ROOT / "scripts" / "build-ragflow-web.ps1"
    patch = ROOT / "patches" / "ragflow" / "0001-localmathrag-offline-ui.patch"
    webdist_compose = ROOT / "docker" / "docker-compose.webdist.yml"
    bootstrap = read_text(ROOT / "scripts" / "bootstrap-ragflow.ps1")
    launcher = read_text(ROOT / "launcher" / "LocalMathRAGFlow" / "Program.cs")
    assert apply_script.exists()
    assert build_script.exists()
    assert patch.exists()
    assert webdist_compose.exists()
    assert "apply-ragflow-patches.ps1" in bootstrap
    assert "docker-compose.webdist.yml" in launcher
    build_text = read_text(build_script)
    assert "--ignore-scripts" in build_text
    assert "--max-old-space-size=8192" in build_text
    patch_text = read_text(patch)
    assert "Discord" in patch_text
    assert "/v1/models/local" in patch_text
    assert "OpenAiAPICompatible" in patch_text
    assert "_check_local_chat_completion" in patch_text
    assert "_defer_local_non_chat_check" in patch_text
    assert "downloadProgress" in patch_text
    assert "Download stores model files only" in patch_text
    assert "configureModel(data.model)" not in patch_text
    assert "嵌入模型" in patch_text


def test_docker_compose_mounts_ragflow_backend_overrides() -> None:
    compose = read_text(ROOT / "docker" / "docker-compose.localmathrag.yml")
    assert "provider_api_service.py:/ragflow/api/apps/services/provider_api_service.py:ro" in compose
    assert "llm_app.py:/ragflow/api/apps/llm_app.py:ro" in compose


def test_agent_rules_document_root_resolution() -> None:
    rules = ROOT / "AGENTS.md"
    assert rules.exists()
    text = read_text(rules)
    assert "Launcher Root Resolution" in text
    assert "RAGFlow Patch Rules" in text
    assert "third_party/ragflow/docker/docker-compose.yml" in text
    assert "不允许从 `dist` 启动时重复下载 RAGFlow 或模型" in text
    assert "Local Model Runtime Rules" in text
    assert "Tray Launcher Rules" in text


def main() -> None:
    test_json_schemas_are_valid()
    test_openapi_and_pipeline_exist()
    test_chinese_text_files_use_utf8_bom()
    test_no_unpruned_bom_scan()
    test_object_service_imports()
    test_windows_launcher_exists()
    test_ragflow_patch_workflow_exists()
    test_docker_compose_mounts_ragflow_backend_overrides()
    print("contract checks passed")


if __name__ == "__main__":
    main()

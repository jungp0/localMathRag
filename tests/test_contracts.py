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
    openapi = read_text(ROOT / "extensions" / "local_math_rag" / "api" / "openapi.yaml")
    assert "/v1/dataset/status" in openapi
    assert "/v1/dataset/files" in openapi
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


def test_ragflow_metadata_progress_patch() -> None:
    patch = read_text(ROOT / "patches" / "ragflow" / "0001-localmathrag-offline-ui.patch")
    assert "Metadata generation {}/{} chunks completed ..." in patch
    assert "Metadata generation {} chunks completed in" in patch
    assert "Error in gen_metadata_task" in patch
    assert "Question generation {} chunks completed in {:.2f}s" in patch


def test_ragflow_embedded_file_skip_patch() -> None:
    patch = read_text(ROOT / "patches" / "ragflow" / "0001-localmathrag-offline-ui.patch")
    assert "Skipped unsupported embedded file" in patch
    assert '"file type not supported yet" not in str(e)' in patch
    assert "logging.warning(error_msg)" in patch


def test_ragflow_vision_enhancement_is_nonfatal() -> None:
    patch = read_text(ROOT / "patches" / "ragflow" / "0001-localmathrag-offline-ui.patch")
    assert "_ensure_vision_runtime" in patch
    assert "Loading local VLM runtime if needed" in patch
    assert "Visual model not ready after waiting; skipping image enhancement" in patch
    assert "_vision_enhancement_callback" in patch
    assert "progress is not None and progress < 0" in patch
    assert "Visual model enhancement skipped" in patch
    assert "callback=_vision_enhancement_callback(callback)" in patch


def test_ragflow_reparse_cancels_previous_unfinished_tasks() -> None:
    patch = read_text(ROOT / "patches" / "ragflow" / "0001-localmathrag-offline-ui.patch")
    assert "cancel_all_unfinished_task_of" in patch
    assert 'REDIS_CONN.set(f"{t.id}-cancel", "x")' in patch
    assert 'cancel_all_unfinished_task_of(doc["id"])' in patch
    assert "cancel_all_unfinished_task_of(doc_id)" in patch


def test_ragflow_auto_metadata_updates_document_snapshots() -> None:
    patch = read_text(ROOT / "patches" / "ragflow" / "0001-localmathrag-offline-ui.patch")
    assert "enabled: Annotated[bool | None" in patch
    assert '"enabled": bool(parser_cfg.get("enable_metadata", False))' in patch
    assert "metadata_parser_cfg = {" in patch
    assert '"enable_metadata": bool(enabled)' in patch
    assert "for doc in DocumentService.query(kb_id=kb.id):" in patch
    assert "_sync_document_metadata_config" in patch
    assert "doc_parser_config.update(metadata_parser_config)" in patch


def test_ragflow_embedding_switch_error_explains_rebuild_vectors() -> None:
    patch = read_text(ROOT / "patches" / "ragflow" / "0002-localmathrag-embedding-switch-check.patch")
    assert "below 0.9, indicating incompatible vector spaces" in patch
    assert "Please delete or reparse existing chunks" in patch
    assert "rebuild the dataset vectors with the new embedding model" in patch
    assert 'check_embedding(dataset_id, tenant_id, {"embd_id": req["embd_id"]})' in patch
    assert "req[\"embd_id\"] != kb.embd_id and kb.chunk_num > 0" in patch
    assert "s.CheckEmbedding(tenantID, datasetID" in patch
    assert "embdID != kb.EmbdID && kb.ChunkNum > 0" in patch
    assert "LOCALMATHRAG_ALLOW_INCOMPATIBLE_EMBEDDING_SWITCH" not in patch


def test_ragflow_default_dataset_manual_refresh_contract() -> None:
    patch = read_text(ROOT / "patches" / "ragflow" / "0003-localmathrag-default-dataset-refresh.patch")
    service = patch
    api = patch
    server = patch
    dataset_service = patch
    web_api = patch
    web_service = patch
    web_hook = patch
    web_page = patch

    assert 'DEFAULT_DATASET_NAME = os.environ.get("LOCALMATHRAG_DEFAULT_DATASET_NAME", "Default")' in service
    assert 'DATASET_DIR = Path(os.environ.get("LOCALMATHRAG_DATASET_DIR", "/localmathrag/dataset"))' in service
    assert 'DATASET_STATE_FILE = Path(os.environ.get("LOCALMATHRAG_DATASET_STATE_FILE", "/localmathrag/cache/dataset-state.json"))' in service
    assert "REFRESH_LOCK = threading.Lock()" in service
    assert "REFRESH_LOCK.acquire(blocking=False)" in service
    assert "ensure_default_dataset" in service
    assert "KnowledgebaseService.create_with_name" in service
    assert "from api.apps.services.local_dataset_service import ensure_default_dataset" in server
    assert "ensure_default_dataset()" in server
    assert "_scan_manifest()" in service
    assert "_cached_subtree" in service
    assert "_deleted_subtree" in service
    assert '"unchanged"' in service and '"new"' in service and '"changed"' in service and '"deleted"' in service
    assert "status\") in {\"new\", \"changed\"}" in service
    assert "FileService().upload_document" in service
    assert "DocumentService.run" in service
    assert "FileService.delete_docs" in service
    assert "LocalDatasetFile.read" in service or "def read(self) -> bytes" in service
    assert "refresh_default_local_dataset" in dataset_service
    assert '"/datasets/<dataset_id>/local-refresh"' in api
    assert "localDatasetRefresh" in web_api
    assert "refreshLocalDataset" in web_service
    assert "useRefreshLocalDataset" in web_hook
    assert "knowledgeDetails.localRefresh" in web_page
    assert "localRefresh: 'Refresh'" in patch
    assert "localRefresh: '刷新'" in patch
    assert "knowledgeBase?.name === 'Default'" in web_page
    assert "kb_ids" not in service
    assert "DialogService" not in service
    assert "ChatService" not in service


def test_object_service_imports() -> None:
    service = ROOT / "services" / "object_service" / "main.py"
    text = read_text(service)
    assert "FastAPI" in text
    assert "/v1/objects/normalize" in text
    assert "/v1/models/local" in text
    assert "/v1/models/recommended" in text
    assert "/v1/models/download" in text
    assert "/v1/models/status" in text
    assert "/v1/dataset/status" in text
    assert "/v1/dataset/files" in text
    assert "LOCALMATHRAG_DATASET_DIR" in text
    assert "DATASET_STATE_FILE" in text
    assert "_scan_dataset_directory" in text
    assert "skipped_directory_count" in text
    assert "/v1/runtime/ensure" in text
    assert "/v1/runtime/status" in text
    assert "/v1/runtime/stop" in text
    assert "/v1/embeddings" in text
    assert "/v1/rerank" in text
    assert "EmbeddingRequest" in text
    assert "RerankRequest" in text
    assert "RuntimeEnsureRequest" in text
    assert "RuntimeStopRequest" in text
    assert "LOCALMATHRAG_DOCKER_SOCKET" in text
    assert "_docker_start_container" in text
    assert "_docker_stop_container" in text
    assert "RUNTIME_START_LOCK" in text
    assert "OPTIONAL_RUNTIME_MAX_ACTIVE" in text
    assert "_runtime_startup_progress" in text
    assert "_runtime_startup_stall_timeout_seconds" in text
    assert "_docker_container_logs" in text
    assert "startup_stalled" in text
    assert "cuda-model-loading" in text
    assert "warming-up" in text
    assert "_balance_optional_runtimes" in text
    assert "_stop_optional_runtime_for_degrade" in text
    assert "_runtime_resource_status" in text
    assert "LOCALMATHRAG_RUNTIME_MIN_AVAILABLE_MEMORY_GB" in text
    assert "EMBEDDING_RUNTIME_BASE_URL" in text
    assert "RERANK_RUNTIME_BASE_URL" in text
    assert "_ensure_runtime_ready" in text
    assert "_optional_runtime_endpoint_status" in text
    assert "_docker_container_state" in text
    assert "_post_runtime_json" in text
    assert "container for {target['compose_service']} was not prepared" in text
    assert "localmathrag-lexical-embedding" in text
    assert "localmathrag-lexical-rerank" in text
    assert "LLAMA_BASE_URL" in text
    assert "EMBEDDING_BASE_URL" in text
    assert "RERANK_BASE_URL" in text
    assert "VISION_BASE_URL" in text
    assert "qwen3-embedding-06b" in text
    assert "qwen3-reranker-06b" in text
    assert "bge-reranker-v2-m3" in text
    assert "qwen3-vl-8b-instruct" in text
    assert "whisper-large-v3-turbo" in text
    assert "cosyvoice2-05b" in text
    assert '"group": "asr"' in text
    assert '"group": "tts"' in text
    assert "runtime_model_name" in text
    assert '"/models/Qwen3-8B-Q4_K_M.gguf"' in text
    assert "payload.get(\"runtime_model_name\")" in text
    assert "payload.get(\"name\")" in text
    assert "_base_url_for_payload" in text
    assert "_model_identity_payload" in text
    assert "_endpoint_key_for_model_type" in text
    assert "_base_url_for_endpoint_key" in text
    assert "def endpoint_ok(endpoint_key: str)" in text
    assert "timeout=0.8" in text
    assert '"asr"' in text and '"tts"' in text
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
    assert icon.stat().st_size > 1000
    project_text = read_text(project)
    assert "UseWindowsForms" in project_text
    assert "ApplicationManifest" not in project_text
    assert "ApplicationIcon" in project_text
    assert "CopyToPublishDirectory" in project_text
    assert "ExcludeFromSingleFile" in project_text
    program_text = read_text(program)
    dev_up_text = read_text(ROOT / "scripts" / "dev-up.ps1")
    assert "Application.Run(new TrayContext())" in program_text
    assert "class LauncherContext" not in program_text
    assert "Application.Run(new LauncherContext())" not in program_text
    assert "BackgroundStartupTask.StartDetached()" in program_text
    assert "StartupStatusServer.StartDetached()" in program_text
    assert "internal static class StartupStatusServer" in program_text
    assert "TcpListener(IPAddress.Loopback" in program_text
    assert "StartupStatusServer.EntryUrl" in program_text
    assert "LocalMathRAGFlow &#27491;&#22312;&#21551;&#21160;" in program_text
    assert "BuildLoadingHtmlPage()" in program_text
    assert 'path.StartsWith("/start"' in program_text
    assert 'path.StartsWith("/stop"' in program_text
    assert "LocalMathRAGFlow \\u6b63\\u5728\\u505c\\u6b62" in program_text
    assert "LocalMathRAGFlow \\u5df2\\u505c\\u6b62" in program_text
    assert 'class="bar" id="bar"' in program_text
    assert "let data;" in program_text
    assert "bar.classList.add('stopped')" in program_text
    assert "startButton.hidden = !data.stopped || data.state === 'stopping'" in program_text
    assert "internal static class BackgroundStartupTask" in program_text
    assert "startup-status.json" in program_text
    assert "startup.log" in program_text
    assert "checking_service" in program_text
    assert "RAGFlow Web is already running." in program_text
    assert 'ready = state is "running"' in program_text
    assert 'stopped = state is "stopped"' in program_text
    assert 'close = state is "stopped_exit"' in program_text
    assert "public static async Task StopServicesAsync(bool exitAfterStop = false)" in program_text
    assert "await BackgroundStartupTask.StopServicesAsync()" in program_text
    assert "exitAfterStop: true" in program_text
    assert "forceStatusPage: true" in program_text
    assert "OpenWeb(bool forceStatusPage = false)" in program_text
    assert "CloseBrowserProcess()" in program_text
    assert "browser-profile-run-" in program_text
    assert "CleanupOldBrowserProfiles(launcherDataDir)" in program_text
    assert "localmathrag_reload" in program_text
    assert "WaitForRagflowWebReadyAsync" in program_text
    assert "ExtractStartupAssetPaths" in program_text
    assert "<div id=\\\"root\\\"></div>" in program_text
    assert "StartServicesAsync(openBrowser: true)" not in program_text
    assert "uiContext.Post(async _ => await StartServicesAsync(openBrowser: true), null)" not in program_text
    assert "StartDockerDesktop" in program_text
    assert "TryBuildAutoLoginUrlAsync" not in program_text
    assert "EncryptRagflowPassword" not in program_text
    assert "ModernMenuRenderer" not in program_text
    assert "MenuGlyph" not in program_text
    assert "NotifyIcon" in program_text
    assert "ContextMenuStrip" in program_text
    assert "ContextMenuStrip = menu" in program_text
    assert "Open dataset" in program_text
    assert '"data", "dataset"' in program_text
    assert "View tray log" not in program_text
    assert "CreateMenuHost" not in program_text
    assert "TRAY manual right click fallback" not in program_text
    assert "SetForegroundWindow(menuHost.Handle)" not in program_text
    assert "BuildMenu()" not in program_text
    assert "Icon = LoadTrayIcon()" in program_text
    assert '"ragflow.ico"' in program_text
    assert "ExtractAssociatedIcon" in program_text
    assert "LoadAppIcon" not in program_text
    assert "Renderer = new ModernMenuRenderer" not in program_text
    assert "Renderer = new MutedTrayMenuRenderer()" in program_text
    assert "tray.MouseClick" in program_text
    assert "tray.DoubleClick" in program_text
    assert "window-state.json" in program_text
    assert "--start-maximized" in program_text
    assert "GetWindowPlacement" in program_text
    assert "ShowWindowMaximized" in program_text
    assert "BrowserWindowState" in program_text
    assert "OpenFromTrayAsync" not in program_text
    assert "menu opening" in program_text
    assert "mouse click" in program_text
    assert "launcher tray icon created" in program_text
    assert "ShowTrayMenu" not in program_text
    assert "OpenOrFocusAppWindow" not in program_text
    assert "SetForegroundWindow" in program_text
    assert "BuildComposeProfiles" not in program_text
    assert "ConfirmLocalModelRuntimeAsync" not in program_text
    assert "DockerImageExistsAsync" not in program_text
    assert "LOCALMATHRAG_GGUF_MODEL" in dev_up_text
    assert "LOCALMATHRAG_EMBEDDING_MODEL" in dev_up_text
    assert "LOCALMATHRAG_RERANK_MODEL" in dev_up_text
    assert "LOCALMATHRAG_VLM_MODEL" in dev_up_text
    assert "LOCALMATHRAG_ASR_MODEL" in dev_up_text
    assert "LOCALMATHRAG_TTS_MODEL" in dev_up_text
    assert '"eager"' in dev_up_text
    assert "LOCALMATHRAG_CTX_SIZE" in dev_up_text
    assert "LOCALMATHRAG_LLAMA_PARALLEL" in dev_up_text
    assert "DOCKER_API_VERSION" in dev_up_text
    assert "create --no-build --pull never" in dev_up_text
    assert "llama-cpp-cpu" in dev_up_text
    assert "llama-cpp-cuda" in dev_up_text
    assert "embedding-cpu" in dev_up_text
    assert "embedding-cuda" in dev_up_text
    assert "rerank-cpu" in dev_up_text
    assert "rerank-cuda" in dev_up_text
    assert "vlm-cuda" in dev_up_text
    assert "asr-local" in dev_up_text
    assert "asr-cuda" in dev_up_text
    assert "tts-local" in dev_up_text
    assert "tts-cuda" in dev_up_text
    assert "Test-DisabledProfile" in dev_up_text
    assert "Test-CpuProfile" in dev_up_text
    assert "LOCALMATHRAG_TEI_CUDA_IMAGE" in read_text(ROOT / "docker" / "docker-compose.localmathrag.yml")
    assert "infiniflow/text-embeddings-inference:cpu-1.8" in read_text(ROOT / "docker" / "docker-compose.localmathrag.yml")
    assert "up\", \"-d\", \"--build" in dev_up_text
    assert "installedRoot" in program_text
    assert program_text.count("var installedRoot") >= 2
    assert "HandleComposeOutput" not in program_text
    assert "downloading Docker images" not in program_text
    assert "WaitForHttpOkAsync" in program_text
    assert "Waiting for RAGFlow Web." in program_text
    assert "third_party" in program_text
    assert "ragflow" in program_text
    build_launcher_text = read_text(build_script)
    assert '"data\\dataset"' in build_launcher_text
    assert '"data\\cache"' in build_launcher_text
    assert "Test-LazyRuntime" in dev_up_text
    assert "create --no-build --pull never localmathrag-embedding" in dev_up_text
    assert "create --no-build --pull never localmathrag-rerank" in dev_up_text
    assert "create --no-build --pull never localmathrag-vlm" in dev_up_text
    assert "stop localmathrag-vlm" in dev_up_text
    runtime_test = ROOT / "scripts" / "test-runtime-balancer.ps1"
    assert runtime_test.exists()
    runtime_test_text = read_text(runtime_test)
    assert "/v1/embeddings" in runtime_test_text
    assert "/v1/rerank" in runtime_test_text
    assert "/v1/runtime/ensure" in runtime_test_text
    assert "/v1/runtime/status" in runtime_test_text
    assert "/v1/runtime/stop" in runtime_test_text
    assert "IncludeVlm" in runtime_test_text
    assert "optional runtime balancer violation" in runtime_test_text


def test_ragflow_patch_workflow_exists() -> None:
    apply_script = ROOT / "scripts" / "apply-ragflow-patches.ps1"
    build_script = ROOT / "scripts" / "build-ragflow-web.ps1"
    patch = ROOT / "patches" / "ragflow" / "0001-localmathrag-offline-ui.patch"
    refresh_patch = ROOT / "patches" / "ragflow" / "0003-localmathrag-default-dataset-refresh.patch"
    dockerfile = read_text(ROOT / "docker" / "Dockerfile.ragflow-local")
    dockerignore = read_text(ROOT / ".dockerignore")
    bootstrap = read_text(ROOT / "scripts" / "bootstrap-ragflow.ps1")
    launcher = read_text(ROOT / "launcher" / "LocalMathRAGFlow" / "Program.cs")
    assert apply_script.exists()
    assert build_script.exists()
    assert patch.exists()
    assert refresh_patch.exists()
    assert "apply-ragflow-patches.ps1" in bootstrap
    assert "docker-compose.webdist.yml" not in launcher
    assert "--unidiff-zero" in read_text(apply_script)
    build_text = read_text(build_script)
    assert "--ignore-scripts" in build_text
    assert "--max-old-space-size=8192" in build_text
    patch_text = read_text(patch)
    assert "Discord" in patch_text
    assert "/v1/models/local" in patch_text
    assert "OpenAiAPICompatible" in patch_text
    assert "_check_local_chat_completion" in patch_text
    assert "api/apps/services/dataset_api_service.py" in dockerfile
    assert "api/apps/services/local_dataset_service.py" in dockerfile
    assert "api/ragflow_server.py" in dockerfile
    assert "api/apps/restful_apis/dataset_api.py" in dockerfile
    assert "api/utils/validation_utils.py" in dockerfile
    assert "!third_party/ragflow/api/apps/services/dataset_api_service.py" in dockerignore
    assert "!third_party/ragflow/api/apps/services/local_dataset_service.py" in dockerignore
    assert "!third_party/ragflow/api/ragflow_server.py" in dockerignore
    assert "!third_party/ragflow/api/apps/restful_apis/dataset_api.py" in dockerignore
    assert "!third_party/ragflow/api/utils/validation_utils.py" in dockerignore
    assert "client.models.list()" in patch_text
    assert "_defer_local_non_chat_check" in patch_text
    assert "deferred local image2text runtime check" not in patch_text
    assert "deferred local OCR runtime check" not in patch_text
    assert "downloadProgress" in patch_text
    assert "Download stores model files only" in patch_text
    assert "configureModel(data.model)" not in patch_text
    assert "vision: ['image2text']" in patch_text
    assert "llm_id: ['chat']" not in patch_text
    assert "-  LLMFactory.OpenAiAPICompatible," in patch_text
    assert "+  LLMFactory.OpenAiAPICompatible," not in patch_text
    assert "嵌入模型" in patch_text
    assert "语音识别" in patch_text
    assert "语音合成" in patch_text
    assert "speech2text" in patch_text
    assert "collectModelPayloads" in patch_text
    assert "model_info: []" in patch_text
    assert "return True, \"success\"" in patch_text
    refresh_patch_text = read_text(refresh_patch)
    assert "DEFAULT_DATASET_NAME" in refresh_patch_text
    assert "local-refresh" in refresh_patch_text
    assert "knowledgeDetails.localRefresh" in refresh_patch_text
    assert "localRefresh: 'Refresh'" in refresh_patch_text
    assert "localRefresh: '刷新'" in refresh_patch_text
    assert "REFRESH_LOCK" in refresh_patch_text


def test_docker_compose_mounts_ragflow_backend_overrides() -> None:
    compose = read_text(ROOT / "docker" / "docker-compose.localmathrag.yml")
    ragflow_dockerfile = ROOT / "docker" / "Dockerfile.ragflow-local"
    localize_script = ROOT / "docker" / "ragflow-localize-conf.py"
    assert ragflow_dockerfile.exists()
    assert localize_script.exists()
    dockerfile_text = read_text(ragflow_dockerfile)
    localize_text = read_text(localize_script)
    assert "COPY third_party/ragflow/api/apps/services/provider_api_service.py" in dockerfile_text
    assert "COPY third_party/ragflow/deepdoc/parser/figure_parser.py" in dockerfile_text
    assert "COPY third_party/ragflow/rag/svr/task_executor.py" in dockerfile_text
    assert "COPY third_party/ragflow/web/dist" in dockerfile_text
    assert "ragflow-localize-conf.py" in dockerfile_text
    assert "Qwen/Qwen3-Embedding-0.6B" in localize_text
    assert "BAAI/bge-reranker-v2-m3" in localize_text
    assert "http://localmathrag-object-service:8088/v1" in localize_text
    assert "http://${TEI_HOST}:80" in localize_text
    assert "dockerfile: docker/Dockerfile.ragflow-local" in compose
    assert "image: localmathrag/ragflow:dev" in compose
    assert "LOCALMATHRAG_LOCAL_ONLY" in compose
    assert "LOCALMATHRAG_DISABLE_LOGIN" in compose
    assert "LOCALMATHRAG_DATASET_DIR" in compose
    assert "LOCALMATHRAG_DATASET_STATE_FILE" in compose
    assert "${LOCALMATHRAG_ROOT}/data/dataset:/localmathrag/dataset:ro" in compose
    assert "${LOCALMATHRAG_ROOT}/data/cache:/localmathrag/cache" in compose
    assert "volumes: !override" in compose
    assert "third_party/ragflow/docker/ragflow-logs" in compose
    assert "service_conf.yaml.template" not in compose
    assert "/ragflow/entrypoint.sh" not in compose
    assert "provider_api_service.py:/ragflow/api/apps/services/provider_api_service.py:ro" not in compose
    assert "llm_app.py:/ragflow/api/apps/llm_app.py:ro" not in compose
    assert "localmathrag-embedding" in compose
    assert "localmathrag-embedding-cpu" in compose
    assert "localmathrag-rerank" in compose
    assert "localmathrag-rerank-cpu" in compose
    assert "/var/run/docker.sock:/var/run/docker.sock" in compose
    assert "LOCALMATHRAG_RUNTIME_LAZY" in compose
    assert "LOCALMATHRAG_OPTIONAL_RUNTIME_MAX_ACTIVE: ${LOCALMATHRAG_OPTIONAL_RUNTIME_MAX_ACTIVE:-1}" in compose
    assert "LOCALMATHRAG_RUNTIME_READY_TIMEOUT_SECONDS: ${LOCALMATHRAG_RUNTIME_READY_TIMEOUT_SECONDS:-240}" in compose
    assert "LOCALMATHRAG_RUNTIME_STARTUP_STALL_TIMEOUT_SECONDS: ${LOCALMATHRAG_RUNTIME_STARTUP_STALL_TIMEOUT_SECONDS:-120}" in compose
    assert "LOCALMATHRAG_VISION_MIN_AVAILABLE_MEMORY_GB: ${LOCALMATHRAG_VISION_MIN_AVAILABLE_MEMORY_GB:-10}" in compose
    assert "LOCALMATHRAG_EMBEDDING_RUNTIME_BASE_URL: http://localmathrag-embedding:8080/v1" in compose
    assert "LOCALMATHRAG_RERANK_RUNTIME_BASE_URL: http://localmathrag-rerank:8080/v1" in compose
    assert "LOCALMATHRAG_EMBEDDING_CONTAINER" in compose
    assert "LOCALMATHRAG_RERANK_CONTAINER" in compose
    assert "LOCALMATHRAG_VISION_CONTAINER" in compose
    assert "LOCALMATHRAG_ASR_CONTAINER" in compose
    assert "LOCALMATHRAG_TTS_CONTAINER" in compose
    assert "${LOCALMATHRAG_TEI_CUDA_IMAGE:-ghcr.io/huggingface/text-embeddings-inference:cuda-latest}" in compose
    assert "${LOCALMATHRAG_TEI_CPU_IMAGE:-infiniflow/text-embeddings-inference:cpu-1.8}" in compose
    assert "entrypoint:" in compose
    assert "text-embeddings-router-$${compute_cap}" in compose
    assert "nvidia-smi --query-gpu=compute_cap" in compose
    assert "LD_LIBRARY_PATH: /usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64" in compose
    assert "${LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS:-16384}" in compose
    assert "/models/${LOCALMATHRAG_RERANK_MODEL:-bge-reranker-v2-m3}" in compose
    assert "${LOCALMATHRAG_TEI_TOKENIZATION_WORKERS:-2}" in compose
    assert "${LOCALMATHRAG_TEI_MAX_CONCURRENT_REQUESTS:-64}" in compose
    assert "${LOCALMATHRAG_TEI_MAX_CLIENT_BATCH_SIZE:-8}" in compose
    assert compose.count('restart: "no"') >= 6
    assert "embedding-cuda" in compose
    assert "rerank-cuda" in compose
    assert "vlm-cuda" in compose
    assert "asr-cuda" in compose
    assert "tts-cuda" in compose
    assert "capabilities: [gpu]" in compose
    assert "${LOCALMATHRAG_EMBEDDING_PORT:-8081}:8080" in compose
    assert "${LOCALMATHRAG_RERANK_PORT:-8082}:8080" in compose
    assert "LOCALMATHRAG_EMBEDDING_BASE_URL: http://localmathrag-object-service:8088/v1" in compose
    assert "LOCALMATHRAG_RERANK_BASE_URL: http://localmathrag-object-service:8088/v1" in compose
    assert "LOCALMATHRAG_VISION_BASE_URL: http://localmathrag-vlm:8000/v1" in compose
    assert "LOCALMATHRAG_ASR_BASE_URL: http://localmathrag-asr:8080/v1" in compose
    assert "LOCALMATHRAG_TTS_BASE_URL: http://localmathrag-tts:8080/v1" in compose
    assert "${LOCALMATHRAG_LLAMA_PARALLEL:-1}" in compose
    assert "localmathrag-vlm" in compose
    assert "localmathrag-asr" in compose
    assert "localmathrag-tts" in compose


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
    test_ragflow_embedding_switch_error_explains_rebuild_vectors()
    test_ragflow_default_dataset_manual_refresh_contract()
    test_object_service_imports()
    test_windows_launcher_exists()
    test_ragflow_patch_workflow_exists()
    test_docker_compose_mounts_ragflow_backend_overrides()
    print("contract checks passed")


if __name__ == "__main__":
    main()

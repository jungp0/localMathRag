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


def test_ragflow_search_summary_has_output_budget() -> None:
    patch = read_text(ROOT / "patches" / "ragflow" / "0004-localmathrag-search-summary-token-budget.patch")
    dynamic_patch = read_text(ROOT / "patches" / "ragflow" / "0014-localmathrag-dynamic-search-token-budget.patch")
    assert 'gen_conf.setdefault("max_completion_tokens", max_tokens)' in patch
    assert 'k in ALLOWED_GEN_CONF_KEYS or k == "max_tokens"' in patch
    assert "_localmathrag_search_token_budgets" in dynamic_patch
    assert "LOCALMATHRAG_SEARCH_DYNAMIC_TOKEN_BUDGET" in dynamic_patch
    assert "LOCALMATHRAG_SEARCH_ANSWER_TOKEN_BUDGET_MAX" in dynamic_patch
    assert "LOCALMATHRAG_SEARCH_KNOWLEDGE_TOKEN_BUDGET_MAX" in dynamic_patch
    assert "Search async_ask token budget: mode=%s max=%s reserved=%s available=%s knowledge=%s answer=%s question=%s" in dynamic_patch
    assert "kb_prompt(kbinfos, knowledge_token_budget)" in patch
    assert '{"temperature": 0.1, "max_tokens": answer_token_budget}' in patch


def test_ragflow_rerank_fallback_contract() -> None:
    patch = read_text(ROOT / "patches" / "ragflow" / "0005-localmathrag-rerank-fallback.patch")
    assert "external rerank failed; falling back to ES/KNN scoring" in patch
    assert "rerank_mdl = None" in patch
    assert "if not rerank_mdl:" in patch


def test_ragflow_sse_requests_abort_on_unmount() -> None:
    patch = read_text(ROOT / "patches" / "ragflow" / "0006-localmathrag-abort-sse-on-unmount.patch")
    assert "activeControllerRef = useRef<AbortController>()" in patch
    assert "activeControllerRef.current?.abort()" in patch
    assert "signal: activeController.signal" in patch
    assert "return () => {" in patch
    assert "clearTimeout(timer.current)" in patch
    assert "useSendMessageWithSse" in patch
    assert "useSendMessageBySSE" in patch


def test_ragflow_rerank_disable_request_overrides_saved_config() -> None:
    patch = read_text(ROOT / "patches" / "ragflow" / "0007-localmathrag-rerank-disable-override.patch")
    dataset_service = read_text(ROOT / "third_party" / "ragflow" / "api" / "apps" / "services" / "dataset_api_service.py")
    bot_api = read_text(ROOT / "third_party" / "ragflow" / "api" / "apps" / "restful_apis" / "bot_api.py")
    assert 'req["rerank_id"] if "rerank_id" in req else search_config.get("rerank_id")' in patch
    assert 'if "rerank_id" not in req:' in patch
    assert dataset_service.count('req["rerank_id"] if "rerank_id" in req else search_config.get("rerank_id")') >= 2
    assert 'search_config.get("rerank_id") or req.get("rerank_id")' not in dataset_service
    assert 'if "rerank_id" not in req:' in bot_api
    assert 'if not req.get("rerank_id"):' not in bot_api


def test_ragflow_search_model_switches_have_backend_guards() -> None:
    patch = read_text(ROOT / "patches" / "ragflow" / "0008-localmathrag-search-model-switch-guards.patch")
    bot_api = read_text(ROOT / "third_party" / "ragflow" / "api" / "apps" / "restful_apis" / "bot_api.py")
    chat_api = read_text(ROOT / "third_party" / "ragflow" / "api" / "apps" / "restful_apis" / "chat_api.py")
    search_api = read_text(ROOT / "third_party" / "ragflow" / "api" / "apps" / "restful_apis" / "search_api.py")
    dockerfile = read_text(ROOT / "docker" / "Dockerfile.ragflow-local")
    dockerignore = read_text(ROOT / ".dockerignore")

    assert 'search_config.get("summary") is False' in patch
    assert 'search_config.get("related_search") is False' in patch
    assert 'search_config.get("query_mindmap") is False' in patch
    assert 'search_config.get("summary") is False' in search_api
    assert 'search_config.get("summary") is False' in bot_api
    assert 'search_config.get("related_search") is False' in bot_api
    assert 'search_config.get("related_search") is False' in chat_api
    assert 'search_config.get("query_mindmap") is False' in bot_api
    assert 'search_config.get("query_mindmap") is False' in chat_api
    assert "COPY third_party/ragflow/api/apps/restful_apis/search_api.py" in dockerfile
    assert "COPY third_party/ragflow/api/apps/restful_apis/chat_api.py" in dockerfile
    assert "!third_party/ragflow/api/apps/restful_apis/search_api.py" in dockerignore
    assert "!third_party/ragflow/api/apps/restful_apis/chat_api.py" in dockerignore


def test_ragflow_runtime_warning_surface_contract() -> None:
    patch = read_text(ROOT / "patches" / "ragflow" / "0009-localmathrag-runtime-warning-surface.patch")
    dockerfile = read_text(ROOT / "docker" / "Dockerfile.ragflow-local")
    dockerignore = read_text(ROOT / ".dockerignore")
    assert "Local rerank runtime degraded" in patch
    assert "runtime_warnings" in patch
    assert "showRuntimeWarnings" in patch
    assert "message.warning(text)" in patch
    assert "Rerank model is configured but unavailable; using ES/KNN scoring" in patch
    assert "COPY third_party/ragflow/rag/llm/rerank_model.py" in dockerfile
    assert "COPY third_party/ragflow/rag/llm/embedding_model.py" in dockerfile
    assert "!third_party/ragflow/rag/llm/rerank_model.py" in dockerignore
    assert "!third_party/ragflow/rag/llm/embedding_model.py" in dockerignore


def test_ragflow_search_progress_and_fast_summary_contract() -> None:
    patch = read_text(ROOT / "patches" / "ragflow" / "0010-localmathrag-search-progress-and-fast-summary.patch")
    dialog_service = read_text(ROOT / "third_party" / "ragflow" / "api" / "db" / "services" / "dialog_service.py")
    logic_hooks = read_text(ROOT / "third_party" / "ragflow" / "web" / "src" / "hooks" / "logic-hooks.ts")
    chat_types = read_text(ROOT / "third_party" / "ragflow" / "web" / "src" / "interfaces" / "database" / "chat.ts")
    search_view = read_text(ROOT / "third_party" / "ragflow" / "web" / "src" / "pages" / "next-search" / "search-view.tsx")
    compose = read_text(ROOT / "docker" / "docker-compose.localmathrag.yml")

    assert "_localmathrag_search_progress" in patch
    dynamic_patch = read_text(ROOT / "patches" / "ragflow" / "0014-localmathrag-dynamic-search-token-budget.patch")
    assert "default_answer_token_budget = min(1024" in dynamic_patch
    assert "default_knowledge_token_budget = min(4096" in dynamic_patch
    assert "LOCALMATHRAG_SEARCH_DYNAMIC_TOKEN_BUDGET" in dynamic_patch
    assert "LOCALMATHRAG_SEARCH_ANSWER_TOKEN_BUDGET_RATIO" in dynamic_patch
    assert "LOCALMATHRAG_SEARCH_KNOWLEDGE_TOKEN_BUDGET_RATIO" in dynamic_patch
    assert "_localmathrag_search_token_budgets(max_tokens, question)" in dialog_service
    assert "budget_mode=budget.get" in dialog_service
    assert '"summary"' in dialog_service and "Generating summary" in dialog_service
    assert '"progress": _localmathrag_search_progress(' in dialog_service
    assert '"citations" if citation_use_model else "references"' in dialog_service
    assert '"Preparing citations" if citation_use_model else "Preparing references"' in dialog_service
    assert "progress: d.progress ?? prev.progress" in logic_hooks
    assert "interface ISearchProgress" in chat_types
    assert "progress?: ISearchProgress" in chat_types
    assert "searchProgress.message || searchProgress.stage" in search_view
    assert "<Progress className=\"mt-2 h-1.5\"" in search_view
    assert "LOCALMATHRAG_SEARCH_DYNAMIC_TOKEN_BUDGET: ${LOCALMATHRAG_SEARCH_DYNAMIC_TOKEN_BUDGET:-1}" in compose
    assert "LOCALMATHRAG_SEARCH_ANSWER_TOKEN_BUDGET_MAX: ${LOCALMATHRAG_SEARCH_ANSWER_TOKEN_BUDGET_MAX:-8192}" in compose
    assert "LOCALMATHRAG_SEARCH_KNOWLEDGE_TOKEN_BUDGET_MAX: ${LOCALMATHRAG_SEARCH_KNOWLEDGE_TOKEN_BUDGET_MAX:-16384}" in compose


def test_ragflow_search_summary_runtime_retry_contract() -> None:
    patch = read_text(ROOT / "patches" / "ragflow" / "0012-localmathrag-search-summary-timeout-retry.patch")
    citation_patch = read_text(ROOT / "patches" / "ragflow" / "0015-localmathrag-search-citation-timeout.patch")
    citation_render_patch = read_text(ROOT / "patches" / "ragflow" / "0017-localmathrag-search-citation-render.patch")
    citation_no_model_patch = read_text(ROOT / "patches" / "ragflow" / "0018-localmathrag-search-citation-no-model-test.patch")
    dialog_service = read_text(ROOT / "third_party" / "ragflow" / "api" / "db" / "services" / "dialog_service.py")
    logic_hooks = read_text(ROOT / "third_party" / "ragflow" / "web" / "src" / "hooks" / "logic-hooks.ts")
    compose = read_text(ROOT / "docker" / "docker-compose.localmathrag.yml")
    dev_up = read_text(ROOT / "scripts" / "dev-up.ps1")

    assert "LocalMathRAGSummaryTimeout" in patch
    assert "LOCALMATHRAG_SEARCH_SUMMARY_FIRST_TOKEN_TIMEOUT_SECONDS" in patch
    assert "LOCALMATHRAG_SEARCH_SUMMARY_RETRY_BUDGET_RATIO" in patch
    assert "answer_reset" in patch
    assert "Summary timed out; reducing context and retrying" in patch
    assert "asyncio.wait_for(iterator.__anext__()" in dialog_service
    assert "_localmathrag_search_citation_timeout_seconds" in citation_patch
    assert "_localmathrag_fast_reference" in citation_patch
    assert "citation_warning" in citation_patch
    assert "asyncio.wait_for(" in citation_patch
    assert "LOCALMATHRAG_SEARCH_CITATION_TIMEOUT_SECONDS" in citation_patch
    citation_embedding_patch = read_text(ROOT / "patches" / "ragflow" / "0016-localmathrag-citation-strong-embedding.patch")
    assert "localmathrag_embedding_purpose" in citation_embedding_patch
    assert "localmathrag_strong_embedding" in citation_embedding_patch
    assert 'localmathrag_embedding_purpose("citation")' in dialog_service
    assert "answer_replace" in citation_render_patch
    assert 'final["answer_replace"] = True' in dialog_service
    assert "d.answer_replace === true" in logic_hooks
    assert "LOCALMATHRAG_SEARCH_CITATION_USE_MODEL" in citation_no_model_patch
    assert "_localmathrag_search_citation_use_model" in dialog_service
    assert "citation_model_disabled" in dialog_service
    assert "citation_use_model = _localmathrag_search_citation_use_model()" in dialog_service
    assert "Preparing references" in dialog_service
    assert "citation_timeout_seconds" in dialog_service
    assert "answer_reset" in dialog_service
    assert "shouldResetAnswer" in logic_hooks
    assert "d.answer_reset === true" in logic_hooks
    assert "LOCALMATHRAG_SEARCH_SUMMARY_MAX_RETRIES: ${LOCALMATHRAG_SEARCH_SUMMARY_MAX_RETRIES:-1}" in compose
    assert "LOCALMATHRAG_SEARCH_DYNAMIC_TOKEN_BUDGET: ${LOCALMATHRAG_SEARCH_DYNAMIC_TOKEN_BUDGET:-1}" in compose
    assert "LOCALMATHRAG_SEARCH_CONTEXT_RESERVED_TOKENS: ${LOCALMATHRAG_SEARCH_CONTEXT_RESERVED_TOKENS:-2048}" in compose
    assert "LOCALMATHRAG_SEARCH_CITATION_TIMEOUT_SECONDS: ${LOCALMATHRAG_SEARCH_CITATION_TIMEOUT_SECONDS:-20}" in compose
    assert "LOCALMATHRAG_SEARCH_CITATION_USE_MODEL: ${LOCALMATHRAG_SEARCH_CITATION_USE_MODEL:-0}" in compose
    assert "LOCALMATHRAG_EMBEDDING_CITATION_READY_TIMEOUT_SECONDS: ${LOCALMATHRAG_EMBEDDING_CITATION_READY_TIMEOUT_SECONDS:-18}" in compose
    assert "local-summary-dynamic" in dev_up


def test_ragflow_runtime_model_switch_contract() -> None:
    patch = read_text(ROOT / "patches" / "ragflow" / "0011-localmathrag-runtime-model-switch.patch")
    service = read_text(ROOT / "third_party" / "ragflow" / "api" / "apps" / "services" / "models_api_service.py")
    dockerfile = read_text(ROOT / "docker" / "Dockerfile.ragflow-local")
    dockerignore = read_text(ROOT / ".dockerignore")
    assert "LOCALMATHRAG_RUNTIME_SWITCH_URL" in patch
    assert "/v1/runtime/switch-model" in patch
    assert "_switch_local_runtime_model_if_needed" in patch
    assert '"start": kind in {"chat", "rerank"}' in patch
    assert "TenantService.update_by_id(tenant_id, {field_name: default_model})" in service
    assert "_switch_local_runtime_model_if_needed(" in service
    assert "COPY third_party/ragflow/api/apps/services/models_api_service.py" in dockerfile
    assert "!third_party/ragflow/api/apps/services/models_api_service.py" in dockerignore


def test_ragflow_docx_formula_image_chunks_contract() -> None:
    patch = read_text(ROOT / "patches" / "ragflow" / "0013-localmathrag-docx-formula-image-chunks.patch")
    naive = read_text(ROOT / "third_party" / "ragflow" / "rag" / "app" / "naive.py")
    figure_parser = read_text(ROOT / "third_party" / "ragflow" / "deepdoc" / "parser" / "figure_parser.py")
    dockerfile = read_text(ROOT / "docker" / "Dockerfile.ragflow-local")

    assert "WORD_INLINE_IMAGE_PLACEHOLDER" in patch
    assert "WORD_EQUATION_PREFIX" in patch
    assert ".//m:oMath | .//m:oMathPara" in patch
    assert "__paragraph_text_with_math_placeholders" in patch
    assert "__image_placeholder" in patch
    assert "last_image_text" in patch
    assert "self.__image_placeholder(p, text)" in patch
    assert '(chunks[idx].get(\'text\') or \'\').rstrip() + "\\n" + description' in patch
    assert "WORD_INLINE_IMAGE_PLACEHOLDER" in naive
    assert ".//m:oMath | .//m:oMathPara" in naive
    assert "last_image_text or self.WORD_INLINE_IMAGE_PLACEHOLDER" in naive
    assert "self.__image_placeholder(p, text)" in naive
    assert "(chunks[idx].get('text') or '').rstrip() + \"\\n\" + description" in figure_parser
    assert "COPY third_party/ragflow/rag/app/naive.py" in dockerfile
    assert "COPY third_party/ragflow/deepdoc/parser/figure_parser.py" in dockerfile


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
    assert "/v1/runtime/switch-model" in text
    assert "/v1/embeddings" in text
    assert "/v1/rerank" in text
    assert "EmbeddingRequest" in text
    assert "RerankRequest" in text
    assert "RuntimeEnsureRequest" in text
    assert "RuntimeSwitchModelRequest" in text
    assert "_switch_runtime_model" in text
    assert "_find_local_model_for_runtime" in text
    assert "_runtime_command_for_model" in text
    assert "_runtime_env_for_model" in text
    assert "LOCALMATHRAG_GGUF_MODEL" in text
    assert "LOCALMATHRAG_EMBEDDING_MODEL" in text
    assert "LOCALMATHRAG_RERANK_MODEL" in text
    assert "Embedding model switched. Existing knowledge-base vectors should be reparsed" in text
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
    assert "RUNTIME_CONFIG_FILE" in text
    assert "LOCALMATHRAG_RUNTIME_CONFIG_FILE" in text
    assert "_load_runtime_config" in text
    assert "_write_runtime_config" in text
    assert "_persist_rerank_runtime_config" in text
    assert "_runtime_config_disabled_payload" in text
    assert "runtime_config" in text
    assert "/v1/chat/completions" in text
    assert "/v1/completions" in text
    assert "LLAMA_RUNTIME_BASE_URL" in text
    assert "LOCALMATHRAG_LLAMA_RUNTIME_BASE_URL" in text
    assert "LLAMA_CONTAINER_NAME" in text
    assert "LOCALMATHRAG_LLAMA_CONTAINER" in text
    assert "LLAMA_COMPOSE_SERVICE" in text
    assert "CHAT_REQUEST_TIMEOUT_SECONDS" in text
    assert "_proxy_openai_generation" in text
    assert "_runtime_chat_model_name" in text
    assert "_generation_body_for_chat_runtime" in text
    assert 'normalized["model"] = _runtime_chat_model_name()' in text
    assert "CHAT_CONTEXT_SIZE" in text
    assert "CHAT_CONTEXT_CLAMP_ENABLED" in text
    assert "_fit_generation_payload_to_chat_context" in text
    assert '}/tokenize"' in text
    assert "CHAT_RUNTIME_REQUEST_RETRIES" in text
    assert "_is_recoverable_chat_runtime_error" in text
    assert "_recover_chat_runtime_for_retry" in text
    assert "_stream_chat_runtime_proxy" in text
    assert "_json_chat_runtime_proxy" in text
    assert "X-LocalMathRAG-Runtime-Retry" in text
    assert "localmathrag_runtime_retry_error" in text
    assert "_stream_runtime_proxy" in text
    assert "RERANK_REQUEST_TIMEOUT_SECONDS" in text
    assert "LOCALMATHRAG_RERANK_REQUEST_TIMEOUT_SECONDS" in text
    assert "RERANK_RUNTIME_BATCH_SIZE" in text
    assert "LOCALMATHRAG_RERANK_RUNTIME_BATCH_SIZE" in text
    assert "RERANK_MODEL_NAME" in text
    assert "LOCALMATHRAG_RERANK_MODEL" in text
    assert "RERANK_PROFILE" in text
    assert "LOCALMATHRAG_RERANK_PROFILE" in text
    assert "RERANK_START_MAX_FAILURES" in text
    assert "LOCALMATHRAG_RERANK_START_MAX_FAILURES" in text
    assert "RERANK_CONTEXT_MIN_TOKENS" in text
    assert "LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS_MIN" in text
    assert "RERANK_CONTEXT_RECREATE_RETRIES" in text
    assert "_recreate_rerank_container_with_lower_context" in text
    assert "_retry_rerank_with_lower_context" in text
    assert "_retry_runtime_with_lower_config" in text
    assert "_runtime_disable_after_failures_enabled" in text
    assert "_runtime_start_max_failures" in text
    assert "_should_persistently_disable_runtime_after_failures" in text
    assert "_docker_create_container_from_inspect" in text
    assert "rerank_context_recreated" in text
    assert "rerank context is at minimum" in text
    assert "rerank context retry limit" in text
    assert "--max-batch-tokens" in text
    assert "runtime_disabled" in text
    assert "RUNTIME_START_FAILURE_COUNTS" in text
    assert "_runtime_disabled_snapshot" in text
    assert "OPTIONAL_RUNTIME_STOP_ON_READY_TIMEOUT" in text
    assert 'LOCALMATHRAG_OPTIONAL_RUNTIME_MAX_ACTIVE", "auto"' in text
    assert "SCHEDULER_POLICY" in text
    assert "_resolve_optional_runtime_policy" in text
    assert "_runtime_host_fingerprint" in text
    assert "_persist_scheduler_runtime_config" in text
    assert "_runtime_scheduler_policy_snapshot" in text
    assert "_maybe_record_chat_embedding_concurrency_observed" in text
    assert "_probe_chat_embedding_runtime" in text
    assert "_reset_scheduler_runtime_policy" in text
    assert "_scheduler_auto_probe_needed" in text
    assert "_run_scheduler_auto_probe" in text
    assert "SCHEDULER_AUTO_PROBE" in text
    assert "LOCALMATHRAG_SCHEDULER_AUTO_PROBE" in text
    assert "RuntimeProbeRequest" in text
    assert "/v1/runtime/policy" in text
    assert "/v1/runtime/probe" in text
    assert "/v1/runtime/policy/reset" in text
    assert "chat and embedding runtimes completed live requests concurrently" in text
    assert "CHAT_BACKGROUND_START" in text
    assert "CHAT_BACKGROUND_START_DELAY_SECONDS" in text
    assert "CHAT_RESTORE_AFTER_EMBEDDING" in text
    assert "CHAT_RESTORE_AFTER_EMBEDDING_DELAY_SECONDS" in text
    assert "_schedule_chat_restore_after_embedding" in text
    assert "EMBEDDING_DEGRADE_SMALL_REQUESTS_WHEN_CHAT_READY" in text
    assert "EMBEDDING_SMALL_REQUEST_MAX_INPUTS" in text
    assert "EMBEDDING_SMALL_REQUEST_MAX_TOKENS" in text
    assert "EMBEDDING_AUTOSTART_COOLDOWN_SECONDS" in text
    assert "EMBEDDING_BACKGROUND_PREWARM" in text
    assert "NO_COLD_START_IN_REQUEST" in text
    assert "AUXILIARY_PREWARM_ENABLED" in text
    assert "RECENT_TASK_WINDOW_SECONDS" in text
    assert "RUNTIME_ACTIVE_REQUESTS" in text
    assert "RUNTIME_RECENT_TASKS" in text
    assert "_runtime_request_context" in text
    assert "_stream_with_runtime_activity" in text
    assert "_persist_recent_runtime_task" in text
    assert "_persist_runtime_prewarm_result" in text
    assert '"recent_tasks"' in text
    assert '"prewarm"' in text
    assert "_recent_task_prewarm_candidates" in text
    assert "_schedule_recent_task_prewarm" in text
    assert "no_cold_start_in_request" in text
    assert "recent_task_counts" in text
    assert "active_requests" in text
    assert "EMBEDDING_DOCUMENT_REQUEST_MIN_INPUTS" in text
    assert "EMBEDDING_DOCUMENT_REQUEST_MIN_TOKENS" in text
    assert "EMBEDDING_DOCUMENT_READY_TIMEOUT_SECONDS" in text
    assert "EMBEDDING_CITATION_READY_TIMEOUT_SECONDS" in text
    assert "EMBEDDING_DOCUMENT_PREEMPT_CHAT" in text
    assert "localmathrag_embedding_purpose" in text
    assert "localmathrag_strong_embedding" in text
    assert "OPTIONAL_RUNTIME_DISABLE_AFTER_FAILURES" in text
    assert "OPTIONAL_RUNTIME_START_MAX_FAILURES" in text
    assert "PERSISTENT_OPTIONAL_RUNTIME_KINDS" in text
    assert '"disabled_runtimes"' in text
    assert "_persist_optional_runtime_disabled_config" in text
    assert "_optional_runtime_disabled_config" in text
    assert "_runtime_start_failure_cooldown_seconds" in text
    assert "_is_document_embedding_request" in text
    assert "_is_quality_embedding_request" in text
    assert "_prepare_quality_embedding_priority" in text
    assert "quality_embedding_priority_reset" in text
    assert "quality_embedding_resource_override" in text
    assert "quality_override_reason" in text
    assert "high priority embedding request reserved resources" in text
    assert "bypass_cached_failure=quality_request" in text
    assert "bypass_runtime_disabled=quality_request" in text
    assert "prefer_quality=quality_request" in text
    assert "allow_background=NO_COLD_START_IN_REQUEST and not quality_request" in text
    assert 'elif purpose == "citation":' in text
    assert "min(ready_timeout_seconds, EMBEDDING_CITATION_READY_TIMEOUT_SECONDS)" in text
    assert '"resource_warning"' in text
    assert '_record_runtime_start_failure("embedding", reason)' in text
    assert "if not quality_request:" in text
    assert '_maybe_record_chat_embedding_concurrency_observed({"embedding_request_seconds": embedding_seconds})' in text
    assert "_should_degrade_small_embedding_for_chat" in text
    assert "_fallback_embedding_response" in text
    assert "RERANK_BACKGROUND_PREWARM" in text
    assert "_runtime_startup_scheduler" in text
    assert "_prewarm_runtime" in text
    assert "_runtime_preemption_order" in text
    assert 'return ["rerank", "chat", "vision", "asr", "tts"]' in text
    assert "_prepare_runtime_start" in text
    assert "balance_actions = _balance_optional_runtimes(target)" in text
    assert "runtime_degradation" in text
    assert "runtime_degradations" in text
    assert "_record_runtime_degradation" in text
    assert "RUNTIME_START_FAILURE_COOLDOWN_SECONDS" in text
    assert "RUNTIME_START_FAILURE_PROBE_TIMEOUT_SECONDS" in text
    assert "RUNTIME_READY_PROBE_TIMEOUT_SECONDS" in text
    assert "RERANK_BACKGROUND_START" in text
    assert "_schedule_runtime_background_start" in text
    assert "background runtime start deferred because user tasks stayed active" in text
    assert "allow_background=NO_COLD_START_IN_REQUEST" in text
    assert "_cached_runtime_start_failure" in text
    assert "_record_runtime_start_failure" in text
    assert "OPTIONAL_RUNTIME_DEGRADE_STOP_TIMEOUT_SECONDS" in text
    assert "_runtime_rerank_response" in text
    assert "_fallback_rerank_response" in text
    assert "rerank runtime failed; falling back" in text
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
    assert '"max_tokens": 16384' in text
    assert "payload.get(\"runtime_model_name\")" in text
    assert "payload.get(\"name\")" in text
    assert "_base_url_for_payload" in text
    assert "_model_identity_payload" in text
    assert "_endpoint_key_for_model_type" in text
    assert "_base_url_for_endpoint_key" in text
    assert "def endpoint_ok(endpoint_key: str)" in text
    assert "_optional_runtime_endpoint_status(endpoint_key, timeout=0.25)" in text
    assert 'endpoint_key != "chat"' in text
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
    dev_down_text = read_text(ROOT / "scripts" / "dev-down.ps1")
    assert "LauncherInstanceCoordinator.TryForwardToExistingInstance()" in program_text
    assert "LauncherInstanceCoordinator.CurrentVersion" in program_text
    assert "PostAsync($\"{StartupStatusServer.EntryUrl}focus\"" in program_text
    assert "PostAsync($\"{StartupStatusServer.EntryUrl}shutdown\"" in program_text
    assert "WaitForPreviousInstanceToClose()" in program_text
    assert "Application.Run(new TrayContext())" in program_text
    assert "class LauncherContext" not in program_text
    assert "Application.Run(new LauncherContext())" not in program_text
    assert "BackgroundStartupTask.StartDetached()" in program_text
    assert "StartupStatusServer.StartDetached()" in program_text
    assert "internal static class StartupStatusServer" in program_text
    assert "TcpListener(IPAddress.Loopback" in program_text
    assert "StartupStatusServer.EntryUrl" in program_text
    assert 'path.StartsWith("/version"' in program_text
    assert 'path.StartsWith("/focus"' in program_text
    assert 'path.StartsWith("/shutdown"' in program_text
    assert "RegisterControlHandlers" in program_text
    assert "ShutdownLauncherOnly()" in program_text
    assert 'data-configured-language="__LANGUAGE__"' in program_text
    assert "ResolveConfiguredLanguage" in program_text
    assert "VITE_DEFAULT_LANGUAGE_CODE" in program_text
    assert "LOCALMATHRAG_LANGUAGE" in program_text
    assert "HtmlAttributeEncode" in program_text
    assert "BuildLoadingHtmlPage()" in program_text
    assert 'path.StartsWith("/start"' in program_text
    assert 'path.StartsWith("/stop"' in program_text
    assert 'path.StartsWith("/reset-runtime-policy"' in program_text
    assert "Reset runtime policy" in program_text
    assert "policy-note" in program_text
    assert "const translations = {" in program_text
    assert "'zh-Hans'" in program_text
    assert "policyNote" in program_text
    assert "probeRagflowWhenStatusIsUnavailable" in program_text
    assert "statusUnavailableStopped" in program_text
    assert "renderStatusUnavailableStopped" in program_text
    assert "if (statusFailures >= 5)" in program_text
    assert "if (statusFailures >= 6) window.close()" in program_text
    assert "mode: 'no-cors'" in program_text
    assert "ResetRuntimePolicyAsync" in program_text
    assert "ResetRuntimeConfigPolicy" in program_text
    assert "resetting_policy" in program_text
    assert 'config.Remove("scheduler")' in program_text
    assert 'config.Remove("rerank")' in program_text
    assert "stoppingTitle" in program_text
    assert "stoppedTitle" in program_text
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
    assert 'Path.Combine(launcherDataDir, "browser-profile")' in program_text
    assert "TryFocusExistingBrowserWindow(state)" in program_text
    assert "TryFindBrowserAppWindow" in program_text
    assert "EnumWindows" in program_text
    assert "focused existing browser app window" in program_text
    assert "browser-profile-run-" in program_text
    assert "CleanupOldBrowserProfiles(launcherDataDir)" in program_text
    assert "BuildBrowserArguments" in program_text
    assert "--no-default-browser-check" in program_text
    assert "--disable-sync" in program_text
    assert "--disable-extensions" not in program_text
    assert "--disable-background-networking" in program_text
    assert "--disable-notifications" in program_text
    assert "msEdgeOnRampFRE" in program_text
    assert "msEdgeSignIn" in program_text
    assert "localmathrag_reload" in program_text
    assert "WaitForRagflowWebReadyAsync" in program_text
    assert "ExtractStartupAssetPaths" in program_text
    assert "<div id=\\\"root\\\"></div>" in program_text
    assert "StartServicesAsync(openBrowser: true)" not in program_text
    assert "uiContext.Post(async _ => await StartServicesAsync(openBrowser: true), null)" not in program_text
    assert "StartDockerDesktop" in program_text
    assert "dockerDesktopStartedByLauncher" in program_text
    assert "ReleaseDockerResourcesOnExitAsync(root)" in program_text
    assert "LOCALMATHRAG_RELEASE_DOCKER_WSL_ON_EXIT" in program_text
    assert "HasRunningDockerContainersAsync" in program_text
    assert '"ps --quiet"' in program_text
    assert "ShutdownDockerDesktopAsync" in program_text
    assert '"DockerCli.exe"' in program_text
    assert '"-Shutdown"' in program_text
    assert "ShutdownWslAsync" in program_text
    assert '"wsl.exe"' in program_text
    assert '"--shutdown"' in program_text
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
    assert "Resolve-LlamaContextSize" in dev_up_text
    assert "LOCALMATHRAG_CTX_SIZE_TARGET" in dev_up_text
    assert "24576" in dev_up_text
    assert "Set-SearchTokenBudgetsFromContext" in dev_up_text
    assert "local-summary-dynamic" in dev_up_text
    assert "LOCALMATHRAG_CTX_SIZE_SOURCE" in dev_up_text
    assert "Resolve-RerankMaxBatchTokens" in dev_up_text
    assert "runtime-config.json" in dev_up_text
    assert "Read-RuntimeConfig" in dev_up_text
    assert "Apply-PersistedRerankRuntimeConfig" in dev_up_text
    assert "Write-RerankRuntimeConfig" in dev_up_text
    assert "LOCALMATHRAG_RERANK_PERSISTED_DISABLED" in dev_up_text
    assert "create-minimum-failed" in dev_up_text
    assert "create-retry-limit" in dev_up_text
    assert "Get-GpuIdentity" in dev_up_text
    assert "Get-GpuMemoryInfo" in dev_up_text
    assert "Get-GpuMemoryGb" in dev_up_text
    assert "Get-TotalMemoryGb" in dev_up_text
    assert "Get-AvailableMemoryGb" in dev_up_text
    assert "Resolve-ResourceTier" in dev_up_text
    assert "AvailableGpuGb" in dev_up_text
    assert "AvailableRamGb" in dev_up_text
    assert "Set-ResourceAwareDefaults" in dev_up_text
    assert "LOCALMATHRAG_RESOURCE_TIER" in dev_up_text
    assert "LOCALMATHRAG_GPU_AVAILABLE_MEMORY_GB" in dev_up_text
    assert "LOCALMATHRAG_RAM_AVAILABLE_GB" in dev_up_text
    assert "available-gpu-ram" in dev_up_text
    assert "LOCALMATHRAG_CTX_GPU_MEMORY_RESERVE_GB" in dev_up_text
    assert "LOCALMATHRAG_RERANK_CONTEXT_GPU_MEMORY_RESERVE_GB" in dev_up_text
    assert "resource-aware" in dev_up_text
    assert "Get-HostFingerprint" in dev_up_text
    assert "LOCALMATHRAG_HOST_FINGERPRINT" in dev_up_text
    assert "Resolve-OptionalRuntimeMaxActive" in dev_up_text
    assert 'return "auto"' in dev_up_text
    assert "adaptive-auto" in dev_up_text
    assert "LOCALMATHRAG_LOW_VRAM_GB" not in dev_up_text
    assert "LOCALMATHRAG_OPTIONAL_RUNTIME_MAX_ACTIVE_SOURCE" in dev_up_text
    assert "LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS" in dev_up_text
    assert "LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS_MIN" in dev_up_text
    assert "LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS_STEP" in dev_up_text
    assert "LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS_SOURCE" in dev_up_text
    assert "Reset-RerankLazyRuntimeContainer" in dev_up_text
    assert "Get-LoweredRerankMaxBatchTokens" in dev_up_text
    assert "LOCALMATHRAG_RERANK_CREATE_RETRY_LIMIT" in dev_up_text
    assert 'Get-EnvInt "LOCALMATHRAG_RERANK_CREATE_RETRY_LIMIT" 2' in dev_up_text
    assert "create-retry" in dev_up_text
    assert "LOCALMATHRAG_LLAMA_PARALLEL" in dev_up_text
    assert "LOCALMATHRAG_LLAMA_RUNTIME_BASE_URL" in dev_up_text
    assert "LOCALMATHRAG_LLAMA_CONTAINER" in dev_up_text
    assert "LOCALMATHRAG_LLAMA_COMPOSE_SERVICE" in dev_up_text
    assert "DOCKER_API_VERSION" in dev_up_text
    assert "Reset-LazyRuntimeContainer" in dev_up_text
    assert "rm --stop --force $Service" in dev_up_text
    assert "create --force-recreate --no-build --pull never $Service" in dev_up_text
    assert 'Reset-LazyRuntimeContainer "localmathrag-llama-cpp-cuda"' in dev_up_text
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
    assert "Remove-StaleRagflowProfileContainers" in dev_up_text
    assert 'foreach ($service in @("ragflow-cpu", "ragflow-gpu"))' in dev_up_text
    assert 'docker compose --profile cpu --profile gpu -f docker-compose.yml -f $Override rm --stop --force $service' in dev_up_text
    assert 'if ($Device -eq "gpu") { "ragflow-gpu" } else { "ragflow-cpu" }' in dev_up_text
    assert '"cpu"' in dev_down_text
    assert '"gpu"' in dev_down_text
    assert '"elasticsearch"' in dev_down_text
    assert '"infinity"' in dev_down_text
    assert '"opensearch"' in dev_down_text
    assert '"llama-cpp-cuda"' in dev_down_text
    assert '"embedding-cuda"' in dev_down_text
    assert '"rerank-cuda"' in dev_down_text
    assert '"vlm-cuda"' in dev_down_text
    assert '"down"' in dev_down_text
    assert '"--remove-orphans"' in dev_down_text
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
    assert 'Reset-LazyRuntimeContainer "localmathrag-embedding"' in dev_up_text
    assert "Reset-RerankLazyRuntimeContainer" in dev_up_text
    assert 'Reset-LazyRuntimeContainer "localmathrag-vlm"' in dev_up_text
    assert "failed to stop lazy $Service" in dev_up_text
    runtime_test = ROOT / "scripts" / "test-runtime-balancer.ps1"
    assert runtime_test.exists()
    runtime_test_text = read_text(runtime_test)
    assert "/v1/embeddings" in runtime_test_text
    assert "/v1/rerank" in runtime_test_text
    assert "/v1/runtime/ensure" in runtime_test_text
    assert "/v1/runtime/status" in runtime_test_text
    assert "/v1/runtime/stop" in runtime_test_text
    assert "IncludeVlm" in runtime_test_text
    assert "MaxActiveRuntime" in runtime_test_text
    assert 'foreach ($kind in @("chat", "embedding", "rerank", "vision", "asr", "tts"))' in runtime_test_text
    assert "embedding should preempt rerank" in runtime_test_text
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
    assert "api/apps/restful_apis/bot_api.py" in dockerfile
    assert "api/utils/validation_utils.py" in dockerfile
    assert "!third_party/ragflow/api/apps/services/dataset_api_service.py" in dockerignore
    assert "!third_party/ragflow/api/apps/services/local_dataset_service.py" in dockerignore
    assert "!third_party/ragflow/api/ragflow_server.py" in dockerignore
    assert "!third_party/ragflow/api/apps/restful_apis/dataset_api.py" in dockerignore
    assert "!third_party/ragflow/api/apps/restful_apis/bot_api.py" in dockerignore
    assert "!third_party/ragflow/api/db/services/dialog_service.py" in dockerignore
    assert "!third_party/ragflow/api/utils/validation_utils.py" in dockerignore
    assert "!third_party/ragflow/rag/llm/" in dockerignore
    assert "!third_party/ragflow/rag/llm/chat_model.py" in dockerignore
    assert "!third_party/ragflow/rag/llm/embedding_model.py" in dockerignore
    assert "!third_party/ragflow/rag/llm/rerank_model.py" in dockerignore
    assert "!third_party/ragflow/rag/nlp/" in dockerignore
    assert "!third_party/ragflow/rag/nlp/search.py" in dockerignore
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
    assert "COPY third_party/ragflow/api/db/services/dialog_service.py" in dockerfile_text
    assert "COPY third_party/ragflow/api/apps/restful_apis/bot_api.py" in dockerfile_text
    assert "COPY third_party/ragflow/deepdoc/parser/figure_parser.py" in dockerfile_text
    assert "COPY third_party/ragflow/rag/llm/chat_model.py" in dockerfile_text
    assert "COPY third_party/ragflow/rag/llm/embedding_model.py" in dockerfile_text
    assert "COPY third_party/ragflow/rag/llm/rerank_model.py" in dockerfile_text
    assert "COPY third_party/ragflow/rag/nlp/search.py" in dockerfile_text
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
    assert "LOCALMATHRAG_RUNTIME_CONFIG_FILE: /data/cache/runtime-config.json" in compose
    assert "LOCALMATHRAG_LLAMA_BASE_URL: http://localmathrag-object-service:8088/v1" in compose
    assert "LOCALMATHRAG_LLAMA_RUNTIME_BASE_URL: ${LOCALMATHRAG_LLAMA_RUNTIME_BASE_URL:-http://localmathrag-llama-cpp-cuda:8080/v1}" in compose
    assert "LOCALMATHRAG_CTX_SIZE: ${LOCALMATHRAG_CTX_SIZE:-24576}" in compose
    assert "${LOCALMATHRAG_ROOT}/data/dataset:/localmathrag/dataset:ro" in compose
    assert "${LOCALMATHRAG_ROOT}/data/cache:/localmathrag/cache" in compose
    assert "${LOCALMATHRAG_CTX_SIZE:-24576}" in compose
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
    assert "LOCALMATHRAG_HOST_FINGERPRINT: ${LOCALMATHRAG_HOST_FINGERPRINT:-}" in compose
    assert "LOCALMATHRAG_OPTIONAL_RUNTIME_MAX_ACTIVE: ${LOCALMATHRAG_OPTIONAL_RUNTIME_MAX_ACTIVE:-auto}" in compose
    assert "LOCALMATHRAG_CHAT_BACKGROUND_START: ${LOCALMATHRAG_CHAT_BACKGROUND_START:-1}" in compose
    assert "LOCALMATHRAG_CHAT_BACKGROUND_START_DELAY_SECONDS: ${LOCALMATHRAG_CHAT_BACKGROUND_START_DELAY_SECONDS:-20}" in compose
    assert "LOCALMATHRAG_CHAT_RESTORE_AFTER_EMBEDDING: ${LOCALMATHRAG_CHAT_RESTORE_AFTER_EMBEDDING:-1}" in compose
    assert "LOCALMATHRAG_CHAT_RESTORE_AFTER_EMBEDDING_DELAY_SECONDS: ${LOCALMATHRAG_CHAT_RESTORE_AFTER_EMBEDDING_DELAY_SECONDS:-30}" in compose
    assert "LOCALMATHRAG_EMBEDDING_DEGRADE_SMALL_REQUESTS_WHEN_CHAT_READY: ${LOCALMATHRAG_EMBEDDING_DEGRADE_SMALL_REQUESTS_WHEN_CHAT_READY:-1}" in compose
    assert "LOCALMATHRAG_EMBEDDING_SMALL_REQUEST_MAX_INPUTS: ${LOCALMATHRAG_EMBEDDING_SMALL_REQUEST_MAX_INPUTS:-2}" in compose
    assert "LOCALMATHRAG_EMBEDDING_SMALL_REQUEST_MAX_TOKENS: ${LOCALMATHRAG_EMBEDDING_SMALL_REQUEST_MAX_TOKENS:-512}" in compose
    assert "LOCALMATHRAG_EMBEDDING_AUTOSTART_COOLDOWN_SECONDS: ${LOCALMATHRAG_EMBEDDING_AUTOSTART_COOLDOWN_SECONDS:-86400}" in compose
    assert "LOCALMATHRAG_EMBEDDING_BACKGROUND_PREWARM: ${LOCALMATHRAG_EMBEDDING_BACKGROUND_PREWARM:-1}" in compose
    assert "LOCALMATHRAG_EMBEDDING_DOCUMENT_REQUEST_MIN_INPUTS: ${LOCALMATHRAG_EMBEDDING_DOCUMENT_REQUEST_MIN_INPUTS:-8}" in compose
    assert "LOCALMATHRAG_EMBEDDING_DOCUMENT_REQUEST_MIN_TOKENS: ${LOCALMATHRAG_EMBEDDING_DOCUMENT_REQUEST_MIN_TOKENS:-2048}" in compose
    assert "LOCALMATHRAG_EMBEDDING_DOCUMENT_READY_TIMEOUT_SECONDS: ${LOCALMATHRAG_EMBEDDING_DOCUMENT_READY_TIMEOUT_SECONDS:-480}" in compose
    assert "LOCALMATHRAG_EMBEDDING_DOCUMENT_PREEMPT_CHAT: ${LOCALMATHRAG_EMBEDDING_DOCUMENT_PREEMPT_CHAT:-1}" in compose
    assert "LOCALMATHRAG_RERANK_BACKGROUND_PREWARM: ${LOCALMATHRAG_RERANK_BACKGROUND_PREWARM:-1}" in compose
    assert "LOCALMATHRAG_RUNTIME_NO_COLD_START_IN_REQUEST: ${LOCALMATHRAG_RUNTIME_NO_COLD_START_IN_REQUEST:-1}" in compose
    assert "LOCALMATHRAG_AUXILIARY_PREWARM_ENABLED: ${LOCALMATHRAG_AUXILIARY_PREWARM_ENABLED:-1}" in compose
    assert "LOCALMATHRAG_AUXILIARY_PREWARM_IDLE_SECONDS: ${LOCALMATHRAG_AUXILIARY_PREWARM_IDLE_SECONDS:-8}" in compose
    assert "LOCALMATHRAG_AUXILIARY_PREWARM_MAX_WAIT_SECONDS: ${LOCALMATHRAG_AUXILIARY_PREWARM_MAX_WAIT_SECONDS:-300}" in compose
    assert "LOCALMATHRAG_RECENT_TASK_WINDOW_SECONDS: ${LOCALMATHRAG_RECENT_TASK_WINDOW_SECONDS:-900}" in compose
    assert "LOCALMATHRAG_RERANK_PREWARM_RECENT_THRESHOLD: ${LOCALMATHRAG_RERANK_PREWARM_RECENT_THRESHOLD:-1}" in compose
    assert "LOCALMATHRAG_EMBEDDING_PREWARM_RECENT_THRESHOLD: ${LOCALMATHRAG_EMBEDDING_PREWARM_RECENT_THRESHOLD:-2}" in compose
    assert "LOCALMATHRAG_SCHEDULER_AUTO_PROBE: ${LOCALMATHRAG_SCHEDULER_AUTO_PROBE:-1}" in compose
    assert "LOCALMATHRAG_SCHEDULER_AUTO_PROBE_TIMEOUT_SECONDS: ${LOCALMATHRAG_SCHEDULER_AUTO_PROBE_TIMEOUT_SECONDS:-240}" in compose
    assert "LOCALMATHRAG_RUNTIME_READY_TIMEOUT_SECONDS: ${LOCALMATHRAG_RUNTIME_READY_TIMEOUT_SECONDS:-240}" in compose
    assert "LOCALMATHRAG_CHAT_READY_TIMEOUT_SECONDS: ${LOCALMATHRAG_CHAT_READY_TIMEOUT_SECONDS:-90}" in compose
    assert "LOCALMATHRAG_CHAT_REQUEST_TIMEOUT_SECONDS: ${LOCALMATHRAG_CHAT_REQUEST_TIMEOUT_SECONDS:-600}" in compose
    assert "LOCALMATHRAG_RUNTIME_READY_PROBE_TIMEOUT_SECONDS: ${LOCALMATHRAG_RUNTIME_READY_PROBE_TIMEOUT_SECONDS:-0.5}" in compose
    assert "LOCALMATHRAG_RUNTIME_START_FAILURE_PROBE_TIMEOUT_SECONDS: ${LOCALMATHRAG_RUNTIME_START_FAILURE_PROBE_TIMEOUT_SECONDS:-0.2}" in compose
    assert "LOCALMATHRAG_RERANK_READY_TIMEOUT_SECONDS: ${LOCALMATHRAG_RERANK_READY_TIMEOUT_SECONDS:-3}" in compose
    assert "LOCALMATHRAG_RERANK_BACKGROUND_START: ${LOCALMATHRAG_RERANK_BACKGROUND_START:-1}" in compose
    assert "LOCALMATHRAG_RERANK_BACKGROUND_READY_TIMEOUT_SECONDS: ${LOCALMATHRAG_RERANK_BACKGROUND_READY_TIMEOUT_SECONDS:-240}" in compose
    assert "LOCALMATHRAG_RUNTIME_STARTUP_STALL_TIMEOUT_SECONDS: ${LOCALMATHRAG_RUNTIME_STARTUP_STALL_TIMEOUT_SECONDS:-120}" in compose
    assert "LOCALMATHRAG_RERANK_REQUEST_TIMEOUT_SECONDS: ${LOCALMATHRAG_RERANK_REQUEST_TIMEOUT_SECONDS:-8}" in compose
    assert "LOCALMATHRAG_RERANK_RUNTIME_BATCH_SIZE: ${LOCALMATHRAG_RERANK_RUNTIME_BATCH_SIZE:-32}" in compose
    assert "LOCALMATHRAG_RERANK_MODEL: ${LOCALMATHRAG_RERANK_MODEL:-bge-reranker-v2-m3}" in compose
    assert "LOCALMATHRAG_RERANK_PROFILE: ${LOCALMATHRAG_RERANK_PROFILE:-cuda}" in compose
    assert "LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS: ${LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS:-16384}" in compose
    assert "LOCALMATHRAG_RERANK_START_MAX_FAILURES: ${LOCALMATHRAG_RERANK_START_MAX_FAILURES:-2}" in compose
    assert "LOCALMATHRAG_RERANK_DISABLE_AFTER_FAILURES: ${LOCALMATHRAG_RERANK_DISABLE_AFTER_FAILURES:-1}" in compose
    assert "LOCALMATHRAG_RERANK_CONTEXT_RECREATE_RETRIES: ${LOCALMATHRAG_RERANK_CONTEXT_RECREATE_RETRIES:-2}" in compose
    assert "LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS_MIN: ${LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS_MIN:-8192}" in compose
    assert "LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS_STEP: ${LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS_STEP:-4096}" in compose
    assert "LOCALMATHRAG_OPTIONAL_RUNTIME_DISABLE_AFTER_FAILURES: ${LOCALMATHRAG_OPTIONAL_RUNTIME_DISABLE_AFTER_FAILURES:-1}" in compose
    assert "LOCALMATHRAG_OPTIONAL_RUNTIME_START_MAX_FAILURES: ${LOCALMATHRAG_OPTIONAL_RUNTIME_START_MAX_FAILURES:-2}" in compose
    assert "LOCALMATHRAG_OPTIONAL_RUNTIME_DEGRADE_STOP_TIMEOUT_SECONDS: ${LOCALMATHRAG_OPTIONAL_RUNTIME_DEGRADE_STOP_TIMEOUT_SECONDS:-1}" in compose
    assert "LOCALMATHRAG_OPTIONAL_RUNTIME_STOP_ON_READY_TIMEOUT: ${LOCALMATHRAG_OPTIONAL_RUNTIME_STOP_ON_READY_TIMEOUT:-0}" in compose
    assert "LOCALMATHRAG_VISION_MIN_AVAILABLE_MEMORY_GB: ${LOCALMATHRAG_VISION_MIN_AVAILABLE_MEMORY_GB:-10}" in compose
    assert "LOCALMATHRAG_EMBEDDING_RUNTIME_BASE_URL: http://localmathrag-embedding:8080/v1" in compose
    assert "LOCALMATHRAG_RERANK_RUNTIME_BASE_URL: http://localmathrag-rerank:8080/v1" in compose
    assert "LOCALMATHRAG_LLAMA_CONTAINER" in compose
    assert "LOCALMATHRAG_LLAMA_COMPOSE_SERVICE" in compose
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
    assert "${LOCALMATHRAG_RERANK_MAX_CLIENT_BATCH_SIZE:-32}" in compose
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
    test_ragflow_search_summary_has_output_budget()
    test_ragflow_rerank_fallback_contract()
    test_ragflow_sse_requests_abort_on_unmount()
    test_ragflow_rerank_disable_request_overrides_saved_config()
    test_ragflow_search_model_switches_have_backend_guards()
    test_ragflow_runtime_warning_surface_contract()
    test_ragflow_search_summary_runtime_retry_contract()
    test_ragflow_runtime_model_switch_contract()
    test_ragflow_docx_formula_image_chunks_contract()
    test_object_service_imports()
    test_windows_launcher_exists()
    test_ragflow_patch_workflow_exists()
    test_docker_compose_mounts_ragflow_backend_overrides()
    print("contract checks passed")


if __name__ == "__main__":
    main()

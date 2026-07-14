from __future__ import annotations

import json
import importlib.util
import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH_ONLY = os.environ.get("LOCALMATHRAG_TEST_PATCH_ONLY") == "1"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def patched_file_text(patch: str) -> str:
    lines: list[str] = []
    for line in patch.splitlines():
        if line.startswith(("--- ", "+++ ", "@@")):
            continue
        if line.startswith("-"):
            continue
        if line.startswith(("+", " ")):
            lines.append(line[1:])
    return "\n".join(lines)


def ragflow_source_available(relative_path: str) -> bool:
    return (ROOT / "third_party" / "ragflow" / relative_path).exists() and not PATCH_ONLY


def read_ragflow_source_or_patch(relative_path: str, patch: str) -> str:
    source_path = ROOT / "third_party" / "ragflow" / relative_path
    if ragflow_source_available(relative_path):
        return read_text(source_path)
    return patched_file_text(patch)


def read_ragflow_source_or_patches(relative_path: str, patches: list[str]) -> str:
    source_path = ROOT / "third_party" / "ragflow" / relative_path
    if ragflow_source_available(relative_path):
        return read_text(source_path)
    return "\n".join(patched_file_text(patch) for patch in patches)


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


def test_ragflow_dataset_upload_ignores_temporary_files_contract() -> None:
    patch = read_text(ROOT / "patches" / "ragflow" / "0042-localmathrag-ignore-temporary-dataset-files.patch")
    dockerfile = read_text(ROOT / "docker" / "Dockerfile.ragflow-local")
    dockerignore = read_text(ROOT / ".dockerignore")

    assert "def is_temporary_file(filename) -> bool:" in patch
    assert '"$$" in basename' in patch
    assert 'basename.startswith("~$")' in patch
    assert 'normalized.startswith(".~lock.")' in patch
    assert '".tmp"' in patch and '".part"' in patch and '".swp"' in patch
    assert '".dwl"' in patch and '".sv$"' in patch
    assert 'TEMPORARY_FILE_NAMES = frozenset({".ds_store", "desktop.ini", "thumbs.db"})' in patch
    assert "if is_temporary_file(file_obj.filename):" in patch
    assert "if child.is_file() and is_temporary_file(child_rel):" in patch
    assert "if is_temporary_file(rel):" in patch
    assert "temporary_previous_entries" in patch
    assert 'deleted["status"] = "deleted"' in patch
    assert 'if filename_type(previous_path) == "other":' in patch
    assert "export const isTemporaryUploadFile" in patch
    assert "!isTemporaryUploadFile(file)" in patch
    assert "if not upload_file_objs:" in patch
    assert "return get_result(data=[])" in patch
    assert "COPY third_party/ragflow/api/utils/file_utils.py /ragflow/api/utils/file_utils.py" in dockerfile
    assert "!third_party/ragflow/api/utils/file_utils.py" in dockerignore


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
    dataset_service = read_ragflow_source_or_patch("api/apps/services/dataset_api_service.py", patch)
    bot_api = read_ragflow_source_or_patch("api/apps/restful_apis/bot_api.py", patch)
    assert 'req["rerank_id"] if "rerank_id" in req else search_config.get("rerank_id")' in patch
    assert 'if "rerank_id" not in req:' in patch
    assert dataset_service.count('req["rerank_id"] if "rerank_id" in req else search_config.get("rerank_id")') >= 2
    assert 'search_config.get("rerank_id") or req.get("rerank_id")' not in dataset_service
    assert 'if "rerank_id" not in req:' in bot_api
    assert 'if not req.get("rerank_id"):' not in bot_api


def test_ragflow_search_model_switches_have_backend_guards() -> None:
    patch = read_text(ROOT / "patches" / "ragflow" / "0008-localmathrag-search-model-switch-guards.patch")
    aux_patch = read_text(ROOT / "patches" / "ragflow" / "0019-localmathrag-search-auxiliary-after-answer.patch")
    bot_api = read_ragflow_source_or_patch("api/apps/restful_apis/bot_api.py", patch)
    chat_api = read_ragflow_source_or_patch("api/apps/restful_apis/chat_api.py", patch)
    search_api = read_ragflow_source_or_patch("api/apps/restful_apis/search_api.py", patch)
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
    assert "LOCALMATHRAG_SEARCH_AUX_TIMEOUT_SECONDS" in aux_patch
    assert "LocalMathRAG related-search auxiliary failed; returning empty result" in aux_patch
    assert "LocalMathRAG mindmap auxiliary failed; returning empty result" in aux_patch
    assert "_localmathrag_related_gen_conf" in aux_patch
    assert "_localmathrag_parse_related_questions" in aux_patch
    assert "_localmathrag_related_fallback_questions" in aux_patch
    assert 'gen_conf.setdefault("max_tokens", 384)' in aux_patch
    assert 'gen_conf.setdefault("max_completion_tokens", gen_conf["max_tokens"])' in aux_patch
    assert 'gen_conf.setdefault("enable_thinking", False)' in aux_patch
    assert "asyncio.wait_for(" in aux_patch
    assert "LocalMathRAG related-search produced no parseable questions" in aux_patch
    assert "/no_think" in aux_patch
    assert "Return 5 numbered questions only." in aux_patch
    assert "Return 5 numbered questions. Do not return an empty answer." in aux_patch
    assert "需要关注哪些典型故障场景" in aux_patch
    assert "json.loads(text)" in aux_patch
    assert "re.split(r\"(?<=[?？])\\s+\", text)" in aux_patch
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
    dynamic_patch = read_text(ROOT / "patches" / "ragflow" / "0014-localmathrag-dynamic-search-token-budget.patch")
    citation_no_model_patch = read_text(ROOT / "patches" / "ragflow" / "0018-localmathrag-search-citation-no-model-test.patch")
    aux_patch = read_text(ROOT / "patches" / "ragflow" / "0019-localmathrag-search-auxiliary-after-answer.patch")
    dialog_service = read_ragflow_source_or_patches(
        "api/db/services/dialog_service.py",
        [patch, dynamic_patch, citation_no_model_patch, aux_patch],
    )
    logic_hooks = read_ragflow_source_or_patch("web/src/hooks/logic-hooks.ts", patch)
    search_hooks = read_ragflow_source_or_patch("web/src/pages/next-search/hooks.ts", aux_patch)
    chat_types = read_ragflow_source_or_patch("web/src/interfaces/database/chat.ts", patch)
    search_view = read_ragflow_source_or_patch("web/src/pages/next-search/search-view.tsx", patch)
    compose = read_text(ROOT / "docker" / "docker-compose.localmathrag.yml")

    assert "_localmathrag_search_progress" in patch
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
    assert "pendingRelatedQuestion" in search_hooks
    assert "pendingSearchParams" in search_hooks
    assert "canFetch" in search_hooks
    assert "setPendingRelatedQuestion(q)" in search_hooks
    assert "fetchRelatedQuestions(pendingRelatedQuestion)" in search_hooks
    assert "_mindmap_sections" in dialog_service
    assert "LOCALMATHRAG_SEARCH_MINDMAP_MAX_CHUNKS" in dialog_service
    assert "LOCALMATHRAG_SEARCH_MINDMAP_MAX_CHARS" in dialog_service
    assert "interface ISearchProgress" in chat_types
    assert "progress?: ISearchProgress" in chat_types
    assert "searchProgress.message || searchProgress.stage" in search_view
    assert "<Progress className=\"mt-2 h-1.5\"" in search_view
    assert "LOCALMATHRAG_SEARCH_DYNAMIC_TOKEN_BUDGET: ${LOCALMATHRAG_SEARCH_DYNAMIC_TOKEN_BUDGET:-1}" in compose
    assert "LOCALMATHRAG_SEARCH_ANSWER_TOKEN_BUDGET_MAX: ${LOCALMATHRAG_SEARCH_ANSWER_TOKEN_BUDGET_MAX:-8192}" in compose
    assert "LOCALMATHRAG_SEARCH_KNOWLEDGE_TOKEN_BUDGET_MAX: ${LOCALMATHRAG_SEARCH_KNOWLEDGE_TOKEN_BUDGET_MAX:-16384}" in compose
    assert "LOCALMATHRAG_SEARCH_AUX_TIMEOUT_SECONDS: ${LOCALMATHRAG_SEARCH_AUX_TIMEOUT_SECONDS:-30}" in compose
    assert "LOCALMATHRAG_SEARCH_MINDMAP_MAX_CHUNKS: ${LOCALMATHRAG_SEARCH_MINDMAP_MAX_CHUNKS:-6}" in compose
    assert "LOCALMATHRAG_SEARCH_MINDMAP_MAX_CHARS: ${LOCALMATHRAG_SEARCH_MINDMAP_MAX_CHARS:-12000}" in compose


def test_ragflow_search_metadata_filter_timeout_contract() -> None:
    patch = read_text(ROOT / "patches" / "ragflow" / "0020-localmathrag-search-metadata-filter-timeout.patch")
    metadata_utils = read_ragflow_source_or_patch("common/metadata_utils.py", patch)
    compose = read_text(ROOT / "docker" / "docker-compose.localmathrag.yml")
    dockerfile = read_text(ROOT / "docker" / "Dockerfile.ragflow-local")
    dockerignore = read_text(ROOT / ".dockerignore")

    assert "LOCALMATHRAG_SEARCH_METADATA_FILTER_TIMEOUT_SECONDS" in patch
    assert "_localmathrag_metadata_filter_timeout_seconds" in metadata_utils
    assert "asyncio.wait_for(" in metadata_utils
    assert "LocalMathRAG metadata filter skipped: no metadata keys available" in metadata_utils
    assert "LocalMathRAG metadata filter generation timed out" in metadata_utils
    assert "continuing without metadata filter" in metadata_utils
    assert "chat model is unavailable" in metadata_utils
    assert "filters and filters.get(\"conditions\")" in metadata_utils
    assert "LOCALMATHRAG_SEARCH_METADATA_FILTER_TIMEOUT_SECONDS: ${LOCALMATHRAG_SEARCH_METADATA_FILTER_TIMEOUT_SECONDS:-8}" in compose
    assert "COPY third_party/ragflow/common/metadata_utils.py" in dockerfile
    assert "!third_party/ragflow/common/metadata_utils.py" in dockerignore


def test_ragflow_search_export_markdown_contract() -> None:
    patch = read_text(ROOT / "patches" / "ragflow" / "0021-localmathrag-search-export-markdown.patch")
    search_view = read_ragflow_source_or_patch("web/src/pages/next-search/search-view.tsx", patch)

    assert "buildSearchExportMarkdown" in search_view
    assert "copyTextToClipboard" in search_view
    assert "downloadFileFromBlob" in search_view
    assert "Clipboard" in search_view
    assert "Download" in search_view
    assert "## AI Summary" in search_view
    assert "## Content List" in search_view
    assert "document_metadata" in search_view
    assert "content_with_weight" in search_view
    assert "text/markdown;charset=utf-8" in search_view
    assert "Copied search results" in search_view
    assert "canShowMindMap" in search_view
    assert "absolute top-16 translate-y-2 right-10 z-30 flex items-center gap-3" in search_view
    assert "size-9 rounded-full p-0" in search_view
    assert "Download Markdown" in search_view


def test_ragflow_search_summary_runtime_retry_contract() -> None:
    patch = read_text(ROOT / "patches" / "ragflow" / "0012-localmathrag-search-summary-timeout-retry.patch")
    citation_patch = read_text(ROOT / "patches" / "ragflow" / "0015-localmathrag-search-citation-timeout.patch")
    citation_embedding_patch = read_text(ROOT / "patches" / "ragflow" / "0016-localmathrag-citation-strong-embedding.patch")
    citation_render_patch = read_text(ROOT / "patches" / "ragflow" / "0017-localmathrag-search-citation-render.patch")
    citation_no_model_patch = read_text(ROOT / "patches" / "ragflow" / "0018-localmathrag-search-citation-no-model-test.patch")
    dialog_service = read_ragflow_source_or_patches(
        "api/db/services/dialog_service.py",
        [patch, citation_patch, citation_embedding_patch, citation_render_patch, citation_no_model_patch],
    )
    logic_hooks = read_ragflow_source_or_patches(
        "web/src/hooks/logic-hooks.ts",
        [patch, citation_patch, citation_render_patch, citation_no_model_patch],
    )
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
    service = read_ragflow_source_or_patch("api/apps/services/models_api_service.py", patch)
    dockerfile = read_text(ROOT / "docker" / "Dockerfile.ragflow-local")
    dockerignore = read_text(ROOT / ".dockerignore")
    assert "LOCALMATHRAG_RUNTIME_SWITCH_URL" in patch
    assert "/v1/runtime/switch-model" in patch
    assert "_switch_local_runtime_model_if_needed" in patch
    assert '"start": kind in {"chat", "rerank"}' in patch
    assert "Local runtime model switch failed" in patch
    assert "switch_result = _switch_local_runtime_model_if_needed(" in service
    update_call = "TenantService.update_by_id(tenant_id, {field_name: default_model})"
    if update_call not in service:
        update_call = "TenantService.update_by_id(tenant_id, updates)"
    assert update_call in service
    assert service.find("switch_result = _switch_local_runtime_model_if_needed(") < service.find(update_call)
    assert "_switch_local_runtime_model_if_needed(" in service
    assert "COPY third_party/ragflow/api/apps/services/models_api_service.py" in dockerfile
    assert "!third_party/ragflow/api/apps/services/models_api_service.py" in dockerignore


def test_local_model_configuration_uses_provider_facing_name() -> None:
    patch = read_text(ROOT / "patches" / "ragflow" / "0027-localmathrag-unify-local-model-identifiers.patch")
    source = read_ragflow_source_or_patch(
        "web/src/pages/user-setting/setting-model/components/un-add-model.tsx", patch
    )
    assert "Persist the same provider-facing identifier for every model type." in patch
    assert "model_name: model.name," in patch
    assert "model_name: model.runtime_model_name || model.name," not in source


def test_model_labels_include_provider_without_changing_model_identity() -> None:
    patch = read_text(ROOT / "patches" / "ragflow" / "0031-localmathrag-model-display-provider.patch")
    label = read_ragflow_source_or_patch(
        "web/src/components/llm-select/llm-label.tsx",
        patch,
    )
    tree = read_ragflow_source_or_patch(
        "web/src/components/model-tree-select.tsx",
        patch,
    )
    assert "providerName !== 'OpenAI-API-Compatible'" in label
    assert "m.provider_name === 'OpenAI-API-Compatible'" in tree
    assert "model_name: modelName" in tree
    assert "model_name: modelName" in patched_file_text(patch)


def test_model_switch_has_visible_progress_feedback() -> None:
    setting = read_ragflow_source_or_patch(
        "web/src/pages/user-setting/setting-model/components/system-setting.tsx",
        read_text(ROOT / "patches" / "ragflow" / "0033-localmathrag-model-switch-progress-feedback.patch"),
    )
    assert "loading: isSwitching" in setting
    assert "disabled={isSwitching}" in setting
    assert "role=\"status\"" in setting
    assert "switchingElapsed" in setting
    assert "setInterval(updateElapsed, 1000)" in setting
    request = read_ragflow_source_or_patches(
        "web/src/hooks/use-llm-request.tsx",
        [
            read_text(ROOT / "patches" / "ragflow" / "0033-localmathrag-model-switch-progress-feedback.patch"),
            read_text(ROOT / "patches" / "ragflow" / "0035-localmathrag-model-switch-readable-error.patch"),
        ],
    )
    assert "onError: () =>" in request
    assert "切换模型失败" in request
    assert "duration: 12" in request


def test_model_switch_runtime_error_is_sanitized() -> None:
    service = read_ragflow_source_or_patch(
        "api/apps/services/models_api_service.py",
        read_text(ROOT / "patches" / "ragflow" / "0035-localmathrag-model-switch-readable-error.patch"),
    )
    assert 're.sub(r"^[\\x00-\\x1f\\ufffd]+", "", reason)' in service


def test_ragflow_docx_formula_image_chunks_contract() -> None:
    patch = read_text(ROOT / "patches" / "ragflow" / "0013-localmathrag-docx-formula-image-chunks.patch")
    naive = read_ragflow_source_or_patch("rag/app/naive.py", patch)
    figure_parser = read_ragflow_source_or_patch("deepdoc/parser/figure_parser.py", patch)
    dockerfile = read_text(ROOT / "docker" / "Dockerfile.ragflow-local")

    assert "WORD_INLINE_IMAGE_PLACEHOLDER" in patch
    assert "WORD_EQUATION_PREFIX" in patch
    assert ".//m:oMath | .//m:oMathPara" in patch
    assert "__image_blobs_from_relation_ids" in patch
    assert "LazyImage.merge(drawing_image, vml_image)" in patch
    assert "__paragraph_text_with_math_placeholders" in patch
    assert "__image_placeholder" in patch
    assert "__cell_text" in patch
    assert 'line = "" if line is None else str(line)' in patch
    assert "self.__clean(m) for m in metadata if self.__clean(m)" in patch
    assert "last_image_text" in patch
    assert "self.__image_placeholder(p, text)" in patch
    assert "cell_text = escape(self.__cell_text(c))" in patch
    assert '(chunks[idx].get(\'text\') or \'\').rstrip() + "\\n" + description' in patch
    assert "WORD_INLINE_IMAGE_PLACEHOLDER" in naive
    assert 'line = "" if line is None else str(line)' in naive
    assert ".//m:oMath | .//m:oMathPara" in naive
    assert "__local_name(element) != \"imagedata\"" in naive
    assert "last_image_text or self.WORD_INLINE_IMAGE_PLACEHOLDER" in naive
    assert "self.__image_placeholder(p, text)" in naive
    assert "cell_text = escape(self.__cell_text(c))" in naive
    assert "(chunks[idx].get('text') or '').rstrip() + \"\\n\" + description" in figure_parser
    assert "COPY third_party/ragflow/rag/app/naive.py" in dockerfile
    assert "COPY third_party/ragflow/deepdoc/parser/figure_parser.py" in dockerfile
    assert "COPY third_party/ragflow/rag/nlp/__init__.py" in dockerfile


def test_ragflow_vision_enhancement_is_opt_in_for_parsing() -> None:
    patch = read_text(ROOT / "patches" / "ragflow" / "0022-localmathrag-vision-enhancement-opt-in.patch")
    figure_parser = read_ragflow_source_or_patch("deepdoc/parser/figure_parser.py", patch)

    assert "LOCALMATHRAG_ENABLE_VISION_ENHANCEMENT" in patch
    assert "Visual model enhancement disabled; preserving image chunks without VLM." in patch
    assert "if not chunks or not idx_lst:" in patch
    assert "if not figures_data:" in patch
    assert "return tbls" in patch
    assert figure_parser.find("get_tenant_default_model_by_type") < figure_parser.find("_ensure_vision_runtime(callback)")
    assert "LOCALMATHRAG_ENABLE_VISION_ENHANCEMENT" in figure_parser
    assert "if not chunks or not idx_lst:" in figure_parser


def test_vlm_model_uses_its_selected_capabilities_without_syncing_defaults() -> None:
    service = read_text(ROOT / "services" / "object_service" / "main.py")
    model_service = read_ragflow_source_or_patches(
        "api/apps/services/models_api_service.py",
        [
            read_text(ROOT / "patches" / "ragflow" / name)
            for name in (
                "0011-localmathrag-runtime-model-switch.patch",
                "0028-localmathrag-shared-vlm-runtime.patch",
                "0029-localmathrag-vlm-chat-model-compatibility.patch",
                "0030-localmathrag-model-capability-validation.patch",
                "0032-localmathrag-runtime-switch-resource-error.patch",
                "0034-localmathrag-role-derived-runtime-scheduling.patch",
            )
        ],
    )
    un_add_model = read_ragflow_source_or_patch(
        "web/src/pages/user-setting/setting-model/components/un-add-model.tsx",
        read_text(ROOT / "patches" / "ragflow" / "0028-localmathrag-shared-vlm-runtime.patch"),
    )

    assert '"model_type": ["chat", "vision"]' in service
    assert 'if "chat" in model_type and any(item in model_type for item in {"vision", "ocr"}):' in service
    assert 'if target["model_type"] not in model_types:' in service
    assert 'updated = _replace_cmd_value(cmd, "--model", container_model_path)' in service
    assert "def _local_runtime_kind_from_endpoint" in model_service
    assert "_runtime_kind_for_model_type" in model_service
    assert "_runtime_roles_for_selection" in model_service
    assert "Model capability validation failed" in read_ragflow_source_or_patch(
        "api/apps/services/provider_api_service.py",
        read_text(ROOT / "patches" / "ragflow" / "0030-localmathrag-model-capability-validation.patch"),
    )
    assert "_display_model_name" in model_service
    assert "Older LocalMathRAG installations" not in model_service
    assert 'TenantService.update_by_id(tenant_id, {field_name: default_model})' in model_service
    assert "model.model_type?.some((type) => type === 'vision' || type === 'image2text')" in un_add_model
    assert "model_type: modelTypes.length ? modelTypes : ragflowTypeByGroup[primaryType]" in un_add_model


def test_vlm_startup_errors_return_before_the_ready_timeout() -> None:
    service = read_text(ROOT / "services" / "object_service" / "main.py")
    compose = read_text(ROOT / "docker" / "docker-compose.localmathrag.yml")
    assert 'if last_progress.get("phase") == "error":' in service
    assert 'last_status["runtime_startup_failed"] = True' in service
    assert "def _runtime_not_ready_reason" in service
    assert "Free memory on device" in service
    assert 'os.environ.get("LOCALMATHRAG_VLM_MAX_MODEL_LEN", "8192")' in service
    assert "--max-model-len" in compose
    assert "LOCALMATHRAG_VLM_MAX_MODEL_LEN:-8192" in compose


def test_shared_runtime_policy_is_derived_from_model_bindings() -> None:
    model_service = read_ragflow_source_or_patch(
        "api/apps/services/models_api_service.py",
        read_text(ROOT / "patches" / "ragflow" / "0034-localmathrag-role-derived-runtime-scheduling.patch"),
    )
    service = read_text(ROOT / "services" / "object_service" / "main.py")
    assert "MODEL_TYPE_TO_RUNTIME_ROLE" in model_service
    assert "_runtime_roles_for_selection" in model_service
    assert '"roles": roles' in model_service
    assert '"replacing_roles"' in model_service
    assert "isinstance(required, (int, float)) and required > 0" in model_service
    assert "RUNTIME_ROLE_PRIORITY" in service
    assert "def _target_runtime_roles" in service
    assert "def _runtime_preemption_order(target" in service
    assert "roles & replacing_roles" in service
    assert "high priority embedding request reserved resources" not in service


def test_model_capabilities_are_selected_then_verified_by_the_system() -> None:
    provider_service = read_ragflow_source_or_patch(
        "api/apps/services/provider_api_service.py",
        read_text(ROOT / "patches" / "ragflow" / "0030-localmathrag-model-capability-validation.patch"),
    )
    model_service = read_ragflow_source_or_patch(
        "api/apps/services/models_api_service.py",
        read_text(ROOT / "patches" / "ragflow" / "0030-localmathrag-model-capability-validation.patch"),
    )
    modal_utils = read_ragflow_source_or_patch(
        "web/src/pages/user-setting/setting-model/modal/provider-modal/field-config/utils.ts",
        read_text(ROOT / "patches" / "ragflow" / "0030-localmathrag-model-capability-validation.patch"),
    )
    local_fields = read_ragflow_source_or_patch(
        "web/src/pages/user-setting/setting-model/modal/provider-modal/field-config/local-llm-configs.ts",
        read_text(ROOT / "patches" / "ragflow" / "0030-localmathrag-model-capability-validation.patch"),
    )

    assert "requested_types" in provider_service
    assert "passed_types" in provider_service
    assert "Model capability validation failed" in provider_service
    assert "get_by_provider_id_and_instance_id_and_model_type_and_model_name" in provider_service
    assert "applyChatToImage2Text" not in modal_utils
    assert "values.vision" not in modal_utils
    assert "name: 'vision'" not in local_fields
    assert "name: 'is_tools'" not in local_fields
    assert "_display_model_name" in model_service
    assert "Existing installations may have stored a local model as /models/*.gguf." in model_service


def test_ragflow_docx_math_ocr_is_optional_for_word_images() -> None:
    patch = read_text(ROOT / "patches" / "ragflow" / "0023-localmathrag-math-ocr-docx-chunks.patch")
    context_patch = read_text(ROOT / "patches" / "ragflow" / "0036-localmathrag-docx-formula-context.patch")
    figure_parser = read_ragflow_source_or_patch("deepdoc/parser/figure_parser.py", patch + "\n" + context_patch)
    compose = read_text(ROOT / "docker" / "docker-compose.localmathrag.yml")

    assert "LOCALMATHRAG_ENABLE_MATH_OCR" in patch
    assert "LOCALMATHRAG_MATH_OCR_URL" in patch
    assert "LOCALMATHRAG_MATH_OCR_STATUS_URL" in patch
    assert "LOCALMATHRAG_MATH_OCR_TIMEOUT_SECONDS" in patch
    assert "LOCALMATHRAG_MATH_OCR_STATUS_TIMEOUT_SECONDS" in patch
    assert "_math_ocr_service_ready" in patch
    assert 'os.environ.get("LOCALMATHRAG_MATH_OCR_URL") or f"{OBJECT_SERVICE_URL}/v1/math-ocr"' in patch
    assert "def _math_ocr_enabled() -> bool:" in patch
    assert 'normalized in {"auto", ""}' in patch
    assert "urllib.request.urlopen(req, timeout=MATH_OCR_TIMEOUT_SECONDS)" in patch
    assert "_first_math_ocr_text" in patch
    assert "[Math OCR LaTeX:" in patch
    assert "_enhance_docx_chunks_with_math_ocr(chunks, idx_lst, callback)" in patch
    assert "open_image_for_processing(ck.get(\"image\"), allow_bytes=True)" in patch
    assert "LOCALMATHRAG_ENABLE_MATH_OCR" in figure_parser
    assert "LOCALMATHRAG_MATH_OCR_URL" in figure_parser
    assert "LOCALMATHRAG_MATH_OCR_STATUS_URL" in figure_parser
    assert "LOCALMATHRAG_MATH_OCR_TIMEOUT_SECONDS" in figure_parser
    assert "formula_indices = _enhance_docx_chunks_with_math_ocr(chunks, idx_lst, callback)" in figure_parser
    assert "idx_lst = _coalesce_docx_formula_chunks(chunks, formula_indices)" in figure_parser
    assert "LOCALMATHRAG_ENABLE_MATH_OCR: ${LOCALMATHRAG_ENABLE_MATH_OCR:-auto}" in compose
    assert "LOCALMATHRAG_MATH_OCR_URL: ${LOCALMATHRAG_MATH_OCR_URL:-}" in compose
    assert "LOCALMATHRAG_MATH_OCR_TIMEOUT_SECONDS: ${LOCALMATHRAG_MATH_OCR_TIMEOUT_SECONDS:-10}" in compose


def test_ragflow_docx_equation_editor_formulas_keep_paragraph_context() -> None:
    patch = read_text(ROOT / "patches" / "ragflow" / "0036-localmathrag-docx-formula-context.patch")
    naive = read_ragflow_source_or_patch("rag/app/naive.py", patch)
    nlp = read_ragflow_source_or_patch("rag/nlp/__init__.py", patch)
    figure_parser = read_ragflow_source_or_patches(
        "deepdoc/parser/figure_parser.py",
        [
            patch,
            read_text(ROOT / "patches" / "ragflow" / "0038-localmathrag-per-formula-ocr.patch"),
            read_text(ROOT / "patches" / "ragflow" / "0039-localmathrag-equation-editor-ocr-quality-gate.patch"),
            read_text(ROOT / "patches" / "ragflow" / "0040-localmathrag-equation-native-mtef.patch"),
        ],
    )

    assert "def __equation_editor_sources" in naive
    assert 'self.__local_name(element) != "OLEObject"' in naive
    assert 'normalized.startswith("equation.")' in naive
    assert '"_docx_paragraph_id": paragraph_id' in naive
    assert '"_docx_anchor_text": self.__clean(anchor_text)' in naive
    assert 'metadata["_docx_formula_sources"] = equation_sources' in naive
    assert "line.get(\"metadata\") or {}" in naive
    assert "section[3] if len(section) > 3" in nlp
    assert '"_docx_paragraph_ids"' in nlp
    assert 'merged[prev_text_ck]["text"] = f"{left}\\n{right}"' in nlp
    assert "for idx in (sorted(formula_indices) if formula_indices else idx_lst)" in figure_parser
    assert "def _coalesce_docx_formula_chunks" in figure_parser
    assert "from rag.utils.lazy_image import LazyImage," in figure_parser
    assert 'paragraph_id in chunk.get("_docx_paragraph_ids", [])' in figure_parser
    assert 'return f"$$\\n{latex}\\n$$"' in figure_parser
    assert "_merge_chunk_image(anchor, formula_chunk)" in figure_parser


def test_ragflow_equation_editor_wmf_previews_are_rasterized_locally() -> None:
    patch = read_text(ROOT / "patches" / "ragflow" / "0037-localmathrag-wmf-equation-rasterization.patch")
    lazy_image = read_ragflow_source_or_patch("rag/utils/lazy_image.py", patch)
    dockerfile = read_text(ROOT / "docker" / "Dockerfile.ragflow-local")
    dockerignore = read_text(ROOT / ".dockerignore")
    compose = read_text(ROOT / "docker" / "docker-compose.localmathrag.yml")
    dev_up = read_text(ROOT / "scripts" / "dev-up.ps1")

    assert 'WMF_PLACEABLE_MAGIC = b"\\xd7\\xcd\\xc6\\x9a"' in lazy_image
    assert 'shutil.which("wmf2gd")' in lazy_image
    assert '"--wmf-sys-fonts"' in lazy_image
    assert 'LOCALMATHRAG_MATH_FONT_DIR' in lazy_image
    assert 'LOCALMATHRAG_WMF_TIMEOUT_SECONDS' in lazy_image
    assert "subprocess.TimeoutExpired" in lazy_image
    assert "def _crop_formula_canvas" in lazy_image
    assert "ImageChops.difference" in lazy_image
    assert "def iter_pil_images" in lazy_image
    assert "libwmf-bin" in dockerfile
    assert "COPY third_party/ragflow/rag/utils/lazy_image.py" in dockerfile
    assert "!third_party/ragflow/rag/utils/lazy_image.py" in dockerignore
    assert "data/fonts/math:/usr/local/share/fonts/localmathrag:ro" in compose
    assert "LOCALMATHRAG_MATH_FONT_DIR: /usr/local/share/fonts/localmathrag" in compose
    assert "function Sync-MathFormulaFonts" in dev_up
    assert '"MTEXTRA.TTF"' in dev_up
    assert '"symbol.ttf"' in dev_up


def test_ragflow_formula_ocr_processes_equations_individually() -> None:
    patch = read_text(ROOT / "patches" / "ragflow" / "0038-localmathrag-per-formula-ocr.patch")
    figure_parser = read_ragflow_source_or_patch("deepdoc/parser/figure_parser.py", patch)

    assert "chunk_image.iter_pil_images()" in figure_parser
    assert "image_cache = {}" in figure_parser
    assert 'hashlib.sha256(encoded.encode("ascii")).digest()' in figure_parser
    assert 'ck["_docx_formula_latex_items"] = latex_items' in figure_parser
    assert 'Local Math OCR: {processed_images}/{total_images} formula images' in figure_parser
    assert 'chunk.get("_docx_formula_latex_items") or []' in figure_parser
    assert 'f"$$\\n{latex}\\n$$" if latex else' in figure_parser


def test_rapid_formula_ocr_does_not_publish_unreliable_equation_editor_latex() -> None:
    patch = read_text(ROOT / "patches" / "ragflow" / "0039-localmathrag-equation-editor-ocr-quality-gate.patch")
    figure_parser = read_ragflow_source_or_patch("deepdoc/parser/figure_parser.py", patch)

    assert "_MATH_OCR_STATUS = {}" in figure_parser
    assert "def _math_ocr_backend_name" in figure_parser
    assert "def _rapid_equation_editor_ocr_allowed" in figure_parser
    assert 'LOCALMATHRAG_ALLOW_RAPID_EQUATION_EDITOR_OCR' in figure_parser
    assert '_math_ocr_backend_name() == "rapid_latex_ocr"' in figure_parser
    assert 'str(source.get("prog_id") or "").lower().startswith("equation.")' in figure_parser
    assert "preserving formula images and context" in figure_parser


def test_equation_native_mtef_is_preferred_over_formula_ocr() -> None:
    patch = read_text(ROOT / "patches" / "ragflow" / "0040-localmathrag-equation-native-mtef.patch")
    naive = read_ragflow_source_or_patch("rag/app/naive.py", patch)
    figure_parser = read_ragflow_source_or_patch("deepdoc/parser/figure_parser.py", patch)
    dockerfile = read_text(ROOT / "docker" / "Dockerfile.ragflow-local")
    parser_root = ROOT / "services" / "mtef_parser"

    assert (parser_root / "LICENSE").is_file()
    assert (parser_root / "localmathrag-mtef-linux-amd64").is_file()
    assert (parser_root / "localmathrag-mtef-linux-amd64").stat().st_size > 1_000_000
    parser_readme = read_text(parser_root / "README.md")
    assert "16d536beaf41e0d88b0963a59e55c36de54e4fba" in parser_readme
    assert "75add02ceb32a9dbfaf46f1a9b80f75ca7cf63fe58f22972ff780881cd3135dc" in parser_readme
    assert "COPY services/mtef_parser/localmathrag-mtef-linux-amd64" in dockerfile
    assert "LOCALMATHRAG_MTEF_PARSER=/usr/local/bin/localmathrag-mtef" in dockerfile
    assert "self._equation_native_cache = {}" in naive
    assert "def __equation_native_latex" in naive
    assert 'input=ole_blob' in naive
    assert 'source["native_latex"] = native_latex' in naive
    assert "and not native_complete" in naive
    assert "native_relationship_ids = set()" in naive
    assert 'pieces.append(f"\\n$$\\n{native_latex}\\n$$\\n")' in naive
    assert 'metadata["_docx_native_formulas_inline"] = all(' in naive
    assert 'chunks[idx]["_docx_formula_latex_items"] = latex_items' in figure_parser
    assert "if idx not in native_indices" in figure_parser
    assert "Equation Native formulas locally (MTEF)" in figure_parser
    assert 'native_formulas_inline = bool(formula_chunk.get("_docx_native_formulas_inline"))' in figure_parser


def test_formula_only_word_layout_tables_keep_text_context() -> None:
    patch = read_text(ROOT / "patches" / "ragflow" / "0041-localmathrag-formula-table-context.patch")
    naive = read_ragflow_source_or_patch("rag/app/naive.py", patch)

    assert "def _coalesce_docx_formula_tables" in naive
    assert 'chunk.get("ck_type") == "table"' in naive
    assert 'chunk.get("image") is None' in naive
    assert 're.sub(r"\\$\\$.*?\\$\\$", "", text, flags=re.DOTALL)' in naive
    assert 'unescape(re.sub(r"<[^>]+>", "", outside_formula))' in naive
    assert 'anchor = next((item for item in reversed(merged) if item.get("ck_type") == "text"), None)' in naive
    assert "chunks = _coalesce_docx_formula_tables(chunks)" in naive


def test_chunk_formula_images_use_bounded_contain_previews() -> None:
    patch = read_text(ROOT / "patches" / "ragflow" / "0053-localmathrag-bounded-chunk-image-preview.patch")
    chunk_card = read_ragflow_source_or_patch(
        "web/src/pages/chunk/parsed-result/add-knowledge/components/knowledge-chunk/components/chunk-card/index.tsx",
        patch,
    )
    dataflow_card = read_ragflow_source_or_patch(
        "web/src/pages/dataflow-result/components/chunk-card/index.tsx",
        patch,
    )

    for component in (chunk_card, dataflow_card):
        assert "flex max-h-80 max-w-56 shrink-0 items-center justify-center overflow-hidden" in component
        assert 'className="!h-auto !w-auto max-h-80 max-w-56 object-contain"' in component
        assert "max-h-[70vh] max-w-[70vw] object-contain" in component
        assert "const narrowThumbnailMaxHeight = 192" in component
        assert "const narrowThumbnailAspectRatio = 0.5" in component
        assert "aspectRatio < narrowThumbnailAspectRatio" in component
        assert "maxHeight: imageMaxHeight" in component
        assert "maxWidth: thumbnailMaxWidth" in component
        assert "handleThumbnailLoad(currentTarget)" in component
        assert "size-28" not in component
    assert "!w-28 object-contain" not in chunk_card
    assert "max-w-[72vw] overflow-hidden bg-white p-2" in chunk_card


def test_parallel_document_toc_is_bounded_and_optional() -> None:
    patch = read_text(ROOT / "patches" / "ragflow" / "0042-localmathrag-bounded-toc-ingestion.patch")
    generator = read_ragflow_source_or_patch("rag/prompts/generator.py", patch)
    executor = read_ragflow_source_or_patch("rag/svr/task_executor.py", patch)
    handler = read_ragflow_source_or_patch("rag/svr/task_executor_refactor/task_handler.py", patch)
    limiter = read_ragflow_source_or_patch("rag/svr/task_executor_limiter.py", patch)
    compose = read_text(ROOT / "docker" / "docker-compose.localmathrag.yml")
    dockerfile = read_text(ROOT / "docker" / "Dockerfile.ragflow-local")

    assert "LOCALMATHRAG_TOC_MAX_OUTPUT_TOKENS" in generator
    assert '"max_tokens": TOC_MAX_OUTPUT_TOKENS' in generator
    assert "TOC_MAX_CONCURRENT_REQUESTS" in generator
    assert "TOC_REQUEST_TIMEOUT_SECONDS" in generator
    assert "request_limiter = asyncio.Semaphore" in generator
    assert "toc_limiter = LoopLocalSemaphore(MAX_CONCURRENT_TOC)" in limiter
    assert "async def build_TOC_guarded" in executor
    assert "async def _build_toc_guarded" in handler
    assert "TOC generation skipped because another document is using the local LLM." in executor
    assert "TOC generation skipped because another document is using the local LLM." in handler
    assert 'os.environ.get("LOCALMATHRAG_TOC_TIMEOUT_SECONDS", "60")' in executor
    assert 'os.environ.get("LOCALMATHRAG_TOC_TIMEOUT_SECONDS", "60")' in handler
    assert "document indexing is complete" in executor
    assert "document indexing is complete" in handler
    assert compose.count("LOCALMATHRAG_MAX_CONCURRENT_TOC: ${LOCALMATHRAG_MAX_CONCURRENT_TOC:-1}") == 2
    assert compose.count("LOCALMATHRAG_TOC_TIMEOUT_SECONDS: ${LOCALMATHRAG_TOC_TIMEOUT_SECONDS:-60}") == 2
    assert "COPY third_party/ragflow/rag/prompts/generator.py" in dockerfile
    assert "COPY third_party/ragflow/rag/svr/task_executor_limiter.py" in dockerfile
    assert "COPY third_party/ragflow/rag/svr/task_executor_refactor/task_handler.py" in dockerfile


def test_expired_executor_pending_tasks_are_requeued() -> None:
    patch = read_text(ROOT / "patches" / "ragflow" / "0043-localmathrag-requeue-expired-executor-tasks.patch")
    executor = read_ragflow_source_or_patches(
        "rag/svr/task_executor.py",
        [
            patch,
            read_text(ROOT / "patches" / "ragflow" / "0051-localmathrag-migrate-legacy-parse-queue.patch"),
            read_text(ROOT / "patches" / "ragflow" / "0052-localmathrag-fix-orphan-consumer-scan.patch"),
        ],
    )

    assert "def requeue_expired_worker_tasks" in executor
    assert "xpending_range(" in executor
    assert 'consumer_name = item.get("consumer")' in executor
    assert "if consumer_name == worker_name" in executor
    assert "target_queue = queue_name_for_task(message, worker_task_type)" in executor
    assert "pipeline.xadd(target_queue, payload)" in executor
    assert "pipeline.xack(queue_name, SVR_CONSUMER_GROUP_NAME, message_id)" in executor
    assert "pipeline.xdel(queue_name, message_id)" in executor
    assert "def requeue_orphaned_pending_tasks" in executor
    assert "xinfo_consumers(queue_name, SVR_CONSUMER_GROUP_NAME)" in executor
    pending_check = executor.index('if int(consumer.get("pending", 0) or 0) <= 0')
    inspected_add = executor.index("inspected.add(worker_name)", pending_check)
    assert pending_check < inspected_add
    assert "worker_name in active_workers" in executor
    assert "WORKER_HEARTBEAT_TIMEOUT * 1000" in executor
    assert "requeue_orphaned_pending_tasks(set(task_executors))" in executor
    assert "requeued = requeue_expired_worker_tasks(worker_name)" in executor
    assert "Requeued %d pending tasks from expired worker %s" in executor


def test_toc_runs_in_a_durable_background_queue() -> None:
    patch = read_text(ROOT / "patches" / "ragflow" / "0044-localmathrag-durable-toc-queue.patch")
    queue_module = read_ragflow_source_or_patch("rag/svr/localmathrag_toc_queue.py", patch)
    executor = read_ragflow_source_or_patch("rag/svr/task_executor.py", patch)
    handler = read_ragflow_source_or_patch("rag/svr/task_executor_refactor/task_handler.py", patch)
    compose = read_text(ROOT / "docker" / "docker-compose.localmathrag.yml")
    dockerfile = read_text(ROOT / "docker" / "Dockerfile.ragflow-local")

    assert 'TOC_QUEUE_NAME = "localmathrag.toc"' in queue_module
    assert 'TOC_CONSUMER_GROUP = "localmathrag_toc_workers"' in queue_module
    assert "def enqueue_toc_task" in queue_module
    assert "async def toc_queue_worker" in queue_module
    assert "xpending_range(TOC_QUEUE_NAME" in queue_module
    assert "pipeline.xadd(TOC_QUEUE_NAME" in queue_module
    assert "pipeline.xdel(TOC_QUEUE_NAME, message_id)" in queue_module
    assert "requested_ids = set(job.get(\"chunk_ids\") or [])" in queue_module
    assert "settings.retriever.chunk_list" in queue_module
    assert "TOC_MAX_RETRIES" in queue_module
    assert "TOC_QUEUE_TIMEOUT_SECONDS" in queue_module
    assert "TOC_RECOVERY_INTERVAL_SECONDS" in queue_module
    assert "time.monotonic() - last_recovery" in queue_module
    assert "if acknowledge:" in queue_module
    assert "REDIS_CONN.REDIS.xdel(TOC_QUEUE_NAME, redis_message.get_msg_id())" in queue_module
    assert "enqueue_toc_task(" in handler
    assert "TOC queued for background generation." in handler
    assert "self._build_toc_guarded(ctx, chunks" not in handler
    assert "toc_queue_worker(f\"{CONSUMER_NAME}_toc\")" in executor
    assert "toc_worker_task.cancel()" in executor
    assert "build_TOC_guarded(task, chunks" not in executor
    assert compose.count("LOCALMATHRAG_TOC_QUEUE_MAX_RETRIES: ${LOCALMATHRAG_TOC_QUEUE_MAX_RETRIES:-2}") == 2
    assert compose.count("LOCALMATHRAG_TOC_QUEUE_TIMEOUT_SECONDS: ${LOCALMATHRAG_TOC_QUEUE_TIMEOUT_SECONDS:-300}") == 2
    assert "COPY third_party/ragflow/rag/svr/localmathrag_toc_queue.py" in dockerfile


def test_toc_prefers_local_document_structure_without_llm() -> None:
    patch = read_text(ROOT / "patches" / "ragflow" / "0045-localmathrag-local-structure-toc.patch")
    quality_patches = [
        patch,
        read_text(ROOT / "patches" / "ragflow" / "0046-localmathrag-rendered-toc-resolution.patch"),
        read_text(ROOT / "patches" / "ragflow" / "0047-localmathrag-toc-body-mapping.patch"),
        read_text(ROOT / "patches" / "ragflow" / "0048-localmathrag-drop-partial-toc-fragments.patch"),
    ]
    builder = read_ragflow_source_or_patches("rag/svr/localmathrag_toc_builder.py", quality_patches)
    queue_module = read_ragflow_source_or_patches(
        "rag/svr/localmathrag_toc_queue.py",
        [read_text(ROOT / "patches" / "ragflow" / "0044-localmathrag-durable-toc-queue.patch"), patch],
    )
    naive = read_ragflow_source_or_patches(
        "rag/app/naive.py",
        [read_text(ROOT / "patches" / "ragflow" / "0001-localmathrag-offline-ui.patch"), patch],
    )
    generator = read_ragflow_source_or_patches(
        "rag/prompts/generator.py",
        [read_text(ROOT / "patches" / "ragflow" / "0042-localmathrag-bounded-toc-ingestion.patch"), patch],
    )
    compose = read_text(ROOT / "docker" / "docker-compose.localmathrag.yml")
    dockerfile = read_text(ROOT / "docker" / "Dockerfile.ragflow-local")
    dockerignore = read_text(ROOT / ".dockerignore")

    assert "def build_local_toc" in builder
    assert "_NUMBERED_PATTERNS" in builder
    assert "_outline_candidates" in builder
    assert "_caption_candidates" in builder
    assert "_resolve_rendered_toc" in builder
    assert "_strip_trailing_page_number" in builder
    assert "_drop_unresolved_toc_fragments" in builder
    assert "_FORMULA_BLOCK_RE" in builder
    assert "from rag.svr.localmathrag_toc_builder import build_local_toc" in queue_module
    assert 'TOC_USE_LLM_LEVELS = _env_enabled("LOCALMATHRAG_TOC_USE_LLM_LEVELS", False)' in queue_module
    assert 'TOC_LLM_FALLBACK = _env_enabled("LOCALMATHRAG_TOC_LLM_FALLBACK", False)' in queue_module
    assert "if not TOC_LLM_FALLBACK:" in queue_module
    assert "Built local TOC" in queue_module
    assert "section_end = max(current + 1, min(following, len(docs)))" in queue_module
    assert "def __heading_level" in naive
    assert 'res[0]["__outline__"] = docx_parser.outlines' in naive
    assert 'LOCALMATHRAG_TOC_MAX_OUTPUT_TOKENS", "256"' in generator
    assert compose.count("LOCALMATHRAG_TOC_USE_LLM_LEVELS: ${LOCALMATHRAG_TOC_USE_LLM_LEVELS:-false}") == 2
    assert compose.count("LOCALMATHRAG_TOC_LLM_FALLBACK: ${LOCALMATHRAG_TOC_LLM_FALLBACK:-false}") == 2
    assert compose.count("LOCALMATHRAG_TOC_LOCAL_MAX_ENTRIES: ${LOCALMATHRAG_TOC_LOCAL_MAX_ENTRIES:-256}") == 2
    assert "COPY third_party/ragflow/rag/svr/localmathrag_toc_builder.py" in dockerfile
    assert "!third_party/ragflow/rag/svr/localmathrag_toc_builder.py" in dockerignore

    source_path = ROOT / "third_party" / "ragflow" / "rag" / "svr" / "localmathrag_toc_builder.py"
    if source_path.exists():
        spec = importlib.util.spec_from_file_location("localmathrag_toc_builder_contract", source_path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        chunks = [
            "\u7cfb\u7edf\u529f\u80fd\u63cf\u8ff0\n\u7b97\u6cd5\u8bbe\u8ba1\n\u8fd9\u662f\u6b63\u6587\u3002",
            "1\uff09\u8054\u90a6\u6ee4\u6ce2\u5668\u5de5\u4f5c\u6d41\u7a0b\n\u2460\u4fe1\u606f\u5206\u914d\n\u8be5\u8fc7\u7a0b\u4e3b\u8981\u4efb\u52a1\u662f\u5206\u914d\u4fe1\u606f\u3002",
            "$$\\hat X_k=P_k$$\nb)\u6a21\u578b\u6761\u4ef6\u6ee4\u6ce2\nc)\u6a21\u578b\u6982\u7387\u66f4\u65b0",
            "\u5bf9\u4e8e\u5404\u4e2a\u6a21\u578b\uff0c\u8ba1\u7b97\u6a21\u578b\u6982\u7387\nd)\u4f30\u8ba1\u878d\u5408",
        ]
        toc = module.build_local_toc(chunks)
        titles = [item["title"] for item in toc]
        assert "\u7b97\u6cd5\u8bbe\u8ba1" in titles
        assert "1\uff09\u8054\u90a6\u6ee4\u6ce2\u5668\u5de5\u4f5c\u6d41\u7a0b" in titles
        assert "b)\u6a21\u578b\u6761\u4ef6\u6ee4\u6ce2" in titles
        assert "c)\u6a21\u578b\u6982\u7387\u66f4\u65b0" in titles
        assert "d)\u4f30\u8ba1\u878d\u5408" in titles
        assert all("hat X" not in title for title in titles)
        outlined = module.build_local_toc(
            chunks,
            [
                {"title": "\u7cfb\u7edf\u529f\u80fd\u63cf\u8ff0", "depth": 0},
                {"title": "\u7b97\u6cd5\u8bbe\u8ba1", "depth": 1},
            ],
        )
        assert outlined[0] == {"level": "1", "title": "\u7cfb\u7edf\u529f\u80fd\u63cf\u8ff0", "chunk_id": "0"}
        assert outlined[1] == {"level": "2", "title": "\u7b97\u6cd5\u8bbe\u8ba1", "chunk_id": "0"}

        rendered_toc = module.build_local_toc(
            [
                "\u76ee\u5f55\n1 \u76ee\u7684\u548c\u8303\u56f4 10\n1.1 \u76ee\u7684 10\n1.2 \u8303\u56f4 11\n"
                "2 \u7cfb\u7edf\u8bbe\u8ba1 12\n2.1 \u529f\u80fd\u63cf\u8ff0 12\n2.2 \u63a5\u53e3\u8bbe\u8ba1 14\n"
                "3 \u9a8c\u8bc1 20\n3.1 \u6d4b\u8bd5\u65b9\u6cd5 21",
                "\uff081\uff09\u5982\u679c\u8bbe\u5907\u6ee1\u8db3\u6761\u4ef6\uff0c\u5219\u7ee7\u7eed\u6267\u884c\u8be5\u6d41\u7a0b\u3002",
                "1 \u76ee\u7684\u548c\u8303\u56f4\n1.1 \u76ee\u7684\n\u6b63\u6587\u3002\n1.2 \u8303\u56f4",
                "2 \u7cfb\u7edf\u8bbe\u8ba1\n2.1 \u529f\u80fd\u63cf\u8ff0\n2.2 \u63a5\u53e3\u8bbe\u8ba1",
                "3 \u9a8c\u8bc1\n3.1 \u6d4b\u8bd5\u65b9\u6cd5",
            ]
        )
        assert len(rendered_toc) == 8
        assert rendered_toc[0] == {"level": "1", "title": "1\u76ee\u7684\u548c\u8303\u56f4", "chunk_id": "2"}
        assert rendered_toc[-1]["chunk_id"] == "4"
        assert all(not re.search(r"\s\d+$", item["title"]) for item in rendered_toc)
        assert all(item["chunk_id"] != "0" for item in rendered_toc)
        assert all("\u5982\u679c\u8bbe\u5907" not in item["title"] for item in rendered_toc)

        partial_toc = module.build_local_toc(
            [
                "\u591a\u6e90\u878d\u5408\u7b97\u6cd5\u8bbe\u8ba1\u65b9\u6848\n\u76ee \u5f55\n6.5.8\t\u5b89\u5168\u4f4d\u7f6e\u9632\u62a4\t16\n\u76ee\u7684\n\u6b63\u6587\u3002",
                "\u8303\u56f4\n\u7b97\u6cd5\u8bbe\u8ba1",
            ]
        )
        assert all("6.5.8" not in item["title"] for item in partial_toc)


def test_auxiliary_indexes_are_queued_below_document_parsing_and_release_on_cancel() -> None:
    patch = read_text(ROOT / "patches" / "ragflow" / "0049-localmathrag-prioritized-auxiliary-index-queue.patch")
    cancellation_patch = read_text(ROOT / "patches" / "ragflow" / "0050-localmathrag-auxiliary-lease-cancellation.patch")
    migration_patch = read_text(ROOT / "patches" / "ragflow" / "0051-localmathrag-migrate-legacy-parse-queue.patch")
    recovery_patch = read_text(ROOT / "patches" / "ragflow" / "0052-localmathrag-fix-orphan-consumer-scan.patch")
    deferred_retry_patch = read_text(ROOT / "patches" / "ragflow" / "0054-localmathrag-deferred-auxiliary-retry.patch")
    stale_cleanup_patch = read_text(ROOT / "patches" / "ragflow" / "0055-localmathrag-clear-stale-auxiliary-tasks.patch")
    restart_cleanup_patch = read_text(ROOT / "patches" / "ragflow" / "0057-localmathrag-restart-clears-prior-index-task.patch")
    stale_lock_patch = read_text(ROOT / "patches" / "ragflow" / "0059-localmathrag-release-stale-graphrag-locks.patch")
    scoped_cleanup_patch = read_text(ROOT / "patches" / "ragflow" / "0062-localmathrag-scope-auxiliary-cleanup-by-task-type.patch")
    patches = [
        patch,
        cancellation_patch,
        migration_patch,
        recovery_patch,
        deferred_retry_patch,
        stale_cleanup_patch,
        restart_cleanup_patch,
        stale_lock_patch,
        scoped_cleanup_patch,
    ]
    priority_queue = read_ragflow_source_or_patches("rag/svr/localmathrag_priority_queue.py", patches)
    task_service = read_ragflow_source_or_patches("api/db/services/task_service.py", [patch, deferred_retry_patch])
    document_service = read_ragflow_source_or_patch("api/db/services/document_service.py", patch)
    executor = read_ragflow_source_or_patches("rag/svr/task_executor.py", patches)
    dataset_service = read_ragflow_source_or_patches(
        "api/apps/services/dataset_api_service.py",
        [stale_cleanup_patch, restart_cleanup_patch, stale_lock_patch, scoped_cleanup_patch],
    )
    task_api = read_ragflow_source_or_patch("api/apps/restful_apis/task_api.py", stale_cleanup_patch)
    compose = read_text(ROOT / "docker" / "docker-compose.localmathrag.yml")
    dockerfile = read_text(ROOT / "docker" / "Dockerfile.ragflow-local")
    dockerignore = read_text(ROOT / ".dockerignore")

    assert "DOCUMENT_PARSE_PRIORITY = 1" in priority_queue
    assert "AUXILIARY_INDEX_PRIORITY = 0" in priority_queue
    assert 'frozenset({"graphrag", "raptor", "mindmap"})' in priority_queue
    assert "def document_parse_queue_busy" in priority_queue
    assert "def queue_name_for_task" in priority_queue
    assert 'group.get("lag", 0)' in priority_queue
    assert 'group.get("pending", 0)' in priority_queue
    assert "def acquire_auxiliary_index_lease" in priority_queue
    assert "def refresh_auxiliary_index_lease" in priority_queue
    assert "def release_auxiliary_index_lease" in priority_queue
    assert "def remove_stale_auxiliary_index_messages" in priority_queue
    assert "task_types: set[str] | None = None" in priority_queue
    assert "if task_types and message_task_type not in task_types:" in priority_queue
    assert "is_pending = normalized_message_id in pending_ids" in priority_queue
    assert "is_queued = _stream_id_tuple(normalized_message_id) > _stream_id_tuple(last_delivered_id)" in priority_queue
    assert 'pipeline.set(f"{message_task_id}-cancel", "x")' in priority_queue
    assert "pipeline.xack(queue_name, SVR_CONSUMER_GROUP_NAME, message_id)" in priority_queue
    assert "pipeline.xdel(queue_name, message_id)" in priority_queue
    assert "priority = DOCUMENT_PARSE_PRIORITY" in task_service
    assert "increment_retry=True" in task_service
    assert "if not increment_retry:" in task_service
    assert "priority = AUXILIARY_INDEX_PRIORITY" in document_service
    assert '"priority": priority' in document_service
    assert "document_parse_queue_busy(TASK_TYPE)" in executor
    assert "acquire_auxiliary_index_lease(task[\"id\"])" in executor
    assert "increment_retry = task_type not in AUXILIARY_INDEX_TASK_TYPES" in executor
    assert "increment_retry=increment_retry" in executor
    assert 'admitted_task = TaskService.get_task(msg["id"], msg.get("doc_ids", []))' in executor
    assert "Deferred %s task %s because %s" in executor
    assert "release_auxiliary_index_lease(auxiliary_lease)" in executor
    assert "lease_heartbeat.cancel()" in executor
    assert "await asyncio.gather(lease_heartbeat, return_exceptions=True)" in executor
    if not ragflow_source_available("rag/svr/task_executor.py"):
        task_type_insert = deferred_retry_patch.index('+    task_type = msg.get("task_type", "")')
        canceled_context = deferred_retry_patch.index("     if task:")
        auxiliary_context = deferred_retry_patch.index("     if task_type in AUXILIARY_INDEX_TASK_TYPES:")
        admitted_context = deferred_retry_patch.index("+        admitted_task = TaskService.get_task")
        assert task_type_insert < canceled_context < auxiliary_context < admitted_context

        heartbeat_cancel = cancellation_patch.index("+            lease_heartbeat.cancel()")
        heartbeat_wait = cancellation_patch.index(
            "+            await asyncio.gather(lease_heartbeat, return_exceptions=True)"
        )
        lease_pop = cancellation_patch.index("         auxiliary_lease = task.pop")
        lease_release_context = cancellation_patch.index(
            "         if auxiliary_lease and not release_auxiliary_index_lease"
        )
        assert heartbeat_cancel < heartbeat_wait < lease_pop < lease_release_context
        assert "+            redis_msg.ack()" in deferred_retry_patch
    else:
        canceled_check = executor.index("if not task or canceled:")
        lease_acquire = executor.index('acquire_auxiliary_index_lease(task["id"])')
        admitted_attempt = executor.index(
            'admitted_task = TaskService.get_task(msg["id"], msg.get("doc_ids", []))'
        )
        lease_release = executor.index("release_auxiliary_index_lease(auxiliary_lease)")
        message_ack = executor.index("redis_msg.ack()", lease_release)
        assert canceled_check < lease_acquire < admitted_attempt < lease_release < message_ack
    assert "stale_task_ids = remove_stale_auxiliary_index_messages(" in dataset_service
    assert "doc_ids=set(document_ids)" in dataset_service
    assert "task_types={task_type}" in dataset_service
    assert "TaskService.delete_by_id(stale_task_id)" in dataset_service
    assert 'REDIS_CONN.set(f"{existing_task_id}-cancel", "x")' in dataset_service
    assert "TaskService.delete_by_id(existing_task_id)" in dataset_service
    assert "A {display_name} Task is already running" not in dataset_service
    assert "def release_stale_auxiliary_index_locks" in priority_queue
    assert "REDIS_CONN.delete_if_equal(" in priority_queue
    assert 'f"batch_merge:{task_id}"' in priority_queue
    assert "release_stale_auxiliary_index_locks(existing_task_id, dataset_id" in dataset_service
    assert "release_stale_auxiliary_index_locks(task_id, dataset_id" in dataset_service
    assert "remove_stale_auxiliary_index_messages(task_ids={task_id})" in dataset_service
    assert "remove_stale_auxiliary_index_messages(task_ids={task_id})" in task_api
    assert compose.count("LOCALMATHRAG_AUX_INDEX_LOCK_TTL_SECONDS: ${LOCALMATHRAG_AUX_INDEX_LOCK_TTL_SECONDS:-86400}") == 2
    assert compose.count("MAX_CONCURRENT_CHATS: ${LOCALMATHRAG_MAX_CONCURRENT_CHATS:-1}") == 2
    assert compose.count("MAX_CONCURRENT_PROCESS_AND_EXTRACT_CHUNK: ${LOCALMATHRAG_MAX_CONCURRENT_PROCESS_AND_EXTRACT_CHUNK:-1}") == 2
    assert "COPY third_party/ragflow/api/db/services/document_service.py" in dockerfile
    assert "COPY third_party/ragflow/rag/svr/localmathrag_priority_queue.py" in dockerfile
    assert "!third_party/ragflow/api/db/services/document_service.py" in dockerignore
    assert "!third_party/ragflow/rag/svr/localmathrag_priority_queue.py" in dockerignore
    assert "COPY third_party/ragflow/api/apps/restful_apis/task_api.py" in dockerfile
    assert "!third_party/ragflow/api/apps/restful_apis/task_api.py" in dockerignore


def test_graphrag_adaptive_execution_contract() -> None:
    patch = read_text(ROOT / "patches" / "ragflow" / "0056-localmathrag-adaptive-graphrag-execution.patch")
    merge_watchdog_patch = read_text(
        ROOT / "patches" / "ragflow" / "0058-localmathrag-adaptive-graphrag-merge-watchdog.patch"
    )
    postprocessing_watchdog_patch = read_text(
        ROOT / "patches" / "ragflow" / "0060-localmathrag-adaptive-graphrag-postprocessing-watchdog.patch"
    )
    adaptive = read_ragflow_source_or_patch("rag/graphrag/adaptive.py", patch)
    index = read_ragflow_source_or_patches(
        "rag/graphrag/general/index.py",
        [patch, merge_watchdog_patch, postprocessing_watchdog_patch],
    )
    extractor = read_ragflow_source_or_patch("rag/graphrag/general/extractor.py", patch)
    light_extractor = read_ragflow_source_or_patch("rag/graphrag/light/graph_extractor.py", patch)
    task_executor = read_ragflow_source_or_patch("rag/svr/task_executor.py", patch)
    compose = read_text(ROOT / "docker" / "docker-compose.localmathrag.yml")
    dev_up = read_text(ROOT / "scripts" / "dev-up.ps1")
    resource_plan = read_text(ROOT / "scripts" / "localmathrag-resource-plan.ps1")
    dockerfile = read_text(ROOT / "docker" / "Dockerfile.ragflow-local")
    dockerignore = read_text(ROOT / ".dockerignore")

    assert "resolve_execution_plan" in adaptive
    assert "math.ceil(chat_slots / chunk_slots)" in adaptive
    assert "AdaptiveActivityWatchdog" in adaptive
    assert "LOCALMATHRAG_MODEL_PARALLEL_SLOTS" in adaptive
    assert "activity_watchdog_factory" in index
    assert "load_chunk_checkpoints" in index
    assert "save_chunk_checkpoint" in index
    assert "clear_chunk_checkpoints" in index
    assert '"subgraph_checkpoint"' in index
    assert "DEFAULT_GRAPHRAG_MERGE_TIMEOUT_SECONDS = 0" in index
    assert "activity_watchdog_setter=set_active_merge_watchdog" in index
    assert "@timeout(60 * 3)\nasync def merge_subgraph" not in index
    assert "DEFAULT_GRAPHRAG_RESOLUTION_TIMEOUT_SECONDS = 0" in index
    assert "DEFAULT_GRAPHRAG_COMMUNITY_TIMEOUT_SECONDS = 0" in index
    assert "activity_watchdog_setter=set_active_resolution_watchdog" in index
    assert "activity_watchdog_setter=set_active_community_watchdog" in index
    assert "@timeout(60 * 30, 1)" not in index
    assert "len(chunks) * build_subgraph_timeout_per_chunk_seconds" not in index
    assert "checkpoint_results" in extractor
    assert "missing_indices" in extractor
    assert "glean_count = 0" in light_extractor
    assert "Knowledge Graph partially completed" in task_executor
    assert '"chunk_checkpoint": True' in task_executor
    assert compose.count("LOCALMATHRAG_GRAPHRAG_MAX_PARALLEL_DOCS: ${LOCALMATHRAG_GRAPHRAG_MAX_PARALLEL_DOCS:-auto}") == 2
    assert compose.count("LOCALMATHRAG_GRAPHRAG_CHUNK_CHECKPOINT: ${LOCALMATHRAG_GRAPHRAG_CHUNK_CHECKPOINT:-true}") == 2
    assert "Resolve-LlamaParallelSlots" in resource_plan
    assert "Resolve-GraphRagAdaptivePlan" in resource_plan
    assert "model-context-capacity" in dev_up
    assert "no-local-model-safe-default" in dev_up
    assert "Adaptive GraphRAG plan" in dev_up
    assert "COPY third_party/ragflow/rag/graphrag/adaptive.py" in dockerfile
    assert "!third_party/ragflow/rag/graphrag/adaptive.py" in dockerignore


def test_raptor_resumable_execution_contract() -> None:
    patch = read_text(ROOT / "patches" / "ragflow" / "0061-localmathrag-resumable-raptor.patch")
    raptor = read_ragflow_source_or_patch("rag/raptor.py", patch)
    service = read_ragflow_source_or_patch(
        "rag/svr/task_executor_refactor/raptor_service.py", patch
    )
    handler = read_ragflow_source_or_patch(
        "rag/svr/task_executor_refactor/task_handler.py", patch
    )
    chunk_service = read_ragflow_source_or_patch(
        "rag/svr/task_executor_refactor/chunk_service.py", patch
    )
    task_executor = read_ragflow_source_or_patch("rag/svr/task_executor.py", patch)
    generate = read_ragflow_source_or_patch(
        "web/src/pages/dataset/dataset/generate-button/generate.tsx", patch
    )
    hook = read_ragflow_source_or_patch(
        "web/src/pages/dataset/dataset/generate-button/hook.ts", patch
    )
    dockerfile = read_text(ROOT / "docker" / "Dockerfile.ragflow-local")
    dockerignore = read_text(ROOT / ".dockerignore")

    assert "if n_clusters >= len(reduced_embeddings):" in raptor
    assert "RAPTOR forced a reducing layer" in raptor
    assert "AdaptiveActivityWatchdog" in raptor
    assert "LOCALMATHRAG_RAPTOR_MIN_INACTIVITY_SECONDS" in raptor
    assert "LOCALMATHRAG_RAPTOR_MAX_INACTIVITY_SECONDS" in raptor
    assert "@timeout(60 * 20)" not in raptor
    assert "@timeout(3600)" not in service
    assert "checkpoint_cb" in service
    assert "checkpoint persisted" in service
    assert "report_progress=False" in handler
    assert 'task_type not in {"graphrag", "raptor", "mindmap"}' in handler
    assert "task_type not in AUXILIARY_INDEX_TASK_TYPES" in task_executor
    assert "wipe: false" in hook
    assert "knowledgeDetails.completed" in generate
    assert "status === generateStatus.completed" in generate
    assert "report_progress: bool = True" in chunk_service
    assert "COPY third_party/ragflow/rag/raptor.py" in dockerfile
    assert "COPY third_party/ragflow/rag/svr/task_executor_refactor/raptor_service.py" in dockerfile
    assert "COPY third_party/ragflow/rag/svr/task_executor_refactor/chunk_service.py" in dockerfile
    assert "!third_party/ragflow/rag/raptor.py" in dockerignore
    assert "!third_party/ragflow/rag/svr/task_executor_refactor/raptor_service.py" in dockerignore
    assert "!third_party/ragflow/rag/svr/task_executor_refactor/chunk_service.py" in dockerignore


def test_ragflow_math_ocr_recommended_install_ui_contract() -> None:
    patch = read_text(ROOT / "patches" / "ragflow" / "0024-localmathrag-math-ocr-recommended-install-ui.patch")
    dropdown_patch = read_text(ROOT / "patches" / "ragflow" / "0032-localmathrag-math-ocr-dropdown-ui.patch")
    auto_register_patch = read_text(
        ROOT / "patches" / "ragflow" / "0035-localmathrag-math-ocr-auto-register.patch"
    )
    un_add_model = read_ragflow_source_or_patch(
        "web/src/pages/user-setting/setting-model/components/un-add-model.tsx",
        patch + "\n" + dropdown_patch + "\n" + auto_register_patch,
    )
    model_settings = read_ragflow_source_or_patch(
        "web/src/pages/user-setting/setting-model/index.tsx", auto_register_patch
    )

    assert "math_ocr: 'Math OCR'" in patch
    assert "| 'math_ocr'" in patch
    assert "'math_ocr'," in patch
    assert "Formula OCR" in patch
    assert "公式 OCR" in patch
    assert "mathOcrComplete" in patch
    assert "h-8 shrink-0" in patch
    assert "math_ocr: 'Math OCR'" in un_add_model
    assert "Formula OCR" in un_add_model
    assert "mathOcrComplete" in un_add_model
    assert "handleAddModel(LLMFactory.OpenAiAPICompatible, makeDefaults(model));" in un_add_model
    assert "math-ocr-config-card" not in un_add_model
    assert "/v1/math-ocr/config" not in un_add_model
    assert "MathOcrStatus" not in un_add_model
    assert "ensureMathOcrAvailable" in auto_register_patch
    assert "math-ocr-auto-registered" in auto_register_patch
    assert "ensureLocalModel(makeDefaults(mathOcrModel))" in auto_register_patch
    assert "ensureLocalModel" in un_add_model
    assert "ensureLocalModel" in model_settings


def test_ragflow_math_ocr_default_setting_ui_contract() -> None:
    patch = read_text(ROOT / "patches" / "ragflow" / "0032-localmathrag-math-ocr-dropdown-ui.patch")
    system_setting = read_ragflow_source_or_patch(
        "web/src/pages/user-setting/setting-model/components/system-setting.tsx", patch
    )
    model_tree_select = read_ragflow_source_or_patch(
        "web/src/components/model-tree-select.tsx", patch
    )
    llm_constants = read_ragflow_source_or_patch("web/src/constants/llm.ts", patch)

    assert "Formula OCR" in patch
    assert "ocr_id" in patch
    assert "ocr_id" in system_setting
    if ragflow_source_available("web/src/pages/user-setting/setting-model/components/system-setting.tsx"):
        assert "ModelTreeSelect" in system_setting
    assert "math-ocr-default-setting" not in system_setting
    assert "/v1/math-ocr/config" not in system_setting
    assert "ocr_id: ['ocr']" in model_tree_select
    assert "ocr: 'ocr_id'" in llm_constants
    assert "ocr_id: 'ocr'" in llm_constants


def test_object_service_imports() -> None:
    service = ROOT / "services" / "object_service" / "main.py"
    text = read_text(service)
    compose = read_text(ROOT / "docker" / "docker-compose.localmathrag.yml")
    dockerfile = read_text(ROOT / "services" / "object_service" / "Dockerfile")
    runner = read_text(ROOT / "services" / "object_service" / "localmathrag_math_ocr.py")
    assert "FastAPI" in text
    assert "/v1/objects/normalize" in text
    assert "/v1/models/local" in text
    assert "/v1/models/recommended" in text
    assert "/v1/models/download" in text
    assert "local-formula-ocr-adapter" in text
    assert "RapidLaTeX OCR (ONNX CPU)" in text
    assert "math_ocr_config" in text
    assert "_install_math_ocr_config" in text
    assert "MathOcrConfigRequest" in text
    assert "_persist_manual_math_ocr_config" in text
    assert "/v1/math-ocr/config" in text
    assert "math_ocr_command_template" in text
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
    assert "/v1/math-ocr" in text
    assert "/v1/math-ocr/status" in text
    assert "EmbeddingRequest" in text
    assert "RerankRequest" in text
    assert "MathOcrRequest" in text
    assert "RuntimeEnsureRequest" in text
    assert "RuntimeSwitchModelRequest" in text
    assert "LOCALMATHRAG_MATH_OCR_COMMAND" in text
    assert "_decode_math_ocr_image" in text
    assert "_math_ocr_command_parts" in text
    assert "_persist_math_ocr_runtime_config" in text
    assert "_math_ocr_runtime_config" in text
    assert "_math_ocr_status_payload" in text
    assert "_is_math_ocr_model" in text
    assert "runtime_config=runtime_config" in text
    assert "subprocess.run(" in text
    assert "NamedTemporaryFile" in text
    assert "_switch_runtime_model" in text
    assert "LOCALMATHRAG_MATH_OCR_COMMAND: ${LOCALMATHRAG_MATH_OCR_COMMAND:-}" in compose
    assert "LOCALMATHRAG_MATH_OCR_TIMEOUT_SECONDS: ${LOCALMATHRAG_MATH_OCR_TIMEOUT_SECONDS:-10}" in compose
    assert "COPY services/object_service/localmathrag_math_ocr.py" in dockerfile
    assert "rapid_latex_ocr" in runner
    assert "MODEL_FILES" in runner
    assert "model_files_status" in runner
    assert "backend_status" in runner
    assert "from rapid_latex_ocr import LaTeXOCR" in runner
    assert "rapid_latex_ocr==0.0.9" in dockerfile
    assert "opencv-python-headless" in dockerfile
    assert '"math_ocr_files"' in text
    assert "_download_url_to_file(str(url), target, job_id)" in text
    assert 'cache_key = hashlib.sha256(b"rapid_latex_ocr:0.0.9\\0" + image_bytes).hexdigest()' in text
    assert 'cached["runtime"]["cache_hit"] = True' in text
    assert "_find_local_model_for_runtime" in text
    assert "requested_keys & _model_switch_keys(payload)" in text
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
    assert "_runtime_model_min_memory_gb" in text
    assert '"min_available_memory_gb": 5.0' not in text
    assert '"min_available_memory_gb": 10.0' not in text
    assert '"model_min_available_memory_gb": None' in text
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
    assert "_runtime_embedding_model_name" in text
    assert "_generation_body_for_chat_runtime" in text
    assert 'normalized["model"] = _runtime_chat_model_name()' in text
    assert 'payload["model"] = _runtime_embedding_model_name()' in text
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
    assert "active_quality_requests" in text
    assert "RUNTIME_ACTIVE_QUALITY_REQUESTS" in text
    assert '_active_quality_runtime_request_count("embedding") > 0' in text
    assert 'candidate["kind"] == "embedding"' in text
    assert '"action": "quality_embedding_direct_load"' in text
    assert '"action": "quality_embedding_direct_load_failed"' in text
    assert '"action": "quality_embedding_exclusive_retry"' in text
    assert '"required document embedding reclaimed chat resources after direct load failed"' in text
    assert "enforce_stall_timeout: bool = True" in text
    assert "enforce_stall_timeout=False" in text
    assert '"fallback": False' in text
    assert "EMBEDDING_DOCUMENT_REQUEST_MIN_INPUTS" in text
    assert "EMBEDDING_DOCUMENT_REQUEST_MIN_TOKENS" in text
    assert "EMBEDDING_DOCUMENT_READY_TIMEOUT_SECONDS" in text
    assert "EMBEDDING_DOCUMENT_REQUEST_TIMEOUT_SECONDS" in text
    assert "EMBEDDING_DOCUMENT_DIRECT_LOAD_TIMEOUT_SECONDS" in text
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
    assert "high priority embedding request reserved resources" not in text
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
    assert "RUNTIME_ROLE_PRIORITY" in text
    assert "roles & replacing_roles" in text
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
    assert 'PackageReference Include="Microsoft.Web.WebView2"' in project_text
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
    assert "LocalMathWebForm" in program_text
    assert "WebView2" in program_text
    assert "CoreWebView2Environment.CreateAsync" in program_text
    assert 'Path.Combine(FindRoot(), "data", "launcher", "webview2-profile")' in program_text
    assert "webViewFallbackActive" in program_text
    assert "OpenBrowserFallback" in program_text
    assert "CloseBrowserProcess()" in program_text
    assert 'Path.Combine(launcherDataDir, "browser-profile")' in program_text
    assert "TryFocusExistingBrowserWindow(state)" in program_text
    assert "TryFindBrowserAppWindow" in program_text
    assert "TryApplyBrowserWindowIcon(browserProcess" in program_text
    assert "ApplyBrowserWindowIcon(handle)" in program_text
    assert "WmSetIcon" in program_text
    assert "IconSmall" in program_text
    assert "IconBig" in program_text
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
    assert '$VlmCandidates = @("Qwen3-VL-4B-Instruct", "Qwen3-VL-8B-Instruct")' in dev_up_text
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
    assert 'throw "Launcher publish failed with exit code $LASTEXITCODE."' in build_launcher_text
    assert 'Test-Path -LiteralPath $LauncherExe' in build_launcher_text
    assert 'Directory -Filter "__pycache__"' in build_launcher_text
    assert '-Include "*.pyc", "*.pyo"' in build_launcher_text
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
    assert "!third_party/ragflow/rag/nlp/__init__.py" in dockerignore
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
    assert "COPY third_party/ragflow/rag/nlp/__init__.py" in dockerfile_text
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
    assert "LOCALMATHRAG_EMBEDDING_DOCUMENT_REQUEST_TIMEOUT_SECONDS: ${LOCALMATHRAG_EMBEDDING_DOCUMENT_REQUEST_TIMEOUT_SECONDS:-300}" in compose
    assert "LOCALMATHRAG_EMBEDDING_DOCUMENT_DIRECT_LOAD_TIMEOUT_SECONDS: ${LOCALMATHRAG_EMBEDDING_DOCUMENT_DIRECT_LOAD_TIMEOUT_SECONDS:-120}" in compose
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
    assert "LOCALMATHRAG_VISION_MIN_AVAILABLE_MEMORY_GB: ${LOCALMATHRAG_VISION_MIN_AVAILABLE_MEMORY_GB:-0}" in compose
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
    assert "LOCALMATHRAG_EMBEDDING_MODEL: ${LOCALMATHRAG_EMBEDDING_MODEL:-bge-m3}" in compose
    assert compose.count("${LOCALMATHRAG_TEI_EMBEDDING_MAX_CLIENT_BATCH_SIZE:-16}") == 2
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
    test_ragflow_dataset_upload_ignores_temporary_files_contract()
    test_ragflow_search_summary_has_output_budget()
    test_ragflow_rerank_fallback_contract()
    test_ragflow_sse_requests_abort_on_unmount()
    test_ragflow_rerank_disable_request_overrides_saved_config()
    test_ragflow_search_model_switches_have_backend_guards()
    test_ragflow_runtime_warning_surface_contract()
    test_ragflow_search_summary_runtime_retry_contract()
    test_ragflow_runtime_model_switch_contract()
    test_local_model_configuration_uses_provider_facing_name()
    test_model_labels_include_provider_without_changing_model_identity()
    test_model_switch_has_visible_progress_feedback()
    test_model_switch_runtime_error_is_sanitized()
    test_model_capabilities_are_selected_then_verified_by_the_system()
    test_ragflow_docx_formula_image_chunks_contract()
    test_ragflow_vision_enhancement_is_opt_in_for_parsing()
    test_vlm_model_uses_its_selected_capabilities_without_syncing_defaults()
    test_vlm_startup_errors_return_before_the_ready_timeout()
    test_shared_runtime_policy_is_derived_from_model_bindings()
    test_ragflow_docx_math_ocr_is_optional_for_word_images()
    test_ragflow_docx_equation_editor_formulas_keep_paragraph_context()
    test_ragflow_equation_editor_wmf_previews_are_rasterized_locally()
    test_ragflow_formula_ocr_processes_equations_individually()
    test_rapid_formula_ocr_does_not_publish_unreliable_equation_editor_latex()
    test_equation_native_mtef_is_preferred_over_formula_ocr()
    test_formula_only_word_layout_tables_keep_text_context()
    test_parallel_document_toc_is_bounded_and_optional()
    test_expired_executor_pending_tasks_are_requeued()
    test_toc_runs_in_a_durable_background_queue()
    test_toc_prefers_local_document_structure_without_llm()
    test_auxiliary_indexes_are_queued_below_document_parsing_and_release_on_cancel()
    test_graphrag_adaptive_execution_contract()
    test_raptor_resumable_execution_contract()
    test_ragflow_math_ocr_recommended_install_ui_contract()
    test_ragflow_math_ocr_default_setting_ui_contract()
    test_object_service_imports()
    test_windows_launcher_exists()
    test_ragflow_patch_workflow_exists()
    test_docker_compose_mounts_ragflow_backend_overrides()
    print("contract checks passed")


if __name__ == "__main__":
    main()

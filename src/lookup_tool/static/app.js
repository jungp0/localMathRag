const state = {
  kbs: [],
  kbId: null,
  projects: [],
  projectId: null,
  documents: [],
  questions: [],
  selectedQuestionId: null,
  lastResult: null,
  modelSettings: null,
  modelStatus: null,
};

const $ = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: options.body instanceof FormData ? {} : { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || `${response.status} ${response.statusText}`);
  }
  return data;
}

function showToast(message) {
  const toast = $("toast");
  toast.textContent = message;
  toast.classList.add("visible");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.remove("visible"), 2400);
}

function activeKb() {
  return state.kbs.find((kb) => kb.id === state.kbId) || null;
}

async function init() {
  bindEvents();
  renderResult({ task: "answer", status: "not_found", items: [], evidence: {} });
  await refreshHealth();
  await refreshKbs();
  setInterval(refreshHealth, 10000);
}

function bindEvents() {
  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => switchTab(button.dataset.tab));
  });
  $("refresh-kbs").addEventListener("click", () => withBusy("refresh-kbs", refreshKbs));
  $("kb-select").addEventListener("change", async (event) => {
    state.kbId = event.target.value;
    state.projectId = null;
    await refreshWorkspace();
  });
  $("create-kb").addEventListener("click", () => withBusy("create-kb", createKb));
  $("migrate-kb").addEventListener("click", () => withBusy("migrate-kb", migrateKb));
  $("create-project").addEventListener("click", () => withBusy("create-project", createProject));
  $("ask-button").addEventListener("click", () => withBusy("ask-button", ask));
  $("copy-result").addEventListener("click", copyResult);
  $("upload-button").addEventListener("click", () => withBusy("upload-button", uploadFiles));
  $("ingest-path-button").addEventListener("click", () => withBusy("ingest-path-button", ingestPath));
  $("refresh-docs").addEventListener("click", () => withBusy("refresh-docs", refreshDocuments));
  $("refresh-history").addEventListener("click", () => withBusy("refresh-history", refreshHistory));
  $("delete-history").addEventListener("click", () => withBusy("delete-history", deleteSelectedHistory));
  $("refresh-model").addEventListener("click", () => withBusy("refresh-model", refreshModelStatus));
  $("save-model").addEventListener("click", () => withBusy("save-model", saveModelSettings));
  $("download-model").addEventListener("click", () => withBusy("download-model", downloadModel));
}

async function withBusy(buttonId, fn) {
  const button = $(buttonId);
  if (!button || button.disabled) return;
  button.disabled = true;
  button.classList.add("busy");
  try {
    await fn();
  } catch (error) {
    showToast(error.message || String(error));
  } finally {
    button.disabled = false;
    button.classList.remove("busy");
  }
}

function switchTab(name) {
  document.querySelectorAll(".tab").forEach((button) => button.classList.toggle("active", button.dataset.tab === name));
  document.querySelectorAll(".tab-page").forEach((page) => page.classList.toggle("active", page.id === `tab-${name}`));
}

async function refreshHealth() {
  try {
    await api("/api/health");
    $("server-status").textContent = "本地服务已连接";
  } catch {
    $("server-status").textContent = "服务未连接";
  }
}

async function refreshKbs() {
  const data = await api("/api/kbs");
  state.kbs = data.items || [];
  if (!state.kbId && state.kbs.length) {
    state.kbId = state.kbs[0].id;
  }
  renderKbs();
  if (state.kbId) {
    await refreshWorkspace();
  }
  await refreshModelSettings();
}

function renderKbs() {
  const select = $("kb-select");
  select.innerHTML = state.kbs.map((kb) => `<option value="${escapeHtml(kb.id)}">${escapeHtml(kb.name)}</option>`).join("");
  if (state.kbId) {
    select.value = state.kbId;
  }
  renderKbDetail();
  renderWorkspaceStatus();
}

async function refreshWorkspace() {
  renderKbs();
  await Promise.all([refreshProjects(), refreshDocuments(), refreshHistory()]);
}

async function createKb() {
  const name = $("new-kb-name").value.trim();
  const rootPath = $("new-kb-path").value.trim();
  if (!name || !rootPath) {
    showToast("需要名称和路径");
    return;
  }
  const data = await api("/api/kbs", {
    method: "POST",
    body: JSON.stringify({ name, root_path: rootPath }),
  });
  state.kbId = data.item.id;
  $("new-kb-name").value = "";
  $("new-kb-path").value = "";
  await refreshKbs();
  showToast("知识库已创建");
}

async function migrateKb() {
  if (!state.kbId) return;
  const rootPath = $("migrate-path").value.trim();
  if (!rootPath) {
    showToast("需要迁移目标路径");
    return;
  }
  await api(`/api/kbs/${state.kbId}/migrate`, {
    method: "POST",
    body: JSON.stringify({ root_path: rootPath }),
  });
  $("migrate-path").value = "";
  await refreshKbs();
  showToast("知识库已迁移");
}

async function refreshProjects() {
  if (!state.kbId) return;
  const data = await api(`/api/kbs/${state.kbId}/projects`);
  state.projects = data.items || [];
  if (!state.projectId && state.projects.length) {
    state.projectId = state.projects[0].id;
  }
  renderProjects();
}

function renderProjects() {
  const list = $("project-list");
  list.innerHTML = state.projects
    .map((project) => {
      const active = project.id === state.projectId ? " active" : "";
      return `<div class="list-item${active}" data-project-id="${escapeHtml(project.id)}">
        <strong>${escapeHtml(project.name)}</strong>
        <div class="row-actions">
          <button class="secondary-button" data-action="select-project" data-id="${escapeHtml(project.id)}">选择</button>
          <button class="danger-button" data-action="delete-project" data-id="${escapeHtml(project.id)}">删除</button>
        </div>
      </div>`;
    })
    .join("");
  list.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", async () => {
      const id = button.dataset.id;
      if (button.dataset.action === "select-project") {
        state.projectId = id;
        renderProjects();
        await refreshHistory();
      } else {
        await api(`/api/kbs/${state.kbId}/projects/${id}`, { method: "DELETE" });
        if (state.projectId === id) state.projectId = null;
        await refreshProjects();
        await refreshHistory();
      }
    });
  });
  const project = state.projects.find((item) => item.id === state.projectId);
  $("active-project-label").textContent = project ? project.name : "";
}

async function createProject() {
  if (!state.kbId) return;
  const name = $("project-name").value.trim();
  if (!name) {
    showToast("需要项目名称");
    return;
  }
  const data = await api(`/api/kbs/${state.kbId}/projects`, {
    method: "POST",
    body: JSON.stringify({ name }),
  });
  state.projectId = data.item.id;
  $("project-name").value = "";
  await refreshProjects();
  showToast("项目已创建");
}

async function ask() {
  if (!state.kbId) return;
  const query = $("query").value.trim();
  if (!query) {
    showToast("请输入问题");
    return;
  }
  const payload = {
    query,
    task: $("task").value,
    top_k: Number($("top-k").value || 12),
    project_id: state.projectId,
  };
  const data = await api(`/api/kbs/${state.kbId}/ask`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
  state.lastResult = data.result;
  renderResult(data.result);
  await refreshHistory();
}

function renderResult(result) {
  $("result-json").textContent = JSON.stringify(result || {}, null, 2);
  const items = (result && result.items) || [];
  if (!items.length) {
    $("result-summary").innerHTML = `<div class="empty-state">暂无结构化结果</div>`;
    return;
  }
  $("result-summary").innerHTML = items
    .slice(0, 6)
    .map((item) => `<div class="result-item">
      <strong>${escapeHtml(item.type || "item")}</strong>
      <div>${escapeHtml(item.predicate || item.latex || item.caption || item.name || item.id || "")}</div>
      <div class="evidence-line">${(item.evidence || [])
        .map((evidence) => `<span class="evidence-chip">${escapeHtml(evidence)}</span>`)
        .join("")}</div>
    </div>`)
    .join("");
}

async function copyResult() {
  await navigator.clipboard.writeText(JSON.stringify(state.lastResult || {}, null, 2));
  showToast("已复制");
}

async function uploadFiles() {
  if (!state.kbId) return;
  const input = $("file-upload");
  if (!input.files.length) {
    showToast("请选择文件");
    return;
  }
  const form = new FormData();
  Array.from(input.files).forEach((file) => form.append("files", file, file.name));
  const data = await api(`/api/kbs/${state.kbId}/upload`, { method: "POST", body: form });
  input.value = "";
  await refreshDocuments();
  showToast(`入库 ${data.documents?.length || 0} 个文档`);
}

async function ingestPath() {
  if (!state.kbId) return;
  const path = $("ingest-path").value.trim();
  if (!path) {
    showToast("请输入路径");
    return;
  }
  const data = await api(`/api/kbs/${state.kbId}/ingest`, {
    method: "POST",
    body: JSON.stringify({ paths: [path], recursive: true }),
  });
  $("ingest-path").value = "";
  await refreshDocuments();
  showToast(`入库 ${data.documents?.length || 0} 个文档`);
}

async function refreshDocuments() {
  if (!state.kbId) return;
  const data = await api(`/api/kbs/${state.kbId}/documents`);
  state.documents = data.items || [];
  renderDocuments();
  renderKbDetail();
  renderWorkspaceStatus();
}

function renderDocuments() {
  const list = $("document-list");
  if (!state.documents.length) {
    list.innerHTML = `<div class="empty-state">还没有入库文档</div>`;
    return;
  }
  list.innerHTML = state.documents
    .map((doc) => `<div class="doc-item">
      <strong>${escapeHtml(shortPath(doc.path))}</strong>
      <div class="meta">blocks ${doc.block_count || 0} · formulas ${doc.equation_count || 0} · tables ${doc.table_block_count || 0} · visuals ${doc.visual_count || 0}</div>
      <div class="meta">${escapeHtml(doc.doc_id)}</div>
      <div class="row-actions">
        <button class="danger-button" data-doc-id="${escapeHtml(doc.doc_id)}">删除索引</button>
      </div>
    </div>`)
    .join("");
  list.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", async () => {
      await api(`/api/kbs/${state.kbId}/documents/${button.dataset.docId}`, { method: "DELETE" });
      await refreshDocuments();
      showToast("文档索引已删除");
    });
  });
}

async function refreshHistory() {
  if (!state.kbId) return;
  const suffix = state.projectId ? `?project_id=${encodeURIComponent(state.projectId)}` : "";
  const data = await api(`/api/kbs/${state.kbId}/questions${suffix}`);
  state.questions = data.items || [];
  renderHistory();
}

function renderHistory() {
  const list = $("history-list");
  if (!state.questions.length) {
    list.innerHTML = `<div class="empty-state">暂无历史记录</div>`;
    $("history-detail").innerHTML = "";
    $("history-json").textContent = "{}";
    state.selectedQuestionId = null;
    return;
  }
  list.innerHTML = state.questions
    .map((record) => `<button class="list-item ${record.id === state.selectedQuestionId ? "active" : ""}" data-id="${escapeHtml(record.id)}">
      <strong>${escapeHtml(record.title || record.query)}</strong>
      <span class="meta">${escapeHtml(record.task)} · ${formatTime(record.created_at)}</span>
    </button>`)
    .join("");
  list.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => selectHistory(button.dataset.id));
  });
  if (!state.selectedQuestionId) {
    selectHistory(state.questions[0].id);
  }
}

function selectHistory(id) {
  state.selectedQuestionId = id;
  const record = state.questions.find((item) => item.id === id);
  if (!record) return;
  $("history-detail").innerHTML = `<strong>${escapeHtml(record.query)}</strong>
    <div class="meta">${escapeHtml(record.task)} · ${formatTime(record.created_at)}</div>`;
  $("history-json").textContent = JSON.stringify(record.result || {}, null, 2);
  renderHistory();
}

async function deleteSelectedHistory() {
  if (!state.kbId || !state.selectedQuestionId) return;
  await api(`/api/kbs/${state.kbId}/questions/${state.selectedQuestionId}`, { method: "DELETE" });
  state.selectedQuestionId = null;
  await refreshHistory();
  showToast("历史记录已删除");
}

function renderKbDetail() {
  const kb = activeKb();
  const detail = $("kb-detail");
  if (!kb || !detail) return;
  detail.innerHTML = [
    ["名称", kb.name],
    ["根目录", kb.root_path],
    ["索引库", kb.db_path],
    ["上传目录", kb.upload_dir],
    ["产物目录", kb.artifact_dir],
    ["文档数", String(state.documents.length)],
  ]
    .map(([key, value]) => `<div class="muted">${escapeHtml(key)}</div><div>${escapeHtml(value || "")}</div>`)
    .join("");
  renderWorkspaceStatus();
}

async function refreshModelSettings() {
  const data = await api("/api/model/settings");
  state.modelSettings = data.item || {};
  renderModelSettings();
  await refreshModelStatus();
}

function renderModelSettings() {
  const settings = state.modelSettings || {};
  $("model-enabled").checked = Boolean(settings.enabled);
  $("model-provider").value = settings.provider || "openai_compatible";
  $("model-base-url").value = settings.base_url || "";
  $("model-id").value = settings.model || "";
  $("model-temperature").value = settings.temperature ?? 0;
  $("model-local-path").value = settings.local_model_path || "";
}

async function saveModelSettings() {
  const payload = {
    enabled: $("model-enabled").checked,
    provider: $("model-provider").value,
    base_url: $("model-base-url").value.trim(),
    model: $("model-id").value.trim(),
    temperature: Number($("model-temperature").value || 0),
    local_model_path: $("model-local-path").value.trim(),
  };
  const data = await api("/api/model/settings", {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
  state.modelSettings = data.item || {};
  renderModelSettings();
  await refreshModelStatus();
  showToast("模型设置已保存");
}

async function refreshModelStatus() {
  const data = await api("/api/model/status");
  state.modelStatus = data.item || {};
  renderModelStatus();
}

function renderModelStatus() {
  const item = state.modelStatus || {};
  const localSize = item.local_model_size ? `${(item.local_model_size / 1024 / 1024 / 1024).toFixed(2)} GB` : "0";
  $("model-status").innerHTML = [
    ["本地 GGUF", item.local_model_exists ? "已就位" : "未找到"],
    ["模型文件", item.local_model_path || ""],
    ["文件大小", localSize],
    ["Ollama", item.ollama_available ? "可用" : "未安装或不在 PATH"],
    ["llama.cpp", item.llama_available ? "可用" : "未安装或不在 PATH"],
    ["端点", item.endpoint_ok ? "在线" : "不可用"],
    ["Ollama 模型", (item.available_ollama_models || []).join(", ") || "无"],
  ]
    .map(([key, value]) => `<div><strong>${escapeHtml(key)}:</strong> ${escapeHtml(value)}</div>`)
    .join("");
  renderWorkspaceStatus();
}

async function downloadModel() {
  showToast("开始下载推荐模型，文件较大，请等待");
  const data = await api("/api/model/download", { method: "POST", body: JSON.stringify({}) });
  await refreshModelStatus();
  showToast(`模型已下载: ${shortPath(data.item?.path || "")}`);
}

function shortPath(path) {
  const parts = String(path || "").split(/[\\/]/);
  return parts.slice(-2).join("/");
}

function formatTime(seconds) {
  if (!seconds) return "";
  return new Date(seconds * 1000).toLocaleString();
}

function renderWorkspaceStatus() {
  const kb = activeKb();
  const kbChip = $("active-kb-chip");
  const docChip = $("doc-count-chip");
  const modelChip = $("model-chip");
  if (kbChip) {
    kbChip.textContent = kb ? `知识库 ${kb.name}` : "未选择知识库";
    kbChip.className = `status-chip${kb ? " good" : " warn"}`;
  }
  if (docChip) {
    const count = state.documents.length;
    docChip.textContent = `文档 ${count}`;
    docChip.className = `status-chip${count ? " good" : ""}`;
  }
  if (modelChip) {
    const enabled = Boolean(state.modelSettings && state.modelSettings.enabled);
    const endpointOk = Boolean(state.modelStatus && state.modelStatus.endpoint_ok);
    const localReady = Boolean(state.modelStatus && state.modelStatus.local_model_exists);
    if (enabled && endpointOk) {
      modelChip.textContent = "模型在线";
      modelChip.className = "status-chip good";
    } else if (enabled && localReady) {
      modelChip.textContent = "模型文件就绪";
      modelChip.className = "status-chip warn";
    } else if (localReady) {
      modelChip.textContent = "模型未启用";
      modelChip.className = "status-chip";
    } else {
      modelChip.textContent = "模型未下载";
      modelChip.className = "status-chip warn";
    }
  }
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

init().catch((error) => showToast(error.message));

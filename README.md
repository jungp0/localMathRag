# Lookup Tool

本项目是一个本地部署的 Formula-aware / Math-aware 技术文档总结与检索工具。目标是给外部 agent 提供稳定、可校验、可引用的机器格式输出，用于代码编写、需求追踪、公式实现和测试生成。

当前实现是可运行 MVP：

- 支持本地解析 `PDF`、`DOCX`、`XLSX`、`CSV`、`TXT/MD`。
- 将文档拆成 `text/table/equation/cell` block，并保存页码、sheet、cell range 等来源。
- 自动识别 Markdown/文本中的 LaTeX 公式、常见数学等式、Excel 公式。
- 使用 SQLite FTS + math-aware rerank 进行检索。
- 输出 agent compact JSON，不偏向人类可读报告。
- 提供 CLI 和无第三方 Web 框架的本地 HTTP API。
- 预留 Docling、PaddleOCR、Ollama/OpenAI-compatible LLM、Qdrant 等增强入口。

## 快速开始

```powershell
# 使用当前 Codex bundled Python 示例
$env:PYTHONPATH="D:\LookupTool\src"
& "C:\Users\kylej\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m lookup_tool ingest examples
& "C:\Users\kylej\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m lookup_tool extract "Kalman filter covariance prediction formula" --task formula_extract
```

如果已经安装到环境：

```powershell
lookup-tool ingest .\docs
lookup-tool search "process noise covariance Q"
lookup-tool extract "Kalman 协方差预测公式" --task formula_extract
lookup-tool serve --port 8765
```

## HTTP API

启动：

```powershell
lookup-tool serve --host 127.0.0.1 --port 8765
```

打开：

```text
http://127.0.0.1:8765
```

WebApp 支持：

- 上传文档并入库。
- 创建知识库，设置知识库根目录。
- 迁移知识库位置。
- 查看并删除文档索引。
- 创建项目/文件夹。
- 在项目下提问。
- 查看、切换、删除历史提问记录。
- 查看 agent compact JSON 结果和 evidence。
- 在设置页切换模型端点、模型 id、本地 GGUF 路径，并查看模型状态。
- 可选连接本机/内网 RAGFlow，支持 `local_only`、`hybrid`、`ragflow_only`。

WebApp 使用 `/api/...` 接口：

- `GET /api/kbs`
- `POST /api/kbs`
- `GET /api/model/settings`
- `PATCH /api/model/settings`
- `GET /api/model/status`
- `POST /api/model/download`
- `GET /api/ragflow/settings`
- `PATCH /api/ragflow/settings`
- `GET /api/ragflow/status`
- `POST /api/kbs/{kb_id}/upload`
- `POST /api/kbs/{kb_id}/ingest`
- `POST /api/kbs/{kb_id}/ask`
- `POST /api/kbs/{kb_id}/ragflow/sync`
- `GET /api/kbs/{kb_id}/documents`
- `GET /api/kbs/{kb_id}/projects`
- `GET /api/kbs/{kb_id}/questions`

示例：

```json
POST /api/kbs/{kb_id}/ask
{
  "query": "Kalman covariance update equation",
  "task": "formula_extract",
  "top_k": 8
}
```

返回格式默认是 agent compact。这里的 `lookup.result.v1` 是结果结构版本，不是旧 HTTP 接口：

```json
{
  "schema": "lookup.result.v1",
  "task": "formula_extract",
  "status": "ok",
  "items": [
    {
      "id": "eq.kalman_filter.covariance_prediction.001",
      "type": "equation",
      "domain": ["kalman_filter", "covariance_prediction"],
      "latex": "P_{k|k-1}=F_kP_{k-1|k-1}F_k^T+Q_k",
      "normalized_latex": "P_{k|k-1}=F_kP_{k-1|k-1}F_k^T+Q_k",
      "vars": {
        "P_{k|k-1}": {"role": "predicted_covariance"},
        "F_k": {"role": "state_transition_matrix"},
        "Q_k": {"role": "process_noise_covariance"}
      },
      "assumptions": ["linear_discrete_system"],
      "evidence": ["ev.block_id"],
      "confidence": 0.93
    }
  ],
  "evidence": {
    "ev.block_id": {
      "doc_id": "doc.x",
      "path": "docs/kalman.pdf",
      "page": 12,
      "section": "3.2",
      "block_id": "block_id"
    }
  },
  "warnings": []
}
```

## 模型设置

默认推荐模型是 `Qwen3-8B-GGUF` 的 `Q4_K_M` 量化文件，适合 32GB RAM
和 4060 级别显卡做本地 RAG 生成补充。WebApp 设置页保存的是可替换配置：

- `provider`: 当前使用 OpenAI-compatible 调用方式。
- `base_url`: 例如 `http://127.0.0.1:8080/v1`。
- `model`: 运行器暴露的模型 id。
- `local_model_path`: 本地 GGUF 文件位置。

本项目会先返回可引用的检索/抽取 JSON。启用模型后，会把 evidence
交给本地 OpenAI-compatible 运行器生成一个 `generated_answer` item。
如果只需要给外部 agent 传结构化证据，可以保持模型关闭。

## RAGFlow 离线整合

RAGFlow 只作为可选的离线检索后端接入。本项目不会嵌入 RAGFlow 的联网能力，
也不会调用公共网络、云搜索或在线模型。默认安全策略：

- `enabled=false`，默认关闭。
- `mode=local_only`，默认只使用本地 SQLite + math/table/visual 抽取。
- `base_url=http://127.0.0.1:9380`。
- 默认只允许 `localhost`、`127.0.0.1`、私有网段、`.local` 或单标签内网主机名。
- 公网域名始终阻止。

三种模式：

- `local_only`: 完全使用当前工具的本地解析、索引和抽取。
- `hybrid`: 当前工具先做公式、表格和图对象精准抽取，再追加 RAGFlow chunk evidence。
- `ragflow_only`: 跳过本地召回，只把 RAGFlow 返回的 chunk 归一化成 `lookup.result.v1`。

RAGFlow 版本间 HTTP API 可能有差异，因此设置页保留了可配置路径：

- `status_path`: 默认 `/api/v1/datasets`。
- `retrieval_path`: 默认 `/api/v1/retrieval`。
- `upload_path_template`: 默认 `/api/v1/datasets/{dataset_id}/documents`。
- `upload_field`: 默认 `file`。

RAGFlow 返回内容会被归一化为：

```json
{
  "id": "ragflow.chunk.001",
  "type": "ragflow_chunk",
  "text": "retrieved chunk text",
  "score": 0.91,
  "source": "spec.pdf",
  "evidence": ["ev.ragflow.chunk-1"]
}
```

这样外部 agent 仍然只需要读取统一的 `lookup.result.v1`。

## Formula-aware 设计

公式不会只作为普通文本保存。每个公式 block 会额外保存：

- `latex`
- `normalized_latex`
- `symbols`
- `operators`
- `source`
- 周边上下文

检索阶段会把公式词、符号重叠、Kalman/variance/covariance 等领域线索纳入 rerank。后续可以接入 PaddleOCR 的 Formula Recognition Pipeline，对扫描 PDF 中的公式进行 LaTeX 化。

## Table-aware Design

Tables are indexed at three levels:

- `table`: the complete table as TSV, useful for broad retrieval.
- `table_row`: one semantic row with `row_label` and normalized key/value text.
- `table_cell`: one non-empty cell with `row_index`, `col_index`, `row_label`,
  and optional `column_name`.

This is important for requirement tables where fields such as `描述`, `输入`,
`输出`, and `异常处理` contain long technical descriptions. During
`requirement_extract`, these rows are split into smaller requirement fragments
and each fragment keeps precise evidence:

```json
{
  "type": "requirement",
  "subject": "描述",
  "predicate": "自动校正后的轮径值应在有效范围内",
  "evidence": ["ev.blk.table_cell"]
}
```

Evidence keeps the table location:

```json
{
  "table_no": "table:1",
  "row_index": 8,
  "col_index": 2,
  "row_label": "描述"
}
```

If a table is only available as a scanned image or screenshot, enable OCR and
install the optional OCR dependencies. The base parser will then attach OCR text
to the `visual_object` and index it as normal text. Full table-grid
reconstruction from images should be handled by the future PaddleOCR table
structure pipeline.

## Visual-aware Design

Figures, charts, screenshots, diagrams, and embedded images are stored as
`visual_object` blocks instead of being folded into plain text. Each visual
object keeps the machine-facing fields that an external agent needs:

- `visual_type`: `chart`, `diagram`, `screenshot`, `photo`, `formula_image`,
  `table_image`, `mixed`, or `unknown`.
- `caption`: explicit figure caption or markdown alt text when available.
- `nearby_text`: neighboring explanatory paragraphs, cells, or page text.
- `asset_path`: local artifact path when the original image can be copied or
  extracted.
- `linked_text_blocks`: text blocks that should be read together with the
  visual object.
- `evidence`: page, sheet, cell range, bbox, or visual number.

Example:

```powershell
lookup-tool extract "Kalman covariance propagation diagram" --task visual_extract
```

Compact output:

```json
{
  "schema": "lookup.result.v1",
  "task": "visual_extract",
  "status": "ok",
  "items": [
    {
      "id": "vis.diagram.001",
      "type": "visual_object",
      "visual_type": "diagram",
      "caption": "Figure 1. Kalman covariance propagation diagram",
      "nearby_text": "Figure 1 explains how covariance prediction propagates uncertainty...",
      "asset_path": "data/artifacts/doc.x/figure_1.png",
      "linked_text_blocks": ["blk.x"],
      "evidence": ["ev.blk.visual"]
    }
  ]
}
```

## 可选增强

推荐部署时安装：

```powershell
pip install -e ".[api,parsing,ocr,ml]"
```

增强后可接入：

- Docling：更好的版面、表格、PDF/Office 结构化解析。
- PaddleOCR / PP-FormulaNet：扫描件公式 OCR。
- Ollama 或 llama.cpp：结构化抽取、变量定义推断。
- Qdrant：大规模向量和混合检索。
- RAGFlow：本机/内网离线部署后，可作为可选混合检索后端。

## 测试

默认测试不依赖网络：

```powershell
$env:PYTHONPATH="D:\LookupTool\src"
& "C:\Users\kylej\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m unittest discover -s tests -v
```

如果需要复现真实公开文档测试，先下载样本：

```powershell
$env:PYTHONPATH="D:\LookupTool\src"
& "C:\Users\kylej\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" tools/fetch_external_samples.py --out data/external_samples
$env:LOOKUP_TOOL_EXTERNAL_SAMPLES="D:\LookupTool\data\external_samples"
& "C:\Users\kylej\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m unittest tests.test_external_samples -v
```

外部测试当前覆盖：

- arXiv Kalman PDF：公式和图对象抽取。
- Calibre DOCX 示例：Word 段落、表格、内嵌图片。
- Microsoft Financial Sample XLSX：宽表格、表头识别、单元格参数抽取。

## 目录

```text
src/lookup_tool/
  app_store.py    # knowledge bases, projects, and question history
  cli.py          # CLI
  config.py       # TOML 配置
  extractor.py    # agent compact 结构化抽取
  formula.py      # 公式识别、规范化、变量角色
  index.py        # SQLite FTS + math-aware rerank
  llm.py          # OpenAI-compatible/Ollama 预留
  models.py       # Pydantic contract
  ocr.py          # optional PaddleOCR hook for image-only tables
  parsers.py      # 文件解析
  ragflow.py      # offline-only RAGFlow adapter
  webapp.py       # local WebApp server and API
  static/         # WebApp HTML/CSS/JS
  visual.py       # visual_object classification and caption linking
```

# RAGFlow 二开完整方案

## 1. 背景

项目改为基于 RAGFlow 二次开发。旧的独立本地 RAG 工具不再作为主线维护，已有 git 历史保留用于回滚，`data/models` 中已下载的 GGUF 模型继续保留。新的主线是 Docker 化 RAGFlow 工作区：RAGFlow 提供知识库、聊天、检索、证据链、文档管理和可视化 chunk；LocalMathRAGFlow 只增强工程文档需要的本地模型、公式、表格、图像和流程图辅助对象。

## 2. 产品定位

主产品形态是工程文档 Chat。用户提问后首先看到简洁、工程风格的自然语言回答。引用、证据链和原文定位沿用 RAGFlow 现有交互。结构化 JSON 对象不是主回答，而是折叠在证据或回答辅助区内，供复制、导出或交给外部表格 agent、代码 agent 使用。

核心原则：

- 回答文字为主，结构化为辅。
- 聊天体验优先，辅助视图不抢主视觉。
- 复用 RAGFlow 的知识库、文档管理、引用和 chunk 可视化。
- 只在必要位置扩展，不重做 RAGFlow 已有交互。
- 本地离线为默认目标，联网行为必须显式触发。

## 3. RAGFlow 保留能力

保留并基于以下能力继续二开：

- Docker 化自托管。
- Dataset/Knowledge Base 管理。
- Chat/Assistant 交互。
- grounded citations。
- key references 快速查看。
- text chunking 可视化。
- Word、Excel、PDF、图片、扫描件等文档入口。
- DeepDoc 文档理解和版面解析。
- 可配置 LLM、embedding、reranker。
- HTTP API 和后续 MCP/agent 能力。

这些能力不重新实现，只做配置、样式和元数据扩展。

## 4. 需要二开的主要内容

### 4.1 离线 Docker 开发栈

RAGFlow 默认是 Docker 路线，因此本项目二开也以 Docker 为主。开发环境由三部分组成：

- `third_party/ragflow`：上游 RAGFlow 源码，默认通过脚本拉取。
- `docker/docker-compose.localmathrag.yml`：覆盖 RAGFlow compose，挂载本地模型、扩展配置和 object sidecar。
- `services/object_service`：结构化对象辅助服务，后续可被 RAGFlow parser 或 chat pipeline 调用。

本地模型优先通过 OpenAI-compatible endpoint 接入。例如 llama.cpp、Ollama、vLLM、LM Studio 或 Xinference。GGUF 文件本身不直接让 RAGFlow 加载，而是由模型服务加载，再在 RAGFlow 里配置 endpoint。

### 4.2 工程聊天风格

系统提示词需要把默认回答风格改成工程文档问答：

- 简洁、直接、以条件和参数为中心。
- 先给结论，再列来源和限制。
- 不做无依据推断。
- 对需求、公式、参数、异常处理、输入输出给出可追溯引用。
- 如果文档信息不足，明确说明缺口。

### 4.3 结构化对象层

结构化对象不是 agent 最终响应格式，而是文档事实对象。它们应挂在 chunk、citation 或 answer metadata 后面，默认折叠展示。

核心对象类型：

- `technical_parameter`：技术参数、范围、单位、适用条件。
- `requirement_table_row`：需求表格行、SIL、确认方式、输入输出、异常处理。
- `formula`：公式、LaTeX、符号解释、上下文。
- `visual_object`：图片、图表、截图、图标、工程图等通用视觉对象。
- `flow_diagram`：流程图节点、边、条件、摘要。
- `table_object`：原始表格结构、表头、单元格、合并关系。

示例：

```json
{
  "id": "fig_3_2",
  "type": "flow_diagram",
  "title": "Kalman Filter Update Flow",
  "nodes": ["Prediction", "Measurement", "Update", "Output"],
  "edges": [
    ["Prediction", "Update"],
    ["Measurement", "Update"],
    ["Update", "Output"]
  ],
  "summary": "该图表示预测结果和观测结果共同进入更新步骤，输出修正后的状态估计。",
  "source_file": "Kalman_Filter_Spec_v1.3.pdf",
  "page": 12,
  "bbox": [120, 220, 890, 760]
}
```

### 4.4 表格增强

表格是重点。很多工程技术要求藏在大表格和长文本单元格中，不能只把表格 flatten 成普通文本。

需要增强：

- 保留单元格坐标、表头、跨行跨列、sheet/page。
- 支持长文本单元格切分，但仍能回到原单元格。
- 行级抽取技术参数、需求条目、输入输出、异常处理。
- 查询时优先返回自然语言回答，同时在折叠对象区提供表格行 JSON。
- 外部 agent 可以按 object id 获取原始表格、单元格和抽取字段。

### 4.5 公式增强

RAGFlow/DeepDoc 可以作为公式区域检测基础，但 Kalman filter 等工程公式需要更强的 math-aware 层。

需要增强：

- 公式区域裁剪保存。
- OCR/LaTeX/MathML 识别。
- 符号表和变量解释抽取。
- 公式与附近文字、图、表的关系。
- 对矩阵、协方差、转置、逆矩阵等符号进行检索增强。
- 支持按符号搜索，例如 `P`、`H`、`R`、`K`。

### 4.6 视觉对象增强

每张图都应成为 `visual_object`，再按类型细分处理。

类型建议：

- `flow_diagram`
- `block_diagram`
- `chart`
- `plot`
- `screenshot`
- `icon`
- `mechanical_drawing`
- `electrical_schematic`
- `photo`
- `generic_figure`

流程图需要抽取 nodes/edges；图表需要轴、单位、图例和趋势；图标需要结合图例和附近解释；普通图片需要 caption、OCR、附近文本和 VLM summary。

### 4.7 前端二开边界

前端不重做主 UI。保留 RAGFlow 的整体审美、导航和聊天流程，只在已有位置扩展：

- Chat answer 下方增加轻量按钮或折叠块：`结构化对象`。
- Citation/reference 面板中增加对象列表。
- 对象默认折叠，展开后可复制单个 JSON。
- 原文定位仍走 RAGFlow 现有引用/文档预览能力。
- 若对象有图片裁剪，展开时显示缩略图、bbox、OCR 和相关公式。

不做：

- 不新增独立的 JSON 主页面。
- 不把 JSON 插进主要回答正文。
- 不破坏 RAGFlow 的会话和引用样式。

### 4.8 外部 agent API

API 以 RAGFlow 原 API 为主，LocalMathRAGFlow 增加辅助 endpoint：

- `GET /v1/schemas`
- `GET /v1/schemas/{name}`
- `POST /v1/objects/normalize`
- `GET /v1/objects/{id}`
- `POST /v1/search/objects`
- `POST /v1/export/objects`

RAGFlow chat 的回答仍是文字，结构化对象通过 metadata/object ids 附带。

### 4.9 评测集

后续需要建立固定评测集：

- 长文本需求表格。
- Excel 多 sheet 参数表。
- Word 表格和嵌套表。
- Kalman filter 公式页。
- 流程图页。
- 图标加文字解释页。
- 扫描 PDF。
- 图片、表格、公式混排页。

指标：

- 表格字段抽取准确率。
- 公式 LaTeX 准确率。
- 符号解释准确率。
- 流程图 nodes/edges 准确率。
- citation page/bbox 准确率。
- JSON schema 合法率。
- 回答是否引用充分。

## 5. 实施阶段

### 阶段 1：Docker 二开工作区

- 删除旧独立实现。
- 保留 `data/models`。
- 建立 RAGFlow bootstrap、Docker override、object sidecar、schema 和二开文档。
- 运行合同测试。

### 阶段 2：RAGFlow 源码 fork

- 将 `third_party/ragflow` 切到自己的 fork。
- 选择稳定 tag 作为基线。
- 建立 upstream 同步流程。
- 固定 Docker image 版本。

### 阶段 3：Parser metadata

- 修改 DeepDoc/parser 输出，增加 `document_objects`。
- 对 table/equation/figure/caption/cell 加 object id。
- 将 bbox、page、source、nearby text 写入 metadata。

### 阶段 4：Chat evidence 扩展

- 检索结果携带 object metadata。
- Prompt 使用对象辅助回答，但不默认输出 JSON。
- citation 面板展示折叠对象。

### 阶段 5：工程对象抽取

- 表格行抽取。
- 公式识别和符号解释。
- 图片 OCR/VLM summary。
- 流程图 nodes/edges。

### 阶段 6：本地模型体验

- 在 RAGFlow 设置中预置 OpenAI-compatible 本地 endpoint 模板。
- 支持 llama.cpp、Ollama、vLLM、LM Studio。
- Docker profile 可选启动 llama.cpp。

## 6. 当前仓库实现状态

当前提交完成阶段 1 的骨架：

- RAGFlow 源码通过 `scripts/bootstrap-ragflow.ps1` 拉取到 `third_party/ragflow`。
- Docker 覆盖文件位于 `docker/docker-compose.localmathrag.yml`。
- 本地 GGUF 模型通过 `data/models` 挂载。
- 结构化对象 sidecar 位于 `services/object_service`。
- schema、prompt、API 合约位于 `extensions/local_math_rag`。

下一步应进入阶段 2：确定 RAGFlow fork/tag，然后开始改 parser 和 frontend citation panel。

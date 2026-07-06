# LocalMathRAGFlow Agent Rules

本仓库已经从独立 Python/EXE RAG 工具切换为基于 RAGFlow Docker 的二开工作区。后续 agent 修改时必须遵守以下规则，避免再次出现从 `dist` 启动后误判根目录、重复下载依赖或模型的问题。

## 目录职责

- `D:\LookupTool` 是开发根目录，也是当前默认 workspace root。
- `data/` 是本地持久化数据目录，始终被 git 忽略。这里保存模型、日志、知识库、运行缓存等，不要删除或提交。
- `data/models/` 保存本地 GGUF 或其他模型文件。已存在模型必须优先复用，不要因为从 release/dist 启动而重复下载。
- `third_party/ragflow/` 保存上游或 fork 后的 RAGFlow 源码，始终被 git 忽略。开发环境优先复用这里的源码。
- `docker/` 保存 LocalMathRAGFlow 对 RAGFlow compose 的覆盖文件。
- `extensions/local_math_rag/` 保存 schema、prompt、pipeline 和 API 合约。
- `services/object_service/` 保存结构化对象 sidecar 服务。
- `launcher/LocalMathRAGFlow/` 保存 Windows 托盘 EXE 启动器源码。
- `dist/` 是构建产物目录，始终被 git 忽略。不要把它当成权威开发根目录，不要提交其中内容。

## Launcher Root Resolution

Windows 启动器必须按以下逻辑选择 root：

1. 从 `AppContext.BaseDirectory` 开始向父目录扫描候选 root。
2. 候选 root 必须同时包含：
   - `docker/docker-compose.localmathrag.yml`
   - `scripts/`
3. 在候选 root 中，优先选择已经存在：
   - `third_party/ragflow/docker/docker-compose.yml`
4. 如果没有任何候选 root 已安装 RAGFlow，才使用第一个候选 root。
5. 只有在选定 root 下缺少 `third_party/ragflow/docker` 时，才允许弹窗询问是否下载 RAGFlow。

这意味着：

- 从 `D:\LookupTool\dist\LocalMathRAGFlow-win-x64\LocalMathRAGFlow.exe` 启动时，如果父级 `D:\LookupTool\third_party/ragflow` 已存在，必须复用 `D:\LookupTool` 作为 root。
- 不允许因为 `dist` 目录中也有 `docker/` 和 `scripts/` 就直接把 `dist` 当作最终 root。
- 不允许从 `dist` 启动时重复下载 RAGFlow 或模型。

## Dependency And Download Rules

- Docker Desktop 未运行时，launcher 可以自动启动 Docker Desktop 并等待 Docker daemon ready。
- Docker Desktop 未安装时，只能提示用户安装，不能静默安装。
- RAGFlow 源码缺失时，必须弹窗确认后再下载。
- 模型缺失时，必须弹窗确认后再下载；已有 `data/models` 中的模型必须优先复用。
- 日常运行应保持离线优先，联网行为必须由用户显式确认。

## Local Model Runtime Rules

- `data/models/*.gguf` 是本地模型的唯一默认发现入口，launcher 和 `scripts/dev-up.ps1` 必须优先扫描这里。
- 如果发现本地 GGUF，默认启动 `llama-cpp-cpu` compose profile，并设置 `LOCALMATHRAG_GGUF_MODEL=<文件名>`。
- 如果本地 GGUF 已存在但 llama.cpp Docker image 尚未安装，launcher 必须弹窗确认后才允许拉取镜像；用户拒绝时只启动 RAGFlow，不启本地模型 endpoint。
- 如需 GPU，使用 `LOCALMATHRAG_LLAMA_PROFILE=cuda` 切换到 `llama-cpp-cuda`；如需禁用本地模型服务，使用 `LOCALMATHRAG_LLAMA_PROFILE=none`。
- 如果没有本地 GGUF，不允许自动下载模型，也不允许强行启动 llama.cpp 容器。
- RAGFlow 模型提供商仍使用 OpenAI-compatible 形式，但 `base_url` 必须指向本地 `http://host.docker.internal:8080/v1` 或等价的本机离线端点。
- `services/object_service` 必须暴露 `/v1/models/status`，用于确认模型文件数量和本地 llama endpoint 状态。
- 当 RAGFlow 报 `No valid response received` 或 `Fail to access model` 时，优先检查 llama.cpp 容器是否已启动、8080 端口是否可访问、`LOCALMATHRAG_GGUF_MODEL` 是否与 `data/models` 文件名一致。

## Tray Launcher Rules

- 托盘菜单必须同时设置 `NotifyIcon.ContextMenuStrip` 和手动右键兜底逻辑，避免 Windows 托盘区域吞掉鼠标事件。
- 手动菜单弹出必须使用隐藏 owner form，并在弹出前 `SetForegroundWindow`，否则右键菜单可能不显示或立即消失。
- 左键双击托盘图标只打开或聚焦现有 WebApp 窗口，不创建多个独立窗口。

## Build And Release Rules

- `scripts/build-launcher.ps1` 只负责生成本地 release 包，不提交 `dist/`。
- release 包可以包含 `docker/`、`scripts/`、`extensions/`、`services/` 等运行所需文件，但这些文件不能改变 root resolution 的优先级。
- 如果修改 launcher root resolution，必须同时更新 `tests/test_contracts.py` 中的约束检查。

## RAGFlow Patch Rules

- `third_party/ragflow/` 是忽略目录，不能依赖直接提交其中的改动。
- 所有 RAGFlow 二开改动必须同步生成到 `patches/ragflow/*.patch`。
- `scripts/apply-ragflow-patches.ps1` 必须能够在干净 RAGFlow checkout 上应用补丁，也必须能在补丁已应用时安全跳过。
- `scripts/build-ragflow-web.ps1` 是前端二开构建入口，负责先应用补丁，再安装依赖并生成 `third_party/ragflow/web/dist`。
- 如果 `third_party/ragflow/web/dist` 存在，launcher 和 `scripts/dev-up.ps1` 必须挂载 `docker/docker-compose.webdist.yml`，让容器使用本地构建后的前端。

## Encoding Rule

含中文的 `.md`、`.txt`、`.ps1`、`.py`、`.yaml`、`.yml` 文件必须使用 UTF-8 with BOM。测试会检查这一点。

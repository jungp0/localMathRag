# LocalMathRAGFlow

## 快速开始

LocalMathRAGFlow 面向 Windows 本地离线运行：下载 release 包、解压、双击启动器即可。启动器会自动发现 `data/models` 中已有模型；缺少 RAGFlow 源码、Docker 镜像或推荐模型时，会弹窗确认后再下载，不会静默联网。

### 需要什么

- Windows 10/11 x64。
- Docker Desktop 4.x 或更新版本，并启用 WSL2 后端。
- Git，用于首次拉取 RAGFlow 源码。
- 约 30 GB 以上可用磁盘空间；如果下载本地大模型，建议预留 80 GB 以上。
- 推荐 NVIDIA GPU + CUDA 驱动。8 GB 显存可跑较小/量化模型；12 GB 到 24 GB 显存体验更稳。没有 CUDA 也可以只启动 RAGFlow，或按需使用 CPU 回退，但解析和问答会慢很多。

### 命令行下载和启动

推荐直接从 GitHub Release 下载最新 Windows 包：

```powershell
$Zip = "LocalMathRAGFlow-win-x64.zip"
$Url = "https://github.com/jungp0/localMathRag/releases/latest/download/$Zip"
$InstallDir = "D:\LocalMathRAGFlow"

New-Item -ItemType Directory -Force $InstallDir | Out-Null
Invoke-WebRequest -Uri $Url -OutFile "$env:TEMP\$Zip"
Expand-Archive -Force "$env:TEMP\$Zip" $InstallDir
Start-Process "$InstallDir\LocalMathRAGFlow.exe"
```

也可以在浏览器打开 GitHub Releases 页面，下载 `LocalMathRAGFlow-win-x64.zip`，解压后双击 `LocalMathRAGFlow.exe`。

### 模型和 CUDA 配置

- 模型默认目录是 `data/models`。已有 `.gguf`、embedding、rerank、VLM/ASR/TTS 模型会被自动发现。
- 如果没有本地模型，启动器会在需要时弹窗确认下载；拒绝下载时仍可启动 RAGFlow，只是不启对应本地模型 endpoint。
- 发现 `.gguf` 后默认使用 `llama-cpp-cuda`；需要 CPU 回退时设置 `LOCALMATHRAG_LLAMA_PROFILE=cpu`。
- 发现 `data/models/bge-m3` 后默认使用 CUDA embedding profile；需要禁用真实 embedding runtime 时设置 `LOCALMATHRAG_EMBEDDING_PROFILE=none`。
- 发现 `data/models/Qwen3-Reranker-0.6B` 后默认使用 CUDA rerank profile；需要禁用时设置 `LOCALMATHRAG_RERANK_PROFILE=none`。

启动后，把文档放进 `data/dataset`，在 RAGFlow 的 `Default` 知识库页面点击 `Refresh local folder`，然后在 RAGFlow 原生聊天/助手页面选择知识库开始问答。

### 源码开发

如果你要改代码或调试开发版，再 clone 仓库：

```powershell
git clone https://github.com/jungp0/localMathRag.git
cd localMathRag
.\scripts\doctor.ps1
.\scripts\bootstrap-ragflow.ps1
.\scripts\dev-up.ps1
```

LocalMathRAGFlow 是基于 RAGFlow 的工程文档问答二开工作区。当前方向不再维护原先的独立 Python/EXE RAG 工具，而是以 RAGFlow 源码为二开基底，构建 `localmathrag/ragflow:dev` 本地镜像，复用其知识库、聊天、引用证据链、chunk 可视化和文档管理能力，并增强工程文档场景需要的表格、公式、图像和流程图辅助对象。

## 产品原则

- 聊天交互为主：用户仍然在 RAGFlow 风格的 Chat 中提问和阅读回答。
- 结构化对象为辅：表格参数、公式、流程图节点边等对象默认折叠，只作为证据增强、复制导出和外部 agent 输入。
- 保持 RAGFlow 一致性：不重做主界面、不另起一套问答页，二开只扩展已有证据链、引用、文档定位和设置流程。
- 完全离线优先：日常运行只依赖本地 Docker 服务、本地知识库和本地模型 endpoint；联网只用于显式拉取镜像、RAGFlow 源码或模型。

完整二开方案见 [docs/ragflow-secondary-development.md](docs/ragflow-secondary-development.md)。

后续 agent 和维护者必须先阅读 [AGENTS.md](AGENTS.md)。其中固化了目录职责、launcher root resolution、依赖下载和 release 构建规则，防止从 `dist` 启动时重复下载 RAGFlow 或模型。

## 仓库结构

```text
launcher/
  LocalMathRAGFlow/                # Windows 托盘 EXE 启动器
docker/
  docker-compose.localmathrag.yml   # RAGFlow Docker 覆盖文件
  Dockerfile.ragflow-local          # LocalMathRAGFlow 二开镜像
docs/
  ragflow-secondary-development.md  # 完整二开文档
extensions/local_math_rag/
  api/openapi.yaml                  # 外部 agent/API 合约
  config/pipeline.yaml              # 工程对象抽取 pipeline 配置
  prompts/engineering_chat_system.md
  schemas/*.json                    # 结构化辅助对象 schema
services/object_service/
  main.py                           # 结构化对象 sidecar 服务
  Dockerfile
scripts/
  bootstrap-ragflow.ps1             # 拉取 RAGFlow 源码到 third_party/ragflow
  dev-up.ps1                        # 启动 Docker 二开环境
  dev-down.ps1                      # 停止 Docker 二开环境
  doctor.ps1                        # 环境检查
tests/
  test_contracts.py                 # schema/文档编码/配置校验
```

`data/` 被保留并继续被 git 忽略。已下载的 `data/models/Qwen3-8B-Q4_K_M.gguf`、`data/models/Qwen3-Embedding-0.6B` 等模型不会被删除，Docker 覆盖文件会把 `data/models` 挂载给本地模型服务。

## 一键启动

本地桌面主入口是 `LocalMathRAGFlow.exe`：

1. 双击 `LocalMathRAGFlow.exe`。
2. 如果 Docker daemon 未运行，启动器会自动打开 Docker Desktop 并等待 Docker ready。
3. 如果 `third_party/ragflow` 不存在，启动器会弹窗确认是否从 GitHub 下载 RAGFlow 源码。
4. 启动器执行 Docker Compose，构建并启动 `localmathrag/ragflow:dev`、MySQL、Redis、MinIO、Elasticsearch、LocalMathRAGFlow object sidecar 和已启用的本地模型服务。
5. 启动完成后自动打开 RAGFlow Web。

托盘菜单支持：

- 打开 RAGFlow。
- 打开 object service。
- 启动、停止、重启服务。
- 打开默认知识库文件夹 `data/dataset`，可以直接放入文件和子文件夹。
- 打开数据目录。
- 退出时选择是否停止 Docker 服务。

默认知识库文件夹为程序目录下的 `data/dataset`。RAGFlow 启动时只确保内置知识库 `Default` 存在，不自动扫描本地文件夹，也不自动绑定 Chat/Dialog；用户仍在 RAGFlow 原生聊天或助手页面自行选择知识库。

在 RAGFlow 的 `Default` 知识库页面点击 `Refresh local folder` 后，后端才会扫描 `/localmathrag/dataset` 并同步新增、变化和删除文件。扫描状态写入 `data/cache/dataset-state.json`，只比较相对路径、类型、size、mtime_ns 和 ctime_ns；目录签名未变化时复用上次子树 manifest，未变化文件不读取内容、不上传、不解析。LocalMathRAGFlow object service 保留 `/v1/dataset/status` 和 `/v1/dataset/files` 作为诊断接口。

构建 EXE：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build-launcher.ps1
```

输出目录：

```text
dist\LocalMathRAGFlow-win-x64
```

Release zip：

```text
dist\LocalMathRAGFlow-win-x64.zip
```

## 开发启动

首次开发需要显式拉取 RAGFlow：

```powershell
.\scripts\bootstrap-ragflow.ps1
```

启动 RAGFlow CPU 开发环境：

```powershell
.\scripts\dev-up.ps1
```

如果希望同时用已保留的 GGUF 模型启动 llama.cpp endpoint，默认会使用 CUDA profile：

```powershell
.\scripts\dev-up.ps1
```

如需 CPU 回退：

```powershell
.\scripts\dev-up.ps1 -Llama cpu
```

RAGFlow Web 默认仍使用其 Docker 配置暴露端口。object sidecar 默认暴露：

```text
http://127.0.0.1:8088/health
```

本地模型端口约定：

- LLM: `8080`，发现本地 GGUF 后默认由 `llama-cpp-cuda` profile 提供；CPU 回退使用 `LOCALMATHRAG_LLAMA_PROFILE=cpu`。
- 嵌入模型: 默认由 object service 的 `http://localmathrag-object-service:8088/v1` fallback endpoint 保证解析链路可用；发现 `bge-m3` 后默认可启动 `embedding-cuda`，端口为 `8081`；禁用真实嵌入 runtime 使用 `LOCALMATHRAG_EMBEDDING_PROFILE=none`。
- 重排模型: `8082`，发现模型后默认使用 `rerank-cuda`；禁用使用 `LOCALMATHRAG_RERANK_PROFILE=none`。
- VLM: `8083`，发现模型后默认使用 `vlm-cuda`；禁用使用 `LOCALMATHRAG_VLM_PROFILE=none`。
- ASR: `8084`，推荐模型有独立 ASR 下载标签；未来接入 runtime 后默认使用 `asr-cuda`。
- TTS: `8085`，推荐模型有独立 TTS 下载标签；未来接入 runtime 后默认使用 `tts-cuda`。

停止环境：

```powershell
.\scripts\dev-down.ps1
```

## 二开边界

本仓库当前默认构建 `localmathrag/ragflow:dev` 二开镜像。短期镜像基于 RAGFlow release image 叠加本仓库 patch，后续可以切到完整 source build；建议把 `third_party/ragflow` 切到自己的 RAGFlow fork，继续保持 upstream 同步能力。主线修改应集中在：

- DeepDoc/parser 输出增加 `document_objects` 元数据。
- Chat retrieval 返回对象引用和折叠结构化证据。
- 前端 citation/reference 面板增加结构化对象折叠区。
- 本地模型配置页默认指向 OpenAI-compatible 本地 endpoint。

## 校验

```powershell
python .\tests\test_contracts.py
```

该测试会检查 JSON schema、OpenAPI/pipeline 文本和含中文文件的 UTF-8 BOM。

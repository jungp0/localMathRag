# LocalMathRAGFlow

LocalMathRAGFlow 是基于 RAGFlow 的工程文档问答二开工作区。当前方向不再维护原先的独立 Python/EXE RAG 工具，而是复用 RAGFlow 的 Docker 化服务、知识库、聊天、引用证据链、chunk 可视化和文档管理能力，在其上增强工程文档场景需要的表格、公式、图像和流程图辅助对象。

## 产品原则

- 聊天交互为主：用户仍然在 RAGFlow 风格的 Chat 中提问和阅读回答。
- 结构化对象为辅：表格参数、公式、流程图节点边等对象默认折叠，只作为证据增强、复制导出和外部 agent 输入。
- 保持 RAGFlow 一致性：不重做主界面、不另起一套问答页，二开只扩展已有证据链、引用、文档定位和设置流程。
- 完全离线优先：日常运行只依赖本地 Docker 服务、本地知识库和本地模型 endpoint；联网只用于显式拉取镜像、RAGFlow 源码或模型。

完整二开方案见 [docs/ragflow-secondary-development.md](docs/ragflow-secondary-development.md)。

## 仓库结构

```text
launcher/
  LocalMathRAGFlow/                # Windows 托盘 EXE 启动器
docker/
  docker-compose.localmathrag.yml   # RAGFlow Docker 覆盖文件
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

`data/` 被保留并继续被 git 忽略。已下载的 `data/models/Qwen3-8B-Q4_K_M.gguf` 不会被删除，Docker 覆盖文件会把 `data/models` 挂载给可选的 llama.cpp 服务。

## 一键启动

本地桌面主入口是 `LocalMathRAGFlow.exe`：

1. 双击 `LocalMathRAGFlow.exe`。
2. 如果 Docker daemon 未运行，启动器会自动打开 Docker Desktop 并等待 Docker ready。
3. 如果 `third_party/ragflow` 不存在，启动器会弹窗确认是否从 GitHub 下载 RAGFlow 源码。
4. 启动器执行 Docker Compose，启动 RAGFlow、MySQL、Redis、MinIO、Elasticsearch 和 LocalMathRAGFlow object sidecar。
5. 启动完成后自动打开 RAGFlow Web。

托盘菜单支持：

- 打开 RAGFlow。
- 打开 object service。
- 启动、停止、重启服务。
- 打开数据目录。
- 查看 launcher/compose 日志。
- 退出时选择是否停止 Docker 服务。

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

如果希望同时用已保留的 GGUF 模型启动 llama.cpp CPU endpoint：

```powershell
.\scripts\dev-up.ps1 -Llama cpu
```

如果 Docker 环境支持 NVIDIA GPU：

```powershell
.\scripts\dev-up.ps1 -Device gpu -Llama cuda
```

RAGFlow Web 默认仍使用其 Docker 配置暴露端口。object sidecar 默认暴露：

```text
http://127.0.0.1:8088/health
```

停止环境：

```powershell
.\scripts\dev-down.ps1
```

## 二开边界

本仓库当前先落地 Docker 二开骨架和工程对象契约。后续真正改 RAGFlow 源码时，建议把 `third_party/ragflow` 切到自己的 RAGFlow fork，继续保持 upstream 同步能力。主线修改应集中在：

- DeepDoc/parser 输出增加 `document_objects` 元数据。
- Chat retrieval 返回对象引用和折叠结构化证据。
- 前端 citation/reference 面板增加结构化对象折叠区。
- 本地模型配置页默认指向 OpenAI-compatible 本地 endpoint。

## 校验

```powershell
python .\tests\test_contracts.py
```

该测试会检查 JSON schema、OpenAPI/pipeline 文本和含中文文件的 UTF-8 BOM。

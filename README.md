**简体中文** | [English](README_EN.md)

# LocalMathRAGFlow

> 在 Windows 上运行的本地工程文档知识库：把 PDF、Word 等资料放进文件夹，就能在浏览器里检索、提问、查看引用，并尽量保留公式、表格和图片上下文。

LocalMathRAGFlow 是基于 [RAGFlow](https://github.com/infiniflow/ragflow) 的本地离线增强版。它不是一个需要上传文件到云端的网站，而是一套运行在你自己电脑上的 Docker 服务，并提供一个 Windows 托盘启动器负责启动、停止和打开界面。

如果你只想使用它，不需要会 Python、Docker 命令或编程。先安装 Docker Desktop 和 Git，再下载 Release 压缩包、解压、双击 EXE 即可。

## 先用一分钟了解它

你可以把它理解成“会查资料、会标出处的本地文档助手”。典型使用过程是：

1. 把规范、手册、论文、计算书或项目资料复制到 `data\dataset`。
2. 在 `Default` 知识库中点击 `Refresh local folder`，让系统只同步新增、修改或删除的文件。
3. 等文档解析完成后，在搜索或聊天页面提问。
4. 阅读答案，同时核对引用的原文片段；需要留档时复制或下载 Markdown。

适合这些场景：

- 在内网或个人电脑上查询不方便上传到云端的资料。
- 从大量工程文档中找参数、条款、公式、表格和上下文。
- 根据文档生成带引用的摘要、对比、检查清单或解释。
- 使用本地 GGUF 大模型，尽量减少日常联网依赖。

它不是“安装后自带全部模型”的成品云服务。Release 不包含体积很大的 AI 模型；本地问答至少需要一个聊天模型。首次准备 RAGFlow 源码、Docker 镜像或模型时可能需要联网，并且启动器会先询问，不会静默下载。

## 实际已经实现的功能

| 功能 | 使用时会看到什么 |
| --- | --- |
| Windows 一键启动 | 双击 `LocalMathRAGFlow.exe` 后自动检查 Docker、启动服务并打开内置 Web 窗口；关闭窗口不会停止后台服务，可从托盘再次打开。 |
| 本地文件夹增量同步 | `Default` 知识库中的 `Refresh local folder` 会递归扫描 `data\dataset`，只处理新增、变化和删除的文件，忽略 Office 临时文件。 |
| RAGFlow 文档知识库 | 保留 RAGFlow 的知识库、文档解析、切片预览、检索、聊天、引用证据和文档管理能力。 |
| 工程公式增强 | 对 DOCX 中的公式图片、Equation Editor/MathType MTEF、WMF 预览和逐公式 OCR 做增强处理，并尽量保留公式所在段落及表格上下文；低质量 OCR 不会冒充可靠公式。 |
| 图片理解可选增强 | 配置 VLM 后可对图片内容进行增强识别；未配置时不会强制启动或下载视觉模型。 |
| 带引用的搜索回答 | 搜索过程显示阶段和进度；回答优先返回，思维导图与相关问题等辅助内容随后生成，辅助步骤超时不会阻塞主答案。 |
| Markdown 留档 | 搜索结果可复制为 Markdown，或下载包含问题、答案和引用片段的 `.md` 文件。 |
| 本地模型发现与切换 | 自动发现 `data\models` 中的聊天、嵌入、重排和视觉模型；模型设置页可查看本地模型、推荐模型、下载进度、切换进度和可读错误。 |
| 资源自适应 | 根据显存、内存、模型大小和上下文长度调整并发与批量大小；可按需启动模型容器，资源不足时降级或停止可选 runtime。 |
| 大型知识库任务恢复 | GraphRAG、RAPTOR、目录结构提取和辅助索引使用限流、租约、看门狗及检查点，减少长任务卡死或重启后全部重做。 |
| 离线兼容 API | object service 提供 OpenAI-compatible 的聊天代理、embedding、rerank，以及模型状态、runtime 管理、结构化对象和数据集诊断接口。 |

当前边界也需要说清楚：

- ASR 和 TTS 已预留模型发现、端口与 Docker profile，但默认镜像仍是占位配置，不应视为开箱即用的语音功能。
- `Qwen3-Embedding-0.6B` 可以保留在模型目录中，但当前默认真实嵌入模型仍是 `bge-m3`。
- OCR、VLM、GraphRAG 和 RAPTOR 会增加显存、内存和处理时间；普通问答不要求全部开启。
- AI 回答可能出错。工程结论应点击引用核对原文，不能把生成结果直接当作规范或计算书签字依据。

## 零门槛安装

### 1. 准备电脑

必须具备：

- Windows 10/11 x64。
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)，使用 WSL2 后端。只需安装，不要求会写 Docker 命令。
- [Git for Windows](https://git-scm.com/download/win)，首次获取 RAGFlow 源码时会用到。
- 至少 30 GB 可用磁盘空间；如果准备多个本地模型，建议预留 80 GB 或更多。

推荐但不是强制：

- NVIDIA 显卡和可用的驱动。8 GB 显存适合较小的量化模型，12–24 GB 显存体验更稳定。
- 没有 NVIDIA 显卡时仍可启动 RAGFlow，也可以使用 CPU 模式，但解析和问答会明显变慢。

不需要另外安装 .NET、Python、Node.js、MySQL、Redis 或 Elasticsearch；Windows Release 和 Docker 环境会负责这些运行组件。

### 2. 下载并解压

打开 [GitHub Releases](https://github.com/jungp0/localMathRag/releases/latest)，下载：

```text
LocalMathRAGFlow-win-x64.zip
```

把压缩包完整解压到一个固定、可写、空间充足的目录，例如：

```text
D:\LocalMathRAGFlow
```

不要直接在压缩包预览窗口中运行 EXE，也不要只复制一个 EXE。`docker`、`scripts`、`services`、`patches` 等目录都是运行所需内容。

也可以用 PowerShell 下载：

```powershell
$Zip = "LocalMathRAGFlow-win-x64.zip"
$Url = "https://github.com/jungp0/localMathRag/releases/latest/download/$Zip"
$InstallDir = "D:\LocalMathRAGFlow"

New-Item -ItemType Directory -Force $InstallDir | Out-Null
Invoke-WebRequest -Uri $Url -OutFile "$env:TEMP\$Zip"
Expand-Archive -Force "$env:TEMP\$Zip" $InstallDir
Start-Process "$InstallDir\LocalMathRAGFlow.exe"
```

### 3. 第一次启动

双击 `LocalMathRAGFlow.exe`。第一次启动通常比以后慢，因为系统需要完成以下工作：

1. 检查 Docker Desktop；如果已安装但未运行，启动器会打开它并等待 Docker daemon 就绪。
2. 寻找已有的 `third_party\ragflow`。只有确实缺少 RAGFlow 时才会询问是否下载源码。
3. 应用本项目补丁并构建 `localmathrag/ragflow:dev` 本地镜像。
4. 启动 RAGFlow、MySQL、Redis、MinIO、Elasticsearch 和 LocalMathRAGFlow object service。
5. 发现本地模型后，按配置准备对应模型 runtime；缺少镜像时会先询问是否拉取。
6. 打开 LocalMathRAGFlow 窗口。默认是本机单用户界面，不需要注册云端账号。

看到下载或拉取确认框时：

- 允许：本次会联网获取明确显示的源码、镜像或模型。
- 拒绝：不会偷偷下载；RAGFlow 主服务通常仍可启动，但相应的本地模型能力不会启用。

以后从 Release 子目录或 `dist` 启动时，启动器会优先复用上级已经安装好的 RAGFlow 和 `data\models`，不会因为启动位置变化而重新下载。

## 第一次使用：从文件到答案

### 第一步：准备至少一个聊天模型

本地问答需要聊天模型。把 GGUF 文件放到安装目录下：

```text
data\models\你的模型.gguf
```

重启服务后，系统会自动发现它并默认使用 llama.cpp CUDA runtime。你也可以在 RAGFlow 的模型设置页查看本地模型与推荐模型。涉及模型下载的按钮都会显示进度，并且只有在你确认后才联网。

建议的模型布局：

| 用途 | 默认发现位置 | 不安装时的影响 |
| --- | --- | --- |
| 聊天/回答 | `data\models\*.gguf` | RAGFlow 可以启动，但无法完成本地生成式问答。 |
| 真实嵌入 | `data\models\bge-m3` | object service 仍提供离线兼容 fallback，便于链路运行；真实语义检索效果建议使用 `bge-m3`。 |
| 重排 | `data\models\Qwen3-Reranker-0.6B` | 检索仍可运行，但少一层精排；服务会使用兼容回退。 |
| 图片理解 | `data\models` 下已配置的 VLM 目录 | 文本问答不受影响，图片增强不会启用。 |
| 数学公式 OCR | 模型设置页中的推荐 Math OCR 项 | 原生公式和普通解析仍工作，图片公式识别能力受限。 |

模型文件通常很大，因此不会放进 Git 仓库或 Release。已经存在于 `data\models` 的文件会被优先复用，不要为升级程序重复下载。

### 第二步：放入文档

在 Windows 右下角托盘找到 LocalMathRAGFlow 图标，右键选择 `Open dataset`，或直接打开：

```text
data\dataset
```

把资料复制进去。可以建立子文件夹整理项目，例如：

```text
data\dataset\
├─ 设计规范\
├─ 设备手册\
├─ 计算书\
└─ 项目会议纪要\
```

支持范围以当前 RAGFlow 文档解析器为准，常见的 PDF、DOCX、XLSX、PPTX、TXT、Markdown、HTML 和图片等均可通过 RAGFlow 导入。公式增强主要针对 DOCX/Office 公式场景。

### 第三步：刷新知识库

1. 打开 RAGFlow。
2. 进入知识库列表并打开内置的 `Default` 知识库。
3. 点击 `Refresh local folder`。
4. 等待新增或变化的文档解析完成。

这一步不是“每次重新上传全部文件”。系统会记录相对路径、大小和修改时间等状态；目录未变化时会复用缓存，未变化文件不会重复读取和解析。删除 `data\dataset` 中的文件后再次刷新，对应知识库文件也会同步删除。

### 第四步：检索和提问

可以使用搜索页，也可以在 RAGFlow 的聊天/助手页面选择 `Default` 知识库后提问。新手可以从这些问题开始：

- “这份规范对环境温度有什么要求？请列出原文出处。”
- “比较 A 型号和 B 型号的额定参数，用表格回答。”
- “解释第 3 章公式中每个符号的含义，并引用公式前后的文字。”
- “找出所有涉及验收条件的条款，按文档和章节分组。”
- “根据这些资料生成检查清单；无法从原文确认的项目请明确标注。”

使用建议：

- 问题中写明文档名、设备名、章节或参数，检索会更准确。
- 先看答案，再点击引用片段核对原文。
- 搜索页的思维导图和相关问题属于辅助结果，可能在答案出现后继续生成。
- 使用复制或下载按钮，可把当前问题、回答和引用保存为 Markdown。

## 托盘图标怎么用

关闭主窗口后，后台服务不会自动消失。右键托盘图标可使用：

| 菜单 | 作用 |
| --- | --- |
| `Open RAGFlow` | 打开或聚焦现有窗口，不会重复创建多个 WebApp 窗口。 |
| `Start services` | 启动 Docker 服务并打开状态页。 |
| `Stop services` | 停止本项目服务。 |
| `Reset runtime policy` | 清除模型 runtime 的失败/降级记忆并重新调度，适合模型切换或资源异常后恢复。 |
| `Open dataset` | 打开需要同步进知识库的文件夹。 |
| `Open data directory` | 打开模型、缓存、日志和知识库数据所在目录。 |
| `Exit` | 退出启动器；根据提示决定是否同时停止 Docker 服务。 |

左键单击或双击托盘图标都会打开或聚焦 LocalMathRAGFlow 窗口。

## 数据放在哪里，会不会上传

所有持久化数据都在安装根目录的 `data` 下，并被 Git 忽略：

```text
data\
├─ models\      # 本地 AI 模型
├─ dataset\     # 等待同步的原始资料
├─ cache\       # 增量扫描、runtime 策略等缓存
├─ logs\        # 本地日志（如组件创建）
└─ launcher\    # 启动器窗口和 WebView2 数据
```

日常解析、检索和问答优先走 Docker 内网中的本地 endpoint。项目本身不会要求把知识库上传到云端。以下操作可能联网，并且应由用户明确触发或确认：

- 首次下载 RAGFlow 源码。
- 拉取尚未安装的 Docker 镜像。
- 在模型设置中下载推荐模型。
- 你自己配置并使用外部模型提供商。

升级前建议备份整个 `data` 目录。不要删除 `data` 来解决普通启动问题，因为其中可能包含模型和知识库持久化数据。

## 常见问题

### 双击后提示找不到 Docker

先安装 Docker Desktop，启动一次并确认状态为 Running，再重新打开 LocalMathRAGFlow。启动器可以自动打开已安装的 Docker Desktop，但不会静默安装它。

### 第一次启动很久没有进入页面

首次构建 RAGFlow 镜像可能需要较长时间。可查看安装目录下的 `launcher-tray.log`，并确认 Docker Desktop 没有在等待 WSL2、网络或许可确认。托盘中的 `Open RAGFlow` 会重新打开当前状态页。

### 能打开页面，但无法回答问题

依次检查：

1. `data\models` 中是否至少有一个完整的 `.gguf` 文件。
2. 模型设置中聊天模型是否已选择为本地模型。
3. 托盘执行 `Reset runtime policy` 后再试。
4. 检查端口 `8080` 的本地聊天 runtime 是否已启动。

### 文档搜索不到，或解析失败

确认文件已放到 `data\dataset`，然后在 `Default` 知识库中点击 `Refresh local folder`。等待文档状态变为成功后再提问。正在编辑的 `~$` Office 临时文件会被自动忽略。

### 出现 `No valid response received`、`Fail to access model` 或 `Fail to bind embedding model`

这通常表示 RAGFlow 数据库仍保存着旧模型地址，或对应模型容器未就绪。先执行 `Reset runtime policy`，再检查模型设置中的地址和以下端口：

| 能力 | 默认端口 |
| --- | ---: |
| 本地 LLM | 8080 |
| Embedding | 8081 |
| Rerank | 8082 |
| VLM | 8083 |
| ASR（预留） | 8084 |
| TTS（预留） | 8085 |
| Object service / fallback API | 8088 |

Embedding 和 rerank 在 RAGFlow 内默认应优先使用 `http://localmathrag-object-service:8088/v1` 作为兼容入口，而不是遗留的宿主机旧地址。

### 显存不够，怎样切换 CPU 或关闭可选模型

从 PowerShell 启动时，可以先设置本次进程的环境变量：

```powershell
$env:LOCALMATHRAG_LLAMA_PROFILE = "cpu"
$env:LOCALMATHRAG_EMBEDDING_PROFILE = "cpu"
$env:LOCALMATHRAG_RERANK_PROFILE = "cpu"
.\LocalMathRAGFlow.exe
```

关闭可选 runtime：

```powershell
$env:LOCALMATHRAG_EMBEDDING_PROFILE = "none"
$env:LOCALMATHRAG_RERANK_PROFILE = "none"
$env:LOCALMATHRAG_VLM_PROFILE = "none"
.\LocalMathRAGFlow.exe
```

CPU 模式会更慢。修改模型或 profile 后，建议从托盘执行一次 `Reset runtime policy`。

## 给开发者

使用源码开发时，从仓库根目录执行：

```powershell
git clone https://github.com/jungp0/localMathRag.git
cd localMathRag
.\scripts\doctor.ps1
.\scripts\bootstrap-ragflow.ps1
.\scripts\dev-up.ps1
```

停止环境：

```powershell
.\scripts\dev-down.ps1
```

运行测试：

```powershell
python .\tests\test_contracts.py
python -m pytest .\tests
```

构建 Windows Release：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build-launcher.ps1
```

输出：

```text
dist\LocalMathRAGFlow-win-x64\
dist\LocalMathRAGFlow-win-x64.zip
```

仓库中的职责边界：

- `docker/`：LocalMathRAGFlow compose 覆盖和二开镜像。
- `extensions/local_math_rag/`：schema、prompt、pipeline 和 API 合约。
- `services/object_service/`：结构化对象、模型 runtime、fallback API 和模型下载 sidecar。
- `services/mtef_parser/`：本地 Equation Editor/MathType MTEF 公式解析组件。
- `launcher/LocalMathRAGFlow/`：Windows 托盘启动器。
- `patches/ragflow/`：所有可提交、可重放的 RAGFlow 二开补丁。
- `third_party/ragflow/`：本地 RAGFlow checkout，始终忽略，不直接依赖其中未生成补丁的修改。
- `data/`：本地持久化数据，始终忽略，不删除、不提交。
- `dist/`：本地构建产物，始终忽略，不作为开发根目录。

进一步了解二开设计请阅读 [docs/ragflow-secondary-development.md](docs/ragflow-secondary-development.md)。修改仓库前必须阅读 [AGENTS.md](AGENTS.md)，特别注意 root resolution、离线下载确认、RAGFlow patch 和中文文件 UTF-8 BOM 规则。

## 许可证与上游

本项目建立在 RAGFlow 之上。使用、分发和二次开发时，请同时遵守本仓库与上游 RAGFlow 的许可证及第三方模型许可证。模型权重不会随本仓库或默认 Release 分发。

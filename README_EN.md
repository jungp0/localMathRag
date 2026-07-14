[简体中文](README.md) | **English**

# LocalMathRAGFlow

> A local knowledge base for engineering documents on Windows. Put PDFs, Word documents, and other files into a folder, then search them, ask questions, inspect citations, and retain as much formula, table, and image context as possible.

LocalMathRAGFlow is an offline-first enhancement of [RAGFlow](https://github.com/infiniflow/ragflow). It is not a website that requires you to upload files to the cloud. It runs as a set of Docker services on your own computer, with a Windows tray launcher that starts, stops, and opens the application.

You do not need to know Python, Docker commands, or programming to use it. Install Docker Desktop and Git, download and extract the Release archive, and double-click the EXE.

## Understand it in one minute

Think of LocalMathRAGFlow as a local document assistant that searches your files and shows where its answers came from.

The usual workflow is:

1. Copy standards, manuals, papers, calculations, or project files into `data\dataset`.
2. Open the built-in `Default` knowledge base and click `Refresh local folder`. Only new, changed, or deleted files are synchronized.
3. Wait for parsing to finish, then ask questions from the search or chat page.
4. Read the answer and verify the cited source passages. Copy or download Markdown when you need a record.

It is useful when you need to:

- Search confidential or internal documents that should not be uploaded to a cloud service.
- Find parameters, clauses, formulas, tables, and surrounding context across many engineering files.
- Create cited summaries, comparisons, checklists, and explanations from your own documents.
- Use a local GGUF language model and reduce everyday network dependency.

This is not a cloud product that includes every AI model after installation. Large model files are not bundled in the Release. Local question answering requires at least one chat model. The first setup may need internet access to obtain RAGFlow source code, Docker images, or models, and the launcher asks before downloading anything.

## Features that are actually implemented

| Feature | What you will see |
| --- | --- |
| One-click Windows launcher | Double-click `LocalMathRAGFlow.exe` to check Docker, start the services, and open the embedded web window. Closing the window leaves the services running; use the tray icon to reopen it. |
| Incremental local-folder sync | `Refresh local folder` in the `Default` knowledge base recursively scans `data\dataset` and processes only new, changed, or deleted files. Office temporary files are ignored. |
| RAGFlow document knowledge base | Keeps RAGFlow's knowledge bases, document parsing, chunk preview, retrieval, chat, citation evidence, and document management features. |
| Engineering formula enhancements | Improves DOCX formula images, Equation Editor/MathType MTEF, WMF previews, and per-formula OCR while retaining paragraph and table context where possible. Low-quality OCR is not presented as a reliable formula. |
| Optional image understanding | A configured VLM can enhance image recognition. No visual model is forced to start or download when one is not configured. |
| Search answers with citations | Search displays stages and progress. The main answer is returned first; mind maps, related questions, and other auxiliary results continue afterward without blocking the answer if they time out. |
| Markdown export | Copy search results as Markdown or download a `.md` file containing the question, answer, and cited passages. |
| Local model discovery and switching | Discovers chat, embedding, rerank, and vision models under `data\models`. The model settings page shows local and recommended models, download progress, switch progress, and readable errors. |
| Adaptive resource planning | Adjusts concurrency and batch sizes according to VRAM, RAM, model size, and context length. Optional model runtimes can start on demand, degrade, or stop when resources are insufficient. |
| Recovery for large knowledge-base jobs | GraphRAG, RAPTOR, table-of-contents extraction, and auxiliary indexing use throttling, leases, watchdogs, and checkpoints to reduce stuck jobs and full restarts after interruption. |
| Offline-compatible APIs | The object service exposes OpenAI-compatible chat proxy, embedding, and rerank endpoints, plus APIs for model status, runtime management, structured objects, and dataset diagnostics. |

Current limitations:

- ASR and TTS model discovery, ports, and Docker profiles are reserved, but their default images are placeholders. Do not treat speech features as ready out of the box.
- `Qwen3-Embedding-0.6B` may remain in the model directory, but `bge-m3` is the current default real embedding model.
- OCR, VLM, GraphRAG, and RAPTOR require additional VRAM, RAM, and processing time. Basic document Q&A does not require all of them.
- AI answers can be wrong. For engineering decisions, open the citations and verify the source. Generated text is not a substitute for an approved standard, calculation, or signed technical document.

## Installation for non-technical users

### 1. Prepare your computer

Required:

- Windows 10/11 x64.
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) using the WSL2 backend. You only need to install it; no Docker command knowledge is required.
- [Git for Windows](https://git-scm.com/download/win), used when RAGFlow source code is obtained for the first time.
- At least 30 GB of free disk space. Reserve 80 GB or more if you plan to keep several local models.

Recommended but optional:

- An NVIDIA GPU with a working driver. 8 GB of VRAM is suitable for smaller quantized models; 12–24 GB provides a more stable experience.
- Without an NVIDIA GPU, RAGFlow can still start and CPU mode is available, but parsing and question answering will be much slower.

You do not need to install .NET, Python, Node.js, MySQL, Redis, or Elasticsearch separately. The Windows Release and Docker environment provide the runtime components.

### 2. Download and extract

Open [GitHub Releases](https://github.com/jungp0/localMathRag/releases/latest) and download:

```text
LocalMathRAGFlow-win-x64.zip
```

Extract the entire archive into a permanent, writable directory with enough space, for example:

```text
D:\LocalMathRAGFlow
```

Do not run the EXE from the archive preview window, and do not copy only the EXE. The `docker`, `scripts`, `services`, `patches`, and other directories are required at runtime.

You can also download it with PowerShell:

```powershell
$Zip = "LocalMathRAGFlow-win-x64.zip"
$Url = "https://github.com/jungp0/localMathRag/releases/latest/download/$Zip"
$InstallDir = "D:\LocalMathRAGFlow"

New-Item -ItemType Directory -Force $InstallDir | Out-Null
Invoke-WebRequest -Uri $Url -OutFile "$env:TEMP\$Zip"
Expand-Archive -Force "$env:TEMP\$Zip" $InstallDir
Start-Process "$InstallDir\LocalMathRAGFlow.exe"
```

### 3. First launch

Double-click `LocalMathRAGFlow.exe`. The first launch is usually slower because the system needs to:

1. Check Docker Desktop. If it is installed but not running, the launcher opens it and waits for the Docker daemon.
2. Look for an existing `third_party\ragflow` checkout. It asks before downloading source code only when RAGFlow is actually missing.
3. Apply this project's patches and build the local `localmathrag/ragflow:dev` image.
4. Start RAGFlow, MySQL, Redis, MinIO, Elasticsearch, and the LocalMathRAGFlow object service.
5. Discover local models and prepare the selected runtimes. If an image is missing, the launcher asks before pulling it.
6. Open the LocalMathRAGFlow window. The default setup is a local single-user interface and does not require a cloud account.

When a download or image-pull confirmation appears:

- Allow: the displayed source code, image, or model is downloaded this time.
- Decline: nothing is downloaded silently. The main RAGFlow service can usually still start, but the related local model capability remains disabled.

If you later start the application from a Release subdirectory or `dist`, the launcher first reuses RAGFlow and `data\models` from an installed parent workspace. A different launch location does not cause duplicate downloads.

## First use: from files to answers

### Step 1: Prepare at least one chat model

Local question answering needs a chat model. Put a GGUF file here:

```text
data\models\your-model.gguf
```

After services restart, the model is discovered automatically and the llama.cpp CUDA runtime is selected by default. You can also inspect local and recommended models on RAGFlow's model settings page. Model download actions display progress and access the internet only after confirmation.

Recommended model layout:

| Purpose | Default discovery location | What happens when it is absent |
| --- | --- | --- |
| Chat and answer generation | `data\models\*.gguf` | RAGFlow can start, but local generative Q&A is unavailable. |
| Real embeddings | `data\models\bge-m3` | The object service keeps an offline-compatible fallback for pipeline continuity. Use `bge-m3` for real semantic retrieval quality. |
| Reranking | `data\models\Qwen3-Reranker-0.6B` | Retrieval still works without the additional precision stage, using a compatible fallback. |
| Image understanding | A configured VLM directory under `data\models` | Text Q&A is unaffected; image enhancement remains disabled. |
| Formula OCR | The recommended Math OCR item on the model settings page | Native formulas and normal parsing still work, but image-formula recognition is limited. |

Models are usually large, so they are not committed to Git or included in the Release. Existing files under `data\models` are reused first. Do not download them again when upgrading the program.

### Step 2: Add documents

Find the LocalMathRAGFlow icon in the Windows notification area, right-click it, and choose `Open dataset`, or open:

```text
data\dataset
```

Copy your files into that directory. Subfolders can be used to organize projects:

```text
data\dataset\
├─ Design standards\
├─ Equipment manuals\
├─ Calculations\
└─ Meeting notes\
```

Supported formats follow the current RAGFlow parsers. Common PDF, DOCX, XLSX, PPTX, TXT, Markdown, HTML, and image files can be imported through RAGFlow. Formula enhancements mainly target DOCX and Office equation scenarios.

### Step 3: Refresh the knowledge base

1. Open RAGFlow.
2. Open the built-in `Default` knowledge base.
3. Click `Refresh local folder`.
4. Wait for new or changed documents to finish parsing.

This does not upload every file again. The system records relative paths, file sizes, modification times, and other state. When the directory is unchanged, cached results are reused and files are not reread or reparsed. If a file is removed from `data\dataset`, refresh again to remove the corresponding knowledge-base document.

### Step 4: Search and ask questions

Use the search page, or select the `Default` knowledge base from a RAGFlow chat or assistant. Good first questions include:

- “What operating temperature does this standard require? Include source citations.”
- “Compare the rated parameters of Model A and Model B in a table.”
- “Explain every symbol in the formula in Chapter 3 and cite the text before and after it.”
- “Find every acceptance criterion and group the results by document and section.”
- “Create an inspection checklist from these files and clearly mark anything that cannot be confirmed from the source.”

Tips:

- Include a document name, equipment name, section, or parameter in the question for more accurate retrieval.
- Read the answer, then open its cited passages and verify the original text.
- Mind maps and related questions are auxiliary results and may continue after the main answer appears.
- Use the copy or download action to save the question, answer, and citations as Markdown.

## Using the tray icon

Closing the main window does not stop the background services. Right-click the tray icon to access:

| Menu item | Action |
| --- | --- |
| `Open RAGFlow` | Opens or focuses the existing window without creating duplicate WebApp windows. |
| `Start services` | Starts the Docker services and opens the status page. |
| `Stop services` | Stops this project's services. |
| `Reset runtime policy` | Clears remembered runtime failures or degradation and schedules models again. Use it after switching models or recovering from resource problems. |
| `Open dataset` | Opens the folder synchronized into the knowledge base. |
| `Open data directory` | Opens the directory containing models, cache, logs, and persistent knowledge-base data. |
| `Exit` | Exits the launcher and asks whether the Docker services should also stop. |

Single-clicking or double-clicking the tray icon opens or focuses the LocalMathRAGFlow window.

## Data location and network behavior

All persistent data is stored under `data` in the installation root and is ignored by Git:

```text
data\
├─ models\      # Local AI models
├─ dataset\     # Source documents waiting to be synchronized
├─ cache\       # Incremental scan and runtime-policy cache
├─ logs\        # Local logs when components create them
└─ launcher\    # Launcher window and WebView2 data
```

Normal parsing, retrieval, and Q&A prefer local endpoints on the Docker network. The project does not require uploading the knowledge base to a cloud service. These actions may access the internet and should be explicitly triggered or confirmed:

- Downloading RAGFlow source code for the first time.
- Pulling a Docker image that is not installed.
- Downloading a recommended model from model settings.
- Using an external model provider that you configure yourself.

Back up the entire `data` directory before an upgrade. Do not delete `data` to fix an ordinary startup issue because it may contain models and persistent knowledge-base data.

## Troubleshooting

### “Docker not found” after double-clicking

Install Docker Desktop, start it once, and confirm that its status is Running. Then reopen LocalMathRAGFlow. The launcher can open an installed Docker Desktop automatically, but it never installs Docker silently.

### The first launch takes a long time

Building the first RAGFlow image can take a while. Check `launcher-tray.log` in the installation directory and make sure Docker Desktop is not waiting for WSL2, network, or license confirmation. `Open RAGFlow` from the tray reopens the current status page.

### The page opens, but questions cannot be answered

Check in this order:

1. Make sure `data\models` contains at least one complete `.gguf` file.
2. Make sure the selected chat model in model settings is the local model.
3. Run `Reset runtime policy` from the tray and try again.
4. Check whether the local chat runtime is listening on port `8080`.

### Documents cannot be found or parsing fails

Make sure the files are under `data\dataset`, then click `Refresh local folder` in the `Default` knowledge base. Wait for the document status to show success before asking questions. Office temporary files beginning with `~$` are ignored automatically.

### `No valid response received`, `Fail to access model`, or `Fail to bind embedding model`

These messages usually mean that the RAGFlow database still contains an old model address, or that a model container is not ready. Run `Reset runtime policy` first, then inspect the model address and these ports:

| Capability | Default port |
| --- | ---: |
| Local LLM | 8080 |
| Embedding | 8081 |
| Rerank | 8082 |
| VLM | 8083 |
| ASR (reserved) | 8084 |
| TTS (reserved) | 8085 |
| Object service / fallback API | 8088 |

Inside RAGFlow, embedding and rerank should normally use `http://localmathrag-object-service:8088/v1` as the compatible endpoint instead of an old host address.

### Switching to CPU or disabling optional models

Set environment variables in PowerShell before launching:

```powershell
$env:LOCALMATHRAG_LLAMA_PROFILE = "cpu"
$env:LOCALMATHRAG_EMBEDDING_PROFILE = "cpu"
$env:LOCALMATHRAG_RERANK_PROFILE = "cpu"
.\LocalMathRAGFlow.exe
```

To disable optional runtimes:

```powershell
$env:LOCALMATHRAG_EMBEDDING_PROFILE = "none"
$env:LOCALMATHRAG_RERANK_PROFILE = "none"
$env:LOCALMATHRAG_VLM_PROFILE = "none"
.\LocalMathRAGFlow.exe
```

CPU mode is slower. After changing a model or profile, run `Reset runtime policy` once from the tray.

## For developers

From the repository root:

```powershell
git clone https://github.com/jungp0/localMathRag.git
cd localMathRag
.\scripts\doctor.ps1
.\scripts\bootstrap-ragflow.ps1
.\scripts\dev-up.ps1
```

Stop the environment:

```powershell
.\scripts\dev-down.ps1
```

Run tests:

```powershell
python .\tests\test_contracts.py
python -m pytest .\tests
```

Build the Windows Release:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build-launcher.ps1
```

Output:

```text
dist\LocalMathRAGFlow-win-x64\
dist\LocalMathRAGFlow-win-x64.zip
```

Repository responsibilities:

- `docker/`: LocalMathRAGFlow Compose overrides and the customized image.
- `extensions/local_math_rag/`: schemas, prompts, pipelines, and API contracts.
- `services/object_service/`: structured objects, model runtimes, fallback APIs, and model-download sidecar.
- `services/mtef_parser/`: local Equation Editor/MathType MTEF parser.
- `launcher/LocalMathRAGFlow/`: Windows tray launcher.
- `patches/ragflow/`: all committed and replayable RAGFlow customizations.
- `third_party/ragflow/`: local RAGFlow checkout, always ignored. Uncommitted customizations inside it must not be relied upon.
- `data/`: local persistent data, always ignored and never deleted or committed.
- `dist/`: local build artifacts, always ignored and never treated as the development root.

Read [docs/ragflow-secondary-development.md](docs/ragflow-secondary-development.md) for the customization design. Before changing the repository, read [AGENTS.md](AGENTS.md), especially the rules for root resolution, offline download confirmation, RAGFlow patches, and UTF-8 BOM encoding.

## License and upstream project

This project is built on RAGFlow. Use, redistribution, and derivative development must follow the licenses of this repository, upstream RAGFlow, and any third-party models. Model weights are not distributed with the repository or the default Release.

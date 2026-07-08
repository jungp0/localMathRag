param(
    [ValidateSet("cpu", "gpu")]
    [string]$Device = "cpu",

    [ValidateSet("auto", "none", "cpu", "cuda")]
    [string]$Llama = "auto",

    [ValidateSet("elasticsearch", "infinity", "opensearch")]
    [string]$DocEngine = "elasticsearch"
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$RagflowDocker = Join-Path $Root "third_party\ragflow\docker"
$Override = Join-Path $Root "docker\docker-compose.localmathrag.yml"

if (!(Test-Path $RagflowDocker)) {
    throw "RAGFlow source is missing. Run .\scripts\bootstrap-ragflow.ps1 first."
}

$env:LOCALMATHRAG_ROOT = $Root.Path
$env:DOC_ENGINE = $DocEngine
$env:DEVICE = $Device
if (-not $env:DOCKER_API_VERSION) {
    $env:DOCKER_API_VERSION = "1.51"
}
if (-not $env:LOCALMATHRAG_CTX_SIZE) {
    $env:LOCALMATHRAG_CTX_SIZE = "8192"
}
if (-not $env:LOCALMATHRAG_LLAMA_PARALLEL) {
    $env:LOCALMATHRAG_LLAMA_PARALLEL = "1"
}
if (-not $env:LOCALMATHRAG_RUNTIME_LAZY) {
    $env:LOCALMATHRAG_RUNTIME_LAZY = "1"
}

$ModelDir = Join-Path $Root "data\models"
$DefaultModel = $null
if (Test-Path $ModelDir) {
    $DefaultModel = Get-ChildItem -Path $ModelDir -Filter "*.gguf" -File | Select-Object -First 1
}
if ($DefaultModel) {
    $env:LOCALMATHRAG_GGUF_MODEL = $DefaultModel.Name
}
$EmbeddingModel = $null
$EmbeddingPath = Join-Path $ModelDir "bge-m3"
if (Test-Path $EmbeddingPath) {
    $EmbeddingModel = "bge-m3"
    $env:LOCALMATHRAG_EMBEDDING_MODEL = $EmbeddingModel
}
elseif (Test-Path (Join-Path $ModelDir "Qwen3-Embedding-0.6B")) {
    $EmbeddingModel = "Qwen3-Embedding-0.6B"
    $env:LOCALMATHRAG_EMBEDDING_MODEL = $EmbeddingModel
    $env:LOCALMATHRAG_EMBEDDING_POOLING = "last-token"
}
$RerankPath = Join-Path $ModelDir "bge-reranker-v2-m3"
if (Test-Path $RerankPath) {
    $env:LOCALMATHRAG_RERANK_MODEL = "bge-reranker-v2-m3"
} elseif (Test-Path (Join-Path $ModelDir "Qwen3-Reranker-0.6B")) {
    $env:LOCALMATHRAG_RERANK_MODEL = "Qwen3-Reranker-0.6B"
}
$VlmPath = Join-Path $ModelDir "Qwen3-VL-4B-Instruct"
if (Test-Path $VlmPath) {
    $env:LOCALMATHRAG_VLM_MODEL = "Qwen3-VL-4B-Instruct"
}
$AsrPath = Join-Path $ModelDir "whisper-large-v3-turbo"
if (Test-Path $AsrPath) {
    $env:LOCALMATHRAG_ASR_MODEL = "whisper-large-v3-turbo"
}
$TtsPath = Join-Path $ModelDir "CosyVoice2-0.5B"
if (Test-Path $TtsPath) {
    $env:LOCALMATHRAG_TTS_MODEL = "CosyVoice2-0.5B"
}
if ($Llama -eq "auto") {
    if ($DefaultModel) {
        $Llama = "cuda"
    }
    else {
        $Llama = "none"
    }
}

function Test-DisabledProfile {
    param([string]$Value)
    return $Value -in @("none", "off", "disabled")
}

function Test-CpuProfile {
    param([string]$Value)
    return $Value -in @("cpu", "tei", "tei-cpu", "local", "local-cpu")
}

function Test-LazyRuntime {
    return $env:LOCALMATHRAG_RUNTIME_LAZY -notin @("0", "false", "False", "FALSE", "no", "off")
}

$profiles = @($DocEngine, $Device)
if ($Llama -eq "cpu") {
    $profiles += "llama-cpp-cpu"
}
elseif ($Llama -eq "cuda") {
    $profiles += "llama-cpp-cuda"
}
if ($EmbeddingModel -and !(Test-DisabledProfile $env:LOCALMATHRAG_EMBEDDING_PROFILE) -and !(Test-LazyRuntime)) {
    if (Test-CpuProfile $env:LOCALMATHRAG_EMBEDDING_PROFILE) {
        $profiles += "embedding-cpu"
    }
    else {
        $profiles += "embedding-cuda"
    }
}
if ($env:LOCALMATHRAG_RERANK_MODEL -and !(Test-DisabledProfile $env:LOCALMATHRAG_RERANK_PROFILE) -and !(Test-LazyRuntime)) {
    if (Test-CpuProfile $env:LOCALMATHRAG_RERANK_PROFILE) {
        $profiles += "rerank-cpu"
    }
    else {
        $profiles += "rerank-cuda"
    }
}
if ($env:LOCALMATHRAG_VLM_MODEL -and !(Test-DisabledProfile $env:LOCALMATHRAG_VLM_PROFILE) -and $env:LOCALMATHRAG_VLM_PROFILE -eq "eager") {
    $profiles += "vlm-cuda"
}
if ($env:LOCALMATHRAG_ASR_MODEL -and !(Test-DisabledProfile $env:LOCALMATHRAG_ASR_PROFILE)) {
    if (Test-CpuProfile $env:LOCALMATHRAG_ASR_PROFILE) {
        $profiles += "asr-local"
    }
    else {
        $profiles += "asr-cuda"
    }
}
if ($env:LOCALMATHRAG_TTS_MODEL -and !(Test-DisabledProfile $env:LOCALMATHRAG_TTS_PROFILE)) {
    if (Test-CpuProfile $env:LOCALMATHRAG_TTS_PROFILE) {
        $profiles += "tts-local"
    }
    else {
        $profiles += "tts-cuda"
    }
}
$env:COMPOSE_PROFILES = ($profiles -join ",")

Push-Location $RagflowDocker
try {
    $composeArgs = @("compose", "-f", "docker-compose.yml", "-f", $Override)
    $composeArgs += @("up", "-d", "--build")
    docker @composeArgs

    if (Test-LazyRuntime) {
        $previousProfiles = $env:COMPOSE_PROFILES
        try {
            if ($EmbeddingModel -and !(Test-DisabledProfile $env:LOCALMATHRAG_EMBEDDING_PROFILE)) {
                $env:COMPOSE_PROFILES = if (Test-CpuProfile $env:LOCALMATHRAG_EMBEDDING_PROFILE) { "embedding-cpu" } else { "embedding-cuda" }
                docker compose -f docker-compose.yml -f $Override create --no-build --pull never localmathrag-embedding
                docker compose -f docker-compose.yml -f $Override stop localmathrag-embedding
            }
            if ($env:LOCALMATHRAG_RERANK_MODEL -and !(Test-DisabledProfile $env:LOCALMATHRAG_RERANK_PROFILE)) {
                $env:COMPOSE_PROFILES = if (Test-CpuProfile $env:LOCALMATHRAG_RERANK_PROFILE) { "rerank-cpu" } else { "rerank-cuda" }
                docker compose -f docker-compose.yml -f $Override create --no-build --pull never localmathrag-rerank
                docker compose -f docker-compose.yml -f $Override stop localmathrag-rerank
            }
        }
        finally {
            $env:COMPOSE_PROFILES = $previousProfiles
        }
    }

    if ($env:LOCALMATHRAG_VLM_MODEL -and !(Test-DisabledProfile $env:LOCALMATHRAG_VLM_PROFILE) -and $env:LOCALMATHRAG_VLM_PROFILE -ne "eager") {
        $vlmImage = if ($env:LOCALMATHRAG_VLM_IMAGE) { $env:LOCALMATHRAG_VLM_IMAGE } else { "vllm/vllm-openai:latest" }
        docker image inspect $vlmImage *> $null
        if ($LASTEXITCODE -eq 0) {
            $previousProfiles = $env:COMPOSE_PROFILES
            try {
                $env:COMPOSE_PROFILES = "vlm-cuda"
                docker compose -f docker-compose.yml -f $Override create --no-build --pull never localmathrag-vlm
                docker compose -f docker-compose.yml -f $Override stop localmathrag-vlm
            }
            finally {
                $env:COMPOSE_PROFILES = $previousProfiles
            }
        }
        else {
            Write-Host "VLM image $vlmImage is not installed; lazy VLM will degrade until the image is installed."
        }
    }
}
finally {
    Pop-Location
}

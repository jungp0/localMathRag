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
$RuntimeConfigFile = Join-Path $Root "data\cache\runtime-config.json"
$MathFontDir = Join-Path $Root "data\fonts\math"
. (Join-Path $PSScriptRoot "localmathrag-resource-plan.ps1")

if (!(Test-Path $RagflowDocker)) {
    throw "RAGFlow source is missing. Run .\scripts\bootstrap-ragflow.ps1 first."
}

function Sync-MathFormulaFonts {
    New-Item -ItemType Directory -Force $MathFontDir | Out-Null
    if ([string]::IsNullOrWhiteSpace($env:WINDIR)) {
        return
    }

    $windowsFontDir = Join-Path $env:WINDIR "Fonts"
    if (!(Test-Path $windowsFontDir)) {
        return
    }

    $fontNames = @(
        "MTEXTRA.TTF",
        "symbol.ttf",
        "times.ttf",
        "timesbd.ttf",
        "timesbi.ttf",
        "timesi.ttf",
        "cambria.ttc",
        "cambriab.ttf",
        "cambriai.ttf",
        "cambriaz.ttf"
    )
    foreach ($fontName in $fontNames) {
        $source = Join-Path $windowsFontDir $fontName
        $target = Join-Path $MathFontDir $fontName
        if ((Test-Path $source) -and !(Test-Path $target)) {
            Copy-Item -LiteralPath $source -Destination $target
        }
    }
}

Sync-MathFormulaFonts

$env:LOCALMATHRAG_ROOT = $Root.Path
$env:DOC_ENGINE = $DocEngine
$env:DEVICE = $Device
if (-not $env:DOCKER_API_VERSION) {
    $env:DOCKER_API_VERSION = "1.51"
}

function Get-EnvInt {
    param(
        [string]$Name,
        [int]$Default
    )

    $value = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrWhiteSpace($value)) {
        return $Default
    }

    $parsed = 0
    if ([int]::TryParse($value, [ref]$parsed)) {
        return $parsed
    }
    return $Default
}

function Get-EnvDouble {
    param(
        [string]$Name,
        [double]$Default
    )

    $value = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrWhiteSpace($value)) {
        return $Default
    }

    $parsed = 0.0
    if ([double]::TryParse($value, [System.Globalization.NumberStyles]::Float, [System.Globalization.CultureInfo]::InvariantCulture, [ref]$parsed)) {
        return $parsed
    }
    return $Default
}

function Get-AvailableMemoryGb {
    try {
        $os = Get-CimInstance Win32_OperatingSystem
        if ($os.FreePhysicalMemory) {
            return [double]$os.FreePhysicalMemory / 1MB
        }
    }
    catch {
        return $null
    }
    return $null
}

function Get-TotalMemoryGb {
    try {
        $computer = Get-CimInstance Win32_ComputerSystem
        if ($computer.TotalPhysicalMemory) {
            return [double]$computer.TotalPhysicalMemory / 1GB
        }
    }
    catch {
        return $null
    }
    return $null
}

function Get-GpuMemoryInfo {
    try {
        $value = nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits 2>$null | Select-Object -First 1
        if ([string]::IsNullOrWhiteSpace($value)) {
            return $null
        }
        $parts = $value.Split(",") | ForEach-Object { $_.Trim() }
        if ($parts.Count -lt 3) {
            return $null
        }
        $totalMb = 0.0
        $usedMb = 0.0
        $freeMb = 0.0
        if (-not [double]::TryParse($parts[0], [System.Globalization.NumberStyles]::Float, [System.Globalization.CultureInfo]::InvariantCulture, [ref]$totalMb)) {
            return $null
        }
        if (-not [double]::TryParse($parts[1], [System.Globalization.NumberStyles]::Float, [System.Globalization.CultureInfo]::InvariantCulture, [ref]$usedMb)) {
            return $null
        }
        if (-not [double]::TryParse($parts[2], [System.Globalization.NumberStyles]::Float, [System.Globalization.CultureInfo]::InvariantCulture, [ref]$freeMb)) {
            return $null
        }
        return [pscustomobject]@{
            TotalGb = $totalMb / 1024.0
            UsedGb = $usedMb / 1024.0
            FreeGb = $freeMb / 1024.0
        }
    }
    catch {
        return $null
    }
    return $null
}

function Get-GpuMemoryGb {
    $info = Get-GpuMemoryInfo
    if ($null -eq $info) {
        return $null
    }
    return $info.FreeGb
}

function Get-GpuIdentity {
    try {
        $value = nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1
        if ([string]::IsNullOrWhiteSpace($value)) {
            return $null
        }
        return $value.Trim()
    }
    catch {
        return $null
    }
    return $null
}

function Get-HostFingerprint {
    $parts = @()
    try {
        $cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
        if ($cpu.Name) {
            $parts += "cpu=$($cpu.Name)"
        }
    }
    catch {
    }
    try {
        $computer = Get-CimInstance Win32_ComputerSystem
        if ($computer.TotalPhysicalMemory) {
            $parts += "mem=$($computer.TotalPhysicalMemory)"
        }
    }
    catch {
    }
    $gpu = Get-GpuIdentity
    if ($gpu) {
        $parts += "gpu=$gpu"
    }
    if ($parts.Count -eq 0) {
        $parts += "machine=$env:COMPUTERNAME"
    }
    $bytes = [System.Text.Encoding]::UTF8.GetBytes(($parts -join "|"))
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        return (($sha.ComputeHash($bytes) | ForEach-Object { $_.ToString("x2") }) -join "")
    }
    finally {
        $sha.Dispose()
    }
}

function Resolve-OptionalRuntimeMaxActive {
    return "auto"
}

function Set-EnvDefault {
    param(
        [string]$Name,
        [object]$Value,
        [string]$Source
    )

    if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($Name))) {
        [Environment]::SetEnvironmentVariable($Name, [string]$Value, "Process")
        if (-not [string]::IsNullOrWhiteSpace($Source)) {
            [Environment]::SetEnvironmentVariable("${Name}_SOURCE", $Source, "Process")
        }
    }
}

function Resolve-ResourceTier {
    param(
        [Nullable[double]]$AvailableGpuGb,
        [Nullable[double]]$AvailableRamGb
    )

    $gpu = if ($null -eq $AvailableGpuGb) { 0.0 } else { [double]$AvailableGpuGb }
    $ram = if ($null -eq $AvailableRamGb) { 0.0 } else { [double]$AvailableRamGb }
    if ($gpu -ge 16.0 -and $ram -ge 48.0) {
        return "high"
    }
    if ($gpu -ge 10.0 -and $ram -ge 32.0) {
        return "balanced"
    }
    if ($gpu -ge 6.0 -and $ram -ge 24.0) {
        return "constrained"
    }
    return "minimal"
}

function Set-ResourceAwareDefaults {
    $gpuInfo = Get-GpuMemoryInfo
    $availableGpuGb = if ($null -eq $gpuInfo) { $null } else { $gpuInfo.FreeGb }
    $totalRamGb = Get-TotalMemoryGb
    $availableRamGb = Get-AvailableMemoryGb
    $tier = Resolve-ResourceTier -AvailableGpuGb $availableGpuGb -AvailableRamGb $availableRamGb

    Set-EnvDefault "LOCALMATHRAG_RESOURCE_TIER" $tier "available-gpu-ram"
    if ($null -ne $gpuInfo) {
        Set-EnvDefault "LOCALMATHRAG_GPU_MEMORY_GB" ([math]::Round($gpuInfo.TotalGb, 2).ToString([System.Globalization.CultureInfo]::InvariantCulture)) "gpu-total"
        Set-EnvDefault "LOCALMATHRAG_GPU_USED_MEMORY_GB" ([math]::Round($gpuInfo.UsedGb, 2).ToString([System.Globalization.CultureInfo]::InvariantCulture)) "gpu-used"
        Set-EnvDefault "LOCALMATHRAG_GPU_AVAILABLE_MEMORY_GB" ([math]::Round($gpuInfo.FreeGb, 2).ToString([System.Globalization.CultureInfo]::InvariantCulture)) "gpu-free"
    }
    if ($null -ne $totalRamGb) {
        Set-EnvDefault "LOCALMATHRAG_RAM_GB" ([math]::Round($totalRamGb, 2).ToString([System.Globalization.CultureInfo]::InvariantCulture)) "ram-total"
    }
    if ($null -ne $availableRamGb) {
        Set-EnvDefault "LOCALMATHRAG_RAM_AVAILABLE_GB" ([math]::Round($availableRamGb, 2).ToString([System.Globalization.CultureInfo]::InvariantCulture)) "ram-available"
    }

    switch ($tier) {
        "high" {
            Set-EnvDefault "LOCALMATHRAG_CTX_SIZE_TARGET" 24576 "resource-aware"
            Set-EnvDefault "LOCALMATHRAG_CTX_SIZE_MAX" 32768 "resource-aware"
            Set-EnvDefault "LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS_MIN" 8192 "resource-aware"
            Set-EnvDefault "LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS_MAX" 24576 "resource-aware"
            Set-EnvDefault "LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS_STEP" 4096 "resource-aware"
        }
        "balanced" {
            Set-EnvDefault "LOCALMATHRAG_CTX_SIZE_TARGET" 16384 "resource-aware"
            Set-EnvDefault "LOCALMATHRAG_CTX_SIZE_MAX" 24576 "resource-aware"
            Set-EnvDefault "LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS_MIN" 8192 "resource-aware"
            Set-EnvDefault "LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS_MAX" 16384 "resource-aware"
            Set-EnvDefault "LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS_STEP" 4096 "resource-aware"
        }
        "constrained" {
            Set-EnvDefault "LOCALMATHRAG_CTX_SIZE_TARGET" 12288 "resource-aware"
            Set-EnvDefault "LOCALMATHRAG_CTX_SIZE_MAX" 16384 "resource-aware"
            Set-EnvDefault "LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS_MIN" 4096 "resource-aware"
            Set-EnvDefault "LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS_MAX" 8192 "resource-aware"
            Set-EnvDefault "LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS_STEP" 2048 "resource-aware"
        }
        default {
            Set-EnvDefault "LOCALMATHRAG_CTX_SIZE_TARGET" 8192 "resource-aware"
            Set-EnvDefault "LOCALMATHRAG_CTX_SIZE_MAX" 12288 "resource-aware"
            Set-EnvDefault "LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS_MIN" 2048 "resource-aware"
            Set-EnvDefault "LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS_MAX" 4096 "resource-aware"
            Set-EnvDefault "LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS_STEP" 1024 "resource-aware"
        }
    }
}

function Read-RuntimeConfig {
    if (!(Test-Path $RuntimeConfigFile)) {
        return $null
    }
    try {
        return Get-Content -Raw -Encoding UTF8 $RuntimeConfigFile | ConvertFrom-Json
    }
    catch {
        Write-Warning "Failed to read runtime config $RuntimeConfigFile; ignoring it."
        return $null
    }
}

function Get-PersistedRerankRuntimeConfig {
    if (-not $env:LOCALMATHRAG_RERANK_MODEL) {
        return $null
    }

    $config = Read-RuntimeConfig
    if ($null -eq $config -or $null -eq $config.rerank) {
        return $null
    }
    if ($config.rerank.model -and $config.rerank.model -ne $env:LOCALMATHRAG_RERANK_MODEL) {
        return $null
    }
    return $config.rerank
}

function Write-RerankRuntimeConfig {
    param(
        [bool]$Disabled,
        [int]$MaxBatchTokens,
        [string]$Reason,
        [string]$Source
    )

    if (-not $env:LOCALMATHRAG_RERANK_MODEL) {
        return
    }

    $cacheDir = Split-Path -Parent $RuntimeConfigFile
    New-Item -ItemType Directory -Force $cacheDir | Out-Null
    $stamp = (Get-Date).ToUniversalTime().ToString("o")
    $profile = if ($env:LOCALMATHRAG_RERANK_PROFILE) { $env:LOCALMATHRAG_RERANK_PROFILE } else { "cuda" }
    $existingConfig = Read-RuntimeConfig
    $payload = [ordered]@{
        version = 1
        updated_at = $stamp
    }
    if ($null -ne $existingConfig -and $null -ne $existingConfig.scheduler) {
        $payload["scheduler"] = $existingConfig.scheduler
    }
    $payload["rerank"] = [ordered]@{
        model = $env:LOCALMATHRAG_RERANK_MODEL
        profile = $profile
        max_batch_tokens = $MaxBatchTokens
        disabled = $Disabled
        reason = $Reason
        source = $Source
        updated_at = $stamp
    }
    $payload | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 $RuntimeConfigFile
}

function Apply-PersistedRerankRuntimeConfig {
    $persisted = Get-PersistedRerankRuntimeConfig
    if ($null -eq $persisted) {
        return
    }

    if (-not $env:LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS -and $persisted.max_batch_tokens) {
        $env:LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS = [string]$persisted.max_batch_tokens
        $env:LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS_SOURCE = "persisted"
    }

    $explicitProfile = [Environment]::GetEnvironmentVariable("LOCALMATHRAG_RERANK_PROFILE")
    if ($persisted.disabled -eq $true -and [string]::IsNullOrWhiteSpace($explicitProfile)) {
        $env:LOCALMATHRAG_RERANK_PROFILE = "none"
        $env:LOCALMATHRAG_RERANK_PERSISTED_DISABLED = "1"
        Write-Warning "Rerank runtime is disabled by persisted config: $($persisted.reason)"
    }
}

function Resolve-LlamaContextSize {
    $minCtx = Get-EnvInt "LOCALMATHRAG_CTX_SIZE_MIN" 8192
    $targetCtx = Get-EnvInt "LOCALMATHRAG_CTX_SIZE_TARGET" 16384
    $maxCtx = Get-EnvInt "LOCALMATHRAG_CTX_SIZE_MAX" 24576
    $step = Get-EnvInt "LOCALMATHRAG_CTX_SIZE_STEP" 1024

    if ($minCtx -lt 2048) {
        $minCtx = 2048
    }
    if ($targetCtx -lt $minCtx) {
        $targetCtx = $minCtx
    }
    if ($maxCtx -lt $minCtx) {
        $maxCtx = $minCtx
    }
    if ($maxCtx -lt $targetCtx) {
        $maxCtx = $targetCtx
    }
    if ($step -lt 256) {
        $step = 256
    }

    $availableGb = Get-AvailableMemoryGb
    if ($null -eq $availableGb) {
        return $maxCtx
    }

    $reserveGb = Get-EnvDouble "LOCALMATHRAG_CTX_MEMORY_RESERVE_GB" 2.0
    $gbPer1k = Get-EnvDouble "LOCALMATHRAG_CTX_MEMORY_GB_PER_1K" 0.25
    if ($gbPer1k -le 0) {
        $gbPer1k = 0.25
    }

    $usableGb = [math]::Max(0.0, $availableGb - $reserveGb)
    $fitRaw = [int]([math]::Floor((($usableGb / $gbPer1k) * 1024.0) / $step) * $step)
    $gpuGb = Get-GpuMemoryGb
    if ($null -ne $gpuGb) {
        $gpuReserveGb = Get-EnvDouble "LOCALMATHRAG_CTX_GPU_MEMORY_RESERVE_GB" 1.0
        $gpuGbPer1k = Get-EnvDouble "LOCALMATHRAG_CTX_GPU_MEMORY_GB_PER_1K" 0.18
        if ($gpuGbPer1k -le 0) {
            $gpuGbPer1k = 0.18
        }
        $gpuUsableGb = [math]::Max(0.0, $gpuGb - $gpuReserveGb)
        $gpuFitRaw = [int]([math]::Floor((($gpuUsableGb / $gpuGbPer1k) * 1024.0) / $step) * $step)
        $fitRaw = [int]([math]::Min($fitRaw, $gpuFitRaw))
    }
    $ctx = [int]([math]::Min($maxCtx, [math]::Max($minCtx, $fitRaw)))

    if ($fitRaw -lt $minCtx) {
        Write-Warning "Available memory supports ctx $fitRaw, below minimum reliable ctx $minCtx; using $minCtx."
    }
    elseif ($ctx -lt $targetCtx) {
        Write-Warning "Available memory supports ctx $ctx, below preferred knowledge ctx $targetCtx."
    }
    return $ctx
}

function Set-SearchTokenBudgetsFromContext {
    if (-not $env:LOCALMATHRAG_CTX_SIZE) {
        return
    }

    $ctx = Get-EnvInt "LOCALMATHRAG_CTX_SIZE" 8192

    if (-not $env:LOCALMATHRAG_SEARCH_DYNAMIC_TOKEN_BUDGET) {
        $env:LOCALMATHRAG_SEARCH_DYNAMIC_TOKEN_BUDGET = "1"
        $env:LOCALMATHRAG_SEARCH_DYNAMIC_TOKEN_BUDGET_SOURCE = "local-summary-dynamic"
    }

    if (-not $env:LOCALMATHRAG_SEARCH_CONTEXT_RESERVED_TOKENS) {
        $reserved = [int]([math]::Max(2048, [math]::Floor($ctx * 0.12)))
        $env:LOCALMATHRAG_SEARCH_CONTEXT_RESERVED_TOKENS = [string]$reserved
        $env:LOCALMATHRAG_SEARCH_CONTEXT_RESERVED_TOKENS_SOURCE = "local-summary-dynamic"
    }

    if (-not $env:LOCALMATHRAG_SEARCH_ANSWER_TOKEN_BUDGET_MAX) {
        $answerMax = [int]([math]::Min(8192, [math]::Max(2048, [math]::Floor($ctx * 0.25))))
        $env:LOCALMATHRAG_SEARCH_ANSWER_TOKEN_BUDGET_MAX = [string]$answerMax
        $env:LOCALMATHRAG_SEARCH_ANSWER_TOKEN_BUDGET_MAX_SOURCE = "local-summary-dynamic"
    }

    if (-not $env:LOCALMATHRAG_SEARCH_KNOWLEDGE_TOKEN_BUDGET_MAX) {
        $knowledgeMax = [int]([math]::Min(16384, [math]::Max(4096, [math]::Floor($ctx * 0.55))))
        $env:LOCALMATHRAG_SEARCH_KNOWLEDGE_TOKEN_BUDGET_MAX = [string]$knowledgeMax
        $env:LOCALMATHRAG_SEARCH_KNOWLEDGE_TOKEN_BUDGET_MAX_SOURCE = "local-summary-dynamic"
    }
}

function Resolve-RerankMaxBatchTokens {
    $minTokens = Get-EnvInt "LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS_MIN" 8192
    $maxTokens = Get-EnvInt "LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS_MAX" 16384
    $step = Get-EnvInt "LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS_STEP" 4096

    if ($minTokens -lt 1024) {
        $minTokens = 1024
    }
    if ($maxTokens -lt $minTokens) {
        $maxTokens = $minTokens
    }
    if ($step -lt 256) {
        $step = 256
    }

    $availableGb = Get-AvailableMemoryGb
    if ($null -eq $availableGb) {
        return $maxTokens
    }

    $reserveGb = Get-EnvDouble "LOCALMATHRAG_RERANK_CONTEXT_MEMORY_RESERVE_GB" 6.0
    $gbPer1k = Get-EnvDouble "LOCALMATHRAG_RERANK_CONTEXT_GB_PER_1K" 0.5
    if ($gbPer1k -le 0) {
        $gbPer1k = 0.5
    }

    $usableGb = [math]::Max(0.0, $availableGb - $reserveGb)
    $fitRaw = [int]([math]::Floor((($usableGb / $gbPer1k) * 1024.0) / $step) * $step)
    $gpuGb = Get-GpuMemoryGb
    if ($null -ne $gpuGb) {
        $gpuReserveGb = Get-EnvDouble "LOCALMATHRAG_RERANK_CONTEXT_GPU_MEMORY_RESERVE_GB" 1.0
        $gpuGbPer1k = Get-EnvDouble "LOCALMATHRAG_RERANK_CONTEXT_GPU_GB_PER_1K" 0.25
        if ($gpuGbPer1k -le 0) {
            $gpuGbPer1k = 0.25
        }
        $gpuUsableGb = [math]::Max(0.0, $gpuGb - $gpuReserveGb)
        $gpuFitRaw = [int]([math]::Floor((($gpuUsableGb / $gpuGbPer1k) * 1024.0) / $step) * $step)
        $fitRaw = [int]([math]::Min($fitRaw, $gpuFitRaw))
    }
    $tokens = [int]([math]::Min($maxTokens, [math]::Max($minTokens, $fitRaw)))

    if ($fitRaw -lt $minTokens) {
        Write-Warning "Available memory supports rerank max-batch-tokens $fitRaw, below minimum reliable value $minTokens; using $minTokens."
    }
    return $tokens
}

Set-ResourceAwareDefaults

if (-not $env:LOCALMATHRAG_CTX_SIZE) {
    $env:LOCALMATHRAG_CTX_SIZE = [string](Resolve-LlamaContextSize)
    $env:LOCALMATHRAG_CTX_SIZE_SOURCE = "dynamic-llm-priority"
}
Set-SearchTokenBudgetsFromContext
if (-not $env:LOCALMATHRAG_HOST_FINGERPRINT) {
    $env:LOCALMATHRAG_HOST_FINGERPRINT = Get-HostFingerprint
}
if (-not $env:LOCALMATHRAG_OPTIONAL_RUNTIME_MAX_ACTIVE) {
    $env:LOCALMATHRAG_OPTIONAL_RUNTIME_MAX_ACTIVE = [string](Resolve-OptionalRuntimeMaxActive)
    $env:LOCALMATHRAG_OPTIONAL_RUNTIME_MAX_ACTIVE_SOURCE = "adaptive-auto"
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
Apply-PersistedRerankRuntimeConfig
if ($env:LOCALMATHRAG_RERANK_MODEL -and -not $env:LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS) {
    $env:LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS = [string](Resolve-RerankMaxBatchTokens)
    $env:LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS_SOURCE = "dynamic"
}
$VlmCandidates = @("Qwen3-VL-4B-Instruct", "Qwen3-VL-8B-Instruct")
$VlmModel = $VlmCandidates | Where-Object { Test-Path (Join-Path $ModelDir $_) } | Select-Object -First 1
if ($VlmModel) {
    $env:LOCALMATHRAG_VLM_MODEL = $VlmModel
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
if ($Llama -eq "cpu") {
    $env:LOCALMATHRAG_LLAMA_RUNTIME_BASE_URL = "http://localmathrag-llama-cpp:8080/v1"
    $env:LOCALMATHRAG_LLAMA_CONTAINER = "docker-localmathrag-llama-cpp-1"
    $env:LOCALMATHRAG_LLAMA_COMPOSE_SERVICE = "localmathrag-llama-cpp"
}
elseif ($Llama -eq "cuda") {
    $env:LOCALMATHRAG_LLAMA_RUNTIME_BASE_URL = "http://localmathrag-llama-cpp-cuda:8080/v1"
    $env:LOCALMATHRAG_LLAMA_CONTAINER = "docker-localmathrag-llama-cpp-cuda-1"
    $env:LOCALMATHRAG_LLAMA_COMPOSE_SERVICE = "localmathrag-llama-cpp-cuda"
}

$modelSizeGb = if ($DefaultModel) { [double]$DefaultModel.Length / 1GB } else { 0.0 }
$gpuInfoForPlan = Get-GpuMemoryInfo
$gpuTotalGbForPlan = if ($null -eq $gpuInfoForPlan) { 0.0 } else { [double]$gpuInfoForPlan.TotalGb }
$availableRamGbForPlan = Get-AvailableMemoryGb
if ($null -eq $availableRamGbForPlan) {
    $availableRamGbForPlan = Get-TotalMemoryGb
}
if ($null -eq $availableRamGbForPlan) {
    $availableRamGbForPlan = 0.0
}
if (-not $env:LOCALMATHRAG_LLAMA_PARALLEL -and -not $DefaultModel) {
    $env:LOCALMATHRAG_LLAMA_PARALLEL = "1"
    $env:LOCALMATHRAG_LLAMA_PARALLEL_SOURCE = "no-local-model-safe-default"
}
elseif (-not $env:LOCALMATHRAG_LLAMA_PARALLEL) {
    $env:LOCALMATHRAG_LLAMA_PARALLEL = [string](Resolve-LlamaParallelSlots `
        -GpuTotalGb $gpuTotalGbForPlan `
        -AvailableRamGb ([double]$availableRamGbForPlan) `
        -ModelSizeGb $modelSizeGb `
        -ContextSize (Get-EnvInt "LOCALMATHRAG_CTX_SIZE" 8192) `
        -LogicalProcessors ([Environment]::ProcessorCount) `
        -UseCuda ($Llama -eq "cuda") `
        -ModelMemoryOverhead (Get-EnvDouble "LOCALMATHRAG_LLAMA_MODEL_MEMORY_OVERHEAD" 1.08) `
        -GpuReserveGb (Get-EnvDouble "LOCALMATHRAG_LLAMA_GPU_RESERVE_GB" 0.75) `
        -RamReserveGb (Get-EnvDouble "LOCALMATHRAG_LLAMA_RAM_RESERVE_GB" 4.0) `
        -ContextMemoryGbPer1K (Get-EnvDouble "LOCALMATHRAG_CTX_GPU_MEMORY_GB_PER_1K" 0.18) `
        -CpuThreadsPerSlot (Get-EnvInt "LOCALMATHRAG_LLAMA_CPU_THREADS_PER_SLOT" 4) `
        -MaximumSlots (Get-EnvInt "LOCALMATHRAG_LLAMA_PARALLEL_MAX" 0))
    $env:LOCALMATHRAG_LLAMA_PARALLEL_SOURCE = "model-context-capacity"
}
$graphRagPlan = Resolve-GraphRagAdaptivePlan `
    -ParallelSlots (Get-EnvInt "LOCALMATHRAG_LLAMA_PARALLEL" 1) `
    -MaximumGleanings (Get-EnvInt "LOCALMATHRAG_GRAPHRAG_MAX_GLEANINGS_MAX" 2)
Set-EnvDefault "LOCALMATHRAG_MODEL_PARALLEL_SLOTS" $graphRagPlan.ChatSlots "llama-capacity"
Set-EnvDefault "LOCALMATHRAG_MAX_CONCURRENT_CHATS" $graphRagPlan.ChatSlots "llama-capacity"
Set-EnvDefault "LOCALMATHRAG_MAX_CONCURRENT_PROCESS_AND_EXTRACT_CHUNK" $graphRagPlan.ChunkSlotsPerDocument "llama-capacity"
Set-EnvDefault "LOCALMATHRAG_GRAPHRAG_MAX_PARALLEL_DOCS" "auto" "chat-chunk-capacity"
Set-EnvDefault "LOCALMATHRAG_GRAPHRAG_MAX_GLEANINGS" $graphRagPlan.MaxGleanings "llama-capacity"
Write-Host (
    "Adaptive GraphRAG plan: llama_parallel={0}, chat_slots={1}, chunk_slots_per_document={2}, document_slots={3}, max_gleanings={4}" -f `
    $env:LOCALMATHRAG_LLAMA_PARALLEL, $graphRagPlan.ChatSlots, $graphRagPlan.ChunkSlotsPerDocument, $graphRagPlan.DocumentSlots, $graphRagPlan.MaxGleanings
)

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

function Remove-StaleRagflowProfileContainers {
    param([string]$TargetService)

    foreach ($service in @("ragflow-cpu", "ragflow-gpu")) {
        if ($service -eq $TargetService) {
            continue
        }

        docker compose --profile cpu --profile gpu -f docker-compose.yml -f $Override rm --stop --force $service
        if ($LASTEXITCODE -ne 0) {
            throw "failed to remove stale $service container with exit code $LASTEXITCODE."
        }
    }
}

function Reset-LazyRuntimeContainer {
    param([string]$Service)

    docker compose -f docker-compose.yml -f $Override rm --stop --force $Service
    if ($LASTEXITCODE -ne 0) {
        throw "failed to remove stale $Service container with exit code $LASTEXITCODE."
    }

    docker compose -f docker-compose.yml -f $Override create --force-recreate --no-build --pull never $Service
    if ($LASTEXITCODE -ne 0) {
        throw "failed to create lazy $Service container with exit code $LASTEXITCODE."
    }

    docker compose -f docker-compose.yml -f $Override stop $Service
    if ($LASTEXITCODE -ne 0) {
        throw "failed to stop lazy $Service container with exit code $LASTEXITCODE."
    }
}

function Get-LoweredRerankMaxBatchTokens {
    $current = Get-EnvInt "LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS" 16384
    $minTokens = Get-EnvInt "LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS_MIN" 8192
    $step = Get-EnvInt "LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS_STEP" 4096

    if ($minTokens -lt 1024) {
        $minTokens = 1024
    }
    if ($step -lt 256) {
        $step = 256
    }
    if ($current -le $minTokens) {
        return $null
    }
    return [int]([math]::Max($minTokens, $current - $step))
}

function Reset-RerankLazyRuntimeContainer {
    $service = "localmathrag-rerank"
    $retryLimit = Get-EnvInt "LOCALMATHRAG_RERANK_CREATE_RETRY_LIMIT" 2
    if ($retryLimit -lt 1) {
        $retryLimit = 1
    }

    docker compose -f docker-compose.yml -f $Override rm --stop --force $service
    if ($LASTEXITCODE -ne 0) {
        throw "failed to remove stale $service container with exit code $LASTEXITCODE."
    }

    for ($attempt = 1; $attempt -le $retryLimit; $attempt++) {
        docker compose -f docker-compose.yml -f $Override create --force-recreate --no-build --pull never $service
        if ($LASTEXITCODE -eq 0) {
            docker compose -f docker-compose.yml -f $Override stop $service
            if ($LASTEXITCODE -ne 0) {
                throw "failed to stop lazy $service container with exit code $LASTEXITCODE."
            }
            Write-RerankRuntimeConfig `
                -Disabled $false `
                -MaxBatchTokens (Get-EnvInt "LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS" 16384) `
                -Reason "lazy rerank container created" `
                -Source $env:LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS_SOURCE
            return
        }

        $nextTokens = Get-LoweredRerankMaxBatchTokens
        if ($null -eq $nextTokens) {
            $reason = "failed to create lazy $service container and rerank max-batch-tokens is already at minimum $env:LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS"
            Write-RerankRuntimeConfig `
                -Disabled $true `
                -MaxBatchTokens (Get-EnvInt "LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS" 16384) `
                -Reason $reason `
                -Source "create-minimum-failed"
            Write-Warning "$reason; rerank will stay disabled until runtime config is reset."
            return
        }
        if ($attempt -ge $retryLimit) {
            $reason = "failed to create lazy $service container after $attempt attempts"
            Write-RerankRuntimeConfig `
                -Disabled $true `
                -MaxBatchTokens (Get-EnvInt "LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS" 16384) `
                -Reason $reason `
                -Source "create-retry-limit"
            Write-Warning "$reason; rerank will stay disabled until runtime config is reset."
            return
        }

        Write-Warning "Failed to create rerank runtime with max-batch-tokens $env:LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS; retrying with $nextTokens."
        $env:LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS = [string]$nextTokens
        $env:LOCALMATHRAG_RERANK_MAX_BATCH_TOKENS_SOURCE = "create-retry"
    }
}

$profiles = @($DocEngine, $Device)
if ($Llama -eq "cpu" -and !(Test-LazyRuntime)) {
    $profiles += "llama-cpp-cpu"
}
elseif ($Llama -eq "cuda" -and !(Test-LazyRuntime)) {
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
    $targetRagflowService = if ($Device -eq "gpu") { "ragflow-gpu" } else { "ragflow-cpu" }
    Remove-StaleRagflowProfileContainers $targetRagflowService

    $composeArgs = @("compose", "-f", "docker-compose.yml", "-f", $Override)
    $composeArgs += @("up", "-d", "--build")
    docker @composeArgs
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose up failed with exit code $LASTEXITCODE."
    }

    if (Test-LazyRuntime) {
        $previousProfiles = $env:COMPOSE_PROFILES
        try {
            if ($Llama -eq "cpu") {
                $env:COMPOSE_PROFILES = "llama-cpp-cpu"
                Reset-LazyRuntimeContainer "localmathrag-llama-cpp"
            }
            elseif ($Llama -eq "cuda") {
                $env:COMPOSE_PROFILES = "llama-cpp-cuda"
                Reset-LazyRuntimeContainer "localmathrag-llama-cpp-cuda"
            }
            if ($EmbeddingModel -and !(Test-DisabledProfile $env:LOCALMATHRAG_EMBEDDING_PROFILE)) {
                $env:COMPOSE_PROFILES = if (Test-CpuProfile $env:LOCALMATHRAG_EMBEDDING_PROFILE) { "embedding-cpu" } else { "embedding-cuda" }
                Reset-LazyRuntimeContainer "localmathrag-embedding"
            }
            if ($env:LOCALMATHRAG_RERANK_MODEL -and !(Test-DisabledProfile $env:LOCALMATHRAG_RERANK_PROFILE)) {
                $env:COMPOSE_PROFILES = if (Test-CpuProfile $env:LOCALMATHRAG_RERANK_PROFILE) { "rerank-cpu" } else { "rerank-cuda" }
                Reset-RerankLazyRuntimeContainer
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
                Reset-LazyRuntimeContainer "localmathrag-vlm"
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

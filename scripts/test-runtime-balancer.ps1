param(
    [string]$ObjectServiceUrl = "http://127.0.0.1:8088",
    [string]$RagflowUrl = "http://127.0.0.1",
    [string]$LlmUrl = "http://127.0.0.1:8080/v1",
    [int]$VisionTimeoutSeconds = 20,
    [switch]$IncludeVlm,
    [switch]$NoCleanup
)

$ErrorActionPreference = "Stop"

function Invoke-Json {
    param(
        [string]$Method,
        [string]$Url,
        [object]$Body = $null,
        [int]$TimeoutSec = 30
    )
    if ($null -eq $Body) {
        return Invoke-RestMethod -Method $Method -Uri $Url -TimeoutSec $TimeoutSec
    }
    return Invoke-RestMethod -Method $Method -Uri $Url -ContentType "application/json" -Body ($Body | ConvertTo-Json -Depth 20) -TimeoutSec $TimeoutSec
}

function Assert-True {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) {
        throw $Message
    }
}

function Get-OptionalRunning {
    param([object]$Status)
    $running = @()
    foreach ($kind in @("embedding", "rerank", "vision", "asr", "tts")) {
        $entry = $Status.endpoint_statuses.PSObject.Properties[$kind].Value
        if ($entry -and $entry.container_running) {
            $running += $kind
        }
    }
    return $running
}

function Assert-Balanced {
    param([string]$Label)
    $status = Invoke-Json -Method GET -Url "$ObjectServiceUrl/v1/models/status" -TimeoutSec 10
    $running = Get-OptionalRunning -Status $status
    Assert-True ($running.Count -le 1) "$Label`: optional runtime balancer violation: $($running -join ', ')"
    return @{
        label = $Label
        running_optional = $running
        endpoint_statuses = $status.endpoint_statuses
    }
}

$results = [ordered]@{}

$ragflow = Invoke-WebRequest -Uri $RagflowUrl -TimeoutSec 15 -UseBasicParsing
Assert-True ($ragflow.StatusCode -lt 500) "RAGFlow is not responding successfully."
$results.ragflow_status = $ragflow.StatusCode

$llm = Invoke-Json -Method GET -Url "$LlmUrl/models" -TimeoutSec 15
Assert-True ($null -ne $llm.data) "LLM /models did not return an OpenAI-compatible model list."
$results.llm_models = $llm.data

$results.initial = Assert-Balanced -Label "initial"
$results.embedding_status_initial = Invoke-Json -Method GET -Url "$ObjectServiceUrl/v1/runtime/status?kind=embedding" -TimeoutSec 10

$embedding = Invoke-Json -Method POST -Url "$ObjectServiceUrl/v1/embeddings" -Body @{
    model = "bge-m3"
    input = @("LocalMathRAGFlow runtime balancer embedding smoke test.")
} -TimeoutSec 300
Assert-True ($embedding.data.Count -ge 1) "Embedding call returned no vectors."
Assert-True ($embedding.data[0].embedding.Count -gt 0) "Embedding vector is empty."
$results.embedding = @{
    model = $embedding.model
    fallback = $embedding.runtime.fallback
    started = $embedding.runtime.started
}
$results.after_embedding = Assert-Balanced -Label "after_embedding"
$results.rerank_status_initial = Invoke-Json -Method GET -Url "$ObjectServiceUrl/v1/runtime/status?kind=rerank" -TimeoutSec 10

$rerank = Invoke-Json -Method POST -Url "$ObjectServiceUrl/v1/rerank" -Body @{
    model = "bge-reranker-v2-m3"
    query = "cuda rerank resource priority"
    documents = @(
        "LLM has highest resource priority.",
        "Optional models are balanced lazily.",
        "Unrelated lexical fallback document."
    )
    top_n = 2
    return_documents = $true
} -TimeoutSec 300
Assert-True ($rerank.results.Count -ge 1) "Rerank call returned no results."
$results.rerank = @{
    model = $rerank.model
    fallback = $rerank.runtime.fallback
    started = $rerank.runtime.started
}
$results.after_rerank = Assert-Balanced -Label "after_rerank"

if ($IncludeVlm) {
    $vision = Invoke-Json -Method POST -Url "$ObjectServiceUrl/v1/runtime/ensure" -Body @{
        kind = "vision"
        timeout_seconds = $VisionTimeoutSeconds
        start = $true
    } -TimeoutSec ($VisionTimeoutSeconds + 20)
} else {
    $vision = Invoke-Json -Method POST -Url "$ObjectServiceUrl/v1/runtime/ensure" -Body @{
        kind = "vision"
        timeout_seconds = 1
        start = $false
    } -TimeoutSec 15
}
Assert-True (($vision.ready -eq $true) -or ($vision.degraded -eq $true)) "Vision runtime ensure returned an invalid state."
$results.vision = @{
    ready = $vision.ready
    degraded = $vision.degraded
    started = $vision.started
    reason = $vision.reason
}
$results.after_vision = Assert-Balanced -Label "after_vision"

if (-not $NoCleanup) {
    foreach ($kind in @("embedding", "rerank", "vision")) {
        $results["cleanup_$kind"] = Invoke-Json -Method POST -Url "$ObjectServiceUrl/v1/runtime/stop" -Body @{ kind = $kind } -TimeoutSec 30
    }
    $results.after_cleanup = Assert-Balanced -Label "after_cleanup"
}

$results | ConvertTo-Json -Depth 20

param(
    [switch]$Volumes
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$RagflowDocker = Join-Path $Root "third_party\ragflow\docker"
$Override = Join-Path $Root "docker\docker-compose.localmathrag.yml"

if (!(Test-Path $RagflowDocker)) {
    throw "RAGFlow source is missing. Nothing to stop."
}

$env:LOCALMATHRAG_ROOT = $Root.Path

Push-Location $RagflowDocker
try {
    $profiles = @(
        "elasticsearch",
        "infinity",
        "opensearch",
        "oceanbase",
        "seekdb",
        "deepdoc",
        "sandbox",
        "cpu",
        "gpu",
        "llama-cpp-cpu",
        "llama-cpp-cuda",
        "embedding-cpu",
        "embedding-cuda",
        "rerank-cpu",
        "rerank-cuda",
        "vlm-cuda",
        "vlm-local",
        "asr-cuda",
        "asr-local",
        "tts-cuda",
        "tts-local"
    )
    $composeArgs = @("compose")
    foreach ($profile in $profiles) {
        $composeArgs += @("--profile", $profile)
    }
    $composeArgs += @("-f", "docker-compose.yml", "-f", $Override, "down", "--remove-orphans")
    if ($Volumes) {
        $composeArgs += "-v"
    }
    docker @composeArgs
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose down failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}

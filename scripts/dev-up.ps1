param(
    [ValidateSet("cpu", "gpu")]
    [string]$Device = "cpu",

    [ValidateSet("none", "cpu", "cuda")]
    [string]$Llama = "none",

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

$profiles = @($DocEngine, $Device)
if ($Llama -eq "cpu") {
    $profiles += "llama-cpp-cpu"
}
elseif ($Llama -eq "cuda") {
    $profiles += "llama-cpp-cuda"
}
$env:COMPOSE_PROFILES = ($profiles -join ",")

Push-Location $RagflowDocker
try {
    docker compose -f docker-compose.yml -f $Override up -d
}
finally {
    Pop-Location
}

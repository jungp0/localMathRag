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
    if ($Volumes) {
        docker compose -f docker-compose.yml -f $Override down -v
    }
    else {
        docker compose -f docker-compose.yml -f $Override down
    }
}
finally {
    Pop-Location
}

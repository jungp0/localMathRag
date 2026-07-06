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
$WebDistOverride = Join-Path $Root "docker\docker-compose.webdist.yml"
$WebDist = Join-Path $Root "third_party\ragflow\web\dist"

if (!(Test-Path $RagflowDocker)) {
    throw "RAGFlow source is missing. Run .\scripts\bootstrap-ragflow.ps1 first."
}

$env:LOCALMATHRAG_ROOT = $Root.Path
$env:DOC_ENGINE = $DocEngine
$env:DEVICE = $Device

$ModelDir = Join-Path $Root "data\models"
$DefaultModel = $null
if (Test-Path $ModelDir) {
    $DefaultModel = Get-ChildItem -Path $ModelDir -Filter "*.gguf" -File | Select-Object -First 1
}
if ($DefaultModel) {
    $env:LOCALMATHRAG_GGUF_MODEL = $DefaultModel.Name
}
if ($Llama -eq "auto") {
    if ($DefaultModel) {
        $Llama = "cpu"
    }
    else {
        $Llama = "none"
    }
}

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
    $composeArgs = @("compose", "-f", "docker-compose.yml", "-f", $Override)
    if ((Test-Path $WebDist) -and (Test-Path $WebDistOverride)) {
        $composeArgs += @("-f", $WebDistOverride)
    }
    $composeArgs += @("up", "-d", "--build")
    docker @composeArgs
}
finally {
    Pop-Location
}

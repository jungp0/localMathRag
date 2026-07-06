$ErrorActionPreference = "Continue"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$RagflowDocker = Join-Path $Root "third_party\ragflow\docker"
$Model = Join-Path $Root "data\models\Qwen3-8B-Q4_K_M.gguf"

Write-Host "LocalMathRAGFlow doctor"
Write-Host "Root: $Root"

if (Get-Command docker -ErrorAction SilentlyContinue) {
    docker --version
}
else {
    Write-Warning "Docker was not found in PATH."
}

if (Get-Command git -ErrorAction SilentlyContinue) {
    git --version
}
else {
    Write-Warning "Git was not found in PATH."
}

if (Test-Path $RagflowDocker) {
    Write-Host "RAGFlow source: present"
}
else {
    Write-Warning "RAGFlow source is missing. Run .\scripts\bootstrap-ragflow.ps1"
}

if (Test-Path $Model) {
    $sizeGb = [math]::Round((Get-Item $Model).Length / 1GB, 2)
    Write-Host "GGUF model: present ($sizeGb GB)"
}
else {
    Write-Warning "GGUF model not found at $Model"
}

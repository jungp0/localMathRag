param(
    [string]$Repo = "https://github.com/infiniflow/ragflow.git",
    [string]$Ref = "v0.26.3"
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Target = Join-Path $Root "third_party\ragflow"

if (Test-Path $Target) {
    Write-Host "RAGFlow source already exists: $Target"
    Push-Location $Target
    git -c safe.directory=$Target fetch --tags --depth 1 origin $Ref
    git -c safe.directory=$Target checkout $Ref
    Pop-Location
    & (Join-Path $PSScriptRoot "apply-ragflow-patches.ps1")
    exit 0
}

New-Item -ItemType Directory -Force (Join-Path $Root "third_party") | Out-Null
git clone --depth 1 --branch $Ref $Repo $Target
& (Join-Path $PSScriptRoot "apply-ragflow-patches.ps1")
Write-Host "RAGFlow source cloned to $Target"

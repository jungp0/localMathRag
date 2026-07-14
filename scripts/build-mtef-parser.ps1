param(
    [string]$GoImage = "golang:1.24-bookworm"
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Source = Join-Path $Root "services\mtef_parser"
$Output = Join-Path $Source "localmathrag-mtef-linux-amd64"

docker run --rm `
    -v "${Source}:/src" `
    -w /src `
    $GoImage `
    go mod tidy

if ($LASTEXITCODE -ne 0) {
    throw "MTEF parser dependency download failed with exit code $LASTEXITCODE."
}

docker run --rm `
    -e CGO_ENABLED=0 `
    -v "${Source}:/src" `
    -w /src `
    $GoImage `
    go build -trimpath "-ldflags=-s -w" -o /src/localmathrag-mtef-linux-amd64 .

if ($LASTEXITCODE -ne 0) {
    throw "MTEF parser build failed with exit code $LASTEXITCODE."
}

Write-Host "Built $Output"

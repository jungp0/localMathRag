$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$WebRoot = Join-Path $Root "third_party\ragflow\web"

& (Join-Path $PSScriptRoot "apply-ragflow-patches.ps1")

if (!(Test-Path $WebRoot)) {
    throw "RAGFlow web source is missing: $WebRoot"
}

Push-Location $WebRoot
try {
    if (!$env:NODE_OPTIONS) {
        $env:NODE_OPTIONS = "--max-old-space-size=8192"
    }

    if (Get-Command pnpm -ErrorAction SilentlyContinue) {
        pnpm install --frozen-lockfile --ignore-scripts
        pnpm build
    }
    elseif (Get-Command corepack -ErrorAction SilentlyContinue) {
        corepack enable
        corepack pnpm install --frozen-lockfile --ignore-scripts
        corepack pnpm build
    }
    else {
        throw "pnpm/corepack not found. Install Node.js 20+ or run through the release build environment."
    }
}
finally {
    Pop-Location
}

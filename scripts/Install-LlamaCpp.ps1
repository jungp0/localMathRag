param(
  [ValidateSet("cuda-12.4", "cpu", "vulkan")]
  [string]$Flavor = "cuda-12.4",
  [string]$Root = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = "Stop"

function Get-AssetPatterns {
  param([string]$SelectedFlavor)
  switch ($SelectedFlavor) {
    "cuda-12.4" {
      return @(
        "^llama-.+-bin-win-cuda-12\.4-x64\.zip$",
        "^cudart-llama-bin-win-cuda-12\.4-x64\.zip$"
      )
    }
    "cpu" {
      return @("^llama-.+-bin-win-cpu-x64\.zip$")
    }
    "vulkan" {
      return @("^llama-.+-bin-win-vulkan-x64\.zip$")
    }
  }
}

$release = Invoke-RestMethod `
  -Uri "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest" `
  -Headers @{ "User-Agent" = "localMathRag-installer" }

$tag = [string]$release.tag_name
$downloadDir = Join-Path $Root "data\runtime\downloads"
$installDir = Join-Path $Root "data\runtime\llama.cpp\$tag-$Flavor"
New-Item -ItemType Directory -Force -Path $downloadDir | Out-Null
New-Item -ItemType Directory -Force -Path $installDir | Out-Null

foreach ($pattern in Get-AssetPatterns $Flavor) {
  $asset = $release.assets | Where-Object { $_.name -match $pattern } | Select-Object -First 1
  if (-not $asset) {
    throw "No llama.cpp release asset matched pattern: $pattern"
  }
  $target = Join-Path $downloadDir $asset.name
  if (-not (Test-Path $target)) {
    Write-Host "Downloading $($asset.name)"
    curl.exe -L --fail --output $target $asset.browser_download_url
  } else {
    Write-Host "Using cached $($asset.name)"
  }
  Expand-Archive -Path $target -DestinationPath $installDir -Force
}

$server = Get-ChildItem -Path $installDir -Recurse -Filter llama-server.exe |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

if (-not $server) {
  throw "llama-server.exe was not found after extraction."
}

Write-Host "Installed llama.cpp $tag ($Flavor)"
Write-Host "llama-server: $($server.FullName)"

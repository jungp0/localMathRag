$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Project = Join-Path $Root "launcher\LocalMathRAGFlow\LocalMathRAGFlow.csproj"
$Out = Join-Path $Root "dist\LocalMathRAGFlow-win-x64"

if (Test-Path $Out) {
    Remove-Item -Recurse -Force $Out
}

dotnet publish $Project `
    -c Release `
    -r win-x64 `
    --self-contained true `
    -p:PublishSingleFile=true `
    -p:IncludeNativeLibrariesForSelfExtract=true `
    -o $Out

if ($LASTEXITCODE -ne 0) {
    throw "Launcher publish failed with exit code $LASTEXITCODE."
}

$LauncherExe = Join-Path $Out "LocalMathRAGFlow.exe"
if (-not (Test-Path -LiteralPath $LauncherExe)) {
    throw "Launcher publish completed without the expected executable: $LauncherExe"
}

Copy-Item -Recurse -Force (Join-Path $Root "docker") $Out
Copy-Item -Recurse -Force (Join-Path $Root "scripts") $Out
Copy-Item -Recurse -Force (Join-Path $Root "extensions") $Out
Copy-Item -Recurse -Force (Join-Path $Root "services") $Out
Copy-Item -Recurse -Force (Join-Path $Root "patches") $Out
Copy-Item -Force (Join-Path $Root "README.md") $Out

Get-ChildItem -Path $Out -Recurse -Directory -Filter "__pycache__" | ForEach-Object {
    Remove-Item -LiteralPath $_.FullName -Recurse -Force
}
Get-ChildItem -Path $Out -Recurse -File -Include "*.pyc", "*.pyo" | ForEach-Object {
    Remove-Item -LiteralPath $_.FullName -Force
}

New-Item -ItemType Directory -Force (Join-Path $Out "data\models") | Out-Null
New-Item -ItemType Directory -Force (Join-Path $Out "data\dataset") | Out-Null
New-Item -ItemType Directory -Force (Join-Path $Out "data\cache") | Out-Null
New-Item -ItemType Directory -Force (Join-Path $Out "third_party") | Out-Null

Compress-Archive -Force -Path (Join-Path $Out "*") -DestinationPath (Join-Path $Root "dist\LocalMathRAGFlow-win-x64.zip")
Write-Host "Built $Out"

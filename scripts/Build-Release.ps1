param(
  [string]$Root = (Split-Path -Parent $PSScriptRoot),
  [string]$PythonCommand = "python",
  [string]$Configuration = "Release",
  [switch]$SkipPythonInstall
)

$ErrorActionPreference = "Stop"

function Resolve-Python {
  param([string]$ConfiguredCommand)
  if ($ConfiguredCommand -and (Test-Path $ConfiguredCommand)) {
    return (Resolve-Path $ConfiguredCommand).Path
  }
  $command = Get-Command $ConfiguredCommand -ErrorAction SilentlyContinue
  if ($command -and $command.Source -and $command.Source -notlike "*\WindowsApps\python*.exe") {
    return $command.Source
  }
  $py = Get-Command "py" -ErrorAction SilentlyContinue
  if ($py -and $py.Source) {
    return $py.Source
  }
  throw "Python was not found. Pass -PythonCommand C:\path\to\python.exe."
}

$Root = (Resolve-Path $Root).Path
$distDir = Join-Path $Root "dist"
$backendDist = Join-Path $distDir "backend-build"
$launcherPublish = Join-Path $distDir "launcher-publish"
$packageDir = Join-Path $distDir "LocalMathRAG-win-x64"
$zipPath = Join-Path $distDir "LocalMathRAG-win-x64.zip"
$venvDir = Join-Path $distDir "build-venv"

New-Item -ItemType Directory -Force -Path $distDir | Out-Null
if (Test-Path $backendDist) { Remove-Item -LiteralPath $backendDist -Recurse -Force }
if (Test-Path $launcherPublish) { Remove-Item -LiteralPath $launcherPublish -Recurse -Force }
if (Test-Path $packageDir) { Remove-Item -LiteralPath $packageDir -Recurse -Force }
if (Test-Path $zipPath) { Remove-Item -LiteralPath $zipPath -Force }

$basePython = Resolve-Python $PythonCommand
if (-not $SkipPythonInstall) {
  if (Test-Path $venvDir) { Remove-Item -LiteralPath $venvDir -Recurse -Force }
  & $basePython -m venv $venvDir
  $python = Join-Path $venvDir "Scripts\python.exe"
  & $python -m pip install --upgrade pip
  & $python -m pip install -e "$Root[packaging]"
} else {
  $python = $basePython
}

& $python -m PyInstaller `
  --noconfirm `
  --clean `
  --onefile `
  --name lookup-tool-server `
  --distpath $backendDist `
  --workpath (Join-Path $distDir "pyinstaller-work") `
  --specpath (Join-Path $distDir "pyinstaller-spec") `
  --paths (Join-Path $Root "src") `
  --add-data "$(Join-Path $Root "src\lookup_tool\static");lookup_tool\static" `
  (Join-Path $Root "tools\lookup_tool_server.py")

dotnet publish (Join-Path $Root "launcher\LocalMathRAG\LocalMathRAG.csproj") `
  -c $Configuration `
  -r win-x64 `
  --self-contained true `
  -p:PublishSingleFile=true `
  -p:PublishTrimmed=false `
  -o $launcherPublish

New-Item -ItemType Directory -Force -Path $packageDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $packageDir "backend") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $packageDir "data") | Out-Null

Copy-Item -LiteralPath (Join-Path $launcherPublish "LocalMathRAG.exe") -Destination (Join-Path $packageDir "LocalMathRAG.exe")
Copy-Item -LiteralPath (Join-Path $backendDist "lookup-tool-server.exe") -Destination (Join-Path $packageDir "backend\lookup-tool-server.exe")
Copy-Item -LiteralPath (Join-Path $Root "README.md") -Destination (Join-Path $packageDir "README.md")
Copy-Item -LiteralPath (Join-Path $Root "LICENSE") -Destination (Join-Path $packageDir "LICENSE")
Copy-Item -LiteralPath (Join-Path $Root "config.example.toml") -Destination (Join-Path $packageDir "config.example.toml")

Compress-Archive -Path (Join-Path $packageDir "*") -DestinationPath $zipPath -Force
Write-Host "Release package: $zipPath"

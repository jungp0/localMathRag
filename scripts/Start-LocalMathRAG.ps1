param(
  [string]$Root = (Split-Path -Parent $PSScriptRoot),
  [string]$HostAddress = "127.0.0.1",
  [int]$WebPort = 8765,
  [int]$LlamaPort = 8080,
  [string]$PythonCommand = "python",
  [string]$ModelPath = "",
  [string]$LlamaServerPath = "",
  [int]$ContextSize = 8192,
  [int]$GpuLayers = 999,
  [switch]$SkipModel,
  [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

function Test-TcpPort {
  param([string]$Address, [int]$Port)
  $client = [System.Net.Sockets.TcpClient]::new()
  try {
    $task = $client.ConnectAsync($Address, $Port)
    if (-not $task.Wait(1000)) {
      return $false
    }
    return $client.Connected
  } catch {
    return $false
  } finally {
    $client.Dispose()
  }
}

function Wait-HttpOk {
  param([string]$Url, [int]$Seconds = 120)
  $deadline = (Get-Date).AddSeconds($Seconds)
  while ((Get-Date) -lt $deadline) {
    try {
      Invoke-RestMethod -Uri $Url -TimeoutSec 5 | Out-Null
      return $true
    } catch {
      Start-Sleep -Seconds 2
    }
  }
  return $false
}

function Resolve-LlamaServer {
  param([string]$RootPath, [string]$ConfiguredPath)
  if ($ConfiguredPath -and (Test-Path $ConfiguredPath)) {
    return (Resolve-Path $ConfiguredPath).Path
  }
  $runtime = Join-Path $RootPath "data\runtime\llama.cpp"
  if (-not (Test-Path $runtime)) {
    return $null
  }
  $server = Get-ChildItem -Path $runtime -Recurse -Filter llama-server.exe |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
  if ($server) {
    return $server.FullName
  }
  return $null
}

function Write-Utf8File {
  param([string]$Path, [string]$Content)
  $encoding = [System.Text.UTF8Encoding]::new($false)
  [System.IO.File]::WriteAllText($Path, $Content, $encoding)
}

$Root = (Resolve-Path $Root).Path
$dataDir = Join-Path $Root "data"
New-Item -ItemType Directory -Force -Path $dataDir | Out-Null

if (-not $ModelPath) {
  $ModelPath = Join-Path $Root "data\models\Qwen3-8B-Q4_K_M.gguf"
}
if ($ModelPath -and (Test-Path $ModelPath)) {
  $ModelPath = (Resolve-Path $ModelPath).Path
}

if (-not $SkipModel) {
  $LlamaServerPath = Resolve-LlamaServer $Root $LlamaServerPath
  if (-not $LlamaServerPath) {
    throw "llama-server.exe was not found. Run scripts\Install-LlamaCpp.ps1 first."
  }
  if (-not (Test-Path $ModelPath)) {
    throw "Model file was not found: $ModelPath"
  }
  if (-not (Test-TcpPort $HostAddress $LlamaPort)) {
    $llamaDir = Split-Path -Parent $LlamaServerPath
    $llamaCmd = Join-Path $dataDir "run-llama-server-$LlamaPort.cmd"
    $llamaLog = Join-Path $dataDir "llama-server-$LlamaPort.out.log"
    $llamaErr = Join-Path $dataDir "llama-server-$LlamaPort.err.log"
    Write-Utf8File $llamaCmd @"
@echo off
set PATH=$llamaDir;%PATH%
cd /d $Root
"$LlamaServerPath" -m "$ModelPath" --host $HostAddress --port $LlamaPort --ctx-size $ContextSize --n-gpu-layers $GpuLayers --reasoning off >> "$llamaLog" 2>> "$llamaErr"
"@
    Start-Process -FilePath "cmd.exe" -ArgumentList "/c start `"LocalMathRAG llama-server`" /MIN `"$llamaCmd`"" -WindowStyle Hidden
  }
  if (-not (Wait-HttpOk "http://$HostAddress`:$LlamaPort/v1/models" 180)) {
    throw "llama-server did not become ready. See data\llama-server-$LlamaPort.err.log"
  }
}

if (-not (Test-TcpPort $HostAddress $WebPort)) {
  $webCmd = Join-Path $dataDir "run-webapp-$WebPort.cmd"
  $webLog = Join-Path $dataDir "webapp-$WebPort.out.log"
  $webErr = Join-Path $dataDir "webapp-$WebPort.err.log"
  $srcPath = Join-Path $Root "src"
  Write-Utf8File $webCmd @"
@echo off
set PYTHONPATH=$srcPath;%PYTHONPATH%
cd /d $Root
"$PythonCommand" -m lookup_tool.cli serve --host $HostAddress --port $WebPort >> "$webLog" 2>> "$webErr"
"@
  Start-Process -FilePath "cmd.exe" -ArgumentList "/c start `"LocalMathRAG WebApp`" /MIN `"$webCmd`"" -WindowStyle Hidden
}

if (-not (Wait-HttpOk "http://$HostAddress`:$WebPort/api/kbs" 120)) {
  throw "WebApp did not become ready. See data\webapp-$WebPort.err.log"
}

if (-not $SkipModel) {
  $payload = @{
    enabled = $true
    provider = "openai_compatible"
    base_url = "http://$HostAddress`:$LlamaPort/v1"
    model = $ModelPath
    temperature = 0.2
    timeout_seconds = 180
    local_models_dir = Split-Path -Parent $ModelPath
    local_model_path = $ModelPath
    llama_server_path = $LlamaServerPath
  } | ConvertTo-Json -Depth 6
  $bytes = [System.Text.Encoding]::UTF8.GetBytes($payload)
  Invoke-RestMethod `
    -Uri "http://$HostAddress`:$WebPort/api/model/settings" `
    -Method Patch `
    -Body $bytes `
    -ContentType "application/json; charset=utf-8" `
    -TimeoutSec 20 | Out-Null
}

$url = "http://$HostAddress`:$WebPort"
Write-Host "LocalMathRAG WebApp: $url"
Write-Host "llama.cpp endpoint: http://$HostAddress`:$LlamaPort/v1"
if (-not $NoBrowser) {
  Start-Process $url
}

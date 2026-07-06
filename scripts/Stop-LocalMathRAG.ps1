param(
  [int[]]$Ports = @(8765, 8080)
)

$ErrorActionPreference = "Stop"

foreach ($port in $Ports) {
  $connections = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
  foreach ($processId in ($connections | Select-Object -ExpandProperty OwningProcess -Unique)) {
    if (-not $processId) {
      continue
    }
    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if ($process) {
      Write-Host "Stopping port $port process $processId ($($process.ProcessName))"
      Stop-Process -Id $processId -Force
    }
  }
}

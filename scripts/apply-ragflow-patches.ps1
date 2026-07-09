$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$RagflowRoot = Join-Path $Root "third_party\ragflow"
$PatchDir = Join-Path $Root "patches\ragflow"

if (!(Test-Path $RagflowRoot)) {
    throw "RAGFlow source is missing: $RagflowRoot"
}

if (!(Test-Path $PatchDir)) {
    Write-Host "No RAGFlow patch directory found: $PatchDir"
    exit 0
}

function Invoke-Git {
    param(
        [Parameter(Mandatory = $true)]
        [string[]] $Arguments
    )

    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & git @Arguments 2>&1 | ForEach-Object { $_.ToString() }
        return @{
            Code = $LASTEXITCODE
            Output = ($output -join [Environment]::NewLine)
        }
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
}

function Test-PatchContentAlreadyApplied {
    param(
        [Parameter(Mandatory = $true)]
        [string] $PatchPath,
        [Parameter(Mandatory = $true)]
        [string] $RepoRoot
    )

    $samplesByPath = @{}
    $currentPath = $null
    foreach ($line in Get-Content $PatchPath) {
        if ($line -match '^\+\+\+ b/(.+)$') {
            $currentPath = $Matches[1]
            if (!$samplesByPath.ContainsKey($currentPath)) {
                $samplesByPath[$currentPath] = New-Object System.Collections.Generic.List[string]
            }
            continue
        }
        if ($line.StartsWith("diff --git ")) {
            $currentPath = $null
            continue
        }
        if (!$currentPath -or !$line.StartsWith("+") -or $line.StartsWith("+++")) {
            continue
        }

        $sample = $line.Substring(1).Trim()
        if ($sample.Length -lt 12) {
            continue
        }
        if ($samplesByPath[$currentPath].Count -lt 120) {
            $samplesByPath[$currentPath].Add($sample)
        }
    }

    $total = 0
    $found = 0
    foreach ($relativePath in $samplesByPath.Keys) {
        $targetPath = Join-Path $RepoRoot $relativePath
        if (!(Test-Path $targetPath)) {
            continue
        }
        $content = Get-Content $targetPath -Raw
        foreach ($sample in $samplesByPath[$relativePath]) {
            $total += 1
            if ($content.Contains($sample)) {
                $found += 1
            }
        }
    }

    if ($total -lt 3) {
        return $false
    }
    return (($found / $total) -ge 0.55)
}

$patches = Get-ChildItem $PatchDir -Filter "*.patch" | Sort-Object Name
foreach ($patch in $patches) {
    Push-Location $RagflowRoot
    try {
        $baseArgs = @("-c", "safe.directory=$RagflowRoot", "apply", "--unidiff-zero")
        $check = Invoke-Git ($baseArgs + @("--check", $patch.FullName))
        if ($check.Code -eq 0) {
            $apply = Invoke-Git ($baseArgs + @($patch.FullName))
            if ($apply.Code -ne 0) {
                throw "Cannot apply $($patch.Name): $($apply.Output)"
            }
            Write-Host "Applied $($patch.Name)"
            continue
        }

        $reverseCheck = Invoke-Git ($baseArgs + @("--reverse", "--check", "--ignore-whitespace", $patch.FullName))
        if ($reverseCheck.Code -eq 0) {
            Write-Host "Already applied $($patch.Name)"
            continue
        }

        if (Test-PatchContentAlreadyApplied -PatchPath $patch.FullName -RepoRoot $RagflowRoot) {
            Write-Host "Already applied $($patch.Name) (content match)"
            continue
        }

        throw "Cannot apply $($patch.Name): $($check.Output)"
    }
    finally {
        Pop-Location
    }
}

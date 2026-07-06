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

$patches = Get-ChildItem $PatchDir -Filter "*.patch" | Sort-Object Name
foreach ($patch in $patches) {
    Push-Location $RagflowRoot
    try {
        $baseArgs = @("-c", "safe.directory=$RagflowRoot", "apply")
        $check = Invoke-Git ($baseArgs + @("--check", $patch.FullName))
        if ($check.Code -eq 0) {
            $apply = Invoke-Git ($baseArgs + @($patch.FullName))
            if ($apply.Code -ne 0) {
                throw "Cannot apply $($patch.Name): $($apply.Output)"
            }
            Write-Host "Applied $($patch.Name)"
            continue
        }

        $reverseCheck = Invoke-Git ($baseArgs + @("--reverse", "--check", $patch.FullName))
        if ($reverseCheck.Code -eq 0) {
            Write-Host "Already applied $($patch.Name)"
            continue
        }

        throw "Cannot apply $($patch.Name): $($check.Output)"
    }
    finally {
        Pop-Location
    }
}

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "..\scripts\localmathrag-resource-plan.ps1")

function Assert-Equal {
    param($Expected, $Actual, [string]$Message)
    if ($Expected -ne $Actual) {
        throw "$Message expected=$Expected actual=$Actual"
    }
}

$constrained = Resolve-LlamaParallelSlots `
    -GpuTotalGb 8 `
    -AvailableRamGb 16 `
    -ModelSizeGb 4.8 `
    -ContextSize 8192 `
    -LogicalProcessors 16 `
    -UseCuda $true
Assert-Equal 1 $constrained "constrained CUDA slots"

$capable = Resolve-LlamaParallelSlots `
    -GpuTotalGb 24 `
    -AvailableRamGb 64 `
    -ModelSizeGb 4.8 `
    -ContextSize 8192 `
    -LogicalProcessors 32 `
    -UseCuda $true
if ($capable -le $constrained) {
    throw "capable hardware should receive more slots"
}

$capped = Resolve-LlamaParallelSlots `
    -GpuTotalGb 24 `
    -AvailableRamGb 64 `
    -ModelSizeGb 4.8 `
    -ContextSize 8192 `
    -LogicalProcessors 32 `
    -UseCuda $true `
    -MaximumSlots 2
Assert-Equal 2 $capped "explicit maximum slots"

$singlePlan = Resolve-GraphRagAdaptivePlan -ParallelSlots 1
Assert-Equal 1 $singlePlan.ChatSlots "single chat slots"
Assert-Equal 1 $singlePlan.ChunkSlotsPerDocument "single chunk slots"
Assert-Equal 1 $singlePlan.DocumentSlots "single document slots"
Assert-Equal 0 $singlePlan.MaxGleanings "single gleanings"

$capablePlan = Resolve-GraphRagAdaptivePlan -ParallelSlots 8
Assert-Equal 8 $capablePlan.ChatSlots "capable chat slots"
Assert-Equal 3 $capablePlan.ChunkSlotsPerDocument "capable chunk slots"
Assert-Equal 3 $capablePlan.DocumentSlots "capable document slots"
Assert-Equal 2 $capablePlan.MaxGleanings "capable gleanings"

Write-Host "resource plan tests passed"

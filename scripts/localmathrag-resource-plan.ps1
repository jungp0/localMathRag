function Resolve-LlamaParallelSlots {
    param(
        [double]$GpuTotalGb,
        [double]$AvailableRamGb,
        [double]$ModelSizeGb,
        [int]$ContextSize,
        [int]$LogicalProcessors,
        [bool]$UseCuda,
        [double]$ModelMemoryOverhead = 1.08,
        [double]$GpuReserveGb = 0.75,
        [double]$RamReserveGb = 4.0,
        [double]$ContextMemoryGbPer1K = 0.18,
        [int]$CpuThreadsPerSlot = 4,
        [int]$MaximumSlots = 0
    )

    $safeContext = [math]::Max(2048, $ContextSize)
    $contextGbPerSlot = [math]::Max(0.125, ($safeContext / 1024.0) * [math]::Max(0.001, $ContextMemoryGbPer1K))
    $residentModelGb = [math]::Max(0.0, $ModelSizeGb) * [math]::Max(1.0, $ModelMemoryOverhead)
    $cpuCapacity = [math]::Max(1, [math]::Floor([math]::Max(1, $LogicalProcessors) / [math]::Max(1, $CpuThreadsPerSlot)))

    $ramCapacity = [math]::Max(1, [math]::Floor(([math]::Max(0.0, $AvailableRamGb - $RamReserveGb - $residentModelGb) / $contextGbPerSlot)))
    $memoryCapacity = $ramCapacity
    if ($UseCuda -and $GpuTotalGb -gt 0) {
        $gpuCapacity = [math]::Max(1, [math]::Floor(([math]::Max(0.0, $GpuTotalGb - $GpuReserveGb - $residentModelGb) / $contextGbPerSlot)))
        $memoryCapacity = [math]::Min($memoryCapacity, $gpuCapacity)
    }

    $slots = [int]([math]::Max(1, [math]::Min($cpuCapacity, $memoryCapacity)))
    if ($MaximumSlots -gt 0) {
        $slots = [int]([math]::Min($slots, $MaximumSlots))
    }
    return $slots
}

function Resolve-GraphRagAdaptivePlan {
    param(
        [int]$ParallelSlots,
        [int]$MaximumGleanings = 2
    )

    $chatSlots = [math]::Max(1, $ParallelSlots)
    $chunkSlots = [int]([math]::Ceiling([math]::Sqrt($chatSlots)))
    $documentSlots = [int]([math]::Ceiling($chatSlots / [double]$chunkSlots))
    $gleanings = if ($chatSlots -le 1) { 0 } else { [int]([math]::Floor([math]::Log($chatSlots, 2))) }
    $gleanings = [int]([math]::Min([math]::Max(0, $MaximumGleanings), [math]::Max(0, $gleanings)))

    return [pscustomobject]@{
        ChatSlots = [int]$chatSlots
        ChunkSlotsPerDocument = $chunkSlots
        DocumentSlots = $documentSlots
        MaxGleanings = $gleanings
    }
}

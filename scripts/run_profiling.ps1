param(
    [ValidateSet('benchmark','profile')]
    [string]$Mode = 'benchmark',
    [string]$FramesDir = 'data\processed\msvd\val\frames\0lh_UWF9ZP4_21_26',
    [string]$Device = 'cuda',
    [string]$Checkpoint = 'checkpoints\msvd_mapper_finetune_v2.pt',
    [string]$BatchSizes = '1',
    [int]$Warmup = 10,
    [int]$Iters = 50,
    [int]$MaxNewTokens = 24,
    [string]$OutputDir = 'reports',
    [switch]$UseAutocastFp16,
    [switch]$AllowOnlineModelChecks
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$CheckScript = Join-Path $PSScriptRoot "check_project_env.ps1"
& powershell -ExecutionPolicy Bypass -File $CheckScript

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$env:PYTHONPATH = $Root

if (-not $AllowOnlineModelChecks) {
    $env:HF_HUB_OFFLINE = '1'
    $env:TRANSFORMERS_OFFLINE = '1'
    Write-Host '[ENV] Hugging Face offline mode enabled for local-model profiling'
}

if (!(Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

if ($Mode -eq 'benchmark') {
    $IsBatchSweep = $BatchSizes.Contains(',')
    $BenchmarkScript = if ($UseAutocastFp16) { 'core\scripts\benchmark_baseline_fp16.py' } else { 'core\scripts\benchmark_baseline.py' }
    $CsvName = if ($UseAutocastFp16) {
        if ($IsBatchSweep) { 'benchmark_bs_comparison_fp16.csv' } else { 'baseline_fp16_iterations.csv' }
    } else {
        if ($IsBatchSweep) { 'benchmark_bs_comparison.csv' } else { 'baseline_iterations.csv' }
    }
    $JsonName = if ($UseAutocastFp16) {
        if ($IsBatchSweep) { 'benchmark_bs_summary_fp16.json' } else { 'baseline_fp16_summary.json' }
    } else {
        if ($IsBatchSweep) { 'benchmark_bs_summary.json' } else { 'baseline_summary.json' }
    }

    $ArgsList = @(
        $BenchmarkScript,
        '--frames-dir', $FramesDir,
        '--ckpt', $Checkpoint,
        '--device', $Device,
        '--batch-sizes', $BatchSizes,
        '--warmup', $Warmup,
        '--iters', $Iters,
        '--max-new-tokens', $MaxNewTokens,
        '--export-csv', (Join-Path $OutputDir $CsvName),
        '--export-json', (Join-Path $OutputDir $JsonName)
    )
    $PrecisionTag = if ($UseAutocastFp16) { 'fp16_autocast' } else { 'fp32_baseline' }
    Write-Host "[RUN] benchmark baseline (precision=$PrecisionTag, batch_sizes=$BatchSizes)"
    & $VenvPython @ArgsList
}
else {
    $ProfileWarmup = [Math]::Min($Warmup, 3)
    $ArgsList = @(
        'core\scripts\profile_nsight.py',
        '--frames-dir', $FramesDir,
        '--ckpt', $Checkpoint,
        '--device', $Device,
        '--warmup', $ProfileWarmup,
        '--max-new-tokens', $MaxNewTokens,
        '--export-json', (Join-Path $OutputDir 'profile_once.json')
    )
    Write-Host "[RUN] single nsight profile"
    & $VenvPython @ArgsList
}


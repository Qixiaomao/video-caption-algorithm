param(
    [ValidateSet('benchmark','profile')]
    [string]$Mode = 'benchmark',
    [string]$FramesDir = 'data\processed\msvd\val\frames\0lh_UWF9ZP4_21_26',
    [string]$Device = 'cuda',
    [string]$Checkpoint = 'checkpoints\msvd_mapper_finetune_v2.pt',
    [int]$Warmup = 10,
    [int]$Iters = 50,
    [int]$MaxNewTokens = 24,
    [string]$OutputDir = 'reports',
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
    $ArgsList = @(
        'core\scripts\benchmark_baseline.py',
        '--frames-dir', $FramesDir,
        '--ckpt', $Checkpoint,
        '--device', $Device,
        '--warmup', $Warmup,
        '--iters', $Iters,
        '--max-new-tokens', $MaxNewTokens,
        '--export-csv', (Join-Path $OutputDir 'baseline_iterations.csv'),
        '--export-json', (Join-Path $OutputDir 'baseline_summary.json')
    )
    Write-Host "[RUN] benchmark baseline"
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


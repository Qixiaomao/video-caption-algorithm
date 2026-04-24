param(
    [string]$FramesDir = 'data\processed\msvd\val\frames\0lh_UWF9ZP4_21_26',
    [string]$Device = 'cuda',
    [string]$Checkpoint = 'checkpoints\msvd_mapper_finetune_v2.pt',
    [int]$Warmup = 3,
    [int]$MaxNewTokens = 24,
    [string]$OutputDir = 'reports',
    [string]$NsightSystemsExe = 'D:\programs\NsightSystems\target-windows-x64\nsys.exe',
    [string]$Trace = 'cuda,nvtx',
    [string]$CaptureRange = 'Inference_Once',
    [switch]$UseNvtxCaptureRange,
    [switch]$UseAutocastFp16,
    [switch]$AllowOnlineModelChecks
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Resolve-NsysCliExe {
    param([string]$InputPath)

    if ([string]::IsNullOrWhiteSpace($InputPath)) {
        return $null
    }

    if (Test-Path $InputPath) {
        $leaf = Split-Path $InputPath -Leaf
        if ($leaf -ieq 'nsys-ui.exe') {
            $candidate = Join-Path (Split-Path $InputPath -Parent) 'nsys.exe'
            if (Test-Path $candidate) {
                Write-Host "[WARN] Received nsys-ui.exe. Switching to CLI binary: $candidate"
                return $candidate
            }

            $targetSibling = Join-Path (Split-Path (Split-Path $InputPath -Parent) -Parent) 'target-windows-x64\nsys.exe'
            if (Test-Path $targetSibling) {
                Write-Host "[WARN] Received nsys-ui.exe. Switching to CLI binary: $targetSibling"
                return $targetSibling
            }
        }
        return $InputPath
    }

    return $InputPath
}

$CheckScript = Join-Path $PSScriptRoot 'check_project_env.ps1'
& powershell -ExecutionPolicy Bypass -File $CheckScript

$VenvPython = Join-Path $Root '.venv\Scripts\python.exe'
$env:PYTHONPATH = $Root

if (-not $AllowOnlineModelChecks) {
    $env:HF_HUB_OFFLINE = '1'
    $env:TRANSFORMERS_OFFLINE = '1'
    Write-Host '[ENV] Hugging Face offline mode enabled for local-model profiling'
}

if ($NsightSystemsExe -match 'nsys-ui\.exe$') {
    Write-Host "[WARN] NsightSystemsExe points to nsys-ui.exe (GUI). CLI args require nsys.exe."
}
$NsightSystemsExe = Resolve-NsysCliExe -InputPath $NsightSystemsExe

if (!(Test-Path $NsightSystemsExe)) {
    Write-Error "Nsight Systems executable not found: $NsightSystemsExe"
}
if (!(Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

$ProfileStem = if ($UseAutocastFp16) { 'profile_once_fp16' } else { 'profile_once' }
$NsysOutputBase = Join-Path $OutputDir $ProfileStem
$ProfileJson = Join-Path $OutputDir ($ProfileStem + '.json')
$ExpectedReports = @(
    "$NsysOutputBase.nsys-rep",
    "$NsysOutputBase.qdrep",
    "$NsysOutputBase.nsys-ui.nsys-rep"
)

$ArgsList = @(
    'profile',
    "--trace=$Trace",
    '--sample=none',
    '--cuda-memory-usage=true',
    '--force-overwrite=true',
    '--wait=primary',
    '-o', $NsysOutputBase,
    $VenvPython,
    'core\scripts\profile_nsight.py',
    '--frames-dir', $FramesDir,
    '--ckpt', $Checkpoint,
    '--device', $Device,
    '--warmup', $Warmup,
    '--max-new-tokens', $MaxNewTokens,
    '--export-json', $ProfileJson
)

if ($UseNvtxCaptureRange) {
    $ArgsList = @(
        'profile',
        "--trace=$Trace",
        '--sample=none',
        '--cuda-memory-usage=true',
        '--force-overwrite=true',
        '--wait=primary',
        '--capture-range=nvtx',
        '--capture-range-end=stop-shutdown',
        '--nvtx-capture', $CaptureRange,
        '-o', $NsysOutputBase,
        $VenvPython,
        'core\scripts\profile_nsight.py',
        '--frames-dir', $FramesDir,
        '--ckpt', $Checkpoint,
        '--device', $Device,
        '--warmup', $Warmup,
        '--max-new-tokens', $MaxNewTokens,
        '--export-json', $ProfileJson
    )
}

if ($UseAutocastFp16) {
    $ArgsList += '--use-autocast-fp16'
}

$PrecisionTag = if ($UseAutocastFp16) { 'fp16_autocast' } else { 'fp32' }
if ($UseNvtxCaptureRange) {
    Write-Host "[RUN] Nsight Systems profile (NVTX capture range: $CaptureRange, precision=$PrecisionTag)"
}
else {
    Write-Host "[RUN] Nsight Systems profile (whole process capture, precision=$PrecisionTag)"
}

& $NsightSystemsExe @ArgsList
if ($LASTEXITCODE -ne 0) {
    throw "Nsight Systems profiling failed with exit code $LASTEXITCODE"
}

$GeneratedReport = $ExpectedReports | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $GeneratedReport) {
    throw "Nsight Systems finished but did not generate a report file under base '$NsysOutputBase'. Try the default whole-process mode first, then inspect Nsight installation/runtime state."
}

Write-Host "[DONE] Nsight Systems report: $GeneratedReport"
Write-Host "[DONE] Profile JSON: $ProfileJson"

param(
    [ValidateSet('ViT_Encoder','GPT2_Decoder_Step')]
    [string]$Target = 'GPT2_Decoder_Step',
    [string]$FramesDir = 'data\processed\msvd\val\frames\0lh_UWF9ZP4_21_26',
    [string]$Device = 'cuda',
    [string]$Checkpoint = 'checkpoints\msvd_mapper_finetune_v2.pt',
    [int]$MaxNewTokens = 24,
    [string]$OutputDir = 'reports',
    [string]$NsightComputeBat = 'D:\programs\Nsight Computer\ncu.bat',
    [switch]$AllowOnlineModelChecks
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$CheckScript = Join-Path $PSScriptRoot 'check_project_env.ps1'
& powershell -ExecutionPolicy Bypass -File $CheckScript

$VenvPython = Join-Path $Root '.venv\Scripts\python.exe'
$env:PYTHONPATH = $Root

if (-not $AllowOnlineModelChecks) {
    $env:HF_HUB_OFFLINE = '1'
    $env:TRANSFORMERS_OFFLINE = '1'
    Write-Host '[ENV] Hugging Face offline mode enabled for local-model profiling'
}

if (!(Test-Path $NsightComputeBat)) {
    Write-Error "Nsight Compute launcher not found: $NsightComputeBat"
}
if (!(Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

if ($Target -eq 'ViT_Encoder') {
    $OutputBase = Join-Path $OutputDir 'ncu_vit_encoder'
    $MetaJson = Join-Path $OutputDir 'ncu_vit_encoder_meta.json'
}
else {
    $OutputBase = Join-Path $OutputDir 'ncu_gpt2_decoder'
    $MetaJson = Join-Path $OutputDir 'ncu_gpt2_decoder_meta.json'
}

$ArgsList = @(
    '--set', 'roofline',
    '--target-processes', 'all',
    '--nvtx',
    '--nvtx-include', $Target,
    '-o', $OutputBase,
    $VenvPython,
    'core\scripts\profile_nsight.py',
    '--frames-dir', $FramesDir,
    '--ckpt', $Checkpoint,
    '--device', $Device,
    '--max-new-tokens', $MaxNewTokens,
    '--export-json', $MetaJson
)

Write-Host "[RUN] Nsight Compute target: $Target"
& $NsightComputeBat @ArgsList

Write-Host "[DONE] Nsight Compute output base: $OutputBase"
Write-Host "[DONE] Metadata JSON: $MetaJson"

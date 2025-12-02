# compare_caption.ps1  (robust, ASCII-safe)
param(
  [string]$Video = ".\some_test.mp4",
  [string]$Stage1Ckpt = ".\checkpoints\msvd_joint_stage1.pt",
  [string]$Stage2Ckpt = ".\checkpoints\msvd_decoder_stage2.pt",
  [int]$NumFrames = 8,
  [int]$PrefixLen = 4
)

$ErrorActionPreference = "Stop"
$env:TF_CPP_MIN_LOG_LEVEL = "2"   # reduce TensorFlow log noise

# 1) Repo root & PYTHONPATH
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$env:PYTHONPATH = $Root

# 2) Log file
$ts = Get-Date -Format yyyyMMdd_HHmmss
if (!(Test-Path "logs")) { New-Item -ItemType Directory -Path "logs" | Out-Null }
$log = "logs\compare_${ts}.txt"

function Invoke-CaptionGen {
  param(
    [string]$Tag,
    [string]$CkptPath
  )

  Write-Host ""
  Write-Host "===== $Tag ====="
  if (!(Test-Path $CkptPath)) {
    Write-Host "WARN: checkpoint not found -> $CkptPath"
    return
  }

  $argsList = @(
    "-m","scripts.generate_caption",
    "--video",$Video,
    "--ckpt",$CkptPath,
    "--cond_mode","prefix",
    "--prefix_len",$PrefixLen,
    "--num_frames",$NumFrames,
    "--max_new_tokens",40,
    "--min_new_tokens",8,
    "--num_beams",5,
    "--temperature",0.8,
    "--top_p",0.9,
    "--no_repeat_ngram_size",3,
    "--repetition_penalty",1.15,
    "--prompt","Describe the video in one simple sentence:"
  )

  # allow stderr without stopping the script
  $prev = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  $out = & python @argsList 2>&1
  $ErrorActionPreference = $prev

  # echo to console
  $out | Write-Host
  # append to log
  "`n===== $Tag =====" | Out-File -FilePath $log -Append -Encoding UTF8
  $out | Out-File -FilePath $log -Append -Encoding UTF8
}

Invoke-CaptionGen -Tag "Stage-1 (joint)" -CkptPath $Stage1Ckpt
Invoke-CaptionGen -Tag "Stage-2 (decoder ft, ViT frozen)" -CkptPath $Stage2Ckpt

Write-Host ""
Write-Host "Compare done. Log saved to: $log"
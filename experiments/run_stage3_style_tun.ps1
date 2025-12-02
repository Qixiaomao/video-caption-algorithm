# run_stage3_style_tune.ps1
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$env:PYTHONPATH = $Root

$ts = Get-Date -Format yyyyMMdd_HHmmss
if (!(Test-Path "logs")) { New-Item -ItemType Directory -Path "logs" | Out-Null }

# 数据与输出
$ANN_TRAIN = ".\data\processed\msvd\train\annotations.json"
$ANN_VAL   = ".\data\processed\msvd\val\annotations.json"
$SAVE_DIR  = ".\checkpoints\gpt2_lm_stage3"

# 训练超参（可改）
$GPT2_NAME = "gpt2"
$EPOCHS    = 6      # 5~8 比较合适
$BATCH     = 16
$MAX_LEN   = 64
$LR        = "2e-5"
$WARMUP    = 0.06

Write-Host "[INFO] Saving to $SAVE_DIR"
Write-Host "[INFO] Logging to logs/style_${ts}.log"
Write-Host ""

$argsList = @(
  "-m", "src.cli.train_decoder_only",
  "--ann_train", $ANN_TRAIN,
  "--ann_val",   $ANN_VAL,
  "--gpt2_name", $GPT2_NAME,
  "--epochs",    $EPOCHS,
  "--batch_size", $BATCH,
  "--max_len",   $MAX_LEN,
  "--lr",        $LR,
  "--warmup_ratio", $WARMUP,
  "--save_dir",  $SAVE_DIR
)

& python @argsList 2>&1 | Tee-Object -FilePath "logs/style_${ts}.log" -Append

Write-Host ""
Write-Host "[DONE] Stage-3 LM finetune finished:"
Write-Host "  - $SAVE_DIR"
Write-Host "  - $SAVE_DIR\best"
Write-Host "  - logs/style_${ts}.log"
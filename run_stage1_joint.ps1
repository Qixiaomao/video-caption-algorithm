# run_stage1_joint.ps1
$ErrorActionPreference = "Stop"

# ---- 项目根目录（脚本所在目录）----
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

# ---- 让 Python 找到 src 包 ----
$env:PYTHONPATH = $Root

# ---- 时间戳 & 目录 ----
$ts = Get-Date -Format yyyyMMdd_HHmmss
if (!(Test-Path "logs")) { New-Item -ItemType Directory -Path "logs" | Out-Null }

# ---- 参数（按需改）----
$ANN_TRAIN = ".\data\processed\msvd\train\annotations.json"
$ANN_VAL   = ".\data\processed\msvd\val\annotations.json"
$RUN_DIR   = ".\runs\${ts}_joint_stage1"
$CKPT_DIR  = ".\checkpoints"
$CKPT_NAME = "msvd_joint_stage1.pt"

$MODEL     = "vit"
$VIT_NAME  = "vit_base_patch16_224"
$BATCH     = 2
$NUM_FRAME = 8
$IMG       = 224
$MAX_LEN   = 48
$LR        = "3e-4"
$EPOCHS    = 10
$VAL_EVERY = 100

Write-Host "[INFO] PYTHONPATH=$env:PYTHONPATH"
Write-Host "[INFO] RUN_DIR=$RUN_DIR"
Write-Host "[INFO] Logging to logs/train_${ts}.log"
Write-Host ""

# ---- 组装参数（避免引号转义问题）----
$argsList = @(
    "-m", "src.cli.train_full",
    "--model", $MODEL,
    "--vit_name", $VIT_NAME,
    "--ann_train", $ANN_TRAIN,
    "--ann_val",   $ANN_VAL,
    "--batch_size", $BATCH,
    "--num_frame",  $NUM_FRAME,
    "--image_size", $IMG,
    "--max_len",    $MAX_LEN,
    "--lr",         $LR,
    "--epochs",     $EPOCHS,
    "--val_every",  $VAL_EVERY,
    "--shuffle",
    "--run_dir",    $RUN_DIR,
    "--ckpt_dir",   $CKPT_DIR,
    "--ckpt_name",  $CKPT_NAME
)

# ---- 运行并同时写日志 ----
& python @argsList 2>&1 | Tee-Object -FilePath "logs/train_${ts}.log" -Append

Write-Host ""
Write-Host "[DONE] Training finished. Check:"
Write-Host "  - logs/train_${ts}.log"
Write-Host "  - $RUN_DIR"
Write-Host "  - $CKPT_DIR\$CKPT_NAME"
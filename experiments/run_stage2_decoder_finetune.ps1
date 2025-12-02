# run_stage2_decoder_finetune.ps1
$ErrorActionPreference = "Stop"

# ---- 项目根目录 ----
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
$RUN_DIR   = ".\runs\${ts}_decoder_finetune"
$CKPT_DIR  = ".\checkpoints"
$CKPT_NAME = "msvd_decoder_stage2.pt"

# ✅ 模型参数（注意这里冻结 ViT）
$MODEL     = "vit"
$VIT_NAME  = "vit_base_patch16_224"
$BATCH     = 2
$NUM_FRAME = 8
$IMG       = 224
$MAX_LEN   = 48
$LR        = "2e-4"         # 🔁 解码器推荐稍低学习率
$EPOCHS    = 5              # 5~8 就够，太多会过拟合
$VAL_EVERY = 100

Write-Host "[INFO] PYTHONPATH=$env:PYTHONPATH"
Write-Host "[INFO] RUN_DIR=$RUN_DIR"
Write-Host "[INFO] Logging to logs/train_${ts}.log"
Write-Host ""

# ---- 组装命令 ----
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
    "--run_dir",    $RUN_DIR,
    "--ckpt_dir",   $CKPT_DIR,
    "--ckpt_name",  $CKPT_NAME,
    "--freeze_vit"             # ✅ 冻结 ViT，只训练解码器
)

# ---- 运行并保存日志 ----
& python @argsList 2>&1 | Tee-Object -FilePath "logs/train_${ts}.log" -Append

Write-Host ""
Write-Host "[DONE] Decoder fine-tuning finished. Check:"
Write-Host "  - logs/train_${ts}.log"
Write-Host "  - $RUN_DIR"
Write-Host "  - $CKPT_DIR\$CKPT_NAME"
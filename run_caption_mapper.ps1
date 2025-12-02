$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$env:PYTHONPATH = $Root

$ANN_TRAIN = ".\data\processed\msvd\train\annotations.json"
$ANN_VAL   = ".\data\processed\msvd\val\annotations.json"
$RUN_DIR   = ".\runs\mapper_ft"
$CKPT_DIR  = ".\checkpoints"
$CKPT_NAME = "msvd_mapper_finetune.pt"

$argsList = @(
  "-m", "src.cli.train_caption_mapper",
  "--ann_train", $ANN_TRAIN, "--ann_val", $ANN_VAL,
  "--batch_size", 8, "--num_frame", 8, "--image_size", 224,
  "--max_len", 48,
  "--epochs", 1, "--val_every", 20,
  "--freeze_vit",
  "--unfreeze_gpt2_last", 0,   # 显存紧就先设 0
  "--run_dir", $RUN_DIR,
  "--ckpt_dir", $CKPT_DIR, "--ckpt_name", $CKPT_NAME
)

& python @argsList
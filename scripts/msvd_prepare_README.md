
# msvd_prepare.py 使用说明

目录约定
```
data/
 ├─ raw/
 │   └─ msvd/
 │       ├─ YouTubeClips/         # 视频（如有）
 │       ├─ captions.csv|json     # 其中之一
 └─ processed/
     └─ msvd/                     # 脚本会写入 train/val/test/annotations.json
```

快速开始
```bash
python scripts/msvd_prepare.py   --raw_dir data/raw/msvd   --out_dir data/processed/msvd   --format grouped   --video_subdir YouTubeClips
```
- `--format grouped` 输出每个视频一条（包含 captions 列表）。若 DataLoader 需要一条 caption 一条，选 `flat`。
- 默认按 8/1/1 划分，可用 `--train_ratio/--val_ratio/--test_ratio` 调整。

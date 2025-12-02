# src/cli/infer_once.py
# 新增命令行推理（单条视频或者多条视频）
# python -m src.cli.infer_once --frames_dir .\data\processed\msvd\frames\0lh_UWF9ZP4_21_26 --stage all

import argparse, json
from pathlib import Path
from inference import generate_caption_stage1, generate_caption_stage2, generate_caption_stage3

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True, help="Dir containing extracted frames, e.g., .../video_001/")
    ap.add_argument("--stage", choices=["1","2","3","all"], default="all")
    args = ap.parse_args()

    fdir = Path(args.frames_dir)
    if args.stage in ["1","all"]:
        print("[S1]", generate_caption_stage1(fdir))
    if args.stage in ["2","all"]:
        print("[S2]", generate_caption_stage2(fdir))
    if args.stage in ["3","all"]:
        print("[S3]", generate_caption_stage3(fdir))

if __name__ == "__main__":
    main()

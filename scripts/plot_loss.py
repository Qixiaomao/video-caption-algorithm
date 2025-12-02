#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用 loss 曲线绘图脚本
用法:
    python -m scripts.plot_loss runs/dryrun_day7/events.csv --out loss_curve.png
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="训练日志 events.csv 文件路径")
    ap.add_argument("--out", default=None, help="输出 PNG 文件名，默认和 csv 同名但后缀改为 .png")
    args = ap.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise SystemExit(f"[ERROR] file not found: {csv_path}")

    df = pd.read_csv(csv_path, header=None, names=["step", "loss"])
    if df.empty:
        raise SystemExit(f"[ERROR] empty file: {csv_path}")

    # 绘制曲线
    plt.figure(figsize=(8,5))
    plt.plot(df["step"], df["loss"], marker="o", linestyle="-")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"Training Loss Curve\n{csv_path.name}")
    plt.grid(True)
    plt.tight_layout()

    # 输出文件路径
    out_path = Path(args.out) if args.out else csv_path.with_suffix(".png")
    plt.savefig(out_path)
    print(f"[OK] saved figure -> {out_path}")

if __name__ == "__main__":
    main()

'''
# 生成 loss_curve.png
python -m scripts.plot_loss runs/dryrun_day7/events.csv

# 或指定输出文件名
python -m scripts.plot_loss runs/dryrun_day7/events.csv --out runs/dryrun_day7/loss_day7.png

'''
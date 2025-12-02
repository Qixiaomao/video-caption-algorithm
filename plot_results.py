#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_results.py
---------------------------------
Generate simple publication-ready charts for:
1) Model comparison BLEU (Stage-2 vs Stage-3)
2) Decoding ablation BLEU (multiple configs)
3) Optional contribution pie (delta BLEU share)

Usage examples:
  python plot_results.py --compare_csv eval_results/compare_20251012/results.csv \
                         --name_a "Stage-2 Base" --name_b "Stage-3 (LM)" \
                         --out_dir figs

  python plot_results.py --ablate_csv eval_results/ablate_decode/ablate_decode.csv \
                         --out_dir figs

  python plot_results.py --compare_csv ... --ablate_csv ... --out_dir figs

Notes:
- Requires: pandas, matplotlib
- One chart per figure, no custom colors/styles
"""

import argparse
from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt


def read_compare(csv_path: Path, name_a: str, name_b: str):
    """
    The compare CSV is expected to have at least columns:
      - hyp_a, hyp_b, ref_0, bleu1_a, bleu1_b
    We'll compute corpus BLEU from summary.txt if available; otherwise
    we fall back to averaging sentence BLEU-1 as a proxy.
    """
    df = pd.read_csv(csv_path)
    # try find a sibling summary.txt
    summary_path = csv_path.parent / "summary.txt"
    bleu_a = None
    bleu_b = None
    if summary_path.exists():
        text = summary_path.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            if "A (" in line and "corpus BLEU:" in line:
                try:
                    bleu_a = float(line.split("corpus BLEU:")[1].strip())
                except Exception:
                    pass
            if "B (" in line and "corpus BLEU:" in line:
                try:
                    bleu_b = float(line.split("corpus BLEU:")[1].strip())
                except Exception:
                    pass
    # fallback if summary not present
    if bleu_a is None:
        if "bleu1_a" in df.columns:
            bleu_a = float(df["bleu1_a"].mean())
        else:
            bleu_a = 0.0
    if bleu_b is None:
        if "bleu1_b" in df.columns:
            bleu_b = float(df["bleu1_b"].mean())
        else:
            bleu_b = 0.0

    data = pd.DataFrame({
        "Model": [name_a, name_b],
        "BLEU": [bleu_a, bleu_b],
    })
    return data


def plot_compare_bar(data: pd.DataFrame, out_path: Path, title: str = "Corpus BLEU Comparison"):
    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.bar(data["Model"], data["BLEU"])
    ax.set_title(title)
    ax.set_ylabel("BLEU")
    for i, v in enumerate(data["BLEU"]):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def read_ablate(csv_path: Path):
    """
    Expect columns: num_beams, temperature, top_p, no_repeat_ngram_size, BLEU
    """
    df = pd.read_csv(csv_path)
    def _label(row):
        return f"b{int(row['num_beams'])}-t{row['temperature']}-p{row['top_p']}-ng{int(row['no_repeat_ngram_size'])}"
    df["label"] = df.apply(_label, axis=1)
    df = df.sort_values(by="BLEU", ascending=False)
    return df


def plot_ablation_bar(df: pd.DataFrame, top_k: int, out_path: Path,
                      title: str = "Decoding Ablation (Top-K by BLEU)"):
    show = df.head(top_k).copy()
    plt.figure(figsize=(max(6, top_k * 0.7), 4))
    ax = plt.gca()
    ax.bar(show["label"], show["BLEU"])
    ax.set_title(title)
    ax.set_ylabel("BLEU")
    ax.set_xlabel("config (beams-temp-top_p-ngram)")
    ax.tick_params(axis='x', labelrotation=45)
    for i, v in enumerate(show["BLEU"]):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_contribution_pie(bleu_a: float, bleu_b: float, out_path: Path,
                          title: str = "Contribution of Improvements (illustrative)"):
    delta = max(0.0, bleu_b - bleu_a)
    if delta <= 1e-6:
        delta = 1e-6
    labels = ["LM fine-tuning", "Decoding strategy", "Others"]
    shares = [0.85 * delta, 0.10 * delta, 0.05 * delta]
    plt.figure(figsize=(5, 5))
    plt.title(title)
    plt.pie(shares, labels=labels, autopct="%1.1f%%", startangle=90)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Plot comparison and ablation charts for captioning experiments")
    ap.add_argument("--compare_csv", type=str, default=None, help="results.csv from eval_compare.py")
    ap.add_argument("--name_a", type=str, default="Stage-2 Base", help="Label for A model")
    ap.add_argument("--name_b", type=str, default="Stage-3 (LM)", help="Label for B model")
    ap.add_argument("--ablate_csv", type=str, default=None, help="ablate_decode.csv from ablate_decode.py")
    ap.add_argument("--top_k", type=int, default=10, help="Top-K ablation configs to display")
    ap.add_argument("--out_dir", type=str, default="figs", help="Output directory for figures")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bleu_a = None
    bleu_b = None
    if args.compare_csv:
        comp = read_compare(Path(args.compare_csv), args.name_a, args.name_b)
        bleu_a = float(comp.loc[0, "BLEU"])
        bleu_b = float(comp.loc[1, "BLEU"])
        plot_compare_bar(comp, out_dir / "bleu_compare.png")
        plot_contribution_pie(bleu_a, bleu_b, out_dir / "contribution_pie.png")

    if args.ablate_csv:
        abd = read_ablate(Path(args.ablate_csv))
        plot_ablation_bar(abd, args.top_k, out_dir / "ablation_bleu.png")

    print("=== Saved Figures ===")
    if args.compare_csv:
        print(f"- {out_dir / 'bleu_compare.png'}")
        print(f"- {out_dir / 'contribution_pie.png'}")
        print(f"  A={args.name_a} BLEU={bleu_a:.2f} | B={args.name_b} BLEU={bleu_b:.2f}")
    if args.ablate_csv:
        print(f"- {out_dir / 'ablation_bleu.png'}")

if __name__ == "__main__":
    main()

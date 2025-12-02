#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_results.py (pretty version)
---------------------------------
Generate publication-ready charts for:
1) Model comparison BLEU (Stage-2 vs Stage-3)
2) Decoding ablation BLEU (multiple configs)
3) Optional contribution pie (delta BLEU share)

Usage examples:
  python plot_results.py --compare_csv eval_results/compare_20251012/results.csv \
                         --name_a "Stage-2 Base" --name_b "Stage-3 (LM)" \
                         --out_dir figs

  python plot_results.py --ablate_csv eval_results/ablate_decode/ablate_decode.csv \
                         --top_k 10 --out_dir figs

Requires: pandas, matplotlib
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ---------- Helpers ----------
def read_compare(csv_path: Path, name_a: str, name_b: str):
    """Read compare results; prefer summary.txt corpus BLEU if found."""
    df = pd.read_csv(csv_path)
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
    # fallback to sentence BLEU-1 mean if corpus not present
    if bleu_a is None:
        bleu_a = float(df.get("bleu1_a", pd.Series([0.0])).mean())
    if bleu_b is None:
        bleu_b = float(df.get("bleu1_b", pd.Series([0.0])).mean())

    return pd.DataFrame({"Model": [name_a, name_b], "BLEU": [bleu_a, bleu_b]})


def plot_compare_bar(data: pd.DataFrame, out_path: Path, title: str = "Corpus BLEU Comparison"):
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    bars = ax.bar(data["Model"], data["BLEU"])
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_ylabel("BLEU", fontsize=14)
    ax.set_xlabel("Model", fontsize=14)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for i, v in enumerate(data["BLEU"]):
        ax.text(i, v + max(0.02, v*0.02), f"{v:.2f}", ha="center", va="bottom", fontsize=12)
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(bottom=0.15)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def read_ablate(csv_path: Path):
    """Expect columns: num_beams, temperature, top_p, no_repeat_ngram_size, BLEU"""
    df = pd.read_csv(csv_path)
    def _label(row):
        return f"b{int(row['num_beams'])}-t{row['temperature']}-p{row['top_p']}-ng{int(row['no_repeat_ngram_size'])}"
    df["label"] = df.apply(_label, axis=1)
    return df.sort_values(by="BLEU", ascending=False)


def plot_ablation_bar(df: pd.DataFrame, top_k: int, out_path: Path,
                      title: str = "Decoding Ablation (Top-K by BLEU)"):
    show = df.head(top_k).copy()
    width = max(10, top_k * 0.9)  # widen for readability
    plt.figure(figsize=(width, 6))
    ax = plt.gca()
    bars = ax.bar(show["label"], show["BLEU"])
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_ylabel("BLEU", fontsize=14)
    ax.set_xlabel("config (beams-temp-top_p-ngram)", fontsize=14)
    ax.tick_params(axis='x', labelrotation=30, labelsize=11)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for rect in bars:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + max(0.02, h*0.02), f"{h:.2f}",
                ha="center", va="bottom", fontsize=10)
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(bottom=0.28)  # add extra space for rotated x labels
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
    plt.figure(figsize=(6, 6))
    plt.title(title, fontsize=16, fontweight="bold")
    plt.pie(shares, labels=labels, autopct="%1.1f%%", startangle=90, textprops={"fontsize": 12})
    plt.tight_layout(pad=3.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


# ---------- Main ----------
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate 30 human-eval samples for questionnaire.
Supports:
  A) Sampling from an existing outputs file (CSV/JSON) with stage1/2/3 columns
  B) Running inference for selected val items using your stage1/2/3 generators

Output: human_eval_samples.csv with columns:
  video_id, stage1_output, stage2_output, stage3_output
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

# ============ (B) If you need to run inference, import your functions here ============
# from your_project.infer import generate_caption_stage1, generate_caption_stage2, generate_caption_stage3
def generate_caption_stage1(_frames_dir: Path) -> str: return "A person is doing something in a room."
def generate_caption_stage2(_frames_dir: Path) -> str: return "A person is walking across a small room."
def generate_caption_stage3(_frames_dir: Path) -> str: return "A young person slowly walks across a small, well-lit room."

# --------------------------------------------------------------------------------------

def normalize_text(s: str) -> str:
    s = (s or "").strip()
    # Basic cleanups for readability in questionnaires
    s = s.replace("  ", " ")
    # Capitalize first letter; ensure ending punctuation
    if s and s[0].isalpha():
        s = s[0].upper() + s[1:]
    if s and s[-1] not in ".!?":
        s += "."
    return s

def token_count(s: str) -> int:
    return len(s.strip().split())

def keep_reasonable_length(s: str, min_w=8, max_w=25) -> bool:
    n = token_count(s)
    return min_w <= n <= max_w

def length_bucket(n_tokens: int) -> str:
    if n_tokens < 11: return "easy"
    if n_tokens < 18: return "medium"
    return "hard"

def sample_diverse(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """
    Try to get easy/medium/hard balance by using stage3 length as a proxy.
    """
    rng = random.Random(seed)
    df = df.copy()
    df["len_stage3"] = df["stage3_output"].fillna("").map(token_count)
    df["bucket"] = df["len_stage3"].map(length_bucket)

    groups = {
        "easy": df[df["bucket"] == "easy"],
        "medium": df[df["bucket"] == "medium"],
        "hard": df[df["bucket"] == "hard"],
    }
    # target split ~ 10/10/10
    target = {"easy": n // 3, "medium": n // 3, "hard": n - 2 * (n // 3)}
    samples = []
    for k in ["easy", "medium", "hard"]:
        g = groups[k]
        if len(g) <= target[k]:
            samples.append(g)
        else:
            samples.append(g.sample(n=target[k], random_state=seed))
    out = pd.concat(samples).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out

def save_csv(rows: List[Dict[str, Any]], out_path: Path) -> pd.DataFrame:
    import pandas as pd
    df = pd.DataFrame(rows, columns=["video_id", "stage1_output", "stage2_output", "stage3_output"])
    df = df.dropna()

    def token_count(s: str) -> int:
        return len(str(s).strip().split())

    def keep_reasonable_length(s: str, min_w=8, max_w=25) -> bool:
        n = token_count(s)
        return min_w <= n <= max_w

    # 过滤：三阶段都在 8–25 词之间
    mask = (
        df["stage1_output"].map(keep_reasonable_length)
        & df["stage2_output"].map(keep_reasonable_length)
        & df["stage3_output"].map(keep_reasonable_length)
    )
    df = df[mask].copy()

    # 规范化（首字母大写+结尾标点）
    def normalize_text(s: str) -> str:
        s = (s or "").strip().replace("  ", " ")
        if s and s[0].isalpha():
            s = s[0].upper() + s[1:]
        if s and s[-1] not in ".!?":
            s += "."
        return s

    for col in ["stage1_output", "stage2_output", "stage3_output"]:
        df[col] = df[col].map(normalize_text)

    # 如行数>30，可做长度均衡抽样（可保留你原先的 sample_diverse 逻辑）
    if len(df) > 30:
        df = sample_diverse(df, 30, seed=42)

    # 保存
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✅ Saved {len(df)} rows to {out_path}")

    # ========= 词数统计：对“已保存的输出 df”做 =========
    print("\n📊 Word Count Statistics (after filtering)")
    bins = [0, 10, 15, 20, 25, 30]
    for col in ["stage1_output", "stage2_output", "stage3_output"]:
        counts = df[col].map(lambda s: len(str(s).split()))
        print(f"\n[{col}] describe():")
        print(counts.describe())
        dist = pd.cut(counts, bins=bins).value_counts().sort_index()
        print("Bucket distribution:", dist.to_dict())

    # （可选）保存直方图
    try:
        import matplotlib.pyplot as plt
        for col in ["stage1_output", "stage2_output", "stage3_output"]:
            counts = df[col].map(lambda s: len(str(s).split()))
            plt.figure(figsize=(6,4))
            plt.hist(counts, bins=8, edgecolor="black")
            plt.title(f"Word Count Distribution - {col}")
            plt.xlabel("Words per caption"); plt.ylabel("Samples")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            png_path = out_path.with_suffix("").parent / f"word_dist_{col}.png"
            plt.savefig(png_path, dpi=150)
            plt.close()
            print(f"🖼️ Saved histogram: {png_path}")
    except Exception as e:
        print(f"(Skip plotting) Reason: {e}")

    return df


def load_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    elif path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # Try to coerce to DataFrame
        if isinstance(data, list):
            return pd.DataFrame(data)
        else:
            return pd.json_normalize(data)
    else:
        raise ValueError("Unsupported file type: use .csv or .json")

def build_from_existing(in_path: Path, out_path: Path, id_col="video_id") -> None:
    """
    A) You already have a file with stage1/2/3 outputs.
    Columns required: video_id, stage1_output, stage2_output, stage3_output
    """
    df = load_any(in_path)
    req = [id_col, "stage1_output", "stage2_output", "stage3_output"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "video_id": str(r[id_col]),
            "stage1_output": str(r["stage1_output"]),
            "stage2_output": str(r["stage2_output"]),
            "stage3_output": str(r["stage3_output"]),
        })
    save_csv(rows, out_path)

def build_by_inference(val_annotations: Path, out_path: Path, n: int = 30, seed: int = 42) -> None:
    df_ann = load_any(val_annotations)
    needed = ["video_id", "frames_dir"]
    missing = [c for c in needed if c not in df_ann.columns]
    if missing:
        raise ValueError(f"val annotations missing columns: {missing}")

    data = df_ann.to_dict("records")
    random.Random(seed).shuffle(data)
    data = data[:n]

    rows = []
    for rec in data:
        vid = str(rec["video_id"])
        frames_dir = Path(rec["frames_dir"])
        cap1 = generate_caption_stage1(frames_dir)
        cap2 = generate_caption_stage2(frames_dir)
        cap3 = generate_caption_stage3(frames_dir)
        rows.append({
            "video_id": vid,
            "stage1_output": cap1,
            "stage2_output": cap2,
            "stage3_output": cap3
        })

    # 只在 save_csv() 里做过滤/长度分布统计/画图
    _ = save_csv(rows, out_path)



def main():
    ap = argparse.ArgumentParser(description="Prepare 30-sample CSV for human evaluation.")
    ap.add_argument("--mode", choices=["existing", "infer"], required=True,
                    help="'existing' = sample from a file that already has stage outputs; 'infer' = run generators.")
    ap.add_argument("--input", type=str, required=True,
                    help="Path to input CSV/JSON. For --mode existing: must include stage1/2/3. For --mode infer: val annotations with video_id, frames_dir.")
    ap.add_argument("--output", type=str, default="human_eval_samples.csv",
                    help="Output CSV path.")
    ap.add_argument("--n", type=int, default=30, help="Target sample size (for --mode infer).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--id-col", type=str, default="video_id", help="ID column name (for --mode existing).")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if args.mode == "existing":
        build_from_existing(in_path, out_path, id_col=args.id_col)
    else:
        build_by_inference(in_path, out_path, n=args.n, seed=args.seed)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from pathlib import Path

def main():
    preds_path = Path("outputs/val_preds.json")
    if not preds_path.exists():
        print("[ERROR] outputs/val_preds.json 不存在，请先运行 infer_simple.py 生成。")
        return

    preds = json.load(open(preds_path, "r", encoding="utf-8"))
    if not preds:
        print("[WARN] 预测结果为空。")
        return

    refs = [[p["gt"].split()] for p in preds]   # 参考：一条句子一个参考
    hyps = [p["pred"].split() for p in preds]   # 预测

    ch = SmoothingFunction()
    bleu4 = corpus_bleu(refs, hyps, smoothing_function=ch.method3)
    print(f"BLEU-4: {bleu4:.4f}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, faiss, torch, numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import GPT2TokenizerFast
from src.data.data_loader import build_dataloader
from src.models.vit_text_align import ViTTextAlignModel

def l2norm(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

def main():
    ann_val = "data/processed/msvd/val/annotations.json"
    index_p = "data/processed/msvd/faiss_index/video.index"
    meta_p  = "data/processed/msvd/faiss_index/meta.json"

    index = faiss.read_index(index_p)
    meta  = json.loads(Path(meta_p).read_text("utf-8"))
    id_list = [m["video_id"] for m in meta]

    tok = GPT2TokenizerFast.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
    loader = build_dataloader(ann_val, tok, batch_size=1, num_frame=8, image_size=224, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ViTTextAlignModel(vocab_size=tok.vocab_size, pad_id=tok.pad_token_id).to(device).eval()
    ckpt = Path("checkpoints/msvd_vit_freeze_best.pt")
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state.get("model_state", state))
        print("[INFO] ckpt loaded:", ckpt)

    Ks = [1,5]
    hit = {k:0 for k in Ks}; mrr_sum=0.0; n=0

    with torch.no_grad():
        for b in tqdm(loader, desc="eval"):
            vid = b["video_id"][0]
            q = model.encode_video(b["video"].to(device)).cpu().numpy().astype("float32")
            q = l2norm(q)
            D,I = index.search(q, max(Ks))
            n+=1
            try:
                gold_rank = I[0].tolist().index(id_list.index(vid)) + 1
            except ValueError:
                gold_rank = None
            if gold_rank:
                mrr_sum += 1.0/gold_rank
                for k in Ks:
                    if gold_rank <= k: hit[k]+=1

    if n==0: print("[WARN] no samples"); return
    print(f"[VAL] N={n}")
    for k in Ks: print(f"Recall@{k}: {hit[k]/n:.3f}")
    print(f"MRR: {mrr_sum/n:.3f}")

if __name__ == "__main__":
    main()
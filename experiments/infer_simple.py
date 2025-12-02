# 推理Demo，这个部分极简模型生成质量
# 几乎为随机句子，目的只是验证推理流程

import json, torch, os
from pathlib import Path

from src.models.simple_vc import SimpleVideoCaptioner
from src.data.data_loader import build_dataloader

save_dir = Path("outputs")
save_dir.mkdir(exist_ok=True)

def get_tokenizer():
    from transformers import BertTokenizerFast
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
    if tok.pad_token_id is None: tok.pad_token_id = 0
    return tok

@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = get_tokenizer()
    vocab_size = tok.vocab_size
    pad_id = tok.pad_token_id

    # 载模型
    ckpt = "checkpoints/msvd_debug/simple_vc_smoke.pt"
    model = SimpleVideoCaptioner(vocab_size=vocab_size, hidden_size=512, max_len=32, pad_id=pad_id).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # 读验证集3条
    ann_path = "data/processed/msvd/val/annotations.json"
    ann = json.load(open(ann_path, "r", encoding="utf-8"))[:3]

    # 写个最简的 sample loader（若你已有 util 可替换）
    from src.data.data_loader import build_dataloader
    loader = build_dataloader(ann_path, tok, batch_size=3, shuffle=False, num_wokers=0, num_frame=8, image_size=224)
    batch = next(iter(loader))
    video = batch["video"].to(device)
    logits = model(video, None)  # [B, L, V]
    pred_ids = logits.argmax(-1) # 贪心
    
    
    
    results = []
    
    for i in range(pred_ids.size(0)):
        gt = ann[i]["caption"]
        try:
            pr =  tok.decode(pred_ids[i].tolist(), skip_special_tokens=True)
        except Exception:
            pr = " ".join(map(str, pred_ids[i].tolist()))
            
        print("GT: ",gt)
        print("PR: ",pr)
        print()
        
        results.append({"gt":gt,"pred":pr})
        
# ====优化：新增保存到 outputs/val_preds.json 功能===========

    save_dir = Path("outputs")
    save_dir.mkdir(exist_ok=True)

    out_path = save_dir / "val_preds.json"
    with open(out_path,"w",encoding="utf-8") as f:
        json.dump(results,f,indent=2,ensure_ascii=False)
        
    print(f"[INFO] Saved predictions to {out_path}")

if __name__ == "__main__":
    main()

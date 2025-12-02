#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最小可跑的 Loader 冒烟测试：
- 读取 data/processed/msvd/train/annotations_filtered.json
- 构造一个可用的 tokenizer（优先 HuggingFace，缺省用极简版）
- 调用 build_dataloader 并打印几个 batch 的 key/shape
"""

from pathlib import Path
import inspect

from src.data.data_loader import build_dataloader  # 保持你项目路径

ANN = Path("data/processed/msvd/train/annotations_frames.json")
if not ANN.exists():
    raise FileNotFoundError(f"{ANN} 不存在；请先生成 filtered 文件。")

# 1) 准备 tokenizer
tokenizer = None
try:
    from transformers import BertTokenizerFast  # type: ignore
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    print("[INFO] Using HuggingFace BertTokenizerFast.")
except Exception:
    print("[WARN] transformers 不可用，改用极简分词器（仅为冒烟测试）。")
    class MinimalTokenizer:
        def __init__(self):
            self.vocab = {"[PAD]":0, "[UNK]":1, "[CLS]":2, "[SEP]":3}
        def __call__(self, text, max_length=32):
            # 非严格实现：空格切分，截断/填充到 max_length
            tokens = str(text).lower().strip().split()
            ids = [2]  # [CLS]
            for w in tokens[:max_length-2]:
                if w not in self.vocab:
                    self.vocab[w] = len(self.vocab)
                ids.append(self.vocab[w])
            ids.append(3)  # [SEP]
            if len(ids) < max_length:
                ids += [0] * (max_length - len(ids))  # pad
            attn = [1 if i!=0 else 0 for i in ids]
            return {"input_ids": ids, "attention_mask": attn}
    tokenizer = MinimalTokenizer()

# 2) 读取签名，构造参数（注意 num_wokers 的拼写）
sig = inspect.signature(build_dataloader)
print("[INFO] build_dataloader signature:", sig)

kwargs = {}
if "batch_size" in sig.parameters: kwargs["batch_size"] = 2
if "shuffle" in sig.parameters: kwargs["shuffle"] = False
if "num_wokers" in sig.parameters: kwargs["num_wokers"] = 0
# 也可以按需指定 max_len/num_frame/image_size，这里用默认

# 3) 创建 DataLoader（第一个必选参数是 ann_path，第二个必选是 tokenizer）
loader = build_dataloader(str(ANN), tokenizer, **kwargs)

print("[INFO] DataLoader created. Iterate a few batches...")
for i, batch in enumerate(loader):
    print(f"---- Batch {i} ----")
    if isinstance(batch, dict):
        for k, v in batch.items():
            shape = getattr(v, "shape", None)
            print(f"{k}: {type(v)} {shape}")
    else:
        try:
            for idx, v in enumerate(batch):
                print(f"item[{idx}]: {type(v)} {getattr(v, 'shape', None)}")
        except Exception as e:
            print(batch)
    if i >= 2:
        break
print("[DONE] loader smoke test finished.")

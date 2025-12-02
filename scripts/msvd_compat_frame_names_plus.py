#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为每个 frames_dir 生成一组常见命名/扩展名的别名，覆盖常见 Loader 规则：
现有：00001.jpg, 00002.jpg, ...
补出：
- image_00001.jpg / img_00001.jpg
- frame_00001.jpg / frame00001.jpg
- 以及对应的 .png 版本（必要时）
优先硬链接，失败再复制；已存在则跳过。
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, shutil
from pathlib import Path
import json

PROC = Path("data/processed/msvd")
SPLITS = ["train","val","test"]

def link_or_copy(src: Path, dst: Path):
    if dst.exists(): return
    try:
        os.link(src, dst)   # NTFS 同分区可硬链接
    except Exception:
        shutil.copy2(src, dst)

def generate_aliases(frames_dir: Path):
    # 原始：00001.jpg 这种 5 位数字
    jpgs = sorted(frames_dir.glob("[0-9][0-9][0-9][0-9][0-9].jpg"))
    if not jpgs: 
        return False
    for src in jpgs:
        stem5 = src.stem                  # 00001
        stem6 = stem5.zfill(6)            # 000001
        # 5位别名
        for prefix in ("image_", "img_", "frame_", "frame"):
            name = f"{prefix}{stem5}.jpg"
            link_or_copy(src, src.with_name(name))
        # 6位别名
        for prefix in ("image_", "img_", "frame_", "frame"):
            name = f"{prefix}{stem6}.jpg"
            link_or_copy(src, src.with_name(name))
    # 可选：首帧生成 .png（只做一张，除非确认 Loader 只认 png）
    try:
        from PIL import Image
        first = jpgs[0]
        png = first.with_name("image_000001.png")
        if not png.exists():
            Image.open(first).save(png)
    except Exception:
        pass
    return True

def process_split(split):
    ann = PROC / split / "annotations_frames.json"
    if not ann.exists():
        print(f"[WARN] {ann} 不存在，跳过 {split}")
        return
    items = json.loads(ann.read_text(encoding="utf-8"))
    cnt = 0
    for it in items:
        fd = Path(it.get("frames_dir",""))
        if not fd.is_dir(): 
            continue
        if generate_aliases(fd):
            cnt += 1
    print(f"[{split}] 兼容命名完成的目录数: {cnt}")

if __name__ == "__main__":
    for sp in SPLITS:
        process_split(sp)
    print("[DONE] 兼容命名 V2 完成")

# scripts/prepare_msvd.py
# Usage:
#   pip install datasets decord pillow tqdm av==10.0.0
#   python scripts/prepare_msvd.py --name friedrichor/MSVD --out data/processed/msvd \
#       --num-frames 12 --size 224 --subset 200 --fps 4 --use-decord

import os, json, random, argparse, shutil
from pathlib import Path
from typing import List, Dict, Any

from tqdm import tqdm
from PIL import Image


# pyav or decord
try:
    import decord
    decord.bridge.set_bridge("torch")
except Exception:
    decord = None

try:
    import av
except Exception:
    av = None
    
from datasets import load_dataset

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    
def load_msvd(name:str):
    """
    期望列结构：
      - 'video' : hf Video对象（包含本地缓存路径）
      - 'video_id' : 唯一ID
      - 'sentences' 或 'captions' : 文本列表
      - 'split' : train/val/test（没有则我们自己切）
    """
    ds  = load_dataset(name)
    # 统一为dict(split_name->list[dict])
    splits = {}
    if isinstance(ds, dict):
        for split_name, d in ds.items():
            splits[split_name] = list(d)
    else:
        splits['train'] = list(ds)
    
    return splits

def get_video_path(example: Dict[str, Any]) -> str:
    v = example.get("video")
    # huggingface Video 对象通常是 {'path': '/.../xxx.mp4', ...}
    if isinstance(v, dict) and "path" in v:
        return v["path"]
    # 有的实现直接是字符串路径
    if isinstance(v, str):
        return v
    # 兜底：尝试常见键
    for k in ["video_path", "path"]:
        if k in example:
            return example[k]
    raise ValueError("Cannot locate video path in example keys:", example.keys())

def get_captions(example: Dict[str, Any]) -> List[str]:
    for key in ["sentences", "captions", "descriptions", "texts"]:
        if key in example and example[key]:
            # 统一清洗：小写、去首尾空白
            return [str(s).strip().lower() for s in example[key] if str(s).strip()]
    # 兜底：若只有单句
    if "sentence" in example:
        return [str(example["sentence"]).strip().lower()]
    return []

def read_frames_decord(video_path: str, num_frames: int, fps: int=0)->List[Image.Image]:
    if decord is None:
        raise RuntimeError("Decord is not installed. Re-run without --use-decord or install decord.")
    vr = decord.VideoReader(video_path)
    total = len(vr)
    if total == 0:
        return []
    # 均匀采样索引
    idxs = uniform_indices(total, num_frames)
    frames = vr.get_batch(idxs).asnumpy()  # (num_frames, H, W, 3), rgb
    imgs = [Image.fromarray(frames[i]) for i in range(frames.shape[0])]
    return imgs


def read_frames_pyav(video_path: str, num_frames: int, fps: int = 0) -> List[Image.Image]:
    if av is None:
        raise RuntimeError("PyAV is not installed. Install `av`.")
    container = av.open(video_path)
    stream = container.streams.video[0]
    # 逐帧读取到列表，再做均匀采样（简化实现，稳定）
    raw = []
    for frame in container.decode(stream):
        img = frame.to_image()  # PIL Image
        raw.append(img)
    container.close()
    if not raw:
        return []
    idxs = uniform_indices(len(raw), num_frames)
    return [raw[i] for i in idxs]


def uniform_indices(total: int, k: int) -> List[int]:
    if k <= 1 or total <= 1:
        return [0]
    if k >= total:
        return list(range(total))
    step = total / float(k)
    return [min(total - 1, int(i * step + step / 2)) for i in range(k)]


def save_frames(imgs: List[Image.Image], out_dir: Path, size: int = 224) -> List[str]:
    ensure_dir(out_dir)
    names = []
    for i, img in enumerate(imgs):
        if size is not None:
            img = img.resize((size, size), Image.BICUBIC)
        fn = out_dir / f"frame_{i:05d}.jpg"
        img.save(fn, quality=95)
        names.append(str(fn))
    return names


def build_annotations(records: List[Dict[str, Any]], save_path: Path):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
        
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name",type=str,default="friedrichor/MSVD",help="HF dataset name")
    ap.add_argument("--out", type=str, required=True, help="output root, e.g., data/processed/msvd")
    ap.add_argument("--num-frames", type=int, default=12)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--subset", type=int, default=0, help="limit number of videos per split (0 = all)")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--fps", type=int, default=0, help="optional, not used in current uniform sampler")
    ap.add_argument("--use-decord", action="store_true", help="use decord; else PyAV")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    
    random.seed(args.seed)

    out_root = Path(args.out)
    if args.overwrite and out_root.exists():
        shutil.rmtree(out_root)
    ensure_dir(out_root)

    print(f"Loading dataset: {args.name}")
    splits = load_msvd(args.name)

    read_fn = read_frames_decord if args.use_decord else read_frames_pyav
    print("Reader:", "decord" if args.use_decord else "pyav")

    for split_name, items in splits.items():
        print(f"\nProcessing split: {split_name} ({len(items)} items)")
        if args.subset and args.subset > 0:
            items = items[: args.subset]
            print(f"  -> subset to {len(items)}")

        split_dir = out_root / split_name
        ensure_dir(split_dir)
        ann_recs = []
        
        for ex in tqdm(items):
            try:
                vid = str(ex.get("video_id") or ex.get("id") or ex.get("name") or "")
                if not vid:
                    # 兜底：从路径推一个 id
                    vp = get_video_path(ex)
                    vid = Path(vp).stem

                video_path = get_video_path(ex)
                caps = get_captions(ex)
                if not caps:
                    # 没字幕就跳过
                    continue

                # 帧目录
                out_dir = split_dir / vid
                if out_dir.exists() and not args.overwrite:
                    # 已处理过：直接写入注释记录
                    frame_files = sorted([str(p) for p in out_dir.glob("frame_*.jpg")])
                    if frame_files:
                        ann_recs.append({
                            "video_id": vid,
                            "split": split_name,
                            "captions": caps,
                            "num_frames": len(frame_files),
                            "fps": args.fps,
                            "frames_dir": str(out_dir)
                        })
                        continue
                    else:
                        shutil.rmtree(out_dir)
                
                # 读取 + 抽帧 + 缩放 + 保存
                imgs = read_fn(video_path, args.num_frames, args.fps)
                if not imgs:
                    continue
                frame_files = save_frames(imgs, out_dir, size=args.size)
                
                ann_recs.append({
                    "video_id": vid,
                    "split": split_name,
                    "captions": caps,
                    "num_frames": len(frame_files),
                    "fps": args.fps,
                    "frames_dir": str(out_dir)
                })
            except Exception as e:
                print("[WARN] Failed:", e)
                continue

        # 写入 split 注释
        build_annotations(ann_recs, split_dir / "annotations.json")
        print(f"Saved {len(ann_recs)} records to {split_dir/'annotations.json'}")

    print("\nDone. Folder layout example:")
    print(f"{out_root}/")
    print("  ├─ train/")
    print("  │   ├─ VID_xxx/ frame_00000.jpg ...")
    print("  │   └─ annotations.json")
    print("  ├─ validation/")
    print("  └─ test/")
    print("\nYou can now point your DataLoader to `split/annotations.json` and per-video frame folders.")
    

if __name__ == "__main__":
    main()
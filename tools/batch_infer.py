# tools/batch_infer.py
import argparse, csv, json, re, sys
from pathlib import Path
import subprocess

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

def parse_args():
    p = argparse.ArgumentParser("Robust batch inference via JSON (S1/S2/S3)")
    p.add_argument("--frames_root", required=True, help="e.g. ./data/processed/msvd/val/frames")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out_csv", default="./data/human_eval_samples/batch_captions.csv")
    p.add_argument("--max_samples", type=int, default=30)
    p.add_argument("--prefix_len", type=int, default=4)
    p.add_argument("--ln_scale", type=float, default=0.9)
    p.add_argument("--in_weight", type=float, default=0.8)
    p.add_argument("--preset1", default="precise")
    p.add_argument("--preset2", default="detailed")
    p.add_argument("--preset3", default="natural")
    p.add_argument("--prompt1", default="Describe the visible action and objects in one short sentence:")
    p.add_argument("--prompt2", default="State the main action and the key object in the scene in one sentence:")
    p.add_argument("--prompt3", default="Write a short, natural caption about what happens in the video:")
    p.add_argument("--timeout", type=int, default=120)
    return p.parse_args()

def has_jpg(frames_dir: Path) -> bool:
    return any(frames_dir.glob("*.jpg"))

def call_infer(frames_dir: Path, ckpt: str, args) -> tuple[str,str,str]:
    """调用 inference，一次拿到 S1/S2/S3（JSON），内部带重试"""
    def _once():
        cmd = [
            sys.executable, "-m", "inference",
            "--frames_dir", str(frames_dir),
            "--stage", "all",
            "--ckpt", ckpt,
            "--prefix_len", str(args.prefix_len),
            "--ln_scale", str(args.ln_scale),
            "--in_weight", str(args.in_weight),
            "--preset1", args.preset1,
            "--preset2", args.preset2,
            "--preset3", args.preset3,
            "--prompt1", args.prompt1,
            "--prompt2", args.prompt2,
            "--prompt3", args.prompt3,
            "--emit_json",
            "--log_level", "ERROR"  # 关键：尽量只输出 JSON
        ]
        proc = subprocess.run(
            cmd, text=True, capture_output=True, timeout=args.timeout
        )
        # 有些环境仍会输出额外日志，取最后一个 JSON 块
        raw = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        m = None
        for m in JSON_RE.finditer(raw):
            pass
        if m:
            try:
                data = json.loads(m.group(0).strip())
                return data.get("S1",""), data.get("S2",""), data.get("S3","")
            except json.JSONDecodeError:
                return "", "", ""
        return "", "", ""

    s1, s2, s3 = _once()
    if not (s1 or s2 or s3):
        # 轻微调参再重试一次（更保守的束搜索/抑制重复）
        bak_preset1, bak_preset2, bak_preset3 = args.preset1, args.preset2, args.preset3
        args.preset1, args.preset2, args.preset3 = "precise", "detailed", "precise"
        s1, s2, s3 = _once()
        # 还不行就返回空
        args.preset1, args.preset2, args.preset3 = bak_preset1, bak_preset2, bak_preset3
    return s1, s2, s3

def main():
    args = parse_args()
    frames_root = Path(args.frames_root)
    folders = sorted([p for p in frames_root.iterdir() if p.is_dir()])

    rows = []
    for f in folders:
        if len(rows) >= args.max_samples:
            break
        # 跳过没有jpg的目录
        if not has_jpg(f):
            print(f"[SKIP] {f.name} (no .jpg frames)", file=sys.stderr)
            rows.append({"video_id": f.name, "stage1_output": "", "stage2_output": "", "stage3_output": ""})
            continue

        s1, s2, s3 = call_infer(f, args.ckpt, args)
        print(f"[{len(rows)+1:02d}] {f.name} -> "
              f"S1:{(s1[:40]+'...') if s1 else '(empty)'} | "
              f"S2:{(s2[:40]+'...') if s2 else '(empty)'} | "
              f"S3:{(s3[:40]+'...') if s3 else '(empty)'}")

        rows.append({
            "video_id": f.name,
            "stage1_output": s1,
            "stage2_output": s2,
            "stage3_output": s3
        })

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as wf:
        w = csv.DictWriter(wf, fieldnames=["video_id", "stage1_output", "stage2_output", "stage3_output"])
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] saved {len(rows)} rows -> {out_path}")

if __name__ == "__main__":
    main()
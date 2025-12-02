# tools/batch_bestof.py
import argparse, csv, json, re, sys
from pathlib import Path
import subprocess

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

# BAD_REGEX v2.0
BAD_REGEX = re.compile(
    r"(http[s]?://\S+|www\.\S+|youtube|facebook|subscribe|channel|report abuse|menu|fullscreen|"
    r"the video (will|begins|shows)|one sentence|the first sentence|"
    r"\bi am\b|\bi'm\b|\bmy\b|\bour\b|"
    r"killed|shot|gun|blood|dead|murder|naked|unclothed|nudity)",
    re.I
)

START_TEMPL_RE = re.compile(r"^(this|that|it)\s+is\s+(the|a)\s+", re.I)

VERB_HINTS = set("""is are was were be being been am are's has have having had
walk walks walking walked run runs running ran talk talks talking talked
look looks looking looked hold holds holding held sit sits sitting sat
stand stands standing stood play plays playing played cook cooks cooking cooked
drive drives driving drove ride rides riding rode cut cuts cutting cut
pour pours pouring poured open opens opening opened close closes closing closed
throw throws throwing threw catch catches catching caught read reads reading read
""".split())
NOUN_HINTS = set("""man woman boy girl person people player child baby dog cat
car ball food phone camera street room kitchen table water animal
""".split())
PLACE_HINTS = set("in on at under with near inside outside into around over by".split())

def parse_args():
    p = argparse.ArgumentParser("Batch inference with best-of-3 selection")
    p.add_argument("--frames_root", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out_csv", default="./data/human_eval_samples/batch_best.csv")
    p.add_argument("--max_samples", type=int, default=30)
    p.add_argument("--prefix_len", type=int, default=4)
    p.add_argument("--ln_scale", type=float, default=0.8)
    p.add_argument("--in_weight", type=float, default=0.8)
    p.add_argument("--preset1", default="precise")
    p.add_argument("--preset2", default="detailed")
    p.add_argument("--preset3", default="natural")
    p.add_argument("--prompt1", default="Describe the visible action and objects in one short sentence:")
    p.add_argument("--prompt2", default="State the main action and the key object in the scene in one sentence:")
    p.add_argument("--prompt3", default="Write a short, natural caption about what happens in the video:")
    p.add_argument("--timeout", type=int, default=120)
    return p.parse_args()

def has_jpg(d: Path) -> bool:
    return any(d.glob("*.jpg"))

def call_infer_once(frames_dir: Path, ckpt: str, args) -> dict:
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
        "--log_level", "ERROR",
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=args.timeout)
    raw = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    m = None
    for m in JSON_RE.finditer(raw):
        pass
    if not m:
        return {"S1":"", "S2":"", "S3":""}
    try:
        data = json.loads(m.group(0).strip())
        return {k: data.get(k,"") for k in ("S1","S2","S3")}
    except json.JSONDecodeError:
        return {"S1":"", "S2":"", "S3":""}

def score_sentence(s: str) -> float:
    if not s: return -1e9
    if BAD_REGEX.search(s): return -100.0

    txt = s.strip()
    # 开头模板惩罚
    pen = -2.0 if START_TEMPL_RE.match(txt) else 0.0

    # 长度得分（8~18最优）
    n = len(txt.split())
    len_score = -abs(n - 13)  # 峰值13词

    # 词性线索（粗略）：动词/名词/介词
    lower = txt.lower().split()
    verb = any(w in lower for w in VERB_HINTS)
    noun = any(w in lower for w in NOUN_HINTS)
    place = any(w in lower for w in PLACE_HINTS)
    bonus = (1.2 if verb else 0.0) + (1.0 if noun else 0.0) + (0.5 if place else 0.0)

    # 收尾标点
    if txt.endswith((".", "!", "?")): bonus += 0.5

    return len_score + bonus + pen

def main():
    args = parse_args()
    root = Path(args.frames_root)
    folders = sorted([p for p in root.iterdir() if p.is_dir()])

    rows = []
    for f in folders:
        if len(rows) >= args.max_samples:
            break
        if not has_jpg(f):
            print(f"[SKIP] {f.name} (no .jpg frames)", file=sys.stderr)
            rows.append({"video_id": f.name, "best": "", "best_from_stage": "",
                         "stage1_output":"", "stage2_output":"", "stage3_output":""})
            continue
        out = call_infer_once(f, args.ckpt, args)
        s1, s2, s3 = out.get("S1",""), out.get("S2",""), out.get("S3","")
        cand = [("S1", s1), ("S2", s2), ("S3", s3)]
        cand.sort(key=lambda kv: score_sentence(kv[1]), reverse=True)
        best_stage, best_sent = cand[0]
        print(f"[{len(rows)+1:02d}] {f.name} -> {best_stage}: {(best_sent[:60]+'...') if best_sent else '(empty)'}")
        rows.append({
            "video_id": f.name,
            "best": best_sent,
            "best_from_stage": best_stage,
            "stage1_output": s1, "stage2_output": s2, "stage3_output": s3
        })

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as wf:
        w = csv.DictWriter(wf, fieldnames=["video_id","best","best_from_stage","stage1_output","stage2_output","stage3_output"])
        w.writeheader(); w.writerows(rows)
    print(f"[OK] saved {len(rows)} rows -> {out_path}")

if __name__ == "__main__":
    main()
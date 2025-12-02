import json,subprocess
from pathlib import Path

def fix(split):
    p = Path(f"./data/processed/msvd/{split}/annotations.json")
    recs=json.loads(p.read_text(encoding="utf-8"))
    fixed=fail=0
    for r in recs:
        v=Path(r.get("video",""))
        d=Path(r.get("frames_dir"),"")
        ok = d.exists() and any(d.glob("frame_*.jpg"))
        if ok or not v.exists():
            continue
        d.mkdir(parents=True,exist_ok=True)
        cmd=["ffmpeg","-y","-i",str(v),"-vf","fps=2",str(d/"frame_%06d.jpg")]
        print("[RUN]"," ".join(cmd))
        ret = subprocess.run(cmd)
        if d.exists() and any(d.glob("frame_*.jpg")):
            fixed+=1
        else:
            fail+=1
    print(f"[{split}] fixed={fixed} fail={fail}")
    
fix("val")
fix("test")
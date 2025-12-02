import json 
from pathlib import Path

def cov(split):
    p = Path(f"./data/processed/msvd/{split}/annotations.json")
    if not p.exists(): return 
    recs = json.loads(p.read_text(encoding="utf-8"))
    ok = sum(1 for r in recs if any(Path(r["frames_dir"]).glob("frame_*.jpg")))
    print(f"{split}:{ok}/{len(recs)}({ok/len(recs):.1%})")
    
for s in ["train","val","test"]:
    cov(s)
    
    
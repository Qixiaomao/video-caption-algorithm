import json
from pathlib import Path

def cov(split):
    p=Path(f"./data/processed/msvd/{split}/annotations.json")
    recs = json.loads(p.read_text(encoding="utf-8"))
    have = 0
    missing=[]
    for r in recs:
        d = Path(r["frames_dir"])
        ok = d.exists() and any(d.glob("frame_*.jpg"))
        have += int(ok)
        if not ok:
            missing.append((r["video_id"],r.get("video",""),r["frames_dir"]))
            
    rate = have/len(recs)
    print(f"{split}:{have}/{len(recs)} ({rate:.1%}) missing={len(missing)}")
    
    for vid, vpath, fdir in missing[:5]:
        print(" -",vid,"video=",vpath,"frames_dir=",fdir)
        
    return missing

m_val = cov("val")
m_test = cov("test")

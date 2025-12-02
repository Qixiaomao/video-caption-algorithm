from pathlib import Path

from pathlib import Path
d = Path(r"D:\Mylearn\INTI courses\Graduation thesis\video-captioning-transformer\data\processed\msvd\frames\4wsuPCjDBc")
print(d.exists(), sum(1 for p in d.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'}))

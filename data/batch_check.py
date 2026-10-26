"""檢查各類別原圖的尺寸與 JPEG 量化表,確認資料是否來自不同批次。"""
import re, hashlib
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image

SRC = Path("/home/rapids/pvc/ear-drum-input-pic")
PREFIX = {"AcuteMyringitis":"AcuteMyringitis","AcuteOtitisMedia":"AcuteOtitis",
 "CerumenImpaction":"CerumenImpaction","ChronicSuppurative":"ChronicSuppurative",
 "Effusion":"Effusion","Normal":"Normal","Otomycosis":"Otomycosis",
 "TympanicMembranePerforation":"TMP","TympanicMembraneRetraction":"TMR",
 "VentilationTube":"VentilationTube"}

print(f"{'類別':<30}{'原圖':>6}  {'尺寸(前2種)':<26}{'量化表指紋(前2種)'}")
print("-"*100)
for d in sorted(SRC.iterdir()):
    if not d.is_dir(): continue
    pre = PREFIX[d.name]
    pat = re.compile(rf"^{re.escape(pre)}_[\d,]+\.jpg(\.jpg)?$")
    files = [f for f in d.iterdir() if pat.match(f.name)]
    sizes, qts = Counter(), Counter()
    for f in files:
        try:
            im = Image.open(f)
            sizes[im.size] += 1
            q = getattr(im, "quantization", None)
            qts[hashlib.md5(str(q).encode()).hexdigest()[:8] if q else "none"] += 1
        except Exception:
            pass
    s = "  ".join(f"{w}x{h}:{n}" for (w,h),n in sizes.most_common(2))
    q = "  ".join(f"{k}:{n}" for k,n in qts.most_common(2))
    print(f"{d.name:<30}{len(files):>6}  {s:<26}{q}")

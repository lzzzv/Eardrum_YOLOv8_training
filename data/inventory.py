"""盤點各類別的原圖與增強影像數量,並以 MD5 計算去重後的原圖數。"""
import re, hashlib
from pathlib import Path
from collections import defaultdict

SRC = Path("/home/rapids/pvc/ear-drum-input-pic")
PREFIX = {   # 資料夾名 -> 檔名前綴
    "AcuteMyringitis": "AcuteMyringitis", "AcuteOtitisMedia": "AcuteOtitis",
    "CerumenImpaction": "CerumenImpaction", "ChronicSuppurative": "ChronicSuppurative",
    "Effusion": "Effusion", "Normal": "Normal", "Otomycosis": "Otomycosis",
    "TympanicMembranePerforation": "TMP", "TympanicMembraneRetraction": "TMR",
    "VentilationTube": "VentilationTube",
}

def md5(p, chunk=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while (b := f.read(chunk)):
            h.update(b)
    return h.hexdigest()

print(f"{'類別':<30}{'原圖':>7}{'增強':>8}{'合計':>8}{'去重後原圖':>12}")
print("-" * 68)

orphans, total_o, total_a, total_u = [], 0, 0, 0
for d in sorted(SRC.iterdir()):
    if not d.is_dir():
        continue
    pre = PREFIX[d.name]
    pat_o = re.compile(rf"^{re.escape(pre)}_[\d,]+\.jpg(\.jpg)?$")
    orig, aug, aug_ids = {}, 0, set()
    for f in d.iterdir():
        if f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        if pat_o.match(f.name):
            gid = re.match(rf"^{re.escape(pre)}_([\d,]+)\.", f.name).group(1)
            orig[gid] = f
        else:
            aug += 1
            m = re.match(rf"^{re.escape(pre)}_([\d,]+)[._]", f.name)
            if m:
                aug_ids.add(m.group(1))
    # 檢查:有增強圖卻找不到原圖的編號
    for gid in aug_ids - set(orig):
        orphans.append(f"{d.name}/{pre}_{gid}")
    uniq = len({md5(p) for p in orig.values()})
    total_o += len(orig); total_a += aug; total_u += uniq
    print(f"{d.name:<30}{len(orig):>7}{aug:>8}{len(orig)+aug:>8}{uniq:>12}")

print("-" * 68)
print(f"{'總計':<30}{total_o:>7}{total_a:>8}{total_o+total_a:>8}{total_u:>12}")

print(f"\n檢查:有增強圖但找不到對應原圖的編號 = {len(orphans)} 組")
for o in orphans[:20]:
    print("  ", o)

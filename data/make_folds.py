"""
grouped 5-fold:以原圖為分組單位(MD5 相同者合併),輪流分配到 5 個 fold。
val 僅含未增強原圖;train 含原圖 + 其增強變體。
若取得病人編號,可改為以病人為分組單位。
"""
import re, hashlib, random
from pathlib import Path
from collections import defaultdict

SRC = Path("/home/rapids/pvc/ear-drum-input-pic")
DST = Path("/home/rapids/pvc/eardrum_train/folds")
N_FOLDS, SEED = 5, 42
random.seed(SEED)

PREFIX = {"AcuteMyringitis":"AcuteMyringitis","AcuteOtitisMedia":"AcuteOtitis",
 "CerumenImpaction":"CerumenImpaction","ChronicSuppurative":"ChronicSuppurative",
 "Effusion":"Effusion","Normal":"Normal","Otomycosis":"Otomycosis",
 "TympanicMembranePerforation":"TMP","TympanicMembraneRetraction":"TMR",
 "VentilationTube":"VentilationTube"}

def md5(p, chunk=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while (b := f.read(chunk)):
            h.update(b)
    return h.hexdigest()

def link(src, dst):
    if not dst.exists():
        dst.symlink_to(src.resolve())

print(f"{'類別':<30}{'原圖':>6}{'合併後':>8}" + "".join(f"{'f'+str(k):>6}" for k in range(N_FOLDS)))
print("-" * 76)

grand = [0]*N_FOLDS
for d in sorted(SRC.iterdir()):
    if not d.is_dir():
        continue
    cls, pre = d.name, PREFIX[d.name]
    pat_o = re.compile(rf"^{re.escape(pre)}_[\d,]+\.jpg(\.jpg)?$")

    orig, aug = {}, defaultdict(list)
    for f in sorted(d.iterdir()):
        if f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        m = re.match(rf"^{re.escape(pre)}_([\d,]+)[._]", f.name)
        if not m:
            continue
        gid = m.group(1)
        if pat_o.match(f.name):
            orig[gid] = f
        else:
            aug[gid].append(f)

    # MD5 合併內容相同的原圖
    parent = {g: g for g in orig}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    seen = {}
    for g, p in orig.items():
        h = md5(p)
        if h in seen:
            ra, rb = find(g), find(seen[h])
            if ra != rb: parent[ra] = rb
        else:
            seen[h] = g

    supers = defaultdict(list)
    for g in orig:
        supers[find(g)].append(g)
    groups = list(supers.values())
    random.shuffle(groups)

    counts = [0]*N_FOLDS
    for i, gids in enumerate(groups):
        k = i % N_FOLDS
        counts[k] += len(gids)
        for gid in gids:
            # 該 fold 當 val:只放原圖
            vd = DST / f"fold{k}" / "val" / cls
            vd.mkdir(parents=True, exist_ok=True)
            link(orig[gid], vd / orig[gid].name)
            # 其餘 fold 當 train:原圖 + 增強
            for j in range(N_FOLDS):
                if j == k: continue
                td = DST / f"fold{j}" / "train" / cls
                td.mkdir(parents=True, exist_ok=True)
                link(orig[gid], td / orig[gid].name)
                for a in aug[gid]:
                    link(a, td / a.name)

    for k in range(N_FOLDS):
        grand[k] += counts[k]
    print(f"{cls:<30}{len(orig):>6}{len(groups):>8}" + "".join(f"{c:>6}" for c in counts))

print("-" * 76)
print(f"{'各 fold 的 val 原圖數':<44}" + "".join(f"{c:>6}" for c in grand))
print()
for k in range(N_FOLDS):
    nt = sum(1 for _ in (DST/f"fold{k}"/"train").rglob("*.jpg"))
    nv = sum(1 for _ in (DST/f"fold{k}"/"val").rglob("*.jpg"))
    print(f"fold{k}:  train {nt:>6} 張   val {nv:>5} 張(全為未增強原圖)")
print(f"\n輸出:{DST}")

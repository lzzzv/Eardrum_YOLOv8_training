"""
整理 ChronicSuppurative 與 TympanicMembranePerforation 兩類供臨床複核。
含:兩類全部原始影像、以及兩類之間的高信心誤判。
"""
import csv, shutil, hashlib
from pathlib import Path

PRED = Path("/home/rapids/pvc/eardrum_train/outputs/cv_predictions.csv")
OUT  = Path("/home/rapids/pvc/review_csp_vs_perforation")
PAIR = {"ChronicSuppurative", "TympanicMembranePerforation"}

def md5(p, chunk=1<<20):
    h=hashlib.md5()
    with open(p,"rb") as f:
        while (b:=f.read(chunk)): h.update(b)
    return h.hexdigest()

rows = list(csv.DictReader(open(PRED)))
seen, uniq = set(), []
for r in rows:
    h = md5(Path(r["src"]))
    if h in seen: continue
    seen.add(h); uniq.append(r)

sub = [r for r in uniq if r["true"] in PAIR]
manifest = []
for r in sub:
    err = r["pred"] in PAIR and r["pred"] != r["true"]
    hi  = err and float(r["conf"]) >= 0.90
    if hi:      folder = "03_高信心互判"
    elif err:   folder = "02_一般互判"
    else:       folder = "01_判斷正確"
    d = OUT / folder / r["true"]
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy2(r["src"], d / r["file"])
    manifest.append([folder, r["true"], r["pred"], r["conf"], r["file"]])

with open(OUT/"manifest.csv","w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["分組","標註類別","模型判為","信心","檔名"])
    w.writerows(sorted(manifest))

from collections import Counter
print(f"兩類獨立原圖共 {len(sub)} 張")
for k, v in sorted(Counter(m[0] for m in manifest).items()):
    print(f"  {k:<16}{v:>4} 張")
print(f"\n輸出:{OUT}")

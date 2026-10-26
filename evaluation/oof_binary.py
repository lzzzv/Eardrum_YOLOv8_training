"""
第一階段(Normal vs Abnormal)的 out-of-fold 指標,
以 MD5 去重後的獨立原圖計算。
"""
import csv, hashlib
from pathlib import Path
from collections import Counter, defaultdict
from ultralytics import YOLO

RUNS = Path("/home/rapids/pvc/eardrum_train/runs/classify")
BIN  = Path("/home/rapids/pvc/eardrum_train/folds_binary")
OUT  = Path("/home/rapids/pvc/eardrum_train/outputs")

def md5(p, chunk=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while (b := f.read(chunk)):
            h.update(b)
    return h.hexdigest()

rows = []
for k in range(5):
    w = RUNS / f"cvbin_fold{k}" / "weights" / "best.pt"
    if not w.exists():
        print(f"cvbin_fold{k} 缺 best.pt"); continue
    m = YOLO(str(w)); names = m.names
    for cd in sorted((BIN / f"fold{k}" / "val").iterdir()):
        if not cd.is_dir():
            continue
        for f in sorted(cd.iterdir()):
            if f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            r = m.predict(str(f), imgsz=320, verbose=False)[0]
            rows.append({"fold": k, "file": f.name, "true": cd.name,
                         "pred": names[int(r.probs.top1)],
                         "conf": round(float(r.probs.top1conf), 4),
                         "src": str(f.resolve())})
    print(f"fold{k} 完成")

print(f"\n原始筆數:{len(rows)}")
seen, uniq = set(), []
for r in rows:
    h = md5(Path(r["src"]))
    if h in seen: continue
    seen.add(h); uniq.append(r)
print(f"MD5 去重後:{len(uniq)}")

with open(OUT/"oof_binary_predictions.csv", "w", newline="") as f:
    w_ = csv.DictWriter(f, fieldnames=["fold","file","true","pred","conf","src"])
    w_.writeheader(); w_.writerows(uniq)

classes = sorted({r["true"] for r in uniq})
N = len(uniq)
correct = sum(1 for r in uniq if r["true"] == r["pred"])
per = {}
for c in classes:
    TP = sum(1 for r in uniq if r["true"]==c and r["pred"]==c)
    FP = sum(1 for r in uniq if r["true"]!=c and r["pred"]==c)
    FN = sum(1 for r in uniq if r["true"]==c and r["pred"]!=c)
    TN = N - TP - FP - FN
    rec = TP/(TP+FN) if TP+FN else 0.0
    pre = TP/(TP+FP) if TP+FP else 0.0
    spe = TN/(TN+FP) if TN+FP else 0.0
    f1  = 2*pre*rec/(pre+rec) if pre+rec else 0.0
    per[c] = dict(n=TP+FN, TP=TP, FP=FP, FN=FN, TN=TN, rec=rec, pre=pre, spe=spe, f1=f1)

macro_f1 = sum(per[c]["f1"] for c in classes)/len(classes)
bal_acc  = sum(per[c]["rec"] for c in classes)/len(classes)

print(f"\n=== 第一階段 out-of-fold(n={N})===")
print(f"accuracy          {correct/N:.4f}")
print(f"macro-F1          {macro_f1:.4f}")
print(f"balanced accuracy {bal_acc:.4f}")
print(f"\n{'類別':<12}{'原圖':>6}{'TP':>6}{'FP':>5}{'FN':>5}{'Recall':>9}{'Prec':>8}{'Spec':>8}{'F1':>8}")
print("-"*70)
for c in classes:
    d = per[c]
    print(f"{c:<12}{d['n']:>6}{d['TP']:>6}{d['FP']:>5}{d['FN']:>5}"
          f"{d['rec']:>9.3f}{d['pre']:>8.3f}{d['spe']:>8.3f}{d['f1']:>8.3f}")

cm = defaultdict(Counter)
for r in uniq:
    cm[r["true"]][r["pred"]] += 1
print(f"\nconfusion matrix(列=實際,欄=預測)")
print(f"{'':<12}" + "".join(f"{c:>12}" for c in classes))
for t in classes:
    print(f"{t:<12}" + "".join(f"{cm[t][p]:>12}" for p in classes))

with open(OUT/"oof_binary_per_class.csv","w",newline="") as f:
    w_ = csv.writer(f)
    w_.writerow(["class","n_originals","TP","FP","FN","TN","recall","precision","specificity","f1"])
    for c in classes:
        d = per[c]
        w_.writerow([c,d['n'],d['TP'],d['FP'],d['FN'],d['TN'],
                     f"{d['rec']:.4f}",f"{d['pre']:.4f}",f"{d['spe']:.4f}",f"{d['f1']:.4f}"])
print(f"\n已存 {OUT}/oof_binary_per_class.csv 與 oof_binary_predictions.csv")

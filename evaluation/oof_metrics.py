"""
以 2,389 張獨立原圖(MD5 去重後)計算 out-of-fold 指標。
輸出:整體指標、各類別 precision/recall/specificity/F1、合併 confusion matrix
"""
import csv, hashlib
from pathlib import Path
from collections import defaultdict, Counter

PRED = Path("/home/rapids/pvc/eardrum_train/outputs/cv_predictions.csv")
OUT  = Path("/home/rapids/pvc/eardrum_train/outputs")

def md5(p, chunk=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while (b := f.read(chunk)):
            h.update(b)
    return h.hexdigest()

rows = list(csv.DictReader(open(PRED)))
print(f"原始筆數:{len(rows)}")

seen, uniq = set(), []
for r in rows:
    h = md5(Path(r["src"]))
    if h in seen:
        continue
    seen.add(h)
    uniq.append(r)
print(f"MD5 去重後:{len(uniq)}")

classes = sorted({r["true"] for r in uniq})
N = len(uniq)
correct = sum(1 for r in uniq if r["true"] == r["pred"])

per = {}
for c in classes:
    TP = sum(1 for r in uniq if r["true"] == c and r["pred"] == c)
    FP = sum(1 for r in uniq if r["true"] != c and r["pred"] == c)
    FN = sum(1 for r in uniq if r["true"] == c and r["pred"] != c)
    TN = N - TP - FP - FN
    rec = TP/(TP+FN) if TP+FN else 0.0
    pre = TP/(TP+FP) if TP+FP else 0.0
    spe = TN/(TN+FP) if TN+FP else 0.0
    f1  = 2*pre*rec/(pre+rec) if pre+rec else 0.0
    per[c] = dict(n=TP+FN, TP=TP, FP=FP, FN=FN, rec=rec, pre=pre, spe=spe, f1=f1)

macro_f1 = sum(per[c]["f1"] for c in classes)/len(classes)
bal_acc  = sum(per[c]["rec"] for c in classes)/len(classes)

print(f"\n=== Out-of-fold(n={N} 張獨立原圖)===")
print(f"accuracy          {correct/N:.4f}")
print(f"macro-F1          {macro_f1:.4f}")
print(f"balanced accuracy {bal_acc:.4f}")

print(f"\n{'類別':<30}{'原圖':>5}{'Recall':>9}{'Prec':>8}{'Spec':>8}{'F1':>8}")
print("-" * 70)
for c in sorted(classes, key=lambda x: -per[x]["f1"]):
    d = per[c]
    print(f"{c:<30}{d['n']:>5}{d['rec']:>9.3f}{d['pre']:>8.3f}{d['spe']:>8.3f}{d['f1']:>8.3f}")

cm = defaultdict(Counter)
for r in uniq:
    cm[r["true"]][r["pred"]] += 1
with open(OUT/"oof_confusion_matrix.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["true\\pred"] + classes)
    for t in classes:
        w.writerow([t] + [cm[t][p] for p in classes])
with open(OUT/"oof_per_class.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["class","n_originals","TP","FP","FN","recall","precision","specificity","f1"])
    for c in classes:
        d = per[c]
        w.writerow([c,d['n'],d['TP'],d['FP'],d['FN'],
                    f"{d['rec']:.4f}",f"{d['pre']:.4f}",f"{d['spe']:.4f}",f"{d['f1']:.4f}"])
print(f"\n已存 {OUT}/oof_confusion_matrix.csv 與 oof_per_class.csv")

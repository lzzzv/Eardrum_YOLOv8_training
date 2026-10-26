"""
彙整 5-fold 結果:整體 top1、balanced accuracy、macro-F1,
以及各類別 recall/precision/F1 的平均值 ± 標準差。
每個 fold 用自己的 best.pt 對該 fold 的 val(未增強原圖)推論。
"""
import statistics as st
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

RUNS  = Path("/home/rapids/pvc/eardrum_train/runs/classify")
FOLDS = Path("/home/rapids/pvc/eardrum_train/folds")
N = 5

top1_list, done = [], []
per_class = defaultdict(dict)     # cls -> fold -> (rec, pre, f1, n)

for k in range(N):
    w = RUNS / f"cv_fold{k}" / "weights" / "best.pt"
    if not w.exists():
        print(f"fold{k}: 尚未完成,跳過")
        continue
    model = YOLO(str(w)); names = model.names
    rows = []
    for cd in sorted((FOLDS / f"fold{k}" / "val").iterdir()):
        if not cd.is_dir():
            continue
        for f in sorted(cd.iterdir()):
            if f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            r = model.predict(str(f), imgsz=320, verbose=False)[0]
            rows.append((cd.name, names[int(r.probs.top1)]))
    t1 = sum(1 for t, p in rows if t == p) / len(rows)
    top1_list.append(t1); done.append(k)
    for c in sorted({t for t, _ in rows}):
        TP = sum(1 for t, p in rows if t == c and p == c)
        FP = sum(1 for t, p in rows if t != c and p == c)
        FN = sum(1 for t, p in rows if t == c and p != c)
        rec = TP/(TP+FN) if TP+FN else 0.0
        pre = TP/(TP+FP) if TP+FP else 0.0
        f1  = 2*pre*rec/(pre+rec) if pre+rec else 0.0
        per_class[c][k] = (rec, pre, f1, TP+FN)
    print(f"fold{k}: val {len(rows)} 張原圖,top1 = {t1:.4f}")

def ms(v):
    if not v: return "-"
    return f"{st.mean(v):.3f} ± {st.stdev(v):.3f}" if len(v) > 1 else f"{v[0]:.3f}"

if top1_list:
    classes = sorted(per_class)
    bal = [st.mean([per_class[c][k][0] for c in classes if k in per_class[c]]) for k in done]
    mf1 = [st.mean([per_class[c][k][2] for c in classes if k in per_class[c]]) for k in done]

    print(f"\n=== {len(done)}-fold 彙整 ===")
    print(f"top1(Normal 約佔 val 七成,僅供參考):{ms(top1_list)}")
    print(f"balanced accuracy(各類 recall 平均):{ms(bal)}")
    print(f"macro-F1(各類 F1 平均)           :{ms(mf1)}")

    print(f"\n{'類別':<30}{'val原圖/fold':>13}{'Recall':>17}{'Precision':>17}{'F1':>17}")
    print("-" * 95)
    for c in classes:
        d = per_class[c]
        n   = st.mean([d[k][3] for k in d])
        rec = [d[k][0] for k in d]
        pre = [d[k][1] for k in d]
        f1  = [d[k][2] for k in d]
        print(f"{c:<30}{n:>13.1f}{ms(rec):>17}{ms(pre):>17}{ms(f1):>17}")

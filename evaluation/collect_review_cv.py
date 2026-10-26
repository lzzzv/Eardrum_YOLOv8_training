"""
以 5-fold 的預測結果重建臨床複核清單。
每張原圖在其所屬 fold 被驗證一次,故可涵蓋全部獨立原圖。
"""
import csv, shutil
from pathlib import Path
from collections import Counter
from ultralytics import YOLO

RUNS  = Path("/home/rapids/pvc/eardrum_train/runs/classify")
FOLDS = Path("/home/rapids/pvc/eardrum_train/folds")
OUT   = Path("/home/rapids/pvc/review_for_clinician_cv")
CSV   = Path("/home/rapids/pvc/eardrum_train/outputs/cv_predictions.csv")

rows = []
for k in range(5):
    w = RUNS / f"cv_fold{k}" / "weights" / "best.pt"
    if not w.exists():
        print(f"fold{k} 缺 best.pt,跳過"); continue
    m = YOLO(str(w)); names = m.names
    for cd in sorted((FOLDS / f"fold{k}" / "val").iterdir()):
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

CSV.parent.mkdir(parents=True, exist_ok=True)
with open(CSV, "w", newline="") as fo:
    w_ = csv.DictWriter(fo, fieldnames=["fold","file","true","pred","conf","src"])
    w_.writeheader(); w_.writerows(rows)

correct = sum(1 for r in rows if r["true"] == r["pred"])
print(f"\n共 {len(rows)} 張獨立原圖,答對 {correct},整體 {correct/len(rows):.4f}")

GROUPS = {
 "01_高信心誤判":        lambda r: r["true"] != r["pred"] and r["conf"] >= 0.90,
 "02_Otomycosis_11":     lambda r: r["file"].startswith("Otomycosis_11"),
 "03_通氣管疑遭遮蔽":     lambda r: r["true"]=="VentilationTube" and r["pred"]=="CerumenImpaction",
 "04_AOM與Effusion互判": lambda r: {r["true"],r["pred"]}=={"AcuteOtitisMedia","Effusion"},
 "05_穿孔與慢性化膿互判": lambda r: {r["true"],r["pred"]}=={"TympanicMembranePerforation","ChronicSuppurative"},
}

manifest = []
for gname, cond in GROUPS.items():
    d = OUT / gname
    d.mkdir(parents=True, exist_ok=True)
    n = 0
    for r in rows:
        if not cond(r):
            continue
        shutil.copy2(r["src"], d / f"{r['true']}__{r['file']}")
        manifest.append([gname, r["true"], r["pred"], r["conf"], r["file"]])
        n += 1
    print(f"{gname:<24}{n:>4} 張")

with open(OUT / "manifest.csv", "w", newline="") as fo:
    w_ = csv.writer(fo)
    w_.writerow(["分組","標註類別","模型預測","信心","檔名"])
    w_.writerows(manifest)

print(f"\n輸出:{OUT}")
print("\n高信心誤判的主要配對:")
for (t,p),n in Counter((r["true"],r["pred"]) for r in rows
                       if r["true"]!=r["pred"] and r["conf"]>=0.90).most_common(10):
    print(f"  {t:<30} → {p:<30}{n}")

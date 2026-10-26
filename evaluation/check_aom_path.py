"""驗證:第二階段 AOM 的改善,是否來自「誤判為 Normal」路徑消失"""
from pathlib import Path
from collections import Counter
from ultralytics import YOLO

RUNS  = Path("/home/rapids/pvc/eardrum_train/runs/classify")
FOLDS = Path("/home/rapids/pvc/eardrum_train/folds")
ST2   = Path("/home/rapids/pvc/eardrum_train/folds_stage2")
TARGET = ["AcuteOtitisMedia", "Otomycosis", "AcuteMyringitis"]

for cls in TARGET:
    flat, st2 = Counter(), Counter()
    for k in range(5):
        for tag, runs_dir, base, bucket in (
            ("flat", f"cv_fold{k}",  FOLDS, flat),
            ("st2",  f"cvst2_fold{k}", ST2, st2)):
            w = RUNS / runs_dir / "weights" / "best.pt"
            d = base / f"fold{k}" / "val" / cls
            if not w.exists() or not d.exists():
                continue
            m = YOLO(str(w)); names = m.names
            for f in sorted(d.iterdir()):
                if f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                r = m.predict(str(f), imgsz=320, verbose=False)[0]
                bucket[names[int(r.probs.top1)]] += 1
    tot = sum(flat.values())
    print(f"\n=== {cls}(共 {tot} 張驗證原圖)===")
    print(f"{'預測為':<32}{'10類模型':>9}{'第二階段':>10}")
    print("-" * 52)
    for p in sorted(set(flat) | set(st2), key=lambda x: -(flat[x] + st2[x])):
        mark = "  ← 正確" if p == cls else ""
        print(f"{p:<32}{flat[p]:>9}{st2[p]:>10}{mark}")

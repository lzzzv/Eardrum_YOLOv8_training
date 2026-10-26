"""由去重切分的資料集建立 Normal / Abnormal 二元資料集(App 部署模型使用)。"""
from pathlib import Path

SRC = Path("/home/rapids/pvc/eardrum_train/dataset_dedup_cls")
DST = Path("/home/rapids/pvc/eardrum_train/dataset_binary_cls")

for split in ("train", "val"):
    for cls_dir in sorted((SRC / split).iterdir()):
        if not cls_dir.is_dir():
            continue
        label = "Normal" if cls_dir.name == "Normal" else "Abnormal"
        out = DST / split / label
        out.mkdir(parents=True, exist_ok=True)
        for f in cls_dir.iterdir():
            link = out / f"{cls_dir.name}__{f.name}"
            if not link.exists():
                link.symlink_to(f.resolve())

for split in ("train", "val"):
    print(split, {d.name: len(list(d.iterdir())) for d in sorted((DST/split).iterdir())})

"""
從既有的 folds 衍生兩套階層式資料集:
  folds_binary : Normal vs Abnormal(9 病灶合併)
  folds_stage2 : 只留 9 種病灶,排除 Normal
val 仍為未增強原圖(繼承自 folds)。
"""
from pathlib import Path

FOLDS = Path("/home/rapids/pvc/eardrum_train/folds")
BIN   = Path("/home/rapids/pvc/eardrum_train/folds_binary")
ST2   = Path("/home/rapids/pvc/eardrum_train/folds_stage2")
N = 5

def link(src, dst):
    if not dst.exists():
        dst.symlink_to(src.resolve())

for k in range(N):
    for split in ("train", "val"):
        for cd in sorted((FOLDS / f"fold{k}" / split).iterdir()):
            if not cd.is_dir():
                continue
            # 二元
            label = "Normal" if cd.name == "Normal" else "Abnormal"
            outb = BIN / f"fold{k}" / split / label
            outb.mkdir(parents=True, exist_ok=True)
            # 第二階段(排除 Normal)
            outs = None
            if cd.name != "Normal":
                outs = ST2 / f"fold{k}" / split / cd.name
                outs.mkdir(parents=True, exist_ok=True)
            for f in cd.iterdir():
                link(f, outb / f"{cd.name}__{f.name}")
                if outs is not None:
                    link(f, outs / f.name)

for name, base in (("binary", BIN), ("stage2", ST2)):
    print(f"--- {name} ---")
    for k in range(N):
        c = {}
        for split in ("train", "val"):
            c[split] = sum(1 for _ in (base/f"fold{k}"/split).rglob("*.jpg"))
        vd = {d.name: len(list(d.iterdir())) for d in sorted((base/f"fold{k}"/"val").iterdir())}
        print(f"  fold{k}: train {c['train']:>5}  val {c['val']:>4}   {vd}")

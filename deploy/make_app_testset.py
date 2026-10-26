#!/usr/bin/env python3
"""挑選 App↔TFLite 一致性測試圖：邊界圖(全部納入)+高信心兩類各數張，
複製到資料夾並產出 TFLite 標準答案表。"""
import csv, shutil
import numpy as np
import tflite_runtime.interpreter as tflite
from ultralytics import YOLO
from pathlib import Path

CSV="outputs/oof_binary_predictions.csv"
TFL="best_float16.tflite"
DST=Path("/home/rapids/pvc/app_testset")
if DST.exists(): shutil.rmtree(DST)
DST.mkdir()

rows=list(csv.DictReader(open(CSV)))
border=sorted([r for r in rows if float(r["conf"])<0.75], key=lambda x:float(x["conf"]))
hi_n=[r for r in rows if r["true"]=="Normal" and float(r["conf"])>0.999][:4]
hi_a=[r for r in rows if r["true"]=="Abnormal" and float(r["conf"])>0.999][:4]
picked = border + hi_n + hi_a

m=YOLO("runs/classify/binary_s320/weights/best.pt"); names=m.names
it=tflite.Interpreter(TFL); it.allocate_tensors()
i=it.get_input_details()[0]; o=it.get_output_details()[0]

table=[]
for idx,r in enumerate(picked):
    src=Path(r["src"])
    newname=f"test_{idx:02d}_{src.name}"
    shutil.copy(src, DST/newname)
    # TFLite 標準答案(用官方前處理)
    res=m.predict(str(src),imgsz=320,device="cpu",verbose=False)[0]
    im=m.predictor.preprocess([res.orig_img])
    xt=im.cpu().numpy().transpose(0,2,3,1).astype(np.float32)
    it.set_tensor(i["index"],xt); it.invoke()
    to=it.get_tensor(o["index"])[0]
    tfl_pred=names[int(to.argmax())]; tfl_conf=float(to.max())
    table.append([newname, r["true"], tfl_pred, f"{tfl_conf:.4f}"])

with open(DST/"tflite_標準答案.csv","w",encoding="utf-8-sig",newline="") as f:
    w=csv.writer(f)
    w.writerow(["檔名","真實類別","TFLite判為","TFLite信心"])
    w.writerows(table)

print(f"測試集: {len(picked)} 張 (邊界{len(border)}+高信心Normal{len(hi_n)}+高信心Abnormal{len(hi_a)})")
print(f"-> 圖片與標準答案: {DST}")
for t in table: print(" ", t)

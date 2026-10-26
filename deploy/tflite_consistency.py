"""以 2,389 張原圖逐張比對 PyTorch 與 TFLite(float16)的判讀結果,兩邊使用相同的前處理張量。"""
import csv,time,numpy as np,tflite_runtime.interpreter as tflite
from ultralytics import YOLO
m=YOLO("runs/classify/binary_s320/weights/best.pt");names=m.names
it=tflite.Interpreter("best_float16.tflite");it.allocate_tensors()
i=it.get_input_details()[0];o=it.get_output_details()[0]
rows=list(csv.DictReader(open("outputs/oof_binary_predictions.csv")))
agree=0;total=0;diffs=[];mism=[];t0=time.time()
for r in rows:
    try: res=m.predict(r["src"],imgsz=320,device="cpu",verbose=False)[0]
    except: continue
    pt=res.probs.data.numpy();pp=names[int(pt.argmax())];pc=float(pt.max())
    im=m.predictor.preprocess([res.orig_img]);xt=im.cpu().numpy().transpose(0,2,3,1).astype(np.float32)
    it.set_tensor(i["index"],xt);it.invoke();to=it.get_tensor(o["index"])[0]
    tp=names[int(to.argmax())];tc=float(to.max())
    total+=1
    if pp==tp: agree+=1
    else: mism.append((r["file"],pp,pc,tp,tc))
    diffs.append(abs(pc-tc))
print(f"比對: {total}, 一致: {agree}/{total} = {agree/total*100:.2f}%")
print(f"prob 平均差 {np.mean(diffs):.5f}, 最大差 {np.max(diffs):.5f}, 耗時 {time.time()-t0:.0f}s")
[print(f"  {f}: PT={pp}({pc:.3f}) TFLite={tp}({tc:.3f})") for f,pp,pc,tp,tc in mism[:20]]

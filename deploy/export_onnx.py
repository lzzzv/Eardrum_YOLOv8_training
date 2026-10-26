from ultralytics import YOLO

m = YOLO("/home/rapids/pvc/eardrum_train/runs/classify/binary_s320/weights/best.pt")
m.export(format="onnx", imgsz=320)

BASE=/home/rapids/pvc/eardrum_train
for k in 0 1 2 3 4; do
  echo "===== binary fold $k ====="
  yolo classify train model=yolov8s-cls.pt data=$BASE/folds_binary/fold$k \
    epochs=50 imgsz=320 batch=64 workers=8 seed=0 name=cvbin_fold$k
done
for k in 0 1 2 3 4; do
  echo "===== stage2 fold $k ====="
  yolo classify train model=yolov8s-cls.pt data=$BASE/folds_stage2/fold$k \
    epochs=50 imgsz=320 batch=64 workers=8 seed=0 name=cvst2_fold$k
done

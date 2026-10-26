FOLDS=/home/rapids/pvc/eardrum_train/folds
for k in 0 1 2 3 4; do
  echo "========== fold $k =========="
  yolo classify train model=yolov8s-cls.pt data=$FOLDS/fold$k \
    epochs=50 imgsz=320 batch=64 workers=8 seed=0 name=cv_fold$k
done

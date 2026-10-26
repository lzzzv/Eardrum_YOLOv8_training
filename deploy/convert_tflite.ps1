# 在獨立的 conda 環境(onnx2tf + tensorflow)執行,避免影響訓練環境的 torch/numpy 版本
# 輸出 tflite_output/best_float32.tflite 與 best_float16.tflite
onnx2tf -i best.onnx -o tflite_output -nuo

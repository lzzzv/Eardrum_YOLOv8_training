# 耳鏡影像 YOLOv8 模型訓練

在臺北榮民總醫院資訊室實習訓練的耳鏡影像分類模型，包含資料整理、模型訓練、驗證分析與轉換成手機格式的程式。訓練好的二元模型（正常／異常）用在 [耳膜判讀 App](https://github.com/lzzzv/Eardrum_interpretation)。

> 僅有程式碼及結果數據。

## 使用技術

Python、Ultralytics YOLOv8（yolov8s-cls）、北榮 A100 GPU 伺服器（JupyterLab）、ONNX、TensorFlow Lite（onnx2tf）

## 程式架構

```
├── data/          資料整理
│   ├── inventory.py         盤點原圖與增強影像數量
│   ├── batch_check.py       檢查影像是否來自不同批次
│   ├── make_folds.py        MD5 合併重複照片後，建立 grouped 5-fold
│   ├── make_folds_hier.py   由 5-fold 建立二元（正常／異常）與 9 類病灶資料集
│   └── make_binary.py       由去重後的單一切分建立 App 用的二元資料集
├── train/         訓練指令
│   ├── run_folds.sh         10 類模型，5 個 fold
│   ├── run_hier_folds.sh    分層實驗（第一、二階段），5 個 fold
│   └── run_binary.sh        App 用的二元模型
├── evaluation/    驗證分析
│   ├── cv_summary.py            5-fold 平均 ± 標準差
│   ├── collect_review_cv.py     產生 5-fold 預測結果，並整理高信心誤判
│   ├── oof_metrics.py           10 類 out-of-fold 指標與 confusion matrix
│   ├── oof_binary.py            二元 out-of-fold 指標
│   ├── check_aom_path.py        分層架構對急性中耳炎等少數類別的影響
│   └── collect_csp_review.py    整理慢性化膿性中耳炎與鼓膜穿孔的互判
└── deploy/        轉換成手機格式
    ├── export_onnx.py           PyTorch-ONNX
    ├── convert_tflite.ps1       ONNX-TensorFlow Lite
    ├── tflite_consistency.py    逐張比對轉換前後的判讀結果
    └── make_app_testset.py      挑選App測試用影像
```

流程：

1. 資料整理：院內 12,046 張影像中，約八成是原圖經資料增強產生的。依檔名分出 2,426 張原圖，再用 MD5 合併內容相同的照片，得到 2,389 張獨立原圖（正常 1,693 張，9 類病灶 696 張）。
2. 訓練與驗證：用 grouped 5-fold 交叉驗證，同一張原圖、它的增強影像，以及內容相同的照片都放在同一組，驗證集只放未增強的原圖。
3. 分層實驗：比較「先判斷正常或異常，再把異常細分成 9 類病灶」的做法。第二階段和直接分 10 類的結果差不多，所以 App 只用正常／異常判讀。
4. 模型轉換：PyTorch → ONNX → TensorFlow Lite，逐張比對轉換前後的結果。

## 結果

以 2,389 張獨立原圖的out-of-fold結果計算：

| | accuracy | macro-F1 | balanced accuracy |
|---|---|---|---|
| 10 類 | 0.957 | 0.845 | 0.812 |
| 正常／異常 | 0.997 | 0.996 | 0.997 |

- 正常／異常：696 張異常原圖有 2 張漏判。
- App用的二元模型（binary_s320）訓練設定相同，但用去重後的單一切分訓練。轉成TensorFlow Lite（float16）後，2,389 張判讀結果 100% 一致。

限制： 急性鼓膜炎、耳黴菌症、急性中耳炎的原圖只有 7～15 張，這幾類的結果只能當參考。院內影像都用在訓練或驗證，沒有另外的測試集，模型在新影像上的表現還沒驗證。
```

- `make_binary.py` 讀的是去重後的單一切分（dataset_dedup_cls），由 split_dedup.py 產生，不在這個專案裡。
- `convert_tflite.ps1` 要在另一個裝 onnx2tf 與 tensorflow 的 conda 環境執行，避免影響訓練環境的套件版本。
- `tflite_consistency.py` 與 `make_app_testset.py` 需要另外安裝 tflite_runtime，並先跑過 `oof_binary.py`。這兩支用相對路徑（runs/、outputs/），要在訓練資料夾下執行，並把 best_float16.tflite 放在同一層。

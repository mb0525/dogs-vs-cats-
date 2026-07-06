# Dogs vs. Cats — YOLO Image Classification

使用 Ultralytics YOLO 預訓練影像分類模型，自動判斷圖片中的動物是貓或狗，並輸出包含預測類別、機率與信心分數的 CSV 報告。

## 功能特色

- 批次辨識 JPG 圖片
- 使用輕量化 `yolo11n-cls.pt` 預訓練模型
- 支援自訂輸入資料夾、輸出位置與批次大小
- 輸出貓／狗機率、最終標籤與信心分數
- 使用固定大小批次，避免大量圖片造成記憶體不足
- 支援 `--limit` 進行少量圖片快速測試

## 專案結構

```text
dogs-vs-cats/
├── cat-or-dog-image-classification-challenge/
│   └── data_mixed/                  # 待辨識圖片
├── classify_cats_dogs_yolo.py       # YOLO 分類主程式
├── cat_dog_predictions.csv          # 完整辨識結果
├── cat_dog_predictions_test.csv     # 少量測試結果
├── sample-how-to-do.ipynb           # Kaggle 參考 Notebook
├── yolo11n-cls.pt                   # YOLO 預訓練權重
└── README.MD
```

資料集 ZIP 與 Kaggle API 回傳的中繼檔可能也會保留在本機，但不是執行分類所必需的檔案。

## 環境需求

- Windows 10 或 Windows 11
- Python 3.12
- 建議至少 8 GB RAM
- 本專案可使用 CPU 執行，不要求 NVIDIA GPU

主要 Python 套件：

- `ultralytics`
- `torch`
- `torchvision`

## 快速開始

### 1. 取得專案

```powershell
git clone <repository-url>
cd dogs-vs-cats
```

若專案已在本機，可直接切換到專案根目錄。

### 2. 建立虛擬環境（建議）

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. 安裝套件

```powershell
python -m pip install --upgrade pip
python -m pip install ultralytics
```

如果 `python` 尚未加入 PATH，可使用目前安裝位置：

```powershell
& "$HOME\AppData\Local\Programs\Python\Python312\python.exe" -m pip install ultralytics
```

### 4. 準備圖片

預設輸入資料夾為：

```text
cat-or-dog-image-classification-challenge/data_mixed/
```

將 JPG 圖片放入此資料夾，或在執行時使用 `--input` 指定其他位置。

### 5. 執行分類

```powershell
python classify_cats_dogs_yolo.py
```

預設結果會輸出至：

```text
cat_dog_predictions.csv
```

## 使用方式

### 少量測試

只辨識前 10 張圖片：

```powershell
python classify_cats_dogs_yolo.py `
  --limit 10 `
  --output cat_dog_predictions_test.csv
```

### 自訂輸入與輸出

```powershell
python classify_cats_dogs_yolo.py `
  --input cat-or-dog-image-classification-challenge/data_mixed `
  --output cat_dog_predictions.csv `
  --batch 32
```

### 命令列參數

| 參數 | 預設值 | 說明 |
| --- | --- | --- |
| `--input` | `cat-or-dog-image-classification-challenge/data_mixed` | JPG 圖片資料夾 |
| `--output` | `cat_dog_predictions.csv` | CSV 輸出位置 |
| `--model` | `yolo11n-cls.pt` | YOLO 分類模型或權重路徑 |
| `--batch` | `32` | 每批處理的圖片數量 |
| `--limit` | 無 | 僅處理前 N 張圖片 |

查看完整說明：

```powershell
python classify_cats_dogs_yolo.py --help
```

## 輸出格式

`cat_dog_predictions.csv` 包含以下欄位：

| 欄位 | 說明 |
| --- | --- |
| `image` | 原始圖片檔名 |
| `img` | 圖片編號或檔名主體 |
| `label` | 預測結果：`cat` 或 `dog` |
| `is_dog` | 正規化後的狗機率，範圍為 0–1 |
| `confidence` | 最終分類的信心分數 |
| `cat_score` | ImageNet 貓類別的原始機率總和 |
| `dog_score` | ImageNet 犬種類別的原始機率總和 |

輸出範例：

```csv
image,img,label,is_dog,confidence,cat_score,dog_score
0.jpg,0,dog,0.999829,0.999829,0.000165,0.963984
1.jpg,1,cat,0.414145,0.585855,0.375915,0.265737
```

## 模型方法

本專案使用 YOLO11 ImageNet 預訓練分類模型，不另外使用此資料集訓練模型。分類流程如下：

1. YOLO 為圖片產生 1,000 個 ImageNet 類別機率。
2. 加總 `tabby`、`tiger cat`、`Persian cat`、`Siamese cat` 與 `Egyptian cat` 等貓類別機率。
3. 加總 Chihuahua 至 Mexican hairless 等犬種類別機率。
4. 將貓狗分數正規化，產生 `is_dog`。
5. `is_dog >= 0.5` 判定為狗，否則判定為貓。

## 目前結果

本機已完成 8,005 張圖片的辨識：

| 項目 | 數量 |
| --- | ---: |
| 圖片總數 | 8,005 |
| 預測為貓 | 3,169 |
| 預測為狗 | 4,836 |
| 信心分數低於 70% | 520 |
| 遺漏圖片 | 0 |
| 重複圖片記錄 | 0 |

以上數字是模型預測統計，不代表資料集的真實類別分布或模型準確率。

## 已知限制

- 模型未針對此 Kaggle 資料集微調。
- 信心分數不是經過校準的真實正確率。
- 圖片若同時包含貓與狗，分類模型仍只會輸出一個主要類別。
- 背景複雜、動物過小、遮擋或非典型品種可能降低辨識品質。
- 建議人工檢查信心分數低於 70% 的圖片。

若需要更可靠的評估，應取得人工標籤，切分訓練／驗證資料，微調 YOLO 分類模型並計算 accuracy、precision、recall 與 confusion matrix。

## 資料來源

- [Cat or Dog: Image Classification Challenge](https://www.kaggle.com/competitions/cat-or-dog-image-classification-challenge)
- [Sample How To Do — Kaggle Notebook](https://www.kaggle.com/code/haradibots/sample-how-to-do)
- [Ultralytics](https://www.ultralytics.com/)

資料集的使用方式與授權條款以 Kaggle 競賽頁面為準。

## 安全注意事項

Kaggle API Token 屬於敏感憑證。請勿將以下檔案加入版本控制或分享給他人：

```text
C:\Users\<你的帳號>\.kaggle\kaggle.json
```

## 授權

此專案目前未附加軟體授權檔。若要公開發布或允許他人使用、修改與散布，請先新增合適的 `LICENSE`，並分別確認 Kaggle 資料集及 YOLO 模型權重的授權條款。

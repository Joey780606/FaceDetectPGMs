# PLAN.md — p06 Haar Cascade + PyTorch CNN

## 專案目標
使用 OpenCV Haar Cascade 做人臉偵測，PyTorch 小型 CNN 做人臉辨識。
授權完全商用安全（Apache 2.0 / BSD / MIT）。

---

## 確認的設計決策

| 項目 | 決定值 |
|------|--------|
| 每人收集張數 | 60 張（augmentation 後 ~300 張） |
| UI 提示 | 顯示「請保持正臉」提醒 |
| CNN 輸入大小 | 96×96（灰階），參數量 ~200K |
| 新增人員策略 | 方案 B：保留舊資料，擴展 softmax 層，fine-tune |
| Unknown 門檻 | 信心值 < 60% 顯示 "Unknown" |

---

## 授權安全確認

| 套件 | 授權 | 商用 |
|------|------|------|
| opencv-python | Apache 2.0 | ✅ |
| PyTorch | BSD 3-Clause | ✅ |
| customtkinter | MIT | ✅ |
| Pillow | HPND (open) | ✅ |
| numpy | BSD | ✅ |
| Haar Cascade XML | Apache 2.0 (OpenCV 內建) | ✅ |

---

## 專案結構

```
p06_HaarPyTorchCNN/
├── main.py              # UI 主程式（CustomTkinter，仿 Refmain.py）
├── face_detector.py     # Haar Cascade 偵測，回傳臉部 ROI
├── face_recognizer.py   # CNN 模型定義 + 訓練 + 推論
├── model_store.py       # 模型 / 標籤 / 訓練資料的儲存與載入
├── requirements.txt     # 套件依賴
├── PLAN.md              # 本文件
├── CLAUDE.md            # Claude Code 專案指引
├── data/
│   └── faces/
│       ├── Joey/        # 每人 60 張 96×96 灰階 ROI (img_001.jpg ...)
│       └── Amy/
└── model/
    └── face_cnn.pth     # 訓練好的模型 + 標籤 dict
```

---

## CNN 架構（FaceCNN）

```
Input:  96×96×1  (灰階)

Conv1:  32 filters, 3×3, padding=1 → BatchNorm → ReLU
MaxPool 2×2  →  48×48×32

Conv2:  64 filters, 3×3, padding=1 → BatchNorm → ReLU
MaxPool 2×2  →  24×24×64

Conv3: 128 filters, 3×3, padding=1 → BatchNorm → ReLU
GlobalAvgPool  →  128-dim 向量       ← 可作為人臉嵌入向量

FC:    128 → N（動態 num_classes）
Softmax
```

參數量估計：~200K，CPU 推論 < 5ms/張。

---

## 資料增強（每人 60 張 → ~300 張）

| 增強方式 | 參數 |
|---------|------|
| 水平翻轉 | 50% 機率 |
| 亮度調整 | ±20% |
| 隨機旋轉 | ±10° |
| 隨機縮放 | ±10% |
| 原始圖 | 1× |

→ 共約 5× augmentation，60 張 → ~300 張有效訓練樣本。

---

## 方案 B：動態擴展策略（新增人員流程）

```
第1人加入：
  - 收集 60 張 → 存 data/faces/Name1/
  - 建立新 CNN（1 class）→ 從頭訓練 50 epochs
  - 儲存 model/face_cnn.pth

第N人加入（N≥2）：
  - 收集 60 張 → 存 data/faces/NameN/
  - 載入舊 model weights（backbone 保留）
  - 擴展最後 FC 層：(N-1) → N classes
  - 讀取所有人的儲存資料（data/faces/所有人/）
  - Fine-tune 20 epochs（比從頭快，從好的初始值出發）
  - 儲存新 model/face_cnn.pth

移除人員：
  - 刪除 data/faces/NameX/
  - 重建 CNN（classes-1）→ 重新訓練 50 epochs（用剩餘所有人資料）
```

---

## UI 設計（仿 Refmain.py，4-row layout）

```
┌─────────────────────────────────────────────────────┐
│ Row0: [Detect]   辨識結果：Joey (92%)                │
├─────────────────────────────────────────────────────┤
│ Row1上: [姓名輸入框]  [Learning]  [Remove]           │
│         「請保持正臉，勿大幅度轉動頭部」（學習時顯示）│
│ Row1下: 剩餘時間: 25s  [========  ]                  │
│         已收集張數: 45  [=======   ] （學習時顯示）   │
├─────────────────────────────────────────────────────┤
│ Row2: Webcam 畫面（人臉框 + 名字標籤）               │
├─────────────────────────────────────────────────────┤
│ Row3: Log 區域（訓練進度、結果摘要）                  │
└─────────────────────────────────────────────────────┘
```

---

## 各模組職責

### `face_detector.py` — HaarFaceDetector
- `__init__()`: 載入 OpenCV 內建 haarcascade_frontalface_default.xml
- `detect(Frame) → list[ROI_96x96]`: 偵測所有人臉，裁切並縮放為 96×96 灰階

### `face_recognizer.py` — FaceRecognizer
- `FaceCNN(nn.Module)`: CNN 模型定義
- `FaceRecognizer`:
  - `train(DataDir, ExistingModelPath) → None`: 訓練 / fine-tune
  - `predict(Roi) → (name, confidence)`: 推論單張 ROI
  - `loadModel(Path) → None`
  - `saveModel(Path) → None`

### `model_store.py` — ModelStore
- `saveTrainingImage(Name, Roi) → None`: 儲存學習時的 ROI 圖片
- `listPersons() → list[str]`: 列出已有人員
- `removePerson(Name) → None`: 刪除資料夾
- `getDataDir() → str`: 回傳 data/faces 路徑
- `getModelPath() → str`: 回傳 model/face_cnn.pth 路徑

### `main.py` — MainApp（CustomTkinter）
- `WebcamManager`: 背景執行緒持續讀取（複用 Refmain.py 架構）
- `MainApp`: 主 UI，整合偵測 / 學習 / 顯示

---

## 學習流程（Learning Mode）

```
1. 使用者輸入姓名 → 點 Learning
2. UI 顯示「請保持正臉」提示
3. 每 500ms 抓一個 frame（LEARN_TICK_MS = 500）
4. Haar 偵測到臉 → 裁切 96×96 → 儲存到 data/faces/{name}/
5. 進度條同步更新（已收集 N/60 張，剩餘秒數）
6. 達到 60 張或逾時（90 秒）→ 觸發訓練
7. 訓練在背景執行緒跑，Log 區顯示 epoch / loss
8. 訓練完成 → 彈出成功訊息
```

---

## 辨識流程（Detect Mode）

```
1. 點 Detect → 每 300ms 推論一次（DETECT_TICK_MS = 300）
2. Haar 偵測臉部 ROI → CNN 推論 → (name, confidence)
3. confidence >= 60%：顯示名字
   confidence <  60%：顯示 "Unknown"
4. Webcam 畫面疊加：人臉框 + 名字標籤
```

---

## 實作順序

- [ ] Step 1: `requirements.txt`
- [ ] Step 2: `face_detector.py`（Haar 偵測）
- [ ] Step 3: `face_recognizer.py`（CNN 模型 + 訓練 + 推論）
- [ ] Step 4: `model_store.py`（資料 / 模型 I/O）
- [ ] Step 5: `main.py`（UI + 整合）
- [ ] Step 6: 測試與調整

---

## 常數定義（main.py 頂部）

```python
LEARN_TARGET_FRAMES   = 60     # 學習目標張數
LEARN_TIMEOUT_SECONDS = 90     # 學習逾時（秒）
LEARN_TICK_MS         = 500    # 學習抓圖間隔（ms）
DETECT_TICK_MS        = 300    # 辨識推論間隔（ms）
UI_REFRESH_MS         = 30     # Webcam 畫面更新間隔（ms）
UNKNOWN_THRESHOLD     = 0.60   # 信心值低於此 → Unknown
CNN_INPUT_SIZE        = 96     # CNN 輸入影像邊長（px）
AUGMENT_FACTOR        = 5      # 資料增強倍率
TRAIN_EPOCHS_FULL     = 50     # 從頭訓練 epoch 數
TRAIN_EPOCHS_FINETUNE = 20     # Fine-tune epoch 數
```

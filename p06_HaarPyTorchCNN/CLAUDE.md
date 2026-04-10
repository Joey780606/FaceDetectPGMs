# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

使用 Haar Cascade 做人臉偵測，PyTorch 小型 CNN 做人臉辨識。
授權完全商用安全（Apache 2.0 / BSD / MIT）。

## Architecture

```
Webcam → HaarFaceDetector → 裁切 96×96 灰階 ROI
                                  ↓
                           FaceRecognizer (CNN)
                           ↓               ↓
                        已知人員        __unknown__
                     (name + conf%)    (conf < 60%)
                                  ↓
                              顯示結果
```

### 模組結構

```
p06_HaarPyTorchCNN/
├── main.py              # UI 主程式（CustomTkinter，4-row layout）
├── face_detector.py     # HaarFaceDetector：偵測人臉，回傳 96×96 灰階 ROI
├── face_recognizer.py   # FaceCNN + FaceDataset + FaceRecognizer
├── model_store.py       # ModelStore：訓練圖片 / 模型檔案 I/O
├── requirements.txt     # 依賴套件
├── data/
│   └── faces/
│       ├── {人名}/      # 每人 60 張 96×96 灰階 ROI
│       └── __unknown__/ # 合成 unknown 樣本（每次訓練自動產生，100 張）
└── model/
    └── face_cnn.pth     # 訓練好的模型 + 標籤 dict
```

### CNN 架構（FaceCNN）

```
Input: 96×96×1（灰階）
Conv(32) + BN + ReLU → MaxPool(2) → 48×48×32
Conv(64) + BN + ReLU → MaxPool(2) → 24×24×64
Conv(128) + BN + ReLU → GlobalAvgPool → 128-dim 嵌入向量
FC: 128 → N+1（N 位已知人員 + 1 個 __unknown__ 類別）
```

## Code Specification

1. 開發語言：Python 3.13
2. UI library：customtkinter
3. 註解請用中文
4. Variable Naming：CamelCase 一致使用
5. Error Handling：所有 API 呼叫必須包含 try-except
6. Function 名稱使用英文，不要用中文

## Design Decisions

### Unknown 人臉處理（方案 B）
- 訓練時自動產生 `__unknown__` 類別（100 張合成樣本），無需外部資料集
- 合成樣本組成：50% 極端變形已知人臉 + 50% 隨機雜訊與幾何圖形
- 推論時雙重判斷：CNN 預測為 `__unknown__` **或** softmax 信心值 < 60% → 顯示 "Unknown"
- `listPersons()` 自動過濾 `__unknown__`，對 UI 透明

### 新增人員策略（Method B Fine-tune）
- 每人收集 60 張臉部 ROI，存入 `data/faces/{人名}/`
- 第 1 人：從頭訓練 50 epochs
- 第 N 人（N ≥ 2）：載入舊 backbone，擴展 FC 層，fine-tune 20 epochs（避免遺忘舊人）
- 資料增強：水平翻轉、亮度對比 ±20%、旋轉 ±10°、縮放 ±10%

### 訓練門檻
- 至少 1 位真實人員即可訓練（`__unknown__` 類別自動補足）

### 辨識設定
- 輸入尺寸：96×96 灰階（CNN_INPUT_SIZE）
- Unknown 門檻：60%（UNKNOWN_THRESHOLD）
- 學習收集張數：60 張（LEARN_TARGET_FRAMES）
- 學習逾時：90 秒（LEARN_TIMEOUT_SECONDS）

## 授權（商用安全）

| 套件 | 授權 |
|------|------|
| opencv-python | Apache 2.0 |
| PyTorch | BSD 3-Clause |
| customtkinter | MIT |
| Pillow | HPND |
| numpy | BSD |
| Haar Cascade XML | Apache 2.0（OpenCV 內建） |

## UI Layout

```
Row0: [Detect]   辨識結果標籤
Row1: [姓名輸入] [Learning] [Remove]  +「請保持正臉」提示（學習時顯示）
      剩餘時間進度條 / 已收集張數進度條（學習時顯示）
Row2: Webcam 畫面（人臉框 + 名字標籤）
Row3: Log 區域（訓練進度、操作記錄）
```

## 參考檔案

- `Refmain.py`：UI 設計參考藍圖（專案完成後由使用者自行刪除）
- `PLAN.md`：完整規劃文件

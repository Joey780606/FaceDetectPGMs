# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

MediaPipe 參考網址: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker

開啟電腦 Webcam，輸入人名後，找出 468 個 3D face landmarks（x, y, z 軸），
萃取 325 維臉部幾何特徵向量，交給單一 SVM 分類器做訓練後，再偵測人臉辨識來者是誰。

## Code Specification

1. 開發語言：Python 3.13
2. UI library：customtkinter
3. 註解請用中文
4. Variable Naming：CamelCase 一致使用
5. Error Handling：所有 API 呼叫必須包含 try-except
6. Function 名稱使用英文，不要用中文

## Design Decisions

1. **臉部寬度歸一化**：所有距離除以左右顴骨間距（臉部寬度），消除臉距鏡頭遠近造成的縮放干擾。
2. **Z-Score + L2 標準化**：消除「人類臉部拓撲相似」造成的干擾，使 SVM 特徵向量真正具有個人區分能力。
3. **辨識引擎（sklearn LinearSVC）**：
   - 多人（≥ 2 人）：OvR LinearSVC（liblinear 座標下降，保證收斂至全域最優，`svm_classifier_np.py`）
   - 單人（1 人）：最大 Cosine 相似度比對
   - C=500（對應 λ=0.001，大邊距），max_iter=2000
   - `class_weight='balanced'`：自動補償 Unknown. 樣本數少於已知人的不平衡問題
4. **姿態正規化（pose normalization）**：
   - `face_feature_3d.py` 以左右顴骨（X 軸）與下巴/額頭（Y 軸）建立旋轉矩陣 R
   - `R.T` 將所有 landmark 旋轉至正臉座標系，消除 Yaw / Pitch / Roll 影響
   - 訓練與偵測走同一條路，特徵一致
5. **單一 SVM**：姿態正規化後只需一個 SVM，所有角度樣本混合訓練。
   訓練時 `FrontalOnly=True`，只收正臉樣本。
6. **Unknown 偵測（四層）**：
   - 第一層：sigmoid 信心度閾值（低於閾值 → "Unknown"）
   - 第二層：margin 分差閾值（正臉時 top-1/top-2 分差小 → "Unknown"；側臉停用）
   - 第三層：餘弦驗證（`COSINE_VERIFY_THRESH`，預設 −1.0 關閉；query 與該人平均向量 cosine < 閾值 → "Unknown"）
   - 第四層：KNN 驗證（`KNN_VERIFY_ENABLED`，預設關閉）
7. **自訓練 Unknown 類別**：
   - 類別名稱固定為 `UNKNOWN_CLASS = "Unknown."` （末尾句點，與 sigmoid 拒絕的 "Unknown" 區別）
   - UI 提供 "Train Unknown." 按鈕，選擇圖片目錄批次訓練
8. **側臉閾值調整**：非正臉時 sigmoid 閾值 −0.1，margin 閾值強制設為 0.0（停用）
9. **時序穩定追蹤（StealEatStep）**：
   - 正臉辨識成功後暫存結果（_StableFace）
   - 後續非正臉 frame 若 bounding box IoU ≥ 0.35，沿用上次正臉結果
   - 連續 10 tick 無偵測後清除快取
   - `StealEatStep = True` 啟用，`False` 停用
10. **商用授權**：MediaPipe Apache 2.0、OpenCV Apache 2.0、NumPy BSD、scikit-learn BSD、CustomTkinter CC0、Pillow MIT，均可安全商用。

## UI
輸入人名，按 Learning 進行正臉學習，按 Detect 推論是誰。
Train Unknown. 按鈕：選擇圖片目錄，批次訓練 Unknown. 類別（負樣本）。
Remove 按鈕：移除指定人物的所有訓練資料。
Export 按鈕：將姓名欄指定人物的訓練資料匯出為獨立 .npz 檔（分散訓練用）。
Import & Merge 按鈕：多選個人 .npz 檔，合併進主模型並重訓儲存。
SVM 信心度閾值 slider（0.10~0.99）、分差閾值 slider（0.0~3.0）、
餘弦驗證閾值 slider（−1.0 關閉 ～ 0.8）可即時調整。

## Unknown. 訓練建議
- Unknown. 訓練樣本應涵蓋多人種、多年齡、多光線條件，越多元效果越好。
- Unknown. 樣本數建議接近「所有已知人樣本總數」，避免 OvR 類別不平衡。
  （已透過 class_weight='balanced' 自動補償，但資料仍越充足越好）
